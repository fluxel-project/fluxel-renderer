# Fluxel Rendering workspace architecture

Fluxel Rendering is a layered Rust workspace for turning renderer-selected
scene data into portable GPU work, then executing that work through a small,
safe native boundary. The layers are deliberately separate: the renderer
chooses *what a frame means*, RenderGraph derives *what work and ordering that
meaning requires*, and RHI performs *how a selected backend owns and executes
the work*.

This document is the architectural entry point for the workspace. It describes
the current system as a whole and its intended extension boundaries. It does
not replace the crate designs, which define each layer in detail, or the ADRs,
which record why durable choices were made.

## Goals and non-goals

The workspace is designed to make GPU correctness inspectable rather than an
accident of callback order or one backend's behavior. Its central goals are:

- portable, explicit resource-access and synchronization semantics;
- safe ownership of device-affine native objects across asynchronous GPU work;
- one immutable execution plan that can be lowered by more than one backend;
- renderer policy that stays separate from resource planning and native API
  details; and
- evidence that distinguishes CPU protocol tests, compile checks, and real GPU
  conformance.

It is not a general graphics API, a scene/asset database, a shader authoring
framework, or a presentation runtime. In particular, there is currently no
general pipeline or bind-group builder, general shader reflection API,
cross-platform surface abstraction, multi-queue scheduler, transient aliasing
implementation, or stable asset/resource handle and cache ABI. The proven
Windows presentation slice supports DX12 and Vulkan through one narrow RHI
surface façade. It handles resize/minimize/restore as generation changes and
independent acquired-frame tickets, while the harness privately proves bounded
frames-in-flight. It is not a general platform API or public frame scheduler.

## Layer model and dependency direction

```text
application / asset system
        |
        v
fluxel-renderer  ---->  fluxel-rendergraph  ---->  fluxel-rhi
 scene policy             portable plan              native execution
```

Dependencies point downward. The renderer may use RenderGraph declarations and
RHI's safe opaque objects; RenderGraph does not depend on RHI or renderer; RHI
implements the execution SPI defined by RenderGraph. Native HAL types never
travel upward, and renderer concepts such as assets, materials, or visibility
never become graph concepts.

| Layer | Owns | Explicitly does not own |
| --- | --- | --- |
| `fluxel-renderer` | Domain inputs, renderer policy, GPU snapshot publication, fixed frame coordination, closed recipes, and the visible fixed-frame transaction | Asset loading/cache identity, graph compilation, native handles, barriers, swapchains, or host/window policy |
| `fluxel-rendergraph` | Logical resources and versions, declared accesses, validation, dependencies, culling, transitions, and immutable execution plans | Scenes, asset handles, shader/pipeline policy, allocation, native handles, queue submission, or readback implementation |
| `fluxel-rhi` | Device-affine native resources, opaque artifacts/bindings, backend lowering, command recording, submission, completion, native diagnostics, and surface/swapchain presentation | Scene selection, asset policy, host/window ownership, general renderer lowering, or a public general graphics API |

Assets cross a repository boundary without moving platform or GPU policy into
one shared crate. `fluxel-bases` owns durable identity, typed handles,
generations, loading state, caching, and reuse contracts. Platform readers live
in `fluxel-host` or `fluxel-jsbridge`; rendering-side adapters own GPU upload,
residency, recreation, and frame-safe retirement. The renderer resolves an
appropriate GPU-ready snapshot for a frame. RenderGraph receives only the
resulting physical binding and its contract. The reason for this boundary is
recorded in [ADR-0001](adr/0001-assets-outside-rendergraph.md). The native
containment boundary is recorded in
[ADR-0002](adr/0002-rhi-unsafe-containment.md).

For layer-specific contracts, see [Renderer design](design-renderer.md),
[RenderGraph design](design-rendergraph.md), and [RHI design](design-rhi.md).

## Frame data flow

The intended frame path has a stable conceptual shape:

```text
application scene/domain data
  -> renderer resolves ready GPU snapshots and frame policy
  -> ordered render packet or closed fixed recipe
  -> acquire one RHI presentable image when presentation is requested
  -> RenderGraph declarations
  -> compiled immutable ExecutionPlan
  -> RHI frame resolution and backend lowering
  -> one serial queue submission
  -> completion, export state, presentation, and retirement
```

### Renderer-side selection

Application code supplies domain data such as cameras, geometry, materials, and
an insertion-ordered `DrawList`. A renderer decides which snapshot generation,
material behavior, and ordering policy are legal for the frame. Persistent CPU
data is validated before it enters the asynchronous GPU path.

The current renderer has deliberately narrow fixed paths: immutable
indexed mesh, texture, normal, and vertex-color snapshots are published only
after their uploads complete, and a fixed-frame coordinator selects one of a
small set of private, closed raster recipes. A recipe jointly specifies the
vertex/texture domain, graph access declarations, RHI kernel and bindings,
reservation topology, and exports. The vertex-color recipe is a representative
closed multi-stream contract: position `f32x3` in slot zero, linear
`UNORM8x4` color in slot one, `u32` indices, and the camera/tint uniform all
move together from immutable snapshot through graph declaration to RHI
recording. It is an internal consistency mechanism, not a configurable material
or pipeline API; see [ADR-0007](adr/0007-closed-fixed-renderer-recipes.md).

For the legacy unlit indexed contract, the renderer lowers a `DrawList` and a
positionally matching sequence of ready indexed snapshots into an owned,
opaque, device-affine `RenderPacket`. Construction copies camera/material
values and each draw's affine model-to-world placement, retains snapshot
leases, and validates exact CPU snapshot metadata; it does not reserve a
generation or expose graph/native objects. The renderer computes column-major
`projection * view * model` into the existing legacy-unlit uniform ABI, so
placement is draw/packet policy rather than a mesh property or an RHI binding
change. Submission deduplicates repeated snapshot generations for graph imports
and reservations, but retains one ordered draw and uniform per list entry. The
real CPU preparation dependencies (shared scene input, per-object work, and
ordered assembly) are privately represented with `slot-graph`; it neither
creates dummy work nor replaces RenderGraph's GPU semantics. It compiles one
graph with one raster pass, advances uniform uploads serially without blocking,
then submits the raster work once. This placement currently
does not extend Lambert normal handling. General material/layout variation,
PBR, scene loading, culling, batching, and a general persistent scene system are
not implemented; the current deterministic three-object scene is only a closed
presentation and preparation proof.

### RenderGraph declaration and compilation

The renderer declares passes rather than recording opaque, unordered native
work. Each pass states which version and range of each logical buffer or
texture it reads, writes, samples, attaches, copies, or accesses read/write.
From those declarations RenderGraph derives the dependency DAG, validity and
initialization checks, dead-pass culling, required usage, and state/memory
transition requirements.

Compilation produces an immutable `ExecutionPlan`-backed graph snapshot. The
snapshot contains portable semantics, not an encoder, command buffer, queue,
or native allocation. Imports are stable slots; each frame binds concrete
resources to them. Exports name roots and carry an outgoing-state contract for
the next graph or external consumer. More detail is in the
[RenderGraph design](design-rendergraph.md).

An import binding identifies a provider-selected physical object and generation,
its actual incoming state, allowed usage, and a completion-safe lease. The
executor checks these facts; an export returns its actual outgoing state rather
than a guessed default. Compatible compiled graphs may privately reuse a
completed transient allocation, but only for exact whole-resource state
carry-over. Reuse is segregated by device and compiled-graph generation; graph
or device invalidation prevents a new checkout while pending or unknown work
continues to retain its old-generation lease. It is not public aliasing or a
caller-visible cache. See [ADR-0009](adr/0009-resource-floor-and-reuse-safety.md).

### RHI resolution, execution, and completion

RHI resolves plan resources against opaque native resources, verifies device
identity, descriptors, incoming state, and actual allowed usage, then lowers
only declared commands. The present implementation uses a serial, one-queue
correctness path. The execution boundary records transitions/order, commands,
submission, and completion while retaining every lease required by native work.

An export reports the actual outgoing state. Readback or a later consumer must
use that state as its incoming state; it must not silently substitute a more
convenient state. Submission is not treated as completion. A known rejection,
accepted-but-unknown submission, successful completion, and terminal failure
are distinct states with different ownership consequences. Unknown accepted
work is quarantined rather than releasing objects or publishing guessed state;
the same conservative rule applies when a completion query itself is `Unknown`.
See [ADR-0004](adr/0004-accepted-unknown-quarantine.md).

The serial lowering is a correctness strategy, not a claim that graph passes
are tied to one encoder, command buffer, or queue. Multi-queue scheduling,
parallel recording, command caching, and aliasing remain possible future
optimizations only after they have a concrete measured benefit and preserve the
same portable plan semantics. See [ADR-0003](adr/0003-serial-execution-lowering.md).

## Lifetime and ownership model

The workspace keeps three lifetimes separate.

| Lifetime | Examples | Owner and rule |
| --- | --- | --- |
| Persistent domain lifetime | Geometry, material inputs, asset identity, renderer caches | Application/asset/renderer policy; not graph state |
| Per-frame logical lifetime | Graph versions, imports, exports, frame inputs, retained pass recipes | RenderGraph declaration/instantiation; a compiled plan contains no native per-frame object |
| Native asynchronous lifetime | Device, buffers, textures, pipelines, bindings, staging data, command objects, leases, completion handles | RHI; objects survive until the relevant work is proven retired |

An immutable GPU snapshot is the bridge between the first and third rows. It
encapsulates a concrete native generation plus leases and reported state, but
does not expose raw native handles. Publication is atomic across the resources
that make up the snapshot. An owned `RenderPacket` bridges the persistent
borrowed draw-list input and one graph execution: it is device-affine, owns
the ready snapshot leases and serialized uniforms, but owns neither a native
reservation nor a native command. Submission reserves each unique generation
as one transaction. Before raster acceptance, failure and drop release the
reservations; after unproven raster acceptance, the generations are poisoned
rather than reused with unknown native state.

RHI resources are device-affine and retain the opened native device through
shared ownership. Cloning a lease extends the resource/native-device lifetime;
the final owner performs destruction. All unsafe and HAL interaction remain in
the private RHI implementation subtree, so higher layers cannot accidentally
outlive, alias, or misuse a raw native object.

## Portable semantics and backend facts

RenderGraph describes portable meaning: resource operations, ranges, ordering,
initial contents, required usages, and exported final states. It does not
pretend that portable states are direct DX12 or Vulkan enumerations.

RHI reports backend facts separately: adapter identity, limits, supported
features, actual allowed resource usage after native normalization, and
validation availability. During frame resolution, the portable requirement
must be a subset of the resource's actual allowed operations. This prevents an
allocator or import binding from merely echoing what the plan requested.

The same `ExecutionPlan` is intended to execute with the same observable
semantics on supported backends. Backend lowering may use different barriers,
resource flags, encoders, or state representations, and a same-state memory
dependency need not imply an identical native barrier. Those are
implementation facts as long as ordering, visibility, lifetime, and results
match the plan contract.

## Validation, errors, and evidence

Validation is layered rather than collapsed into one “it worked” result:

1. **Domain validation** rejects invalid scene, geometry, image, uniform, and
   fixed-recipe inputs before a GPU operation begins.
2. **Graph validation** rejects invalid versions, ranges, initialization,
   capability requirements, resource conflicts, and undeclared recording.
3. **RHI boundary validation** rejects foreign devices, invalid descriptors,
   usage/state/alignment violations, unavailable backends, and unsupported
   native prerequisites before unsafe calls.
4. **Completion validation** distinguishes rejection, pending work, proven
   completion, and unproven/failed accepted work without fabricating state.
5. **Native diagnostics and conformance** collect backend validation output
   and compare hardware readback against an independent CPU oracle.

Errors are structured at the layer that owns the failed contract. Renderer
errors never need to expose native handles; graph errors explain declaration
semantics; RHI errors identify native/open/resource/recording/submission or
completion boundaries without leaking unsafe types.

CPU-only tests and `TestRhi` prove compiler and protocol behavior. Compilation,
linking, and CI prove build coverage. Neither proves native GPU correctness.
A native conformance claim requires the same compiled plan on each supported
backend, required validation where available, collected diagnostics, exact
inputs/outputs, and an independent CPU oracle. The evidence policy is in
[ADR-0005](adr/0005-gpu-conformance-evidence.md).

## Repository and module map

```text
crates/
  renderer/       application-facing domain types, snapshots, packets, fixed-frame policy
    src/upload/   immutable snapshot upload/publication domains
    src/fixed_frame/ closed recipes, owned packets, and fixed-frame submissions
    src/shader/   private fixed shader sources/selection
  rendergraph/    portable graph declaration, validation, compiler, execution SPI
    src/compile/  dependency, validation, culling, transition-plan compilation
    src/execution/ frame instantiation and portable execution protocol
    src/plan/     immutable plan data and contracts
    src/test_rhi/  deterministic CPU-only execution-protocol backend
  rhi/            safe native resource and execution facade
    src/resource/ owned resources, uploads, leases, fixed artifacts
    src/execution/ plan providers, fixed command recording, completion helpers
    src/imp/      private HAL/native implementation and platform stubs
documents/
  design-*.md     current-state layer designs
  adr/            durable architectural decisions and alternatives
  draft/          local, uncommitted plan/review working material
```

Each source module has one clear responsibility. Composition modules expose
only declarations, narrow shared contracts, and re-exports; implementation is
split by independently changing concerns. This preserves the `imp` subtree as
the sole unsafe/native containment boundary and keeps a public facade stable
while internal implementation evolves.

## Platform boundary

The portable graph and default renderer-domain model are not tied to a native
API. Native execution is Windows-focused, with headless DX12/Vulkan feature
selection and a backend-neutral DX12/Vulkan surface-generation/ticket path. The
Windows proof harness owns its bounded three-slot admission and back-pressure
policy; RHI only guards native image availability and completion-driven teardown.
Non-Windows native requests fail explicitly rather than silently emulating a
backend. Lost native surface/device recovery and a general cross-platform
surface API remain absent. The Stage 2.1 WebGL2 and Stage 2.2 WebGPU paths are
closed browser adapters, not a general web renderer; DOM lifecycle and RAF
ownership remain in `fluxel-jsbridge`. The common fixed resource floor is
available on DX12, Vulkan, WebGPU, and WebGL2. Compute/storage-buffer and
writable storage-texture recipes are present on DX12, Vulkan, and WebGPU;
readable storage texture is currently Vulkan-only. DX12 reports its observed
read limitation and fails closed; WebGPU read is not promised. WebGL2 rejects
compute and every storage operation with structured capability evidence before
context/resource side effects, with no emulation. WebGPU separately owns a
private device-generation/canvas-epoch state machine, opaque per-key resource
registry, ticket-held leases, asynchronous recovery, and terminal disposal;
that browser-only contract does not broaden native Surface.

Windows MSVC is the primary Windows development/native test environment. Linux
must be tested natively (for example in WSL2/Ubuntu), because it exercises
`cfg(not(windows))`, fallback, and feature paths that a Windows cross-build
cannot prove. When Android support is introduced, it requires an explicit NDK
target gate and executable emulator/device coverage for platform code. Platform
stubs must return their documented structured errors, not merely compile. The
rationale and testing rule are in [ADR-0008](adr/0008-native-platform-test-gates.md).

## Extension principles

New work should extend the lowest layer that naturally owns the new fact and
must not use a convenient lower-level escape hatch to bypass an existing
contract.

- Add scene/asset policy, snapshot selection, transforms, materials, and
  culling in the renderer; resolve assets before graph binding.
- Add portable resource-access semantics, compiler validation, and immutable
  plan contracts in RenderGraph; do not add native handles, barriers, or queue
  operations there.
- Add backend resource creation, artifacts, recording, synchronization,
  submission, and completion in RHI; keep raw native types and `unsafe`
  private.
- Add a closed recipe when a vertical slice proves a new combination. Do not
  turn a single proven combination into a general pipeline/material API without
  evidence for its ownership, reflection, layout, and portability contracts;
  see [ADR-0006](adr/0006-no-general-pipeline-yet.md).
- Treat performance work as a lowering optimization with benchmark and profiling
  evidence, never as a change to portable graph meaning.
- Record a short ADR when a decision will constrain more than the immediate
  implementation; keep current behavior in a design document and release
  investigation in local plan/review material.

This separation lets the rendering kernel grow from fixed headless
evidence-backed slices toward a general embeddable renderer without making
application code depend on backend accidents or transient implementation
policy.
