# Fluxel RenderGraph Architecture

## Purpose and boundary

`fluxel-rendergraph` is the portable, in-frame planner for GPU work. An
application declares resources and pass accesses; the crate derives ordering,
validates the declaration against observed device facts, removes dead work,
and produces an immutable execution plan. Its purpose is to make resource
semantics—not incidental callback or command-recording order—the authority for
what a frame may do.

The crate owns logical resource versions, access ranges and semantics,
dependency and hazard analysis, root-driven culling, semantic state planning,
creation-time operation requirements, and the portable execution protocol. It
does not own native devices, native handles, memory allocation policy, barriers
as API-specific objects, command buffers, queue submission implementation,
readback implementation, asset loading, scene policy, shader compilation,
pipeline construction, descriptor allocation, or presentation. Those are
renderer and RHI concerns. The graph describes *what must be true*; an RHI
backend is responsible for safely making it true on a device.

## Public API boundary

Consumers use the deliberately selected crate-root API, for example
`fluxel_rendergraph::RenderGraph`, `TextureDesc`, and `FrameExecutor`.
Implementation modules such as `pass`, `compile`, and `backend` are private:
their layout is not a SemVer commitment, and their public types are re-exported
from the crate root where they form part of the supported contract. The sole
public module is `test_rhi`, a deterministic CPU-only test backend; it is not
evidence of native GPU correctness.

```text
Renderer: scene/frame policy, snapshots, opaque pipeline and binding recipes
                              |
                              v
RenderGraph: declarations -> validation -> immutable portable ExecutionPlan
                              |
                              v
RHI: native resources, lowering, barriers, command recording, submit, completion
```

Assets remain outside the graph: renderer code resolves an asset or cache entry
to a GPU-ready snapshot, then binds that snapshot to a graph import for the
frame. The rationale and rejected alternatives are in
[ADR-0001](adr/0001-assets-outside-rendergraph.md). Native handles and `unsafe`
are likewise deliberately contained by RHI; see
[ADR-0002](adr/0002-rhi-unsafe-containment.md).

## Declaration model

### Typed identities, versions, and pass-local authority

`TextureVersion` and `BufferVersion` describe immutable logical contents. A
read borrows a version. A write consumes a version and yields its successor.
This linear story prevents an implicit merge of competing writer branches:
two writers cannot consume the same whole-resource version, even if their
current ranges do not overlap. The compiler is still authoritative for stale,
foreign, and branching use, because ownership alone cannot prove every graph
construction error.

During setup, a pass builder also returns non-forgeable, pass-local handles:
`TextureRead`, `TextureWrite`, `TextureReadWrite`, and their buffer
counterparts. The matching execute callback receives a `PassResourceResolver`
that can resolve only those handles. A callback therefore cannot obtain an
arbitrary graph resource or invent a dependency after compilation. Opaque
`RasterPipelineId`, `ComputePipelineId`, and `BindingSetId` identify
renderer-owned objects, but never authorize undeclared GPU resource access.

Versions are whole-resource content lineage; access and validity are more
precise. `TextureRange` describes aspect/mip/layer subresources and
`BufferRange` describes bytes. A partial successor inherits untouched contents
and validity. `WriteCoverage::Full` proves initialization of the *declared
range*, not the entire resource; `Unknown` adds no initialization fact. A
read-write access must first read defined contents and then inherits valid
contents into its successor. This keeps state legality separate from the
stronger question of whether contents are known initialized.

### Pass declaration and retained recipes

`RenderGraph<F>` is a mutable authoring object. `add_raster_pass`,
`add_compute_pass`, and `add_copy_pass` each take:

1. a one-time setup closure, which declares accesses and returns downstream
   versions plus static retained data; and
2. a repeatable execute closure, which records commands later for one frame.

Retained data and execute closures are `Send + Sync + 'static`; compilation
never invokes an execute closure. `FrameInputs<F>` owns dynamic per-frame data
and import bindings. `CompiledGraph::instantiate_local` accepts owner-thread
frame data and carries a structurally non-`Send` marker, while `instantiate_send`
requires `F: Send + Sync`; neither choice weakens the retained-recipe requirement.
This division lets a compiled topology
be reused without putting acquired images, encoders, or mutable native state in
the compiler snapshot.

Raster attachment behavior is setup-time data. `LoadOp::Load`, `Clear`, and
`DontCare`, together with `StoreOp::Store` or `Discard`, contribute to content
validity and attachment state. A clear recorded later in an opaque callback
cannot retroactively establish the declaration's initialization contract.

`depends_on` exists only for a declared external-protocol or diagnostic edge
that no GPU resource can express. It constrains the pass DAG but contributes no
resource access, lifetime, state, or memory-ordering fact. Similarly,
`mark_side_effect` retains a pass only for a named non-resource observable
effect. Neither mechanism may hide a missing resource declaration.

### Imports, exports, and physical identity

Transients are logical graph declarations. Persistent objects enter through
`ImportTextureSlot` or `ImportBufferSlot`, whose contracts fix descriptor,
incoming state, caller ownership, and `InitialContents`. Ordinary imports reject
surface ownership; acquired surface textures use the dedicated surface import API.
A readable state does not imply defined contents, so state and initialization are
explicitly independent.
Concrete renderer-selected objects are bound per frame through opaque
`TextureBindingId` and `BufferBindingId` values.

Exports are roots with an explicit final state. `FrameExports` returns the
physical exported object, descriptor, caller-owned lease, and its
`outgoing_state`. A consumer such as an RHI readback helper must use that
reported state as its actual incoming state; it must not assume or silently
repair a convenient state. Surface imports and `present` are portable
declaration vocabulary. An acquired image enters through a one-shot opaque
presentation token; a retained surface must close to exactly one Present root,
and the executor transfers that token to backend submission only after the
final presentation transition is recorded. Native acquisition, swapchain
policy, and the actual Present call remain RHI responsibilities.

At frame resolution, `FrameResourceProvider` converts opaque IDs to
`BoundTexture`/`BoundBuffer`. The provider supplies a device identity, a
physical-generation identity, actual descriptor, actual incoming state,
actual allowed operations, and a lease. The executor checks device and
descriptor contracts, incoming state, complete binding coverage, and
`required_usage subset_of actual_usage`. Distinct live logical resources may
not resolve to the same physical generation because the current plan contains
no physical-alias model. Before invoking the provider or allocating a transient,
the executor first proves that every retained ordinary import has a frame binding;
these checks reject incomplete frame input without provider side effects and before
an encoder is opened.

Provider identity is physical, not a descriptive label: a recreated object has
a new generation even when it has the same descriptor and contents contract.
The provider owns persistent-object lifetime and supplies a lease for the
entire execution. The executor preserves the provider's actual incoming state;
an export similarly reports its actual outgoing state to the next consumer.
State is never reset merely because a resource crossed a frame boundary.

## Compilation

Compilation is deterministic planning, not execution:

```text
declared accesses and roots
        -> version/reference/initialization validation
        -> resource dependencies + explicit-order edges
        -> root-driven liveness and culling
        -> topological execution order
        -> capability validation and semantic transitions
        -> immutable CompiledGraph + ExecutionPlan + CompileReport
```

Dependencies arise from declared version use and hazards, not from the source
order of execute callbacks. Culling starts from exports, presentation targets,
and marked side effects; dead work cannot enlarge transient creation
requirements. `CompileReport` exposes culled passes, inferred dependencies,
retained explicit orders and side effects, and any safe capability fallback so
these decisions are inspectable rather than implicit.

Validation rejects stale or foreign versions, reads before initialization,
writer branches, conflicting accesses, invalid ranges, dependency cycles,
incomplete imports, unsupported semantic requirements, and invalid export or
present roots. Device capabilities must also assign each `QueueId` exactly once so
queue selection and backend submission refer to one unambiguous capability record.
`CompileError` and `DiagnosticContext` carry stable categories
and relevant pass/resource/slot/capability evidence. Recording errors are a
separate domain: undeclared pass handles, declared-use mismatch, absent frame
bindings, invalid command arguments, incompatible binding recipes, and
post-validation backend-object creation failure are `RecordingError` values.

### Immutable outputs and the capability fingerprint

`CompiledGraph` retains only the live pass recipes, resources, roots,
deterministic execution order, a normalized capability fingerprint, and one
`ExecutionPlan`. Later changes to `RenderGraph` cannot mutate that snapshot.
`ExecutionPlan` contains the selected logical queue, planned passes, semantic
transitions before each pass, final export transitions, raster attachment
descriptors, and per-live-resource `TextureUsage`/`BufferUsage` requirements.

`DeviceCapabilities` is a set of observed facts consumed by validation:
logical queues, recording, transition and synchronization models, transient
resource facts, limits, buffer support, texture-format support, and optional
surface facts. It is deliberately demand-driven rather than a speculative
feature checklist. Compilation canonicalizes those facts into a private
fingerprint. Execution compares the fingerprint with the backend's current
capabilities, preventing a plan compiled for one semantic device contract from
running on another.

The plan records semantic transitions, not API-specific barrier commands. An
equal before/after state is meaningful: overlapping writes can need a
same-state memory dependency even without a named state transition. Conversely,
consecutive reads and disjoint ranges do not manufacture such a dependency.
The RHI must preserve this semantic distinction when lowering; see
[ADR-0003](adr/0003-serial-execution-lowering.md).

## Execution protocol

`FrameExecutor<B>` is a serial, single-queue adapter around an
`ExecutionBackend`. For one `FrameExecution`, it:

1. verifies the compiled-graph identity and capability fingerprint;
2. resolves every live import, allocates live transients, and validates leases,
   usage, device identity, state, descriptor, and non-aliasing contracts;
3. begins one encoder on the plan's logical queue;
4. emits planned transitions, opens the appropriate pass scope, and invokes
   retained callbacks in deterministic plan order;
5. emits final export transitions, finishes the encoder, and submits it; and
6. returns `ExecutedFrame`, which combines exports with a `FrameSubmission`.

The current serial lowering is a correctness baseline, not a claim that logical
passes map one-to-one to native encoders, command buffers, recording jobs, or
hardware queues. Multi-queue scheduling, parallel recording, aliasing,
recording caches, and GPU-performance claims remain future lowering choices
that must preserve the portable plan and be justified by measurement.

### Private transient reuse

The executor may retain completed transient allocations for a compatible later
instantiation. This is private allocation policy, not a physical-alias model or
a caller-visible resource cache. A reusable slot is segregated by device
identity, compiled-graph generation, and logical resource identity. It is
eligible only where the executor can carry one exact whole-resource state from
the prior execution. The first transition of a reused resource begins at that
remembered state, not at fabricated `Undefined`.

Resources with partial or mixed subresource state do not enter this pool until
the execution contract can represent their complete physical state history.
Graph or device invalidation removes a slot from future checkout but marks it
for retirement; pending and accepted-unknown work remains quarantined with its
lease. Thus invalidating a graph/device generation cannot free old physical
resources before their submission reaches a known terminal outcome. `TestRhi`
persists physical state for this protocol check, but does not establish native
GPU correctness.

`ExecutionBackend` is the narrow backend SPI. It owns transient allocation,
encoder/pass lifecycle, planned transition lowering, opaque pipeline/binding
selection, draw/dispatch/copy commands, submit, completion polling, and
retirement. `RenderObjectProvider` resolves opaque pipeline and binding IDs
against graph-authorized physical ranges and semantics. The graph does not
parse WGSL, create native pipelines, choose descriptors, or expose a general
pipeline API. The choice to keep proven combinations closed is recorded in
[ADR-0006](adr/0006-no-general-pipeline-yet.md).

### Completion, leases, re-entrancy, and failure

Submission acceptance and completion are distinct. `CompletionStatus` is
`Pending`, `Unknown`, `Complete`, or a structured terminal `Failed` state. A
backend may return submit `Err` only when it knows no work was accepted. An
`Unknown` status means acceptance or completion cannot yet be proved; it is
not terminal and retains the same leases as `Pending`. Every future/unrecognized
nonterminal status is treated conservatively until retirement is safe. This
rule is essential for native lifetime safety; its rationale is in
[ADR-0004](adr/0004-accepted-unknown-quarantine.md) and the reuse implications
are recorded in [ADR-0009](adr/0009-resource-floor-and-reuse-safety.md).

`FrameSubmission` owns executor-held leases until terminal completion. Dropping
a pending submission is non-blocking: it moves its completion and leases to an
executor retirement inbox, which the next executor operation transfers to the
backend's retirement queue. Caller-owned export leases are separate and remain
valid independently. Executor operations use non-blocking backend acquisition;
re-entrant use returns `ExecutionError::ExecutorBusy` rather than waiting while
a backend lock is held. `ExecutionError` keeps frame-binding, recording,
capability, wrong-graph, unsupported-execution, executor-busy, and backend
failures distinguishable.

## Reference backend and evidence

`TestRhi` implements the same `ExecutionBackend` and provider contracts with a
deterministic CPU-only registry, trace, leases, injected failures, and manually
advanced completion. It proves graph/executor protocol properties: pass order,
transitions, resource and binding checks, submission classification, and
completion-based retirement. It does not execute shaders, emulate GPU memory,
validate native API calls, establish API-specific barriers, or measure GPU
performance.

Evidence therefore has explicit levels:

- Compiler and validation tests prove portable declaration and diagnostic
  rules.
- `TestRhi` tests prove execution-protocol and lifetime state machines.
- RHI unit/negative tests prove safe native-boundary rejection paths.
- Native conformance runs execute the same immutable `ExecutionPlan` on DX12
  and Vulkan with required validation, a CPU oracle, recorded input/output,
  completion, export state, and diagnostics.

Only the final category supports a native GPU correctness claim. Compile-only,
mock, or one-backend success never substitutes for it; see
[ADR-0005](adr/0005-gpu-conformance-evidence.md). Platform-specific paths also
require native platform gates rather than cross-compilation assumptions; see
[ADR-0008](adr/0008-native-platform-test-gates.md).

## Module map

```text
access.rs / handles.rs / resource.rs / rhi.rs
    Portable vocabulary: ranges, uses, versions, contracts, capabilities/states.
graph.rs / pass/ / recipe.rs
    Mutable authoring plus separated pass vocabulary, builders, commands and resolvers.
internal/
    Private declarations, compiler model and retained callback adapters.
compile/
    Dependency derivation and orchestration, with reference/range/conflict validation leaves.
plan/
    Backend-neutral execution-plan and usage-requirement construction.
execution/
    Frame instantiation, resolution, transition/pass orchestration, exports and retirement.
backend/
    ExecutionBackend SPI plus provider, physical-resource, and execution-error contracts.
test_rhi/
    Deterministic CPU protocol reference backend and inspection fixtures.
```

This decomposition intentionally keeps the core crate platform-neutral and
free of a production GPU runtime dependency. The public semantic center is
declaration and compilation; the execution SPI is deliberately narrow while
native integration evolves. Historical alternatives, trade-offs, and durable
constraints belong in [the ADR index](adr/README.md), not in this current-state
design description.

## Related documents

- [RenderGraph crate guide](../crates/rendergraph/README.md)
- [RHI architecture](design-rhi.md)
- [Renderer architecture](design-renderer.md)
- [Architecture Decision Records](adr/README.md)
