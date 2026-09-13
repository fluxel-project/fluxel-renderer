# Fluxel Renderer design

## Purpose

`fluxel-renderer` is the application-facing renderer layer in the
Fluxel workspace. It owns renderer policy: it accepts typed scene-domain data,
coordinates immutable GPU snapshots, lowers supported scene input into owned
packets, selects proven raster contracts, and declares the corresponding frame
graph. It never exposes native handles to its callers.

The crate provides portable domain objects (`Camera`, `Geometry`, `Mesh`,
`BasicMaterial`, and insertion-ordered `DrawList`); opt-in non-blocking uploads
that publish immutable mesh and texture snapshots after GPU completion; an
owned `RenderPacket` for ordered legacy unlit indexed draws; and seven closed
single-draw indexed contracts which yield opaque offscreen `FrameImage`
metadata after completion. The visible slice reuses the legacy-unlit
camera/material recipe for one acquired presentable image on DX12 or Vulkan;
an ordered packet may contain the same deterministic multi-object scene. The default crate has no graphics-backend dependency;
GPU coordination and fixed drawing require the `gpu-upload` feature.

This document is a description of the current system, not a release history.
The reasons for durable architectural choices are in the
[ADRs](adr/README.md).

## Layer boundaries

```text
application
    | Camera, Geometry, Mesh, BasicMaterial, upload inputs
    v
fluxel-renderer
    | snapshot lifetime, fixed draw policy, graph declaration
    v
fluxel-rendergraph
    | validate declarations, derive dependencies/transitions, immutable plan
    v
fluxel-rhi
    | resources, native recording/submission/completion, DX12/Vulkan
    v
native GPU API
```

The renderer chooses *what* one fixed frame means: it creates
renderer-owned declarations, imports ready snapshots, binds a closed RHI
artifact, and translates completion into renderer-visible status.

RenderGraph owns portable resource-access semantics. It validates declarations,
derives dependencies and state transitions, culls unused work, and creates an
immutable `ExecutionPlan`. It does not own assets, materials, native objects,
barriers, submission, or readback implementation.

RHI owns the native boundary: allocation, resource leases, native recording,
one-queue submission, completion observation, and all HAL/unsafe code.
Renderer code sees only safe opaque RHI types and the explicitly provisional
`adapter::fixed_artifacts::RasterBackend`/`RasterKernel` contract.

Persistent asset identity, loading, cache eviction, and hot reload are outside
the graph. The 0.14 renderer-private residency layer may resolve a fixed
renderer GPU generation before frame construction, but it is not a graph node
or graph resource owner. See [ADR-0001](adr/0001-assets-outside-rendergraph.md)
and [ADR-0010](adr/0010-renderer-private-fixed-asset-residency.md).

## Non-goals and current limits

The current renderer intentionally has no:

- stable public asset/resource handles, a general asset cache, or a generic
  resource-residency protocol;
- arbitrary shaders, reflection, pipeline layouts, bindings, vertex layouts,
  samplers, views, or material parameters;
- transformed-normal Lambert shading, lights, depth, blending, culling,
  batching, instancing, PBR, animation, or scene-file loading;
- surface acquisition, swapchain policy, resize, or lost-surface recovery; or
- multi-queue scheduling, parallel recording, aliasing, or performance policy.

`slot-graph` is a renderer-private CPU preparation implementation detail for
the real per-object dependency DAG: shared scene input fans out to object
preparation and fans back in to ordered assembly. It does not model GPU
dependencies, submission, completion, or synchronization. `async-runtime` is
not a renderer dependency; host-native scheduling and shutdown policy remain
outside this crate.

The fixed API is a correctness vertical slice, not an early general pipeline
API. See [ADR-0006](adr/0006-no-general-pipeline-yet.md) and
[ADR-0007](adr/0007-closed-fixed-renderer-recipes.md).

## Module map

```text
crates/renderer/src/
  lib.rs                 public domain model and feature-gated facade
  shader/                private stage-selection boundary
  frame_uniform.rs       fixed camera/material uniform ABI
  upload/
    indexed.rs           position/index snapshot upload and publication
    textured.rs          position/index/UV snapshot upload and publication
    normal.rs            position/index/normal snapshot upload and publication
    vertex_color/        position/index/RGBA8-color domain and upload lifecycle
    texture/             separate linear-UNORM and sRGB texture domains
    shared.rs            generation allocation and snapshot-use gate
  fixed_frame/
    recipe.rs            seven closed renderer-to-RHI raster mappings
    renderer/            public draw facade, validation and transaction startup
    graph.rs             closed-recipe graph declarations and export contracts
    provider.rs          snapshot-to-graph import-slot binding adapter
    submission.rs        two-phase submission/completion lifecycle
    ids.rs               private fixed pipeline and binding identifiers
    packet/              owned legacy-unlit draw-list packet lowering and lifecycle
    recipe/tests/        closed recipe mapping contract tests
    tests/               CPU contracts plus Windows native conformance suites
  tests/                 crate-level domain and facade contract tests
```

The `upload/tests/` tree groups publication, gate, geometry, payload, and
texture contracts without moving private-contract tests outside their owning
module. `fixed_frame/tests/` similarly separates clip validation, the legacy
unlit fixture, and the fixed raster conformance suites. Each module has one
responsibility. The `mod.rs` files are facades that retain public paths rather
than accumulating implementation logic. `shader` is a private boundary, not a
public shader system: it keeps stage selection out of the public domain API
until a supported renderer shader contract exists.

## Domain model and public API boundary

`Camera` stores column-major view and projection matrices. `BasicMaterial`
stores a linear RGBA base color. `Geometry` owns CPU `f32x3` positions and may
own validated `u32` indices; `Mesh` pairs one geometry with one basic material.
Fields are private, so constructors and accessors, not struct literals, define
the supported contract.

`DrawList<'a>` borrows a camera and meshes for one preparation scope and
preserves insertion order. Each `DrawItem` owns a finite affine
column-major `ModelTransform` from model space to world space; `push` supplies
identity and `push_transformed` supplies an explicit placement. The transform
belongs to an ordered draw rather than `Mesh`, allowing one mesh to be reused
at different placements without changing immutable geometry/material data. For
the legacy unlit indexed recipe,
`FixedFrameRenderer::lower_draw_list` accepts a positionally corresponding
slice of ready `IndexedMeshSnapshot` values and creates an opaque owned,
non-`Clone`, device-affine `RenderPacket`. It compares positions by float bit
pattern and indices exactly, then retains snapshot leases and serialized
per-draw uniforms. The packet holds no graph slot, RHI/native handle, or
reservation, so construction and drop do not change snapshot gates. This is a
narrow transition model, not a stable asset-handle renderer ABI; it does not
sort, cull, or synthesize material policy.

For packet construction, preparation validates each object, computes its
`projection * view * model` uniform, and preserves insertion order. The private
`slot-graph` DAG expresses these actual CPU dependencies rather than inventing
tasks to demonstrate the dependency library; no node identity or graph API
crosses the renderer boundary.

With `gpu-upload`, the crate also exposes opaque-ready resource families:

| Input domain | Upload operation | Ready snapshot | Fixed stream ABI |
| --- | --- | --- | --- |
| indexed geometry | `IndexedMeshUpload` | `IndexedMeshSnapshot` | position `f32x3`, index `u32` |
| indexed geometry plus UV | `TexturedIndexedMeshUpload` | `TexturedIndexedMeshSnapshot` | position `f32x3`, index `u32`, UV `f32x2` |
| indexed geometry plus normals | `NormalIndexedMeshUpload` | `NormalIndexedMeshSnapshot` | position `f32x3`, index `u32`, normal `f32x3` |
| indexed geometry plus vertex colors | `VertexColorIndexedMeshUpload` | `VertexColorIndexedMeshSnapshot` | position `f32x3`, linear color `UNORM8x4`, index `u32` |
| linear RGBA8 image | `BaseColorTextureUpload` | `BaseColorTextureSnapshot` | `Rgba8Unorm` |
| sRGB RGBA8 image | `SrgbBaseColorTextureUpload` | `SrgbBaseColorTextureSnapshot` | `Rgba8UnormSrgb` |

`TexturedBasicMaterial` accepts only a linear texture snapshot, while
`SrgbTexturedBasicMaterial` accepts only an sRGB snapshot. That type split
prevents a draw from silently changing a texture's color-space meaning. Native
buffers, textures, leases, imported graph handles, and RHI bindings remain
crate-private.

Start and terminal errors are structured enums. `DrawStartError` retains graph,
pipeline, provider, and upload error types rather than formatting them into
strings; `FixedFrameFailure` similarly retains structured completion and
execution causes. Upload
start errors report rejection before work is accepted; `FixedFrameFailure` and
upload failure enums retain completion, observation, and later-start causes
rather than flattening them into strings.

## Immutable uploads and snapshot ownership

An upload begins native immutable uploads without a host wait. `poll()` observes
each accepted upload non-blockingly and returns `Pending`, `Ready`, or
structured `Failed`. A ready snapshot is cloneable and contains an opaque
generation, immutable RHI result(s), CPU metadata used by fixed validation, and
a shared use gate.

Publication is atomic at the renderer level:

- indexed mesh: both position and index uploads complete;
- textured mesh: position, index, and UV uploads complete;
- normal mesh: position, index, and normal uploads complete; and
- vertex-color mesh: position, color, and index uploads complete; and
- texture: its upload completes.

If a later upload cannot start after an earlier one is accepted, the operation
fails but continues retaining and observing accepted siblings. A partial
generation is never published and accepted RHI work is never dropped early.
Role-specific stream slots remain typed because position, index, UV, and normal
failures have distinct meanings.

Every snapshot generation has one `SnapshotUseGate`, shared by clones:

```text
Ready --reserve--> InFlight --proven complete--> Ready
                         \--uncertain/drop-----> Poisoned
```

Only one fixed draw can use a generation at once. A graph/native rejection
before acceptance releases its reservation; proven completion also releases it.
An accepted submission whose final outgoing state cannot be proven, or an
accepted submission dropped before proof, permanently poisons that generation.
This fail-closed rule prevents later work from assuming a native resource state
which may be unknown; see [ADR-0004](adr/0004-accepted-unknown-quarantine.md).

The generation counter is process-local opaque identity, not an asset handle or
persistence protocol. This crate does not implement hot reload, eviction, or
cross-frame asset selection.

## Fixed raster contracts

`FixedFrameRenderer` owns a clone of a safe `Device`, a capability snapshot,
and a `FrameExecutor<RasterBackend>`. It compiles graphs against capabilities
from that same executor/device. A draw start checks extent, device identity,
index count, finite camera/material data, and its recipe preconditions before
work is accepted.

Private `RasterRecipe` is the sole mapping point for seven audited contracts. It
has private fields and exactly seven associated constants. A recipe atomically
selects the RHI `RasterKernel`, fixed pipeline/binding IDs, vertex ABI, texture
interpretation, graph shape, snapshot domain, reservation topology, and export
topology. Callers cannot manufacture a new combination by mixing optional
resources, shaders, layouts, or bindings.

| Public start method | Vertex input | Texture operation | Shading result |
| --- | --- | --- | --- |
| `draw` | position, index | none | uniform base color |
| `draw_textured` | position, index | position-derived clamped mip-zero integer `textureLoad` | texture × base color |
| `draw_textured_uv` | position, index, UV | explicit-UV clamped mip-zero integer `textureLoad` | texture × base color |
| `draw_textured_uv_linear_clamp` | position, index, UV | linear UNORM, linear min/mag, nearest mip, clamp-to-edge, level zero | texture × base color |
| `draw_textured_uv_linear_clamp_srgb` | position, index, UV | sRGB view with the same fixed sampler policy | decoded/filter result × base color |
| `draw_lambert` | position, index, normal | none | base color RGB × fixed `+Z` Lambert factor |
| `draw_vertex_color` | position, index, linear `UNORM8x4` color | none | perspective-interpolated vertex color × linear tint |

All paths are indexed triangle draws. The fixed renderer does not expose a
sampler, LOD, wrap mode, filter choice, vertex-slot selection, shader source,
or pipeline-state selector. Provider/binding maps are private graph details and
are checked against their selected recipe.

## Frame declaration and execution flow

```text
ready snapshot(s) + Camera + material + extent
  -> validate and reserve every required generation
  -> serialize fixed uniform and build a graph declaration
  -> compile immutable ExecutionPlan against executor capabilities
  -> submit uniform upload (phase 1, non-blocking)
  -> when it completes, execute raster graph (phase 2, non-blocking)
  -> poll completion
  -> release reservations + FrameImage, or retain terminal failure
```

The graph imports immutable streams with their reported `CopyDestination`
outgoing state, declares the recipe's vertex/index/uniform reads and, where
required, its sampled texture read, creates a linear offscreen `Rgba8Unorm`
target, and exports that target for copy-source readback. Imported snapshots
are exported back in their declared outgoing state. The vertex-color recipe
imports three streams—position, color, and index—and restores all three to
`CopyDestination`; its color stream is a vertex input, not a texture or
material binding. RenderGraph reports portable semantics; RHI records
transitions and fixed native commands from that plan.

Uniform upload and raster execution are separate accepted operations.
`FixedFrameSubmission::poll` does not block the CPU and starts raster only
after uniform completion succeeds. `FrameImage` appears only after raster
completion and reveals metadata, not a host mapping or native texture.
CPU-oracle readback is RHI test support, never a production renderer API.

Queue synchronization, encoders, barriers, leases, fences, and validation
capture remain RHI work. See [ADR-0002](adr/0002-rhi-unsafe-containment.md) and
[ADR-0003](adr/0003-serial-execution-lowering.md).

### Ordered packet flow

```text
DrawList + matching ready indexed snapshots
  -> validate and copy into owned RenderPacket
  -> reserve each unique generation once
  -> compile one graph with one legacy-unlit raster pass
  -> upload each draw uniform serially, without a host wait
  -> submit the raster work once
  -> release all reservations on proven completion, otherwise poison after acceptance
```

The packet path deliberately supports only the existing position/index legacy
unlit recipe. Repeated references to one snapshot generation share one graph
position/index import pair and one reservation, because RenderGraph forbids
distinct logical imports of the same physical generation. They remain separate
ordered draws with separate uniforms and binding entries. The first uniform
upload is part of start; a later upload starts only after its predecessor has
proved complete. Therefore a later rejection is terminal but does not abandon
unobserved accepted uploads. Until raster acceptance, every failure releases
the reservation transaction; after it, any failed, unobservable, or dropped
outcome poisons every reserved generation.

## Data semantics

### Uniform ABI

`FrameUniform` is exactly 80 bytes: `0..64` is the column-major object-to-clip
matrix and `64..80` is linear RGBA base color. Single-draw fixed recipes
serialize `projection * view`; each legacy-unlit packet draw serializes
`projection * view * model` with its own affine `ModelTransform`. The latter
reuses the same ABI and does not add a per-draw RHI binding contract. Camera
input, model input, the computed matrix, and each color component must be
finite; color is in `[0, 1]`. The payload is visible to the fixed vertex and
fragment stages and is not a general uniform-layout API.

### Geometry, UVs, and clipping

Positions are tightly packed `f32x3`; indices are tightly packed `u32`.
Textured geometry adds one finite `f32x2` UV per position. Normal geometry adds
one finite normal per position, canonicalized to unit `f32x3` at construction.
Vertex-color geometry adds exactly one linear encoded `UNORM8x4` color per
position. The bytes are normalized by the fixed vertex input and are then
perspective-interpolated by the rasterizer; they are not sRGB texture data.
The textured and Lambert paths validate their fixed clip-space input before
acceptance, including non-finite values, invalid homogeneous `w`, and clipping
outside the closed contract. UV paths use perspective-correct center
interpolation.

### Color spaces and normals

`Rgba8Image` represents linear UNORM texels; `Srgba8Image` represents encoded
sRGB texels. Both upload original RGBA8 bytes, but the sRGB path chooses an sRGB
native view: RGB is decoded before linear filtering, alpha stays linear, and
the target stays linear `Rgba8Unorm`. UNORM and sRGB filterability are queried
independently from actual device capabilities.

`draw_lambert` perspective-interpolates canonical object-space normals, safely
normalizes a nonzero interpolation, applies a fixed object-space `+Z` Lambert
factor to RGB, and preserves alpha. Per-draw model transforms currently do not
apply to this path: it has no transformed normals, normal matrix,
caller-visible lights, or PBR semantics.

`draw_vertex_color` uses the existing 80-byte camera/material ABI: bytes
`0..64` contain `projection * view`, and bytes `64..80` contain a finite linear
RGBA tint in `[0, 1]`. Its vertex shader consumes tightly packed position
`f32x3` from slot zero and normalized color `UNORM8x4` from slot one; the
fragment result is the interpolated linear color multiplied by that tint and
written to the linear `Rgba8Unorm` target. It has no model transform, texture,
sampler, alpha/blend policy, or configurable color-space conversion.

## Errors, lifetime, and concurrency

The API separates failures by ownership transition:

- invalid input, foreign device, unavailable capability, graph construction, or
  reservation conflict fail before raster acceptance;
- uniform failure means raster did not start and releases reservations;
- raster rejection before native acceptance releases reservations; and
- accepted work with failed/unobservable completion is terminal and poisons all
  reserved generations.

`FixedFrameSubmission` retains the snapshots, reservations, uniform operation,
graph, and RHI submission required by its state. `RenderPacketSubmission`
retains the packet, all unique reservations, completed uniform uploads, graph,
provider, and accepted raster submission required by its state. Snapshot clones
retain RHI leases; device clones in the renderer/executor path keep the native
device alive until dependent work completes.

The mutex-protected gate safely coordinates snapshot clones, but is deliberately
conservative: it is not a promise of parallel rendering. Native queue
serialization and accepted-unknown handling are RHI concerns. Renderer code
never repairs a poisoned snapshot by guessing state.

## Validation and evidence

Stage 2 extracts the legacy-unlit CPU work into `PreparedBasicScene`. A real
`slot-graph` DAG fans one owned scene into per-draw validation/uniform work and
then assembles results by insertion index. Native packet lowering reuses that
result and only associates device-local snapshots; the WASM binding performs a
field-preserving borrow into the RHI-owned browser fixed ABI. Neither path
recomputes P*V*M, clip acceptance, material color, or ordering.

The native and browser paths also call the same private fixed raster-pass
declaration. Browser compilation selects a renderer-owned closed
`PresentationProfile` for only `Rgba8Unorm` or `Bgra8Unorm`, with fixed usage
and capability facts. WebGL2 uses the former; the WebGPU binding exhaustively
maps the RHI-reported canvas format to one of the two. Both use imported buffers
and one imported presentable resource; RHI validates the identical compiled
topology before lowering its closed browser recipe.
These closed adapter types prove one retained scene and are not a configurable
pipeline, material, or browser host API.

Portable unit tests cover domain validation, payload packing, publication,
reservations, recipe mapping, graph declaration, and failure paths. They do
not prove DX12/Vulkan correctness.

Windows hardware fixtures run the fixed contracts on both supported native
backends with required validation and compare readback to CPU oracles. Cases
distinguish perspective interpolation, integer versus linear sampling, clamp,
sRGB decode-before-filter, normal-stream binding, Lambert shading,
separate-slot linear vertex-color interpolation and tint multiplication, and
pre-accept versus accepted-unknown faults. The legacy-unlit packet fixture also
uses non-commuting camera and model matrices to compare per-draw placement with
an independent CPU `projection * view * model` oracle. Unexpected validation
diagnostics are failures. RHI readback consumes the graph-exported outgoing
state as its actual incoming state, so it cannot silently repair a wrong export.

See [ADR-0005](adr/0005-gpu-conformance-evidence.md) and
[ADR-0008](adr/0008-native-platform-test-gates.md). Native conformance is
separate from compile/link and CPU-only graph evidence.

## Extension boundaries

New renderer work needs a narrow tested contract. Scene data must resolve to
renderer-owned packets before graph declaration; it must not move asset lifetime
or native commands into RenderGraph. A general material,
shader, or pipeline API must intentionally replace/extend the closed recipe
boundary with explicit layout, binding, capability, lifetime, and cross-backend
semantics.

## Fixed-asset residency (0.14)

0.14 adds a private reuse layer only for fixed `MeshAsset`/`Geometry` and
`ImageAsset`/linear `Rgba8Image` inputs. It is not an application-visible
asset cache and does not change the existing fixed recipe, general shader,
general/native RHI, or RenderGraph APIs. The sibling
`fluxel-rendering-wasm` adapter adds only a closed experimental browser-
residency seam with opaque tokens; they are not RHI or graph handles. The
renderer records residency with the exact key:

```text
(AssetId, ContentGeneration, DeviceIdentity)
```

An `AssetId` names the logical source, `ContentGeneration` names one immutable
content revision, and `DeviceIdentity` names the device for which the GPU
representation was created. No shortened key is sufficient: an asset ID alone
can select stale contents, and an asset/content pair alone can select a native
resource for the wrong device.

Preparation is the only asset-resolution boundary. It takes an immutable CPU
`AssetSnapshot`, resolves or starts its fixed upload, and gives graph
declaration a renderer-owned GPU snapshot plus its RHI lease. A raster pass
has no `AssetStore`: it neither loads nor resolves assets, and therefore still
declares only concrete imported resources and their states.

```text
AssetStore --AssetSnapshot--> renderer prepare
    -> key lookup / fixed upload
    -> renderer-owned GPU snapshot + RHI lease
    -> graph declaration -> raster pass (no AssetStore)
```

Each private entry follows this small lifecycle:

```text
PendingUpload --upload completion--> Committed --replacement/device change/eviction--> RetireCandidate
```

Only `Committed` entries may satisfy preparation. `PendingUpload` cannot be
drawn. A `RetireCandidate` retains its native resource while any accepted
submission still has an RHI lease; submission completion, as observed by that
lease, is the sole authority for final retirement. Replacing an asset, ending a
frame, or observing device loss is not evidence that an old submission is
complete.

When a device is recreated, CPU `AssetSnapshot` values are retained and their
fixed mesh/image contents are uploaded under the new `DeviceIdentity`. Old
device entries are not transplanted or guessed safe; they remain retirement
candidates until their original leases complete. This preserves cross-device
safety without adding a recovery protocol to the general/native RHI or
RenderGraph. Browser-specific recovery uses only the sibling adapter's closed
experimental residency seam and opaque tokens.

See [ADR-0010](adr/0010-renderer-private-fixed-asset-residency.md).

The fixed visible-frame path sits at the renderer/RHI boundary above private
swapchain details. Renderer accepts one acquired opaque binding for the entire
ordered packet; RenderGraph imports it as a presentable resource and plans its
use plus final presentation intent; RHI owns acquire/submit/present, generation
retirement, and completion on DX12 or Vulkan. This is not a general surface or
window API. Bounded frames-in-flight, slot reuse, and back pressure are private
scheduler policy; scheduling, parallel recording, transient aliasing, and caches
do not alter declared frame semantics.

Until those contracts exist, the seven-recipe system is the supported renderer
implementation and remains deliberately closed.
