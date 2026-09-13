# Fluxel Renderer

`fluxel-renderer` defines the user-facing scene inputs `Camera`, `Geometry`,
`Mesh`, `BasicMaterial`, and insertion-ordered `DrawList`. Its optional
`gpu-upload` feature also turns validated indexed geometry into an opaque,
immutable GPU-ready snapshot without blocking the frame-building thread. A
`FixedFrameRenderer` can lower a draw list and its matching ready indexed
snapshots into an owned opaque `RenderPacket`, then non-blockingly submit its
ordered legacy unlit draws as one compiled graph, raster pass, and native
submission. The feature also supports deliberately closed single-draw
`f32x3/u32` indexed, textured, vertex-color, and Lambert paths that return
opaque offscreen image metadata. The Stage 1 visible path uses the same
legacy-unlit camera/material recipe to draw one acquired DX12 or Vulkan
presentable image and returns a non-blocking presentation submission. An ordered
packet can carry the deterministic multi-object proof scene; multiple submissions may retain
independent immutable snapshot read leases; a private proof scheduler, not the
renderer API, bounds frames in flight.
It can separately upload canonical unit normals and execute one closed,
non-textured fixed-Lambert draw.

It is intentionally not a complete GPU renderer: it does not compile general
material shaders, expose configurable samplers/PBR or a general
pipeline/bind-group API, or own windows and swapchains.

## Optional GPU upload

```toml
[dependencies.fluxel-renderer]
git = "https://github.com/fluxel-project/fluxel-rendering"
tag = "v0.12.0"
features = ["gpu-upload"]
```

`IndexedMeshUpload::begin` serializes positions as tightly packed
little-endian `f32x3` and indices as little-endian `u32`. Polling is
non-blocking. A cloneable `IndexedMeshSnapshot` appears only after both native
submissions complete; failure and partial acceptance never expose a half-ready
generation. Buffers, resource states, mapping, and native handles remain
renderer-private.

`FixedFrameRenderer::draw` accepts a ready snapshot, `Camera`, `BasicMaterial`,
and a nonzero extent. It serializes `projection * view` and the linear
`base_color` into one immutable 80-byte uniform, starts its upload without
blocking, then submits one fixed triangle-list Raster recipe only after that
upload completes. Poll the returned `FixedFrameSubmission`: `Pending` covers
either uniform upload or Raster completion, while retryable `Busy` retains the
operation. `Submitted` reports that raster submission was accepted and the
acquired image was consumed by the native presentation path, but does not claim
present or GPU completion. `Complete` yields an opaque `FrameImage`; terminal
or otherwise unproven accepted Raster work poisons that snapshot generation.
Snapshot clones share one generation that permits concurrent immutable readers;
each reader releases only after its own known completion, while any unknown
accepted result poisons the generation monotonically. No native texture or
buffer handle is exposed.

`FixedFrameRenderer::lower_draw_list` accepts an insertion-ordered `DrawList`
and one ready `IndexedMeshSnapshot` for each draw at the same positional index.
It checks exact CPU geometry metadata and copies the camera, material uniforms,
per-draw model transform, and snapshot leases into a non-`Clone`, device-affine
`RenderPacket`; building or dropping the packet does not reserve a snapshot.
`DrawList::push_transformed` associates one finite affine `ModelTransform` with
one draw, while `push` retains identity placement. A transform belongs to the
draw/packet rather than `Mesh`, so one mesh can be placed independently by
multiple ordered draws. For this legacy-unlit packet path, the renderer lowers
the column-major transform as `projection * view * model` into the existing
80-byte `mat4x4<f32> + vec4<f32>` uniform ABI; it does not add a new RHI binding
layout. `submit_packet` reserves each unique snapshot generation once, uploads
draw uniforms one at a time without blocking, and submits one legacy-unlit
raster graph only after all uniforms complete. Raster completion releases
reservations; an unproven accepted raster outcome poisons them. Multiple packet
entries may reference one snapshot generation while retaining their own ordered
draws and uniforms.

Packet preparation uses private `slot-graph` only for the real CPU dependency
shape: shared scene input fans out to per-object validation/uniform preparation
and fans back in to insertion-ordered assembly. It neither exposes graph/node
identities nor models GPU submission, completion, or synchronization.
`async-runtime` is not a renderer dependency; native scheduling and shutdown
belong to the host layer.

`Rgba8Image` validates a nonzero two-dimensional tight RGBA8 payload.
`BaseColorTextureUpload::begin` borrows it and starts a non-blocking immutable
upload; only proven completion publishes `BaseColorTextureSnapshot`.
`TexturedBasicMaterial` couples that snapshot to `BasicMaterial`, and
`FixedFrameRenderer::draw_textured` uses a fixed mip-zero `textureLoad` mapping
derived from model-space `position.xy`. `TexturedGeometry` and
`TexturedIndexedMeshUpload` add an isolated three-stream position/index/UV
generation; `draw_textured_uv` consumes its explicit finite `f32x2` coordinates
through a closed second vertex slot. `draw_textured_uv_linear_clamp` adds a
separate fixed `textureSampleLevel(..., 0.0)` path with private linear filtering
and clamp-to-edge state. `Srgba8Image` and its separate upload/snapshot/material
types retain encoded base-color bytes; `draw_textured_uv_linear_clamp_srgb`
uses a true sRGB source view so decode occurs per texel before filtering, while
the target stays linear `Rgba8Unorm`. These APIs expose no configurable sampler,
color space, mip/LOD, or general binding contract. The textured paths accept
only finite, positive-w triangles wholly inside the clip volume, and mesh/texture
generations share one atomic draw reservation outcome.

`NormalGeometry` couples indexed geometry to one canonical finite unit
`f32x3` normal per position. `NormalIndexedMeshUpload` atomically publishes its
position/index/normal generation, and `FixedFrameRenderer::draw_lambert` uses a
fixed object-space `+Z` direction. The fragment shader perspective-interpolates
and safely normalizes the normal, multiplies only linear RGB by Lambert, and
preserves material alpha. Model transforms currently apply only to the
legacy-unlit packet path: Lambert drawing has no transformed normals, normal
matrix, light, normal-map, or PBR descriptor.

`VertexColorGeometry` adds a separate linear `UNORM8x4` color stream to indexed
geometry, and `VertexColorIndexedMeshUpload` publishes position, color, and
index buffers only after all three uploads complete. `draw_vertex_color`
combines the perspective-interpolated color with a finite linear RGBA tint from
`VertexColorMaterial`. The closed ABI fixes position at slot 0, color at slot 1,
and the camera/tint uniform at 80 bytes; it does not expose a general vertex
layout, shader, material, or pipeline descriptor.

## Use today

Build a mesh and schedule it for a camera:

```rust
use fluxel_renderer::{BasicMaterial, Camera, DrawList, Geometry, Mesh, ModelTransform};

let camera = Camera::default();
let geometry = Geometry::from_positions(vec![
    [-0.5, -0.5, 0.0],
    [0.5, -0.5, 0.0],
    [0.0, 0.5, 0.0],
])
.with_indices(vec![0, 1, 2])?;
let mesh = Mesh::new(
    geometry,
    BasicMaterial::new([0.2, 0.7, 1.0, 1.0])?,
);

let mut draws = DrawList::new(&camera);
draws.push(&mesh);
let translated = ModelTransform::from_column_major([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.25, 0.0, 0.0, 1.0],
])?;
draws.push_transformed(&mesh, translated);
assert_eq!(draws.len(), 2);
# Ok::<(), Box<dyn std::error::Error>>(())
```

`Geometry::with_indices` checks indices when the geometry is constructed.
`DrawList` preserves insertion order, which makes packet draw order explicit
without adding sorting policy prematurely.

## Headless frame example

On a configured Windows device, this example performs one full public lifecycle:
immutable mesh upload, non-blocking polling, ready snapshot publication, one
closed draw, further polling, and `FrameImage` completion. It prints structured
errors and requires a nonzero target extent with native validation required.

```powershell
cargo run -p fluxel-renderer --features gpu-upload --example 01_headless_frame -- dx12
cargo run -p fluxel-renderer --features gpu-upload --example 01_headless_frame -- vulkan
```

It is neither a window/swapchain demo nor GPU conformance evidence: it does not
use `test-support`, read pixels back, or compare a CPU oracle. Run every ignored
hardware fixture through the workspace's single release gate instead:

```powershell
./scripts/conformance.ps1
```

## Performance and compatibility

With default features the crate has no dependencies, no `unsafe`, and no native
API calls, and builds wherever Rust 1.87 supports the workspace. `gpu-upload`
adds the safe `fluxel-rhi` boundary; native execution is currently implemented
on Windows DX12/Vulkan. The `camera_material` fixtures compare the Camera/material
uniform fixed offscreen draw byte-for-byte with CPU pixel oracles on both
backends under Required native validation. The renderer crate itself contains
no `unsafe`.

The `texture_load` fixtures compare this textured draw byte-for-byte with an
independent CPU oracle on DX12 and Vulkan under Required validation.
The `uv_texture_load` fixtures separately prove explicit UV perspective interpolation
against position-derived and linear CPU counter-oracles on both backends.
The `linear_clamp_sampling` fixtures additionally prove fixed linear filtering and independently
observable U/V clamp behavior on DX12 and Vulkan.
The `srgb_sampling` fixtures distinguish decode-before-filter from encoded-space
filtering, nearest/repeat, and affine UV counter-oracles on both backends.
The `normal_lambert` fixtures distinguish perspective normal interpolation and
fragment renormalization from counter-oracles and exercise the exact-zero
fallback on both backends.
The `vertex_color` fixtures prove linear `UNORM8x4` transport, perspective color
interpolation, tint application, three-stream binding, and full-readback parity
against an independent CPU oracle on both backends.

The internal `shader` module is an ownership boundary for material shader
modules. Shader compilation and reflection are deliberately future backend
work.

The [renderer design](https://github.com/fluxel-project/fluxel-rendering/blob/main/documents/design-renderer.md)
explains the ownership split and the evidence gates for future GPU integration.

## License

Licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at
your option.
