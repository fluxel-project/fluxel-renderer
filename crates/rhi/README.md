# fluxel-rhi

`fluxel-rhi` opens one headless Direct3D 12 or Vulkan device on Windows and
creates opaque owned buffers and 2D textures through a safe portable contract.
Its fixed-artifact Raster, Compute, and Copy backend executes the same compiled
RenderGraph plan on both APIs, including semantic transition lowering, one
queue submission, completion, and retirement.

Renderer-shaped raster artifacts are deliberately closed and live under
`fluxel_rhi::adapter::fixed_artifacts`. They are closed integration recipes,
not the stable RHI facade or a general pipeline API.

It is deliberately a narrow native boundary, not a general graphics API.
`CopyBackend` implements only Copy commands; the distinct `ComputeBackend`
adds only closed `ComputeKernel` artifacts and their single RW storage-buffer
binding recipe. `RasterBackend` adds only the fixed R01/R02 raster artifacts,
the closed U02 renderer snapshot recipe, its closed U03 Camera/material-uniform
variant, the closed uniform-plus-texture raster variant, and the closed X01 texture-pack
compute recipe needed for one
Raster→Compute→Copy chain. It does **not** expose general mapping/readback,
general acquire/present abstractions, a general shader/pipeline API, renderer lowering,
multiple queues, or transient aliasing.

`Device::upload_immutable_buffer` and the closed
`Device::upload_immutable_texture` are the production upload primitives. They
atomically create a device-local destination and return an owning pending
operation. Only proven completion can yield `UploadedBuffer`; accepted-unknown
work retains its target, staging allocation, command objects, and device rather
than publishing guessed contents or state. The finalized incoming state is
exactly `CopyDestination`. Texture upload accepts only a whole tight
single-mip/layer/sample D2 linear `Rgba8Unorm` or encoded `Rgba8UnormSrgb`
image with CopyDestination and Sampled usage; its private staging rows are
padded to the native 256-byte requirement.

## Installation

```toml
[dependencies.fluxel-rhi]
git = "https://github.com/fluxel-project/fluxel-rendering"
tag = "v0.12.0"
```

The crate is not published on crates.io yet, so the Git dependency is the
current installation path. The default feature set enables both Windows backends.
To select one explicitly:

```toml
[dependencies.fluxel-rhi]
git = "https://github.com/fluxel-project/fluxel-rendering"
tag = "v0.12.0"
default-features = false
features = ["dx12"]
```

| Feature | Effect on Windows |
| --- | --- |
| `dx12` | Compile Direct3D 12 device bootstrap. |
| `vulkan` | Compile Vulkan device bootstrap. |

The non-default `test-support` feature is reserved for workspace conformance
fixtures. Its doc-hidden exact-state readback and diagnostics capture are not a
production mapping/readback API.

Both features are enabled by default. A build with a requested backend omitted
returns `OpenError::BackendDisabled`; a non-Windows build returns
`OpenError::PlatformUnsupported`. Adding the dependency opens no device. Rust
1.87 and edition 2024 are required.

The Windows presentation slice accepts only standard window/display handles.
Backend-neutral `presentation::Surface` opens a fixed RGBA8 FIFO target on
DX12 or Vulkan and exposes an opaque generation plus Active, Suspended,
Poisoned, or Closed lifecycle state. Zero-sized targets suspend acquisition;
a later non-zero extent creates a fresh generation only after known retirement
of the old one. A single private acquire lease protects the HAL's one-unpresented-
image contract; after present, a separate capacity-bounded ticket remains until
completion-driven destruction of derived views. Vulkan uses the surface's
required current extent and quarantines an unpresented dropped image because
wgpu-hal 30 cannot safely recycle that acquire semaphore. Resize and shutdown
refuse while any ticket remains live; native swapchain objects, ticket capacity,
image indices, synchronization, and HWND remain private. Renderer
bounded frames-in-flight and back pressure are separate private scheduler policy.
`Surface::open` is a single-surface bootstrap, not a general surface API or a
commitment to an eventual multi-surface topology.

## Open a device

Select a backend explicitly, open adapter zero, then inspect the native facts:

```rust,no_run
use fluxel_rhi::{Backend, Device, DeviceOptions};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::open(Backend::Dx12, DeviceOptions::default())?;
    println!("{:#?}", device.hardware());
    println!("{:#?}", device.capabilities());
    Ok(())
}
```

On a Windows machine with a suitable driver, run the complete selectable
example from the workspace root:

```powershell
cargo run -p fluxel-rhi --example 01_open_device -- dx12
cargo run -p fluxel-rhi --example 01_open_device -- vulkan
```

`Device::create_buffer` and `Device::create_texture` validate RenderGraph's
typed descriptors and usage sets before entering the native boundary. Returned
resources expose only descriptor, actual allowed usage, opaque identity and a
cloneable lease. The final resource or lease destroys the native object once;
it also retains the owning device. Requested usage is a minimum: native
normalization may report a wider allowed set, such as buffer `StorageWrite`
becoming storage read/write. There are no raw handles.

`CopyBackend` implements the execution SPI with one logical queue and one
command buffer per graph execution. It permanently rejects compute and raster
families. `ComputeBackend` separately enables Copy plus the fixed Compute
subset. Embedded WGSL is parsed and validated as Naga IR, then lowered by the
HAL for DX12 or Vulkan; Fluxel's portable identity is source hash, entry point,
workgroup shape, and binding-recipe version, never a native-binary hash.
Pipelines and bindings are opaque device-affine values with cloneable leases;
accepted submissions retain every referenced lease until retirement is safe.

`RasterBackend` is intentionally not a general draw interface. It supports a
single-sampled, single-mip/layer `Rgba8Unorm` color target, fixed vertex/index
recipes, exact load/store combinations, and R01 clear/triangle plus R02
indexed viewport/scissor fixtures. U02 adds a separate renderer snapshot recipe
with tightly packed `float32x3` positions, `uint32` indices, and a fixed opaque
fragment color. U03 is a separate closed variant with exactly one 80-byte
static group-0/binding-0 uniform: column-major `projection * view` and linear
RGBA base color, visible to both vertex and fragment stages. Its opaque binding
is device- and pipeline-affine, retains matching pipeline and buffer leases
through completion, and accepts no dynamic offset or configurable binding
layout. The textured variant adds a fragment-visible whole `texture_2d<f32>` at
binding 1 and a fixed mip-zero integer `textureLoad` mapping; its portable
identity explicitly records binding numbers, visibility, dimension, format,
sample type, mip and mapping version. It does not add a sampler. Neither recipe
makes those choices configurable.

The explicit-UV raster artifact uses two vertex streams:
slot 0 is tightly packed `f32x3` position at shader location 0 and slot 1 is
tightly packed `f32x2` texture coordinate at location 1. Its binding retains
the expected physical identities for both streams, and safe/native recording
rejects swapped, missing, repeated, partial, or foreign roles before submission.
It preserves the same no-sampler mip-zero `textureLoad` fragment recipe.

The explicit-UV linear-clamp artifact has a
binding layout fixes a filterable `Rgba8Unorm` texture and a private filtering
sampler, while its shader uses explicit level zero. The opaque binding owns the
sampler through terminal completion; it is neither a public descriptor nor a
RenderGraph resource. Adapter `SAMPLED_LINEAR` support is reported as a raw
fact, propagated into the graph fingerprint, and rechecked before native
creation.

The type-distinct `Rgba8UnormSrgb` artifact preserves that closed
artifact. Encoded upload bytes are preserved, the sampled native view performs
per-texel sRGB decode before filtering, alpha remains linear, and the color
target remains `Rgba8Unorm`. sRGB linear-filter support is queried as its own
raw adapter fact and is never inferred from the UNORM format.

The non-textured normal-Lambert artifact uses slot 0
is tight position `f32x3`, slot 1 is tight canonical unit normal `f32x3`, and
the existing 80-byte camera/material uniform remains the only binding. The
shader perspective-interpolates the object-space normal, normalizes it only
when its exact squared length is positive, applies fixed `+Z` Lambert to RGB,
and preserves alpha. Strong portable identity and safe/native role validation
keep this recipe distinct from every UV layout.

For reproducible DX12 validation the workspace carries the upstream
gfx-rs/wgpu#10221 fix on its vendored `wgpu-hal` 30.0.1 source. See
`vendor/wgpu-hal-30.0.1/FLUXEL-PATCH.md`; the patch can be removed once a
Rust-1.87-compatible release includes it.

X01 samples that complete color texture in
a closed compute artifact, packs row-major RGBA8 pixels into an RW storage
buffer, and copies the result to an export. Texture-row padding is stripped for
the exact CPU oracle. Readback consumes the exported outgoing state and lease
as its actual input; it cannot conceal an incorrect graph transition.

## Core concepts

`Backend` is an explicit choice between `Dx12` and `Vulkan`; Fluxel never
silently falls back. `DeviceOptions::adapter_index` selects the backend-native
enumeration index, with zero as the default. It is a bootstrap/probing
mechanism, not yet a polished cross-platform adapter-selection API.

`Device::hardware()` returns unmodified driver identity: backend, name,
vendor/device IDs, device kind, PCI bus ID where available, and driver strings.
`Device::capabilities()` returns raw adapter limits currently needed by the
bootstrap contract, including fixed-Compute dispatch, workgroup-dimension, and
total-invocation bounds. These facts are not a portability profile, feature
tier, or promise that a particular Fluxel graph can execute; a future renderer
makes those policy decisions. Zero or excessive dispatch dimensions fail before
recording; the fixed shader workgroup is checked again before native creation.

`OpenError` distinguishes an omitted feature, unsupported platform, absent
adapter, unavailable validation facility, insufficient baseline limits, and
native loader/driver failure. Include its display text in diagnostics, but do
not parse it as a stable native error code.

## Native validation

The default `Validation::Disabled` does not request native validation. During
development, request fail-closed validation:

```rust,no_run
use fluxel_rhi::{Backend, Device, DeviceOptions, Validation};

let device = Device::open(
    Backend::Vulkan,
    DeviceOptions {
        validation: Validation::Required,
        ..DeviceOptions::default()
    },
)?;
# let _: fluxel_rhi::Device = device;
# Ok::<(), fluxel_rhi::OpenError>(())
```

For DX12, `Required` needs the D3D12 debug layer and verifies that the opened
device exposes its information queue. For Vulkan, it needs
`VK_LAYER_KHRONOS_validation` and the layer's
`VK_EXT_validation_features`; the internal HAL request enables synchronization
validation when that facility is available. If Fluxel cannot positively
establish the requested facility, opening fails with
`OpenError::ValidationUnavailable` rather than continuing with a weaker
configuration.

The conformance fixtures collect DX12 information-queue and Vulkan validation
diagnostics programmatically while checking command states and synchronization.
C01/C02/C03 cover Copy and K01/K02 cover fixed Compute. R01, R02, and X01 are
the fixed Raster→Compute→Copy release fixtures. On the recorded AMD
Radeon 780M hardware, each passes its exact CPU oracle on DX12 and Vulkan with
collected Required-validation diagnostics empty.

## Platform compatibility

| Platform | DX12 | Vulkan |
| --- | --- | --- |
| Windows | Supported when its feature, loader, driver, and selected adapter are available | Supported when its feature, loader, driver, and selected adapter are available |
| macOS, Linux, other targets | Returns `PlatformUnsupported` | Returns `PlatformUnsupported` |

`Device::open` remains headless: it creates no window or surface. On Windows,
the separately constructed `presentation::Surface` is the Stage-1 fixed RGBA8
FIFO surface/swapchain path for the chosen DX12 or Vulkan backend. It retains
an `Arc` window owner until acquired work is discarded or retired; it is not a
general surface or presentation API.

## Performance and scope

Opening a native device is setup work, not a rendering benchmark. This crate
makes no frame-time, throughput, allocation, synchronization, or
GPU-performance claim. It keeps native objects in a private backend module to
establish a safe ownership boundary for later vertical slices.

The current scope ends after fixed Raster, Compute, and Copy transition/order
lowering, command recording, one submission, completion, lease retirement, and
crate-private exact test readback. A readback consumes the export's reported
outgoing state as its real incoming state; it cannot silently repair a
graph-state bug. The lifetime/error/unsafe contract requires all pipeline,
binding, attachment, encoder, command-buffer, and resource leases to survive
until terminal completion; rejected and accepted-unknown submission paths stay
distinct.
Buffer Copies require 4-byte-aligned source offset, destination offset, and
size, checked in both graph recording and RHI lowering. Queue operations are
serialized per opened device, including error-path idle waits. Surface/present
outside the one fixed DX12/Vulkan path, general shaders/pipelines, renderer lowering,
multi-queue, parallel recording, aliasing, and performance work remain out of scope.

## Testing and development

Ordinary tests verify API behavior and native-backend compile/link coverage.
Tests that actually open hardware are ignored because they depend on the local
driver and validation installation. The workspace has one release gate for all
such fixtures; run it deliberately on a clean, configured Windows checkout:

```powershell
./scripts/conformance.ps1
```

The script injects the checked-out commit as `FLUXEL_TEST_COMMIT` and preserves
the complete workspace result and manifest under `target/conformance/<sha>/`.

The DX12 required-validation case needs the D3D12 debug layer. The Vulkan
required-validation case needs LunarG's Khronos validation layer and its
validation-features extension. CI compiling the crate is not evidence that a
GitHub-hosted runner opened a real GPU device.

See the
[RHI architecture design](https://github.com/fluxel-project/fluxel-rendering/blob/main/documents/design-rhi.md)
for ownership, validation, and evolution decisions. See the
[`fluxel-rendergraph` user guide](https://github.com/fluxel-project/fluxel-rendering/tree/main/crates/rendergraph)
for graph authoring and its numbered examples.

## License

Licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at
your option.
