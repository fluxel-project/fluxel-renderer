# fluxel-rendergraph

`fluxel-rendergraph` is a typed, retained render graph for planning GPU work.
You declare resource accesses during graph setup; the compiler derives pass
dependencies, validates resource versions and device capabilities, removes dead
work, and produces an immutable plan. It is for renderers that need pass
ordering and resource-state requirements to be explicit rather than encoded in
ad-hoc command recording order.

The current release supplies the declaration/compiler contract and a
single-queue execution SPI verified by the deterministic CPU-only `TestRhi`.
The companion
[`fluxel-rhi`](https://github.com/fluxel-project/fluxel-rendering/tree/main/crates/rhi)
crate implements fixed Raster, Compute, and Copy subsets on native DX12 and
Vulkan through the same immutable plan. The native profile is limited to
R01 clear/triangle, R02 indexed viewport/scissor, and X01's closed
Raster→Compute→Copy recipe; surfaces, renderer lowering, and a general shader
or pipeline API remain outside it.

## Installation

```toml
[dependencies.fluxel-rendergraph]
git = "https://github.com/fluxel-project/fluxel-rendering"
tag = "v0.14.0"
```

The crate is not published on crates.io yet, so the Git dependency is the
current installation path. It requires Rust 1.87
or newer, uses edition 2024, and has no production GPU dependency.

## Quick start

The smallest complete example declares an uninitialized transient buffer,
initializes it in a compute pass, exports its successor version, and compiles
the graph against the application's observed capabilities:

```rust
use fluxel_rendergraph::{
    BufferCapabilities, BufferDesc, BufferRange, BufferWriteUse, CompileError,
    DeviceCapabilities, ExportBufferContract, QueueCapabilities,
    QueueDescriptor, QueueId, RenderGraph, ResourceAccessState, WriteCoverage,
};

fn main() -> Result<(), CompileError> {
    let capabilities = DeviceCapabilities::builder()
        .queue(QueueDescriptor::new(
            QueueId::new(0),
            QueueCapabilities::new(false, true, false, false),
        ))
        .buffers(BufferCapabilities::new(false, true, false))
        .build();

    let mut graph = RenderGraph::<()>::new();
    let output = graph.create_buffer("output", BufferDesc { size: 256 });
    let output = graph.add_compute_pass(
        "initialize",
        |pass| {
            let (next, write) = pass.write_buffer(
                output,
                BufferWriteUse::Storage,
                BufferRange::whole(),
                WriteCoverage::Full,
            );
            (next, write)
        },
        |_commands, _resolver, _write, _frame| Ok(()),
    );

    graph.export_buffer(
        output.output,
        ExportBufferContract {
            final_state: ResourceAccessState::ShaderStorageWrite,
        },
    );
    let compiled = graph.compile(&capabilities)?;
    println!("retained passes: {}", compiled.graph.execution_order().len());
    Ok(())
}
```

`capabilities` is application/RHI-owned. The example uses a portable fixture;
a real integration supplies facts observed from its selected device. The same
program is checked in as [`00_minimal_compile.rs`](examples/00_minimal_compile.rs);
from this repository, run it with:

```sh
cargo run -p fluxel-rendergraph --example 00_minimal_compile
```

When compilation returns `CompileErrorKind::UnsupportedSemanticRequirement`,
`error.context.unsupported` contains a typed `CapabilityRequirement` and the
full observed `DeviceCapabilities` snapshot. Use those fields for fallback or
user-facing diagnostics; `DiagnosticContext::detail` is descriptive text, not
a matching contract. This reports declared resource and queue semantics only,
and does not introduce a general pipeline API.

## Core concepts

### Versions and typed access handles

`TextureVersion` and `BufferVersion` represent logical contents. A read borrows
a version; a write consumes it and returns its successor. Setup also returns a
pass-local `*Read`, `*Write`, or `*ReadWrite` handle. The execute callback can
resolve and use only those declared handles. This keeps data dependencies
visible to the compiler while pipelines, descriptors, samplers, and native
resources stay outside the graph.

See [`10_resource_versions.rs`](examples/10_resource_versions.rs) for a
read/write chain and [`08_multi_reader.rs`](examples/08_multi_reader.rs) for
fan-out reads.

### Passes, roots, and culling

`add_raster_pass`, `add_compute_pass`, and `add_copy_pass` separate setup
(declaration) from repeatable execution. Exported versions, presentation
targets, and declared side effects are graph roots. Only work required by a
root is retained; `CompileReport::culled_passes` explains what was removed.

[`09_culling.rs`](examples/09_culling.rs) demonstrates root-driven culling.
Use `depends_on` only for external protocol or diagnostic ordering that no GPU
resource can express—never as a substitute for a resource access.

### Imports, exports, and frames

Transient resources are graph-owned declarations. Persistent GPU resources are
declared as import slots with descriptor, initial state, ownership, and content
contracts, then bound for each frame by the renderer. Exports specify the
required final state. A compiled graph remains reusable: per-frame values and
import bindings belong to `FrameInputs`, not to graph compilation.

Every bound physical resource reports its actual allowed domain operations.
Before recording, frame resolution checks that this set covers the compiled
requirement for both imports and backend-created transients. Providers derive
allowed operations from creation and native facts; they must not copy the
compiled requirement into the binding.

Renderer-private GPU residency is resolved before this boundary. A frame binds
only the selected concrete snapshot, its state, and lease; graph setup and pass
callbacks receive neither `AssetStore` nor an asset lookup handle. Residency
does not add an asset identity, cache, or generic shader contract to
RenderGraph.

Start with [`01_copy_buffer.rs`](examples/01_copy_buffer.rs) for an import and
export, then [`14_two_frame_dynamic.rs`](examples/14_two_frame_dynamic.rs) for
one graph instantiated with distinct frame inputs.

### Execution integration

`FrameExecutor` resolves a compiled graph through an `ExecutionBackend`, a
`FrameResourceProvider`, and a renderer-owned `RenderObjectProvider`. The
provided `TestRhi` validates protocol order, transitions, binding checks, and
completion-based retirement; it does not run shaders or emulate GPU memory.
For a present root, the resource provider resolves an acquired presentable
image together with an opaque one-shot token. The executor records the graph's
final `Present` transition and transfers that token, paired with its graph root,
only to `ExecutionBackend::submit`; native surface and swapchain objects never
enter graph declarations or pass callbacks.
On Windows, `fluxel-rhi` separately executes the same immutable plan on DX12
and Vulkan for fixed Raster, Compute, and Copy release fixtures. Its support
is not a general graph shader surface: a renderer/RHI provider registers opaque
fixed pipeline and binding objects, while the graph continues to own only
declared resource accesses, ordering, states, and dispatch validation. X01 is
one closed sampled `Rgba8Unorm` texture-to-packed-storage-buffer recipe, not a
general texture-compute feature.
[`20_headless_frame_pipeline.rs`](examples/20_headless_frame_pipeline.rs) is
the full CPU-only Raster → Compute → Copy integration fixture.

## Examples

The numbered examples are a progressive tour and run with
`cargo run --example <name>`.

- `01_copy_buffer`, `02_compute_buffer`, `03_raster_triangle`: executable
  TestRhi command-recording paths.
- `04_frame_pipeline`, `05_import_export_history`, `11_subresource`: resource
  contracts, attachment/range semantics, and persistent history.
- `06_explicit_order`, `07_compile_diagnostics`, `09_culling`: compiler
  diagnostics and graph retention.
- `12_capability_fallback`, `15_sampled_indexed_draw`, `18_backend_variants`:
  capability-dependent declarations and renderer-facing bindings.
- `19_invalid_graphs` and `20_headless_frame_pipeline`: failure shapes and the
  most complete executable protocol fixture.

## Performance characteristics

Compilation is CPU-side planning. It derives dependencies and transitions from
declared accesses, culls unreachable work, and produces an immutable snapshot
that can be instantiated repeatedly. The current executor lowers one serial,
single-queue plan; it does not claim multi-queue overlap, transient aliasing,
parallel recording, native GPU throughput, or GPU timing performance.

Repository benchmarks measure compiler and CPU `TestRhi` protocol work, not
GPU performance.

## Platform compatibility

The core crate is platform-neutral and its normal library build has no native
GPU dependency. It supports platforms on which Rust 1.87 is supported. Native
device discovery is deliberately separate in `fluxel-rhi`; its current
headless bootstrap targets DX12 on Windows and Vulkan where supported by that
crate and its drivers.

## Current limits

- Real-GPU execution is limited to the fixed Raster, Compute, and Copy profile
  on Windows DX12/Vulkan. R01/R02/X01 pass their exact CPU oracles on both
  backends with Required validation on the recorded release hardware.
- No native surface acquisition, resize/recreation, or present execution in
  RenderGraph itself. It only declares an imported presentable resource and
  final present intent; RHI owns native presentation.
- No native multi-queue lowering, resource aliasing, or GPU conformance claim.
- The execution SPI is provisional while the first native backend is built;
  declaration semantics and compiler diagnostics are the stable center.

## Verification

From the repository root:

```sh
cargo test
cargo run --example 00_minimal_compile
```

For architecture and rationale, read
[`documents/design-rendergraph.md`](https://github.com/fluxel-project/fluxel-rendering/blob/main/documents/design-rendergraph.md).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
