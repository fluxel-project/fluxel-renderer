# Fluxel Rendering

`fluxel-rendering` is Fluxel's host-agnostic Rust rendering kernel. Its typed
render graph, native RHI boundary, and renderer layer provide portable drawing
semantics and GPU execution without owning an application host. Cross-frame
asset identity, platform services, runtime packaging, and language SDKs belong
to adjacent Fluxel repositories described in the
[Fluxel ecosystem roadmap](https://github.com/fluxel-project/.github/blob/main/ROADMAP.md).

The kernel accepts a host-supplied surface target, resize information, resource
bytes, scene or canvas updates, and explicit render calls. It does not create a
window, run an event loop, read files or URLs, collect input, provide storage,
audio, video, networking, or decide when the next frame runs. Browser and
mini-game adapters drive its WASM form; native hosts drive its library form.

## Workspace

The workspace is organised around four crates:

| Crate | Responsibility |
| --- | --- |
| `fluxel-rendergraph` | Typed resource declarations, dependency compilation, validation, immutable execution plans, and the CPU-only `TestRhi` protocol. |
| `fluxel-rhi` | DX12/Vulkan native ownership, fixed RenderGraph execution, and the narrow backend-neutral presentation boundary. |
| `fluxel-renderer` | Scene/domain data, immutable GPU snapshots, ordered render packets, fixed indexed draws, and visible-frame transactions. |
| `fluxel-rendering-wasm` | Non-published wasm-bindgen capsule for the named WebGL2 and WebGPU browser proofs; JavaScript remains the RAF and DOM lifecycle owner. |

The optional `gpu-upload` slice publishes immutable GPU snapshot generations
after native uploads complete. It can lower an insertion-ordered `DrawList`
and its matching ready indexed snapshots into an owned, opaque, device-affine
`RenderPacket`, then execute all of its legacy unlit indexed draws through one
compiled graph, raster pass, and submission. Existing single-draw paths also
cover closed textured, vertex-color, and fixed-Lambert recipes.
Each legacy-unlit packet draw may carry an independent affine model-to-world
placement; the renderer lowers it into the existing camera/material uniform
contract without exposing a general scene or pipeline API.
The default build remains the portable headless domain model.

The fixed renderer and RHI slices are split into single-responsibility modules.
Seven proven raster paths share one private, closed recipe mapping.

Package READMEs are the detailed user documentation published with each crate;
this file is only the workspace entry point.

## Released source dependency

The workspace releases its three publishable crates together. Git consumers must pin the
release tag rather than follow `main`:

```toml
fluxel-rendergraph = { git = "https://github.com/fluxel-project/fluxel-rendering", tag = "v0.12.0" }
fluxel-rhi = { git = "https://github.com/fluxel-project/fluxel-rendering", tag = "v0.12.0" }
fluxel-renderer = { git = "https://github.com/fluxel-project/fluxel-rendering", tag = "v0.12.0" }
```

`v0.12.0` and each publishable package's `0.12.0` version identify the same workspace
release. See [RELEASING.md](RELEASING.md) for the release gate.

## Documentation

- [Workspace architecture](documents/design-overview.md)
- [RenderGraph design](documents/design-rendergraph.md)
- [RHI design](documents/design-rhi.md)
- [Renderer design](documents/design-renderer.md)
- [Architecture decisions](documents/adr/README.md)
- [Fluxel ecosystem roadmap](https://github.com/fluxel-project/.github/blob/main/ROADMAP.md)
- [RenderGraph guide](crates/rendergraph/README.md)
- [RHI guide](crates/rhi/README.md)

## Quick verification

Rust MSRV is 1.87 (edition 2024).

```sh
cargo +1.87.0 test --workspace --all-targets --all-features --locked
cargo +1.87.0 clippy --workspace --all-targets --all-features --locked -- -D warnings
```

The graph compiler and `TestRhi` are CPU-only. `fluxel-rhi` provides
headless DX12 and Vulkan fixed Raster, Compute, and Copy execution on Windows
when the native loader and driver are available. CI is a compile/link gate;
real-GPU conformance remains an explicit local release gate.

On a clean Windows `x86_64-pc-windows-msvc` checkout, the release-only GPU
gate is:

```powershell
./scripts/conformance.ps1
```

It runs every workspace ignored fixture, injects the checked-out commit through
`FLUXEL_TEST_COMMIT`, and preserves a manifest and complete log under
`target/conformance/<sha>/`, including failures. Successful release artifacts
are retained on the matching [GitHub Release](https://github.com/fluxel-project/fluxel-rendering/releases).

## Roadmap

The organization
[roadmap](https://github.com/fluxel-project/.github/blob/main/ROADMAP.md) is the
only stage/status authority. This README records only the workspace's current
supported paths and recommended entry points.

Current retained closures are:

- a fixed resource floor on DX12, Vulkan, WebGPU, and WebGL2;
- fixed compute plus storage-buffer paths on DX12, Vulkan, and WebGPU;
- fixed storage-texture writes on DX12, Vulkan, and WebGPU, and fixed
  storage-texture reads on Vulkan only;
- structured, zero-side-effect WebGL2 rejection for Compute and every storage
  resource operation;
- headless DX12/Vulkan execution and real-GPU conformance;
- visible Windows DX12/Vulkan surface lifecycle and bounded completion;
- the retained RGB scene on the named Chrome/WebGL2 target; and
- the same scene on the named Chrome/WebGPU target, including canvas epochs,
  bounded completion, controlled destroyed-device recovery, and async disposal.

This is a capability matrix, not a portable feature tier: DX12 storage-texture
read fails closed on the observed driver, and WebGPU storage-texture read is
not currently promised. WebGL2 does not emulate Compute or storage. Unsupported
graph declarations expose a typed `UnsupportedCapability` requirement with the
observed capability snapshot, so a host can choose a fallback before context or
resource work begins.

For application integration, begin with the typed RenderGraph declaration and
compile it against the selected backend's observed capabilities; bind persistent
objects through provider-owned imports and consume each export's reported
outgoing state. The executor may privately reuse compatible whole-resource
transients, but callers must never depend on a transient's physical identity.
Details and lifetime limits are in [the RenderGraph design](documents/design-rendergraph.md),
[the RHI design](documents/design-rhi.md), and
[ADR-0009](documents/adr/0009-resource-floor-and-reuse-safety.md).

The corresponding supported-target limits and release evidence are maintained
in the ecosystem roadmap and GitHub Releases. The resource floor remains a
closed set of recipes; it does not add a general shader, pipeline, descriptor,
or resource-builder API.

### Unscheduled optimizations

- [ ] Batch compatible render packets while preserving declared ordering.
- [ ] Lower compatible repeated draws as instances when the selected recipe
  permits it.
- [ ] Reuse compatible renderer-side pipelines and bindings across a frame.
- [ ] Add multi-queue scheduling, parallel command recording, recording caches,
  transient resource aliasing, or more aggressive GPU scheduling and memory
  optimization only after representative benchmarks or profiling demonstrate a
  concrete need.

### Out of scope

Durable asset identity and caching, general file or URL loading, broad scene
graphs, animation, application host policy, and higher-level JS or UI APIs
belong to other Fluxel layers. This workspace owns backend bring-up and the
minimal presentation boundary needed to prove RHI and renderer behavior; it
does not own a general platform runtime or host services.

## License

Licensed under either [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at
your option.
