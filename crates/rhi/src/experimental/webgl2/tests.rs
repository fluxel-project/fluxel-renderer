//! Browser contracts for WebGL2 capability rejection before session opening.

use fluxel_rendergraph::{
    BufferDesc, BufferRange, BufferReadUse, BufferReadWriteUse, BufferWriteUse,
    CapabilityRequirement, CompileErrorKind, Extent3d, ExternalOwnership, ImportBufferContract,
    ImportTextureContract, InitialContents, PassKind, RenderGraph, ResourceAccessState,
    SideEffectReason, TextureDesc, TextureDimension, TextureFormat, TextureRange, TextureReadUse,
    TextureReadWriteUse, TextureWriteUse, WriteCoverage,
};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_test::*;

use super::*;

wasm_bindgen_test_configure!(run_in_browser);

fn canvas() -> web_sys::HtmlCanvasElement {
    let document = web_sys::window()
        .expect("browser window")
        .document()
        .expect("browser document");
    let canvas = document
        .create_element("canvas")
        .expect("create canvas")
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .expect("canvas element");
    canvas.set_width(1);
    canvas.set_height(1);
    canvas
}

#[derive(Clone, Copy)]
enum StorageAccess {
    Read,
    Write,
    ReadWrite,
}

fn texture() -> TextureDesc {
    TextureDesc {
        dimension: TextureDimension::D2,
        extent: Extent3d {
            width: 4,
            height: 4,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
    }
}

fn assert_rejected(graph: &RenderGraph<()>, requirement: CapabilityRequirement) {
    let error = match WebGl2Session::new_for_graph(JsValue::NULL, graph) {
        Err(WebGl2GraphOpenError::Compile(error)) => error,
        Err(WebGl2GraphOpenError::Session(error)) => {
            panic!("compile rejection must precede the invalid canvas: {error:?}")
        }
        Ok(_) => panic!("WebGL2 must fail closed"),
    };
    assert_eq!(error.kind, CompileErrorKind::UnsupportedSemanticRequirement);
    let unsupported = error
        .context
        .unsupported
        .as_deref()
        .expect("unsupported capability must be structured");
    assert_eq!(unsupported.requirement, requirement);
    assert_eq!(*unsupported.observed, webgl2_capabilities());
    assert_eq!(
        browser_side_effects(),
        BrowserSideEffects::default(),
        "compile-time rejection must not create a context/resource, record, or submit"
    );
}

fn compute_graph() -> RenderGraph<()> {
    let mut graph = RenderGraph::new();
    let pass = graph.add_compute_pass(
        "unsupported-compute",
        |_| ((), ()),
        |_commands, _resolver, _data, _frame| Ok(()),
    );
    graph.mark_side_effect(
        pass.id,
        SideEffectReason::Diagnostic("retain compute capability witness".into()),
    );
    graph
}

fn storage_buffer_graph(access: StorageAccess) -> RenderGraph<()> {
    let mut graph = RenderGraph::new();
    let buffer = graph.import_buffer_slot(
        "storage-buffer",
        ImportBufferContract {
            descriptor: BufferDesc { size: 16 },
            initial_state: ResourceAccessState::CopySource,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let pass = graph.add_raster_pass(
        "unsupported-storage-buffer",
        |pass| {
            match access {
                StorageAccess::Read => {
                    let _ = pass.read_buffer(
                        &buffer.version,
                        BufferReadUse::Storage,
                        BufferRange::whole(),
                    );
                }
                StorageAccess::Write => {
                    let _ = pass.write_buffer(
                        buffer.version,
                        BufferWriteUse::Storage,
                        BufferRange::whole(),
                        WriteCoverage::Full,
                    );
                }
                StorageAccess::ReadWrite => {
                    let _ = pass.read_write_buffer(
                        buffer.version,
                        BufferReadWriteUse::Storage,
                        BufferRange::whole(),
                    );
                }
            }
            ((), ())
        },
        |_commands, _resolver, _data, _frame| Ok(()),
    );
    graph.mark_side_effect(
        pass.id,
        SideEffectReason::Diagnostic("retain storage-buffer capability witness".into()),
    );
    graph
}

fn storage_texture_graph(access: StorageAccess) -> RenderGraph<()> {
    let mut graph = RenderGraph::new();
    let image = graph.import_texture_slot(
        "storage-texture",
        ImportTextureContract {
            descriptor: texture(),
            initial_state: ResourceAccessState::ColorAttachmentWrite,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let pass = graph.add_raster_pass(
        "unsupported-storage-texture",
        |pass| {
            match access {
                StorageAccess::Read => {
                    let _ = pass.read_texture(
                        &image.version,
                        TextureReadUse::Storage,
                        TextureRange::whole(),
                    );
                }
                StorageAccess::Write => {
                    let _ = pass.write_texture(
                        image.version,
                        TextureWriteUse::Storage,
                        TextureRange::whole(),
                        WriteCoverage::Full,
                    );
                }
                StorageAccess::ReadWrite => {
                    let _ = pass.read_write_texture(
                        image.version,
                        TextureReadWriteUse::Storage,
                        TextureRange::whole(),
                    );
                }
            }
            ((), ())
        },
        |_commands, _resolver, _data, _frame| Ok(()),
    );
    graph.mark_side_effect(
        pass.id,
        SideEffectReason::Diagnostic("retain storage-texture capability witness".into()),
    );
    graph
}

#[wasm_bindgen_test]
fn compute_is_rejected_before_any_browser_side_effect() {
    reset_browser_side_effects();
    assert_rejected(
        &compute_graph(),
        CapabilityRequirement::Queue {
            pass_kinds: vec![PassKind::Compute],
            present: false,
        },
    );
}

#[wasm_bindgen_test]
fn every_storage_buffer_access_is_structured_and_side_effect_free() {
    for (access, state) in [
        (StorageAccess::Read, ResourceAccessState::ShaderStorageRead),
        (
            StorageAccess::Write,
            ResourceAccessState::ShaderStorageWrite,
        ),
        (
            StorageAccess::ReadWrite,
            ResourceAccessState::ShaderStorageReadWrite,
        ),
    ] {
        reset_browser_side_effects();
        assert_rejected(
            &storage_buffer_graph(access),
            CapabilityRequirement::BufferState { state },
        );
    }
}

#[wasm_bindgen_test]
fn every_storage_texture_access_is_structured_and_side_effect_free() {
    for (access, state) in [
        (StorageAccess::Read, ResourceAccessState::ShaderStorageRead),
        (
            StorageAccess::Write,
            ResourceAccessState::ShaderStorageWrite,
        ),
        (
            StorageAccess::ReadWrite,
            ResourceAccessState::ShaderStorageReadWrite,
        ),
    ] {
        reset_browser_side_effects();
        assert_rejected(
            &storage_texture_graph(access),
            CapabilityRequirement::TextureState {
                format: TextureFormat::Rgba8Unorm,
                sample_count: 1,
                state,
            },
        );
    }
}

#[wasm_bindgen_test]
fn capability_factory_advertises_only_the_fixed_default_framebuffer_recipe() {
    let caps = webgl2_capabilities();
    let queue = &caps.queues[0].capabilities;
    let rgba8 = caps
        .texture_formats
        .iter()
        .find(|facts| facts.format == TextureFormat::Rgba8Unorm)
        .expect("closed default-framebuffer format is advertised");
    assert!(queue.raster && queue.present);
    assert!(!queue.compute && !queue.copy);
    assert!(!rgba8.sampled && !rgba8.copy_source && !rgba8.copy_destination);
    assert!(!caps.transient_resources.cross_frame_object_pooling);
    assert!(!caps.transient_resources.in_frame_object_reuse);
    assert!(!caps.transient_resources.aliased_memory);
    assert!(
        !caps
            .surface
            .as_ref()
            .expect("surface facts")
            .copy_destination
    );
}

#[wasm_bindgen_test]
fn fixed_resource_floor_uses_real_webgl2_upload_copy_depth_sampling_and_readback() {
    let canvas = canvas();
    let mut session = WebGl2Session::new(canvas.into()).expect("open real WebGL2 session");
    let evidence = session
        .run_resource_floor_fixture()
        .expect("fixed common-resource recipe must execute");
    assert_eq!(evidence.sampled_default_pixel, [51, 102, 153, 255]);
    assert_eq!(evidence.copied_buffer_bytes, [51, 102, 153, 255]);
    assert_eq!(evidence.copied_texture_pixel, [51, 102, 153, 255]);
    assert!(evidence.depth32float_attachment_usable);
    assert_eq!(
        session
            .run_resource_floor_fixture()
            .expect("retained fixture"),
        evidence,
        "the recipe's resources remain session-owned until disposal"
    );
    session.dispose().expect("dispose resource-floor objects");
}
