//! Shared native witness setup and entry points.

use super::native_compute::{
    build_k01, build_k02, build_x01, hardware_guard, run_c02, run_compute_case_bundle, run_k02,
    run_s01, run_t01_transient_reuse, run_x01, run_x01_case,
};
use super::native_compute_negative::run_compute_negative_paths;
use super::native_copy::{run_c01, run_c03};
use super::native_raster::{
    build_r01, build_r02, run_k01, run_r01, run_r01_case, run_r02, run_r02_case,
};
use super::native_raster_state::run_raster_negative_paths;
use crate::*;
use fluxel_rendergraph::*;
use std::collections::HashMap;

pub(super) struct Resources {
    pub(super) device: fluxel_rendergraph::DeviceIdentity,
    pub(super) buffers: HashMap<BufferBindingId, (Buffer, ResourceAccessState)>,
    pub(super) textures: HashMap<TextureBindingId, (Texture, ResourceAccessState)>,
}

impl FrameResourceProvider<CopyBackend> for Resources {
    fn texture(
        &self,
        id: fluxel_rendergraph::TextureBindingId,
    ) -> Result<BoundTexture<Texture, ResourceLease>, FrameBindingError> {
        let (texture, state) = self.textures.get(&id).ok_or_else(|| {
            missing(
                FrameBindingErrorKind::MissingTexture,
                "texture not registered",
            )
        })?;
        Ok(BoundTexture {
            device: self.device,
            identity: texture.identity(),
            physical: texture.clone(),
            descriptor: texture.descriptor().texture,
            usage: texture.allowed_usage(),
            initial_state: *state,
            lease: texture.lease().into(),
        })
    }
    fn buffer(
        &self,
        id: BufferBindingId,
    ) -> Result<BoundBuffer<Buffer, ResourceLease>, FrameBindingError> {
        let (buffer, state) = self.buffers.get(&id).ok_or_else(|| {
            missing(
                FrameBindingErrorKind::MissingBuffer,
                "buffer not registered",
            )
        })?;
        Ok(BoundBuffer {
            device: self.device,
            identity: buffer.identity(),
            physical: buffer.clone(),
            descriptor: buffer.descriptor().buffer,
            usage: buffer.allowed_usage(),
            initial_state: *state,
            lease: buffer.lease().into(),
        })
    }
}

impl FrameResourceProvider<ComputeBackend> for Resources {
    fn texture(
        &self,
        id: fluxel_rendergraph::TextureBindingId,
    ) -> Result<BoundTexture<Texture, ResourceLease>, FrameBindingError> {
        <Self as FrameResourceProvider<CopyBackend>>::texture(self, id)
    }
    fn buffer(
        &self,
        id: BufferBindingId,
    ) -> Result<BoundBuffer<Buffer, ResourceLease>, FrameBindingError> {
        <Self as FrameResourceProvider<CopyBackend>>::buffer(self, id)
    }
}

impl FrameResourceProvider<RasterBackend> for Resources {
    fn texture(
        &self,
        id: fluxel_rendergraph::TextureBindingId,
    ) -> Result<BoundTexture<Texture, ResourceLease>, FrameBindingError> {
        <Self as FrameResourceProvider<CopyBackend>>::texture(self, id)
    }
    fn buffer(
        &self,
        id: BufferBindingId,
    ) -> Result<BoundBuffer<Buffer, ResourceLease>, FrameBindingError> {
        <Self as FrameResourceProvider<CopyBackend>>::buffer(self, id)
    }
}

fn missing(kind: FrameBindingErrorKind, detail: &str) -> FrameBindingError {
    FrameBindingError {
        kind,
        texture_slot: None,
        buffer_slot: None,
        resource: None,
        surface_binding: None,
        detail: detail.into(),
    }
}

pub(super) struct NoObjects;
impl RenderObjectProvider<CopyBackend> for NoObjects {
    fn raster_pipeline(
        &self,
        _: RasterPipelineId,
    ) -> Result<BoundRasterPipeline<UnsupportedRasterPipeline, ResourceLease>, RecordingError> {
        Err(no_object())
    }
    fn compute_pipeline(
        &self,
        _: ComputePipelineId,
    ) -> Result<BoundComputePipeline<UnsupportedComputePipeline, ResourceLease>, RecordingError>
    {
        Err(no_object())
    }
    fn bindings(
        &self,
        _: BindingSetId,
        _: &[ResolvedBindingResource<'_, Texture, Buffer>],
        _: &[u32],
    ) -> Result<BoundBindings<UnsupportedBindings, ResourceLease>, RecordingError> {
        Err(no_object())
    }
}

// T01 uses ComputeBackend solely because its test-only observation sink lives
// there; its graph contains only copy passes, so every object lookup remains a
// fail-closed guard.
impl RenderObjectProvider<ComputeBackend> for NoObjects {
    fn raster_pipeline(
        &self,
        _: RasterPipelineId,
    ) -> Result<BoundRasterPipeline<UnsupportedRasterPipeline, ResourceLease>, RecordingError> {
        Err(no_object())
    }
    fn compute_pipeline(
        &self,
        _: ComputePipelineId,
    ) -> Result<BoundComputePipeline<ComputePipeline, ResourceLease>, RecordingError> {
        Err(no_object())
    }
    fn bindings(
        &self,
        _: BindingSetId,
        _: &[ResolvedBindingResource<'_, Texture, Buffer>],
        _: &[u32],
    ) -> Result<BoundBindings<ComputeBindings, ResourceLease>, RecordingError> {
        Err(no_object())
    }
}
fn no_object() -> RecordingError {
    RecordingError {
        kind: RecordingErrorKind::MissingFrameBinding,
        context: DiagnosticContext {
            passes: vec![],
            resource: None,
            texture_slot: None,
            buffer_slot: None,
            detail: "copy fixture has no render objects".into(),
            unsupported: None,
        },
    }
}

pub(super) struct CopyData {
    source: BufferRead,
    destination: BufferWrite,
    source_offset: u64,
    destination_offset: u64,
    size: u64,
}

pub(super) struct TextureCopyData {
    pub(super) source: TextureRead,
    pub(super) destination: TextureWrite,
    pub(super) region: TextureCopyRegion,
}

pub(super) fn add_copy(
    graph: &mut RenderGraph,
    name: &str,
    source: &fluxel_rendergraph::BufferVersion,
    destination: fluxel_rendergraph::BufferVersion,
    source_offset: u64,
    destination_offset: u64,
    size: u64,
) -> fluxel_rendergraph::BufferVersion {
    graph
        .add_copy_pass(
            name,
            |pass| {
                let src = pass.read_buffer(
                    source,
                    BufferRange::Bytes {
                        offset: source_offset,
                        size,
                    },
                );
                let (out, dst) = pass.write_buffer(
                    destination,
                    BufferRange::Bytes {
                        offset: destination_offset,
                        size,
                    },
                    WriteCoverage::Full,
                );
                (
                    out,
                    CopyData {
                        source: src,
                        destination: dst,
                        source_offset,
                        destination_offset,
                        size,
                    },
                )
            },
            |commands, _resolver: &mut PassResourceResolver<'_>, data, _frame| -> RecordResult {
                commands.copy_buffer(
                    &data.source,
                    &data.destination,
                    BufferCopyRegion {
                        source_offset: data.source_offset,
                        destination_offset: data.destination_offset,
                        size: data.size,
                    },
                )
            },
        )
        .output
}

macro_rules! native_case {
    ($name:ident, $runner:ident, $backend:expr) => {
        #[test]
        #[ignore = "requires native validation plus a real GPU"]
        fn $name() {
            let _guard = hardware_guard();
            crate::imp::initialize_validation_capture();
            $runner($backend);
        }
    };
}

native_case!(c01_dx12_partial_buffer_copy, run_c01, crate::Backend::Dx12);
native_case!(
    c01_vulkan_partial_buffer_copy,
    run_c01,
    crate::Backend::Vulkan
);
native_case!(
    c02_dx12_texture_copy_row_padding,
    run_c02,
    crate::Backend::Dx12
);
native_case!(
    c02_vulkan_texture_copy_row_padding,
    run_c02,
    crate::Backend::Vulkan
);
native_case!(
    c03_dx12_same_state_overlapping_waw,
    run_c03,
    crate::Backend::Dx12
);
native_case!(
    c03_vulkan_same_state_overlapping_waw,
    run_c03,
    crate::Backend::Vulkan
);
native_case!(k01_dx12_wrapping_add, run_k01, crate::Backend::Dx12);
native_case!(k01_vulkan_wrapping_add, run_k01, crate::Backend::Vulkan);
native_case!(k02_dx12_ordered_add_multiply, run_k02, crate::Backend::Dx12);
native_case!(
    k02_vulkan_ordered_add_multiply,
    run_k02,
    crate::Backend::Vulkan
);
native_case!(
    t01_dx12_cross_frame_transient_reuse,
    run_t01_transient_reuse,
    crate::Backend::Dx12
);
native_case!(
    t01_vulkan_cross_frame_transient_reuse,
    run_t01_transient_reuse,
    crate::Backend::Vulkan
);
native_case!(
    s01_dx12_storage_texture_store_load,
    run_s01,
    crate::Backend::Dx12
);
native_case!(
    s01_vulkan_storage_texture_store_load,
    run_s01,
    crate::Backend::Vulkan
);
native_case!(r01_dx12_clear_triangle, run_r01, crate::Backend::Dx12);
native_case!(r01_vulkan_clear_triangle, run_r01, crate::Backend::Vulkan);
native_case!(
    r02_dx12_indexed_viewport_scissor,
    run_r02,
    crate::Backend::Dx12
);
native_case!(
    r02_vulkan_indexed_viewport_scissor,
    run_r02,
    crate::Backend::Vulkan
);
native_case!(x01_dx12_raster_compute_copy, run_x01, crate::Backend::Dx12);
native_case!(
    x01_vulkan_raster_compute_copy,
    run_x01,
    crate::Backend::Vulkan
);
#[test]
#[ignore = "requires native validation plus a real GPU"]
fn r01_paired_same_compiled_plan() {
    let _guard = hardware_guard();
    crate::imp::initialize_validation_capture();
    let case = build_r01();
    run_r01_case(crate::Backend::Dx12, &case, "paired-same-compiled-plan");
    run_r01_case(crate::Backend::Vulkan, &case, "paired-same-compiled-plan");
}
#[test]
#[ignore = "requires native validation plus a real GPU"]
fn r02_paired_same_compiled_plan() {
    let _guard = hardware_guard();
    crate::imp::initialize_validation_capture();
    let case = build_r02();
    run_r02_case(crate::Backend::Dx12, &case, "paired-same-compiled-plan");
    run_r02_case(crate::Backend::Vulkan, &case, "paired-same-compiled-plan");
}
#[test]
#[ignore = "requires native validation plus a real GPU"]
fn x01_paired_same_compiled_plan() {
    let _guard = hardware_guard();
    crate::imp::initialize_validation_capture();
    let case = build_x01();
    run_x01_case(crate::Backend::Dx12, &case, "paired-same-compiled-plan");
    run_x01_case(crate::Backend::Vulkan, &case, "paired-same-compiled-plan");
}
#[test]
#[ignore = "requires native validation plus a real GPU"]
fn k01_paired_same_compiled_plan() {
    let _guard = hardware_guard();
    crate::imp::initialize_validation_capture();
    let case = build_k01();
    run_compute_case_bundle("K01", crate::Backend::Dx12, &case);
    run_compute_case_bundle("K01", crate::Backend::Vulkan, &case);
}
#[test]
#[ignore = "requires native validation plus a real GPU"]
fn k02_paired_same_compiled_plan() {
    let _guard = hardware_guard();
    crate::imp::initialize_validation_capture();
    let case = build_k02();
    run_compute_case_bundle("K02", crate::Backend::Dx12, &case);
    run_compute_case_bundle("K02", crate::Backend::Vulkan, &case);
}
native_case!(
    negative_dx12_compute_fail_closed,
    run_compute_negative_paths_exact,
    crate::Backend::Dx12
);
native_case!(
    negative_vulkan_compute_fail_closed,
    run_compute_negative_paths_exact,
    crate::Backend::Vulkan
);
native_case!(
    negative_dx12_raster_fail_closed,
    run_raster_negative_paths_exact,
    crate::Backend::Dx12
);
native_case!(
    negative_vulkan_raster_fail_closed,
    run_raster_negative_paths_exact,
    crate::Backend::Vulkan
);
native_case!(
    failure_dx12_submission_contract,
    run_submission_failure_paths,
    crate::Backend::Dx12
);
native_case!(
    failure_vulkan_submission_contract,
    run_submission_failure_paths,
    crate::Backend::Vulkan
);

#[test]
fn negative_commit_helper_accepts_only_full_hex_sha_values() {
    assert!(negative_commit_is_valid(&"0".repeat(40)));
    assert!(negative_commit_is_valid(
        "0123456789abcdef0123456789abcdef01234567"
    ));
    assert!(!negative_commit_is_valid("working-tree"));
    assert!(!negative_commit_is_valid(&"0".repeat(39)));
    assert!(!negative_commit_is_valid(&"g".repeat(40)));
}

pub(super) fn run_submission_failure_paths(backend_kind: crate::Backend) {
    let device = Device::open(
        backend_kind,
        crate::DeviceOptions {
            validation: crate::Validation::Required,
            ..crate::DeviceOptions::default()
        },
    )
    .unwrap();
    crate::imp::clear_validation_diagnostics(&device.inner);
    let mut backend = CopyBackend::new(device.clone());

    crate::imp::inject_submit_rejected_once();
    let encoder = backend.begin_encoder(QueueId::new(0)).unwrap();
    let command = backend.finish_encoder(encoder).unwrap();
    assert!(matches!(
        backend.submit(QueueId::new(0), command, Vec::new()),
        Err(NativeExecutionError::SubmitRejected(_))
    ));

    crate::imp::inject_submit_accepted_unknown_once();
    let encoder = backend.begin_encoder(QueueId::new(0)).unwrap();
    let command = backend.finish_encoder(encoder).unwrap();
    let completion = backend
        .submit(QueueId::new(0), command, Vec::new())
        .unwrap();
    assert_eq!(
        backend.completion_status(&completion),
        CompletionStatus::Failed(CompletionFailure::DeviceLost)
    );
    assert_eq!(
        backend.wait(&completion, Duration::from_secs(1)),
        Err(WaitError::Failed(CompletionFailure::DeviceLost))
    );
    drop(completion);

    let encoder = backend.begin_encoder(QueueId::new(0)).unwrap();
    let command = backend.finish_encoder(encoder).unwrap();
    let completion = backend
        .submit(QueueId::new(0), command, Vec::new())
        .unwrap();
    crate::imp::inject_completion_pending_once();
    assert_eq!(
        backend.completion_status(&completion),
        CompletionStatus::Pending
    );
    backend.wait(&completion, Duration::from_secs(10)).unwrap();

    // The accepted completion owns its native bundle independently from
    // the backend; dropping either side first is nonblocking and safe.
    drop(backend);
    drop(completion);
    assert!(crate::imp::validation_diagnostics(&device.inner).is_empty());
}

/// Exact-SHA evidence is opt-in: an all-zero marker explicitly labels a
/// working-tree run, while any nonzero marker must prove it is the clean
/// checked-out HEAD before a negative hardware fixture opens a device.
pub(super) fn negative_exact_commit() -> String {
    let commit = std::env::var("FLUXEL_TEST_COMMIT").unwrap_or_else(|_| "0".repeat(40));
    assert!(
        negative_commit_is_valid(&commit),
        "negative hardware evidence requires FLUXEL_TEST_COMMIT to be exactly 40 hexadecimal characters"
    );
    if commit.bytes().all(|byte| byte == b'0') {
        return "working-tree".into();
    }
    let head = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .expect("negative exact-SHA evidence needs git rev-parse HEAD");
    assert!(
        head.status.success(),
        "negative exact-SHA evidence git rev-parse HEAD failed"
    );
    assert_eq!(
        String::from_utf8(head.stdout)
            .expect("git rev-parse HEAD emitted non-UTF-8 output")
            .trim(),
        commit,
        "negative exact-SHA evidence commit differs from HEAD"
    );
    let status = std::process::Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=normal"])
        .output()
        .expect("negative exact-SHA evidence needs git status");
    assert!(
        status.status.success(),
        "negative exact-SHA evidence git status failed"
    );
    assert!(
        status.stdout.is_empty(),
        "negative exact-SHA evidence requires a clean worktree: {}",
        String::from_utf8_lossy(&status.stdout)
    );
    commit
}

pub(super) fn negative_commit_is_valid(commit: &str) -> bool {
    commit.len() == 40 && commit.bytes().all(|byte| byte.is_ascii_hexdigit())
}

pub(super) fn run_compute_negative_paths_exact(backend_kind: crate::Backend) {
    let commit = negative_exact_commit();
    run_compute_negative_paths(backend_kind);
    eprintln!(
        "artifact case=negative-compute backend={backend_kind:?} commit={commit} exact_sha={} diagnostics=empty completion=no-submit",
        commit != "working-tree"
    );
}

pub(super) fn run_raster_negative_paths_exact(backend_kind: crate::Backend) {
    let commit = negative_exact_commit();
    run_raster_negative_paths(backend_kind);
    eprintln!(
        "artifact case=negative-raster backend={backend_kind:?} commit={commit} exact_sha={} diagnostics=empty completion=no-submit",
        commit != "working-tree"
    );
}
