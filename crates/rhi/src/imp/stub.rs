//! Unsupported-platform and disabled-backend implementation of the private RHI boundary.

use crate::{
    Backend, BufferDescriptor, BufferUploadStage, DeviceOptions, HardwareCapabilities,
    HardwareInfo, OpenError, ResourceCreateError, ResourceLease, TextureDescriptor,
    TextureUploadStage,
};
use fluxel_rendergraph::{
    BufferCopyRegion, BufferUsage, CompletionFailure, CompletionStatus, ResourceAccessState,
    TextureCopyRegion, TextureDesc, TextureRange, TextureUsage,
};
use std::sync::Arc;
pub(crate) struct OpenedDevice {
    // `OpenedDevice` is re-exported by the `imp` façade and its facts are
    // consumed by the parent RHI module.  This remains crate-private; using
    // `pub(crate)` here would stop at `imp`, not reach that parent.
    pub(crate) hardware: HardwareInfo,
    pub(crate) capabilities: HardwareCapabilities,
}
pub(crate) struct OwnedBuffer;
pub(crate) struct OwnedTexture;
pub(crate) struct CopyEncoder;
pub(crate) struct CopyCommandBuffer;
pub(crate) struct NativeComputePipeline;
pub(crate) struct NativeComputeBindings;
pub(crate) struct NativeTexturePackBindings;
pub(crate) struct NativeRasterPipeline;
pub(crate) struct NativeRasterUniformBindings;
pub(crate) struct NativeRasterTextureBindings;
#[derive(Clone)]
pub(crate) struct NativeCompletion;
pub(crate) fn open(backend: Backend, _: DeviceOptions) -> Result<OpenedDevice, OpenError> {
    #[cfg(windows)]
    {
        Err(OpenError::BackendDisabled { backend })
    }
    #[cfg(not(windows))]
    {
        Err(OpenError::PlatformUnsupported { backend })
    }
}
pub(crate) fn create_buffer(
    owner: &Arc<OpenedDevice>,
    _: BufferDescriptor,
) -> Result<(OwnedBuffer, BufferUsage), ResourceCreateError> {
    Err(ResourceCreateError::NativeFailure {
        backend: owner.hardware.backend,
        reason: "native resources are only supported on Windows".into(),
    })
}
pub(crate) fn create_texture(
    owner: &Arc<OpenedDevice>,
    _: TextureDescriptor,
) -> Result<(OwnedTexture, TextureUsage), ResourceCreateError> {
    Err(ResourceCreateError::NativeFailure {
        backend: owner.hardware.backend,
        reason: "native resources are only supported on Windows".into(),
    })
}
pub(crate) fn upload_immutable_buffer(
    _: &Arc<OpenedDevice>,
    _: &OwnedBuffer,
    _: ResourceLease,
    _: &[u8],
) -> Result<NativeCompletion, (BufferUploadStage, String)> {
    Err((
        BufferUploadStage::Staging,
        "native uploads are unavailable without a Windows backend".into(),
    ))
}
pub(crate) fn upload_immutable_texture(
    _: &Arc<OpenedDevice>,
    _: &OwnedTexture,
    _: ResourceLease,
    _: TextureDesc,
    _: &[u8],
) -> Result<NativeCompletion, (TextureUploadStage, String)> {
    Err((
        TextureUploadStage::Staging,
        "native uploads are unavailable without a Windows backend".into(),
    ))
}
#[cfg(feature = "test-support")]
#[cfg_attr(
    any(
        not(windows),
        all(windows, not(any(feature = "dx12", feature = "vulkan")))
    ),
    allow(
        dead_code,
        reason = "the unsupported-platform or disabled-backend façade never initializes native capture"
    )
)]
pub(crate) fn initialize_validation_capture() {}
#[cfg(feature = "test-support")]
#[cfg_attr(
    any(
        not(windows),
        all(windows, not(any(feature = "dx12", feature = "vulkan")))
    ),
    allow(
        dead_code,
        reason = "the unsupported-platform or disabled-backend façade returns diagnostics directly"
    )
)]
pub(crate) fn clear_validation_diagnostics(_: &Arc<OpenedDevice>) {}
#[cfg(feature = "test-support")]
#[cfg_attr(
    any(
        not(windows),
        all(windows, not(any(feature = "dx12", feature = "vulkan")))
    ),
    allow(
        dead_code,
        reason = "the unsupported-platform or disabled-backend façade returns diagnostics directly"
    )
)]
pub(crate) fn validation_diagnostics(_: &Arc<OpenedDevice>) -> Vec<String> {
    Vec::new()
}
#[cfg(any(test, feature = "test-support"))]
#[cfg_attr(
    any(
        not(windows),
        all(windows, not(any(feature = "dx12", feature = "vulkan")))
    ),
    allow(
        dead_code,
        reason = "the unsupported-platform or disabled-backend façade rejects readback before this hook"
    )
)]
pub(crate) fn readback_buffer_for_test(
    _: &Arc<OpenedDevice>,
    _: &OwnedBuffer,
    _: ResourceLease,
    _: ResourceAccessState,
    _: u64,
) -> Result<Vec<u8>, String> {
    Err("native readback is unavailable without a Windows backend".into())
}
#[cfg(any(test, feature = "test-support"))]
#[allow(
    dead_code,
    reason = "the no-backend test-support stub preserves the native helper shape"
)]
pub(crate) struct TextureReadback {
    pub(crate) tight: Vec<u8>,
    pub(crate) padded: Vec<u8>,
    pub(crate) bytes_per_row: u32,
}
#[cfg(any(test, feature = "test-support"))]
#[allow(
    dead_code,
    reason = "the no-backend test-support stub preserves a fail-closed readback path"
)]
pub(crate) fn readback_texture_for_test(
    _: &Arc<OpenedDevice>,
    _: &OwnedTexture,
    _: ResourceLease,
    _: TextureDesc,
    _: ResourceAccessState,
) -> Result<TextureReadback, String> {
    Err("native texture readback is unavailable without a Windows backend".into())
}
pub(crate) fn begin_copy_encoder(_: &Arc<OpenedDevice>) -> Result<CopyEncoder, String> {
    Err("native execution is only supported on Windows".into())
}
pub(crate) fn create_compute_pipeline(
    _: &Arc<OpenedDevice>,
    _: &str,
    _: &str,
) -> Result<NativeComputePipeline, String> {
    Err("native compute is only supported on Windows".into())
}
pub(crate) fn create_compute_bindings(
    _: &Arc<OpenedDevice>,
    _: &NativeComputePipeline,
    _: &OwnedBuffer,
    _: u64,
    _: u64,
) -> Result<NativeComputeBindings, String> {
    Err("native compute is only supported on Windows".into())
}
pub(crate) fn create_texture_pack_bindings(
    _: &Arc<OpenedDevice>,
    _: &NativeComputePipeline,
    _: &OwnedTexture,
    _: &OwnedBuffer,
    _: u64,
    _: u64,
) -> Result<NativeTexturePackBindings, String> {
    Err("native compute is only supported on Windows".into())
}
pub(crate) fn create_raster_pipeline(
    _: &Arc<OpenedDevice>,
    _: &str,
    _: &str,
    _: &str,
    _: crate::RasterKernel,
) -> Result<NativeRasterPipeline, String> {
    Err("native raster is only supported on Windows".into())
}
pub(crate) fn create_raster_uniform_bindings(
    _: &Arc<OpenedDevice>,
    _: &NativeRasterPipeline,
    _: &OwnedBuffer,
) -> Result<NativeRasterUniformBindings, String> {
    Err("native raster is only supported on Windows".into())
}
#[allow(
    clippy::too_many_arguments,
    reason = "the closed native normal recipe passes all independently validated role facts"
)]
pub(crate) fn create_raster_normal_bindings(
    _: &Arc<OpenedDevice>,
    _: &NativeRasterPipeline,
    _: &OwnedBuffer,
    _: fluxel_rendergraph::PhysicalResourceIdentity,
    _: u64,
    _: fluxel_rendergraph::PhysicalResourceIdentity,
    _: u64,
) -> Result<NativeRasterUniformBindings, String> {
    Err("native raster is only supported on Windows".into())
}
#[allow(
    clippy::too_many_arguments,
    reason = "matches the closed native vertex-color ABI"
)]
pub(crate) fn create_raster_vertex_color_bindings(
    _: &Arc<OpenedDevice>,
    _: &NativeRasterPipeline,
    _: &OwnedBuffer,
    _: fluxel_rendergraph::PhysicalResourceIdentity,
    _: u64,
    _: fluxel_rendergraph::PhysicalResourceIdentity,
    _: u64,
) -> Result<NativeRasterUniformBindings, String> {
    Err("native raster is only supported on Windows".into())
}
pub(crate) fn create_raster_texture_bindings(
    _: &Arc<OpenedDevice>,
    _: &NativeRasterPipeline,
    _: &OwnedBuffer,
    _: &OwnedTexture,
) -> Result<NativeRasterTextureBindings, String> {
    Err("native raster is only supported on Windows".into())
}
#[allow(
    clippy::too_many_arguments,
    reason = "the closed native UV recipe passes all independently validated role facts"
)]
pub(crate) fn create_raster_uv_texture_bindings(
    _: &Arc<OpenedDevice>,
    _: &NativeRasterPipeline,
    _: &OwnedBuffer,
    _: &OwnedTexture,
    _: fluxel_rendergraph::PhysicalResourceIdentity,
    _: u64,
    _: fluxel_rendergraph::PhysicalResourceIdentity,
    _: u64,
) -> Result<NativeRasterTextureBindings, String> {
    Err("native raster is only supported on Windows".into())
}
#[allow(
    clippy::too_many_arguments,
    reason = "the closed native linear-clamp UV recipe passes all independently validated role facts"
)]
pub(crate) fn create_raster_uv_linear_clamp_texture_bindings(
    _: &Arc<OpenedDevice>,
    _: &NativeRasterPipeline,
    _: &OwnedBuffer,
    _: &OwnedTexture,
    _: fluxel_rendergraph::PhysicalResourceIdentity,
    _: u64,
    _: fluxel_rendergraph::PhysicalResourceIdentity,
    _: u64,
) -> Result<NativeRasterTextureBindings, String> {
    Err("native raster is only supported on Windows".into())
}
#[allow(
    clippy::too_many_arguments,
    reason = "the closed native sRGB linear-clamp UV recipe passes all independently validated role facts"
)]
pub(crate) fn create_raster_uv_linear_clamp_srgb_texture_bindings(
    _: &Arc<OpenedDevice>,
    _: &NativeRasterPipeline,
    _: &OwnedBuffer,
    _: &OwnedTexture,
    _: fluxel_rendergraph::PhysicalResourceIdentity,
    _: u64,
    _: fluxel_rendergraph::PhysicalResourceIdentity,
    _: u64,
) -> Result<NativeRasterTextureBindings, String> {
    Err("native raster is only supported on Windows".into())
}
pub(crate) fn transition_texture(
    _: &mut CopyEncoder,
    _: &OwnedTexture,
    _: TextureDesc,
    _: TextureRange,
    _: ResourceAccessState,
    _: ResourceAccessState,
) -> Result<(), String> {
    Err("native execution is only supported on Windows".into())
}
pub(crate) fn transition_buffer(
    _: &mut CopyEncoder,
    _: &OwnedBuffer,
    _: ResourceAccessState,
    _: ResourceAccessState,
) -> Result<(), String> {
    Err("native execution is only supported on Windows".into())
}
pub(crate) fn begin_compute(_: &mut CopyEncoder, _: &str) -> Result<(), String> {
    Err("native compute is only supported on Windows".into())
}
#[allow(
    clippy::too_many_arguments,
    reason = "fail-closed stub mirrors the native raster boundary exactly"
)]
pub(crate) fn begin_raster(
    _: &mut CopyEncoder,
    _: &OwnedTexture,
    _: TextureDesc,
    _: Option<[f32; 4]>,
    _: bool,
    _: bool,
    _: Option<(&OwnedTexture, TextureDesc, Option<f32>, bool)>,
    _: &str,
) -> Result<(), String> {
    Err("native raster is only supported on Windows".into())
}
pub(crate) fn end_raster(_: &mut CopyEncoder) -> Result<(), String> {
    Err("native raster is only supported on Windows".into())
}
pub(crate) fn end_compute(_: &mut CopyEncoder) -> Result<(), String> {
    Err("native compute is only supported on Windows".into())
}
pub(crate) fn set_compute_pipeline(
    _: &mut CopyEncoder,
    _: &NativeComputePipeline,
) -> Result<(), String> {
    Err("native compute is only supported on Windows".into())
}
pub(crate) fn set_compute_bindings(
    _: &mut CopyEncoder,
    _: &NativeComputeBindings,
) -> Result<(), String> {
    Err("native compute is only supported on Windows".into())
}
pub(crate) fn set_texture_pack_bindings(
    _: &mut CopyEncoder,
    _: &NativeTexturePackBindings,
) -> Result<(), String> {
    Err("native compute is only supported on Windows".into())
}
pub(crate) fn set_raster_pipeline(
    _: &mut CopyEncoder,
    _: &NativeRasterPipeline,
) -> Result<(), String> {
    Err("native raster is only supported on Windows".into())
}
pub(crate) fn set_raster_uniform_bindings(
    _: &mut CopyEncoder,
    _: &NativeRasterUniformBindings,
) -> Result<(), String> {
    Err("native raster is only supported on Windows".into())
}
pub(crate) fn set_raster_texture_bindings(
    _: &mut CopyEncoder,
    _: &NativeRasterTextureBindings,
) -> Result<(), String> {
    Err("native raster is only supported on Windows".into())
}
pub(crate) fn set_raster_uv_texture_bindings(
    _: &mut CopyEncoder,
    _: &NativeRasterTextureBindings,
) -> Result<(), String> {
    Err("native raster is only supported on Windows".into())
}
pub(crate) fn set_raster_uv_linear_clamp_texture_bindings(
    _: &mut CopyEncoder,
    _: &NativeRasterTextureBindings,
) -> Result<(), String> {
    Err("native raster is only supported on Windows".into())
}
pub(crate) fn set_raster_uv_linear_clamp_srgb_texture_bindings(
    _: &mut CopyEncoder,
    _: &NativeRasterTextureBindings,
) -> Result<(), String> {
    Err("native raster is only supported on Windows".into())
}
/// Mirrors the native closed-slot minimum range rule on unsupported platforms.
#[allow(
    dead_code,
    reason = "the non-Windows mirror is consumed only by cross-platform contract tests"
)]
pub(crate) const fn raster_vertex_minimum_size(kernel: crate::RasterKernel, slot: u32) -> u64 {
    match (kernel, slot) {
        (
            crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUv
            | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
            | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb,
            1,
        ) => 8,
        (crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialVertexColor, 1) => 4,
        _ => 12,
    }
}
#[allow(
    clippy::too_many_arguments,
    reason = "the closed native UV recipe passes all independently validated role facts"
)]
pub(crate) fn set_vertex_buffer(
    _: &mut CopyEncoder,
    _: &OwnedBuffer,
    _: u64,
    _: u64,
    _: crate::RasterKernel,
    _: u32,
    _: fluxel_rendergraph::PhysicalResourceIdentity,
    _: Option<&NativeRasterTextureBindings>,
    _: Option<&NativeRasterUniformBindings>,
    _: Option<&NativeRasterUniformBindings>,
) -> Result<(), String> {
    Err("native raster is only supported on Windows".into())
}
pub(crate) fn set_index_buffer(
    _: &mut CopyEncoder,
    _: &OwnedBuffer,
    _: u64,
    _: u64,
    _: fluxel_rendergraph::IndexFormat,
) -> Result<(), String> {
    Err("native raster is only supported on Windows".into())
}
pub(crate) fn set_viewport(
    _: &mut CopyEncoder,
    _: f32,
    _: f32,
    _: f32,
    _: f32,
    _: f32,
    _: f32,
) -> Result<(), String> {
    Err("native raster is only supported on Windows".into())
}
pub(crate) fn set_scissor(
    _: &mut CopyEncoder,
    _: u32,
    _: u32,
    _: u32,
    _: u32,
) -> Result<(), String> {
    Err("native raster is only supported on Windows".into())
}
pub(crate) fn draw(_: &mut CopyEncoder, _: u32, _: u32, _: u32, _: u32) -> Result<(), String> {
    Err("native raster is only supported on Windows".into())
}
pub(crate) fn draw_indexed(
    _: &mut CopyEncoder,
    _: u32,
    _: u32,
    _: i32,
    _: u32,
    _: u32,
) -> Result<(), String> {
    Err("native raster is only supported on Windows".into())
}
pub(crate) fn dispatch(_: &mut CopyEncoder, _: [u32; 3]) -> Result<(), String> {
    Err("native compute is only supported on Windows".into())
}
pub(crate) fn copy_texture(
    _: &mut CopyEncoder,
    _: &OwnedTexture,
    _: &OwnedTexture,
    _: TextureDesc,
    _: TextureCopyRegion,
) -> Result<(), String> {
    Err("native execution is only supported on Windows".into())
}
pub(crate) fn copy_buffer(
    _: &mut CopyEncoder,
    _: &OwnedBuffer,
    _: &OwnedBuffer,
    _: BufferCopyRegion,
) -> Result<(), String> {
    Err("native execution is only supported on Windows".into())
}
pub(crate) fn finish_copy_encoder(_: CopyEncoder) -> Result<CopyCommandBuffer, String> {
    Err("native execution is only supported on Windows".into())
}
pub(crate) fn submit_copy(
    _: CopyCommandBuffer,
    _: Vec<ResourceLease>,
) -> Result<NativeCompletion, String> {
    Err("native execution is only supported on Windows".into())
}
pub(crate) fn completion_status(_: &NativeCompletion) -> Result<CompletionStatus, String> {
    Ok(CompletionStatus::Failed(CompletionFailure::DeviceLost))
}
pub(crate) fn wait_completion(
    _: &NativeCompletion,
    _: core::time::Duration,
) -> Result<CompletionStatus, String> {
    Ok(CompletionStatus::Failed(CompletionFailure::DeviceLost))
}
