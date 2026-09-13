//! Raster capability derivation from native device facts.
//!
//! The resulting profile advertises only the fixed raster recipes this backend
//! can validate and record. Command recording and submission are owned by the
//! sibling operation modules.

use super::*;
use crate::execution::helpers::require_device;

impl RasterBackend {
    pub(in crate::execution) fn is_uv_kernel(kernel: crate::RasterKernel) -> bool {
        matches!(
            kernel,
            crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUv
                | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
                | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
        )
    }

    pub(in crate::execution) fn is_normal_kernel(kernel: crate::RasterKernel) -> bool {
        kernel == crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialNormalLambert
    }
    pub(in crate::execution) fn is_vertex_color_kernel(kernel: crate::RasterKernel) -> bool {
        kernel == crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialVertexColor
    }

    /// Creates the only backend profile that can record the fixed raster slice.
    pub fn new(device: Device) -> Self {
        Self {
            capabilities: raster_capabilities(&device),
            device,
            retired: Vec::new(),
        }
    }

    /// Creates the fixed backend only for a surface-compatible device.
    #[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
    pub(crate) fn for_surface(device: Device) -> Self {
        Self {
            capabilities: raster_surface_capabilities(&device),
            device,
            retired: Vec::new(),
        }
    }

    /// Returns the conservative cross-backend capability profile.
    pub fn portable_capabilities() -> DeviceCapabilities {
        raster_capabilities_from_limit([65_535; 3])
    }

    /// Waits outside graph execution for an accepted submission.
    pub fn wait(&self, completion: &NativeCompletion, timeout: Duration) -> Result<(), WaitError> {
        CopyBackend::new(self.device.clone()).wait(completion, timeout)
    }

    pub(in crate::execution) fn copy(&self) -> CopyBackend {
        CopyBackend {
            device: self.device.clone(),
            capabilities: copy_capabilities(),
            retired: Vec::new(),
        }
    }
    pub(in crate::execution) fn check_encoder(
        &self,
        encoder: &CopyEncoder,
    ) -> Result<(), NativeExecutionError> {
        require_device(
            encoder.device,
            self.device.identity(),
            NativeExecutionError::ForeignEncoder,
        )
    }
    pub(in crate::execution) fn check_buffer(
        &self,
        buffer: &Buffer,
    ) -> Result<(), NativeExecutionError> {
        require_device(
            buffer.device_identity(),
            self.device.identity(),
            NativeExecutionError::ForeignResource,
        )
    }
    pub(in crate::execution) fn check_texture(
        &self,
        texture: &Texture,
    ) -> Result<(), NativeExecutionError> {
        require_device(
            texture.device_identity(),
            self.device.identity(),
            NativeExecutionError::ForeignResource,
        )
    }
}

fn raster_capabilities(device: &Device) -> DeviceCapabilities {
    let facts = device.capabilities();
    raster_capabilities_from_limit_and_filterability(
        facts.max_compute_workgroups_per_dimension,
        facts.rgba8_unorm_filterable,
        facts.rgba8_unorm_srgb_filterable,
    )
}

#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
fn raster_surface_capabilities(device: &Device) -> DeviceCapabilities {
    let facts = device.capabilities();
    let mut capabilities = raster_capabilities_from_limit_and_filterability(
        facts.max_compute_workgroups_per_dimension,
        facts.rgba8_unorm_filterable,
        facts.rgba8_unorm_srgb_filterable,
    );
    capabilities.queues[0].capabilities.present = true;
    capabilities.surface = Some(SurfaceCapabilities::new(
        vec![TextureFormat::Rgba8Unorm],
        true,
        false,
    ));
    capabilities
}

fn raster_capabilities_from_limit(maximum: [u32; 3]) -> DeviceCapabilities {
    // The portable profile preserves prior fixed fixtures: it is an abstract
    // compile target, not a claim about a particular adapter's sampler fact.
    raster_capabilities_from_limit_and_filterability(maximum, true, true)
}

pub(in crate::execution) fn raster_capabilities_from_limit_and_filterability(
    maximum: [u32; 3],
    rgba8_unorm_filterable: bool,
    rgba8_unorm_srgb_filterable: bool,
) -> DeviceCapabilities {
    DeviceCapabilities::builder()
        .queue(QueueDescriptor::new(
            QueueId::new(0),
            QueueCapabilities::new(true, true, true, false),
        ))
        .recording(RecordingCapabilities::new(
            RecordingModel::DeferredCommandBuffers,
            false,
        ))
        .transitions(TransitionCapabilities::GraphManagedExplicit)
        .synchronization(SynchronizationCapabilities::SingleQueueOrdering)
        .timestamps(TimestampCapabilities::Unsupported)
        .transient_resources(TransientResourceCapabilities::new(false, false, false))
        .limits(DeviceLimits::new(1, 256).with_max_compute_workgroups_per_dimension(maximum))
        .buffers(BufferCapabilities::new(true, true, false))
        .texture_format(
            TextureFormatCapabilities::builder(TextureFormat::Rgba8Unorm)
                .sampled(true, rgba8_unorm_filterable)
                .attachments(true, false, vec![1])
                .copies(true, true)
                .build(),
        )
        .texture_format(
            TextureFormatCapabilities::builder(TextureFormat::Rgba8UnormSrgb)
                .sampled(true, rgba8_unorm_srgb_filterable)
                .copies(true, true)
                .build(),
        )
        .texture_format(
            // The native raster pass creates a Depth32Float D2 view and the
            // fixed depth-pipeline sibling declares exactly this format.
            TextureFormatCapabilities::builder(TextureFormat::Depth32Float)
                .attachments(false, true, vec![1])
                .build(),
        )
        .build()
}
