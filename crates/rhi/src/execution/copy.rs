//! Copy-only execution backend.

use super::*;
use crate::execution::helpers::{require_device, validate_buffer_copy, validate_texture_copy};

pub(in crate::execution) fn copy_capabilities() -> DeviceCapabilities {
    DeviceCapabilities::builder()
        .queue(QueueDescriptor::new(
            QueueId::new(0),
            QueueCapabilities::new(false, false, true, false),
        ))
        .recording(RecordingCapabilities::new(
            RecordingModel::DeferredCommandBuffers,
            false,
        ))
        .transitions(TransitionCapabilities::GraphManagedExplicit)
        .synchronization(SynchronizationCapabilities::SingleQueueOrdering)
        .timestamps(TimestampCapabilities::Unsupported)
        // Native DeviceOnly resources have stable physical identities and their
        // leases remain owned by the submitted command buffer.  They may
        // therefore be reused by the executor only after completion.  We do
        // not expose in-frame aliasing: the fixed native recorders do not have
        // an aliasing-barrier contract.
        .transient_resources(TransientResourceCapabilities::new(true, false, false))
        .limits(DeviceLimits::new(0, 256))
        .buffers(BufferCapabilities::new(false, false, false))
        .texture_format(
            TextureFormatCapabilities::builder(TextureFormat::Rgba8Unorm)
                .copies(true, true)
                .build(),
        )
        .build()
}

impl ExecutionBackend for CopyBackend {
    type Texture = Texture;
    type Buffer = Buffer;
    type RasterPipeline = UnsupportedRasterPipeline;
    type ComputePipeline = UnsupportedComputePipeline;
    type Bindings = UnsupportedBindings;
    type Encoder = CopyEncoder;
    type CommandBuffer = CopyCommandBuffer;
    type Completion = NativeCompletion;
    type PresentationToken = PresentationToken;
    type Lease = ResourceLease;
    type Error = NativeExecutionError;

    fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }
    fn device_identity(&self) -> fluxel_rendergraph::DeviceIdentity {
        self.device.identity()
    }

    fn create_transient_texture(
        &mut self,
        descriptor: TextureDesc,
        usage: TextureUsage,
    ) -> Result<BoundTexture<Self::Texture, Self::Lease>, Self::Error> {
        let texture = self.device.create_texture(TextureDescriptor {
            texture: descriptor,
            usage,
            memory: MemoryPolicy::DeviceOnly,
        })?;
        Ok(BoundTexture {
            device: self.device.identity(),
            identity: texture.identity(),
            physical: texture.clone(),
            descriptor,
            usage: texture.allowed_usage(),
            initial_state: ResourceAccessState::Undefined,
            lease: texture.lease().into(),
        })
    }
    fn create_transient_buffer(
        &mut self,
        descriptor: BufferDesc,
        usage: BufferUsage,
    ) -> Result<BoundBuffer<Self::Buffer, Self::Lease>, Self::Error> {
        let buffer = self.device.create_buffer(BufferDescriptor {
            buffer: descriptor,
            usage,
            memory: MemoryPolicy::DeviceOnly,
        })?;
        Ok(BoundBuffer {
            device: self.device.identity(),
            identity: buffer.identity(),
            physical: buffer.clone(),
            descriptor,
            usage: buffer.allowed_usage(),
            initial_state: ResourceAccessState::Undefined,
            lease: buffer.lease().into(),
        })
    }
    fn begin_encoder(&mut self, queue: QueueId) -> Result<Self::Encoder, Self::Error> {
        if queue != QueueId::new(0) {
            return Err(NativeExecutionError::UnknownQueue(queue));
        }
        crate::imp::begin_copy_encoder(&self.device.inner)
            .map(|native| CopyEncoder {
                native,
                device: self.device.identity(),
                leases: Vec::new(),
                active_compute: None,
                bound_compute_pipeline: None,
                active_raster: None,
                bound_raster_uniform: None,
                bound_raster_texture: None,
                bound_raster_uv_texture: None,
                bound_raster_normal: None,
                bound_raster_vertex_color: None,
                raster_uv_epoch: 0,
                raster_uv_binding_epoch: None,
                raster_uv_vertex_slots: [None, None],
                raster_uv_index_ready: false,
                raster_normal_epoch: 0,
                raster_normal_binding_epoch: None,
                raster_normal_vertex_slots: [None, None],
                raster_normal_index_ready: false,
                raster_vertex_color_epoch: 0,
                raster_vertex_color_binding_epoch: None,
                raster_vertex_color_slots: [None, None],
                raster_vertex_color_index_ready: false,
                vertex_buffer: None,
                index_buffer: None,
                raster_extent: None,
                compute_open: false,
                copy_open: false,
            })
            .map_err(NativeExecutionError::Recording)
    }
    fn transition_texture(
        &mut self,
        encoder: &mut Self::Encoder,
        texture: &Self::Texture,
        range: TextureRange,
        before: ResourceAccessState,
        after: ResourceAccessState,
    ) -> Result<(), Self::Error> {
        self.check_encoder(encoder)?;
        if encoder.copy_open || encoder.compute_open || encoder.raster_extent.is_some() {
            return Err(NativeExecutionError::CopyStateMismatch);
        }
        self.check_texture(texture)?;
        crate::imp::transition_texture(
            &mut encoder.native,
            texture.native(),
            texture.descriptor().texture,
            range,
            before,
            after,
        )
        .map_err(NativeExecutionError::Recording)?;
        encoder.leases.push(texture.lease().into());
        Ok(())
    }
    fn transition_buffer(
        &mut self,
        encoder: &mut Self::Encoder,
        buffer: &Self::Buffer,
        _range: BufferRange,
        before: ResourceAccessState,
        after: ResourceAccessState,
    ) -> Result<(), Self::Error> {
        self.check_encoder(encoder)?;
        if encoder.copy_open || encoder.compute_open || encoder.raster_extent.is_some() {
            return Err(NativeExecutionError::CopyStateMismatch);
        }
        self.check_buffer(buffer)?;
        crate::imp::transition_buffer(&mut encoder.native, buffer.native(), before, after)
            .map_err(NativeExecutionError::Recording)?;
        encoder.leases.push(buffer.lease().into());
        Ok(())
    }
    fn begin_raster(
        &mut self,
        _: &mut Self::Encoder,
        _: &RasterPassDescriptor<'_, Self::Texture>,
    ) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn end_raster(&mut self, _: &mut Self::Encoder) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn begin_compute(&mut self, _: &mut Self::Encoder, _: &str) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn end_compute(&mut self, _: &mut Self::Encoder) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn begin_copy(&mut self, encoder: &mut Self::Encoder, _: &str) -> Result<(), Self::Error> {
        self.check_encoder(encoder)?;
        if encoder.copy_open || encoder.compute_open || encoder.raster_extent.is_some() {
            return Err(NativeExecutionError::CopyStateMismatch);
        }
        encoder.copy_open = true;
        Ok(())
    }
    fn end_copy(&mut self, encoder: &mut Self::Encoder) -> Result<(), Self::Error> {
        self.check_encoder(encoder)?;
        if !encoder.copy_open {
            return Err(NativeExecutionError::CopyStateMismatch);
        }
        encoder.copy_open = false;
        Ok(())
    }
    fn set_raster_pipeline(
        &mut self,
        _: &mut Self::Encoder,
        _: &Self::RasterPipeline,
    ) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn set_compute_pipeline(
        &mut self,
        _: &mut Self::Encoder,
        _: &Self::ComputePipeline,
    ) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn set_bindings(
        &mut self,
        _: &mut Self::Encoder,
        _: &Self::Bindings,
    ) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn set_vertex_buffer(
        &mut self,
        _: &mut Self::Encoder,
        _: u32,
        _: &Self::Buffer,
        _: u64,
    ) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn set_index_buffer(
        &mut self,
        _: &mut Self::Encoder,
        _: &Self::Buffer,
        _: u64,
        _: IndexFormat,
    ) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn set_viewport(&mut self, _: &mut Self::Encoder, _: Viewport) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn set_scissor(&mut self, _: &mut Self::Encoder, _: ScissorRect) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn draw(
        &mut self,
        _: &mut Self::Encoder,
        _: Range<u32>,
        _: Range<u32>,
    ) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn draw_indexed(
        &mut self,
        _: &mut Self::Encoder,
        _: Range<u32>,
        _: i32,
        _: Range<u32>,
    ) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn dispatch(&mut self, _: &mut Self::Encoder, _: [u32; 3]) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn copy_texture(
        &mut self,
        encoder: &mut Self::Encoder,
        source: &Self::Texture,
        destination: &Self::Texture,
        region: TextureCopyRegion,
    ) -> Result<(), Self::Error> {
        self.check_encoder(encoder)?;
        if !encoder.copy_open {
            return Err(NativeExecutionError::CopyStateMismatch);
        }
        self.check_texture(source)?;
        self.check_texture(destination)?;
        validate_texture_copy(
            source.descriptor().texture,
            destination.descriptor().texture,
            region,
        )?;
        crate::imp::copy_texture(
            &mut encoder.native,
            source.native(),
            destination.native(),
            source.descriptor().texture,
            region,
        )
        .map_err(NativeExecutionError::Recording)?;
        encoder.leases.push(source.lease().into());
        encoder.leases.push(destination.lease().into());
        Ok(())
    }
    fn copy_buffer(
        &mut self,
        encoder: &mut Self::Encoder,
        source: &Self::Buffer,
        destination: &Self::Buffer,
        region: BufferCopyRegion,
    ) -> Result<(), Self::Error> {
        self.check_encoder(encoder)?;
        if !encoder.copy_open {
            return Err(NativeExecutionError::CopyStateMismatch);
        }
        self.check_buffer(source)?;
        self.check_buffer(destination)?;
        validate_buffer_copy(
            source.descriptor().buffer,
            destination.descriptor().buffer,
            region,
        )?;
        crate::imp::copy_buffer(
            &mut encoder.native,
            source.native(),
            destination.native(),
            region,
        )
        .map_err(NativeExecutionError::Recording)?;
        encoder.leases.push(source.lease().into());
        encoder.leases.push(destination.lease().into());
        Ok(())
    }
    fn finish_encoder(
        &mut self,
        encoder: Self::Encoder,
    ) -> Result<Self::CommandBuffer, Self::Error> {
        self.check_encoder(&encoder)?;
        if encoder.copy_open {
            return Err(NativeExecutionError::CopyStateMismatch);
        }
        if encoder.compute_open || encoder.raster_extent.is_some() {
            return Err(NativeExecutionError::RasterStateMismatch);
        }
        let CopyEncoder {
            native,
            device,
            leases,
            ..
        } = encoder;
        crate::imp::finish_copy_encoder(native)
            .map(|native| CopyCommandBuffer {
                native,
                device,
                leases,
            })
            .map_err(NativeExecutionError::Recording)
    }
    fn submit(
        &mut self,
        queue: QueueId,
        command_buffer: Self::CommandBuffer,
        presentations: Vec<PresentationSubmission<Self::PresentationToken>>,
    ) -> Result<Self::Completion, Self::Error> {
        if !presentations.is_empty() {
            return Err(NativeExecutionError::SubmitRejected(
                "copy backend does not present acquired surface images".into(),
            ));
        }
        if queue != QueueId::new(0) {
            return Err(NativeExecutionError::UnknownQueue(queue));
        }
        require_device(
            command_buffer.device,
            self.device.identity(),
            NativeExecutionError::ForeignCommandBuffer,
        )?;
        let CopyCommandBuffer {
            native,
            device: _,
            leases,
        } = command_buffer;
        crate::imp::submit_copy(native, leases)
            .map(NativeCompletion)
            .map_err(NativeExecutionError::SubmitRejected)
    }
    fn completion_status(&self, completion: &Self::Completion) -> CompletionStatus {
        // ExecutionBackend requires a total query. A native observation error
        // proves neither completion nor terminal failure, so keep the work
        // quarantined as Unknown.
        crate::imp::completion_status(&completion.0).unwrap_or(CompletionStatus::Unknown)
    }
    fn retire(&mut self, completion: Self::Completion, leases: Vec<Self::Lease>) {
        self.retired.push(Retired { completion, leases });
    }
    fn collect_retired(&mut self) -> Result<usize, Self::Error> {
        let before = self.retired.len();
        let mut query_error = None;
        // Query failure does not prove that accepted GPU work is finished. Keep
        // the entry and its leases quarantined instead of freeing live handles.
        self.retired.retain(|entry| {
            let _ = entry.leases.len();
            match crate::imp::completion_status(&entry.completion.0) {
                Ok(CompletionStatus::Pending | CompletionStatus::Unknown) => true,
                Ok(CompletionStatus::Complete | CompletionStatus::Failed(_)) => false,
                Ok(_) => true,
                Err(error) => {
                    query_error.get_or_insert(error);
                    true
                }
            }
        });
        if let Some(error) = query_error {
            Err(NativeExecutionError::Completion(error))
        } else {
            Ok(before - self.retired.len())
        }
    }
}
