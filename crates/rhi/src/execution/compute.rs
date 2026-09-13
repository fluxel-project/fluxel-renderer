//! Fixed compute execution.
//!
//! This module owns the compute-and-copy profile and validates only generic
//! compute dispatch constraints. Raster recipe rules live with the raster
//! profile so compute-only backends do not define raster semantics.

use super::*;
use crate::execution::helpers::require_device;

/// A serial DX12/Vulkan backend for the fixed Compute-and-Copy slice.
pub struct ComputeBackend {
    device: Device,
    capabilities: DeviceCapabilities,
    retired: Vec<Retired>,
    #[cfg(test)]
    transient_observations: TestTransientObservations,
}

/// Test-only evidence emitted by native transient allocation and barrier
/// recording.  This deliberately stays crate-private: it is not an RHI API.
#[cfg(test)]
#[derive(Clone, Debug, Default)]
pub(in crate::execution) struct TestTransientObservations {
    pub(in crate::execution) allocations: Vec<fluxel_rendergraph::PhysicalResourceIdentity>,
    pub(in crate::execution) transitions: Vec<TestTransientTransition>,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::execution) struct TestTransientTransition {
    pub(in crate::execution) identity: fluxel_rendergraph::PhysicalResourceIdentity,
    pub(in crate::execution) before: ResourceAccessState,
    pub(in crate::execution) after: ResourceAccessState,
}

impl ComputeBackend {
    /// Creates a compute-and-copy backend over an already opened native device.
    pub fn new(device: Device) -> Self {
        Self {
            capabilities: compute_capabilities(&device),
            device,
            retired: Vec::new(),
            #[cfg(test)]
            transient_observations: TestTransientObservations::default(),
        }
    }

    /// Returns the normalized capability profile used by fixed compute fixtures.
    pub fn portable_capabilities() -> DeviceCapabilities {
        // The cross-backend profile is conservative. A real backend instance
        // additionally carries its verified native workgroup count.
        compute_capabilities_from_limit([65_535, 65_535, 65_535])
    }

    /// Waits outside graph execution for an accepted submission.
    pub fn wait(&self, completion: &NativeCompletion, timeout: Duration) -> Result<(), WaitError> {
        CopyBackend::new(self.device.clone()).wait(completion, timeout)
    }
}
fn compute_capabilities(device: &Device) -> DeviceCapabilities {
    let portable_limit = [65_535; 3];
    debug_assert!(
        device
            .capabilities()
            .max_compute_workgroups_per_dimension
            .into_iter()
            .zip(portable_limit)
            .all(|(native, required)| native >= required),
        "Device::open validates the portable dispatch baseline"
    );
    let mut capabilities = compute_capabilities_from_limit(portable_limit);
    if let Some(format) = capabilities
        .texture_formats
        .iter_mut()
        .find(|format| format.format == TextureFormat::Rgba8Unorm)
    {
        format.storage_read = device.capabilities().rgba8_unorm_storage_read_enabled;
        format.storage_write = device.capabilities().rgba8_unorm_storage_write;
    }
    capabilities
}

fn compute_capabilities_from_limit(maximum: [u32; 3]) -> DeviceCapabilities {
    DeviceCapabilities::builder()
        .queue(QueueDescriptor::new(
            QueueId::new(0),
            QueueCapabilities::new(false, true, true, false),
        ))
        .recording(RecordingCapabilities::new(
            RecordingModel::DeferredCommandBuffers,
            false,
        ))
        .transitions(TransitionCapabilities::GraphManagedExplicit)
        .synchronization(SynchronizationCapabilities::SingleQueueOrdering)
        .timestamps(TimestampCapabilities::Unsupported)
        // See CopyBackend: these are owned native DeviceOnly allocations, so
        // completion-gated cross-frame reuse is sound.  Alias barriers are
        // deliberately not part of this fixed compute profile.
        .transient_resources(TransientResourceCapabilities::new(true, false, false))
        .limits(DeviceLimits::new(0, 256).with_max_compute_workgroups_per_dimension(maximum))
        .buffers(BufferCapabilities::new(true, true, false))
        .texture_format(
            TextureFormatCapabilities::builder(TextureFormat::Rgba8Unorm)
                .copies(true, true)
                .build(),
        )
        .build()
}

impl ExecutionBackend for ComputeBackend {
    type Texture = Texture;
    type Buffer = Buffer;
    type RasterPipeline = UnsupportedRasterPipeline;
    type ComputePipeline = ComputePipeline;
    type Bindings = ComputeBindings;
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
    ) -> Result<BoundTexture<Texture, ResourceLease>, Self::Error> {
        let texture = self.device.create_texture(TextureDescriptor {
            texture: descriptor,
            usage,
            memory: MemoryPolicy::DeviceOnly,
        })?;
        #[cfg(test)]
        self.transient_observations
            .allocations
            .push(texture.identity());
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
    ) -> Result<BoundBuffer<Buffer, ResourceLease>, Self::Error> {
        let buffer = self.device.create_buffer(BufferDescriptor {
            buffer: descriptor,
            usage,
            memory: MemoryPolicy::DeviceOnly,
        })?;
        #[cfg(test)]
        self.transient_observations
            .allocations
            .push(buffer.identity());
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
    fn begin_encoder(&mut self, queue: QueueId) -> Result<CopyEncoder, Self::Error> {
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
        encoder: &mut CopyEncoder,
        texture: &Texture,
        range: TextureRange,
        before: ResourceAccessState,
        after: ResourceAccessState,
    ) -> Result<(), Self::Error> {
        let result = self
            .copy()
            .transition_texture(encoder, texture, range, before, after);
        #[cfg(test)]
        if result.is_ok() {
            self.transient_observations
                .transitions
                .push(TestTransientTransition {
                    identity: texture.identity(),
                    before,
                    after,
                });
        }
        result
    }
    fn transition_buffer(
        &mut self,
        encoder: &mut CopyEncoder,
        buffer: &Buffer,
        range: BufferRange,
        before: ResourceAccessState,
        after: ResourceAccessState,
    ) -> Result<(), Self::Error> {
        let result = self
            .copy()
            .transition_buffer(encoder, buffer, range, before, after);
        #[cfg(test)]
        if result.is_ok() {
            self.transient_observations
                .transitions
                .push(TestTransientTransition {
                    identity: buffer.identity(),
                    before,
                    after,
                });
        }
        result
    }
    fn begin_raster(
        &mut self,
        _: &mut CopyEncoder,
        _: &RasterPassDescriptor<'_, Texture>,
    ) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn end_raster(&mut self, _: &mut CopyEncoder) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn begin_compute(&mut self, encoder: &mut CopyEncoder, label: &str) -> Result<(), Self::Error> {
        self.check_encoder(encoder)?;
        if encoder.copy_open || encoder.compute_open || encoder.raster_extent.is_some() {
            return Err(NativeExecutionError::ComputeBindingMismatch);
        }
        crate::imp::begin_compute(&mut encoder.native, label)
            .map_err(NativeExecutionError::Recording)?;
        encoder.active_compute = None;
        encoder.bound_compute_pipeline = None;
        encoder.compute_open = true;
        Ok(())
    }
    fn end_compute(&mut self, encoder: &mut CopyEncoder) -> Result<(), Self::Error> {
        self.check_encoder(encoder)?;
        if !encoder.compute_open {
            return Err(NativeExecutionError::ComputeBindingMismatch);
        }
        crate::imp::end_compute(&mut encoder.native).map_err(NativeExecutionError::Recording)?;
        encoder.active_compute = None;
        encoder.bound_compute_pipeline = None;
        encoder.compute_open = false;
        Ok(())
    }
    fn begin_copy(&mut self, encoder: &mut CopyEncoder, label: &str) -> Result<(), Self::Error> {
        self.copy().begin_copy(encoder, label)
    }
    fn end_copy(&mut self, encoder: &mut CopyEncoder) -> Result<(), Self::Error> {
        self.copy().end_copy(encoder)
    }
    fn set_raster_pipeline(
        &mut self,
        _: &mut CopyEncoder,
        _: &UnsupportedRasterPipeline,
    ) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn set_compute_pipeline(
        &mut self,
        encoder: &mut CopyEncoder,
        pipeline: &ComputePipeline,
    ) -> Result<(), Self::Error> {
        self.check_encoder(encoder)?;
        if !encoder.compute_open {
            return Err(NativeExecutionError::ComputeBindingMismatch);
        }
        self.check_pipeline(pipeline)?;
        crate::imp::set_compute_pipeline(&mut encoder.native, pipeline.native())
            .map_err(NativeExecutionError::Recording)?;
        encoder.active_compute = Some(pipeline.clone());
        encoder.bound_compute_pipeline = None;
        encoder.leases.push(pipeline.lease().into());
        Ok(())
    }
    fn set_bindings(
        &mut self,
        encoder: &mut CopyEncoder,
        bindings: &ComputeBindings,
    ) -> Result<(), Self::Error> {
        self.check_encoder(encoder)?;
        if !encoder.compute_open {
            return Err(NativeExecutionError::ComputeBindingMismatch);
        }
        if bindings.device_identity() != self.device.identity() {
            return Err(NativeExecutionError::ForeignResource);
        }
        let pipeline = encoder
            .active_compute
            .as_ref()
            .ok_or(NativeExecutionError::ComputeBindingMismatch)?;
        if !bindings.pipeline().same_object(pipeline) {
            return Err(NativeExecutionError::ComputeBindingMismatch);
        }
        crate::imp::set_compute_bindings(&mut encoder.native, bindings.native())
            .map_err(NativeExecutionError::Recording)?;
        encoder.bound_compute_pipeline = Some(pipeline.clone());
        encoder.leases.push(bindings.lease().into());
        Ok(())
    }
    fn set_vertex_buffer(
        &mut self,
        _: &mut CopyEncoder,
        _: u32,
        _: &Buffer,
        _: u64,
    ) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn set_index_buffer(
        &mut self,
        _: &mut CopyEncoder,
        _: &Buffer,
        _: u64,
        _: IndexFormat,
    ) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn set_viewport(&mut self, _: &mut CopyEncoder, _: Viewport) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn set_scissor(&mut self, _: &mut CopyEncoder, _: ScissorRect) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn draw(
        &mut self,
        _: &mut CopyEncoder,
        _: Range<u32>,
        _: Range<u32>,
    ) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn draw_indexed(
        &mut self,
        _: &mut CopyEncoder,
        _: Range<u32>,
        _: i32,
        _: Range<u32>,
    ) -> Result<(), Self::Error> {
        Err(NativeExecutionError::UnsupportedCommandFamily)
    }
    fn dispatch(&mut self, encoder: &mut CopyEncoder, groups: [u32; 3]) -> Result<(), Self::Error> {
        self.check_encoder(encoder)?;
        if !encoder.compute_open {
            return Err(NativeExecutionError::ComputeBindingMismatch);
        }
        if !valid_compute_dispatch(
            groups,
            self.capabilities
                .limits
                .max_compute_workgroups_per_dimension,
        ) {
            return Err(NativeExecutionError::InvalidDispatch);
        }
        if encoder.active_compute.is_none() || encoder.bound_compute_pipeline.is_none() {
            return Err(NativeExecutionError::ComputeBindingMismatch);
        }
        crate::imp::dispatch(&mut encoder.native, groups).map_err(NativeExecutionError::Recording)
    }
    fn copy_texture(
        &mut self,
        encoder: &mut CopyEncoder,
        source: &Texture,
        destination: &Texture,
        region: TextureCopyRegion,
    ) -> Result<(), Self::Error> {
        self.copy()
            .copy_texture(encoder, source, destination, region)
    }
    fn copy_buffer(
        &mut self,
        encoder: &mut CopyEncoder,
        source: &Buffer,
        destination: &Buffer,
        region: BufferCopyRegion,
    ) -> Result<(), Self::Error> {
        self.copy()
            .copy_buffer(encoder, source, destination, region)
    }
    fn finish_encoder(&mut self, encoder: CopyEncoder) -> Result<CopyCommandBuffer, Self::Error> {
        self.copy().finish_encoder(encoder)
    }
    fn submit(
        &mut self,
        queue: QueueId,
        command_buffer: CopyCommandBuffer,
        presentations: Vec<PresentationSubmission<Self::PresentationToken>>,
    ) -> Result<NativeCompletion, Self::Error> {
        self.copy().submit(queue, command_buffer, presentations)
    }
    fn completion_status(&self, completion: &NativeCompletion) -> CompletionStatus {
        // ExecutionBackend requires a total query. A native query error cannot
        // prove progress, so it remains non-terminal and quarantined.
        crate::imp::completion_status(&completion.0).unwrap_or(CompletionStatus::Unknown)
    }
    fn retire(&mut self, completion: NativeCompletion, leases: Vec<ResourceLease>) {
        self.retired.push(Retired { completion, leases });
    }
    fn collect_retired(&mut self) -> Result<usize, Self::Error> {
        let before = self.retired.len();
        let mut query_error = None;
        // Query failure does not prove that accepted GPU work is finished. Keep
        // the entry and its leases quarantined instead of freeing live handles.
        self.retired.retain(
            |entry| match crate::imp::completion_status(&entry.completion.0) {
                Ok(CompletionStatus::Pending | CompletionStatus::Unknown) => true,
                Ok(_) => false,
                Err(error) => {
                    query_error.get_or_insert(error);
                    true
                }
            },
        );
        match query_error {
            Some(error) => Err(NativeExecutionError::Completion(error)),
            None => Ok(before - self.retired.len()),
        }
    }
}

pub(in crate::execution) fn valid_compute_dispatch(groups: [u32; 3], maximum: [u32; 3]) -> bool {
    groups
        .into_iter()
        .zip(maximum)
        .all(|(given, maximum)| given != 0 && given <= maximum)
}

impl ComputeBackend {
    #[cfg(test)]
    pub(in crate::execution) fn test_transient_observations(&self) -> TestTransientObservations {
        self.transient_observations.clone()
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
    pub(in crate::execution) fn check_pipeline(
        &self,
        pipeline: &ComputePipeline,
    ) -> Result<(), NativeExecutionError> {
        require_device(
            pipeline.device_identity(),
            self.device.identity(),
            NativeExecutionError::ForeignResource,
        )
    }
}
