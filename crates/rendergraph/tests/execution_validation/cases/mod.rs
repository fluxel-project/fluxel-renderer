//! Shared fixtures and validation groups at the execution boundary.

//! Focused validation coverage for the execution boundary.

use fluxel_rendergraph::{
    test_rhi::{TestBuffer, TestRegistry, TestRhi, TestTexture, TestTraceEvent},
    *,
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};

fn device() -> DeviceIdentity {
    DeviceIdentity::new(88)
}

fn buffer(size: u64) -> BufferDesc {
    BufferDesc { size }
}

fn texture(format: TextureFormat) -> TextureDesc {
    TextureDesc {
        dimension: TextureDimension::D2,
        extent: Extent3d {
            width: 8,
            height: 8,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        sample_count: 1,
        format,
    }
}

fn caps() -> DeviceCapabilities {
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
        .limits(DeviceLimits::new(4, 256))
        .buffers(BufferCapabilities::new(true, true, true))
        .texture_format(
            TextureFormatCapabilities::builder(TextureFormat::Rgba8Unorm)
                .sampled(true, true)
                .storage(true, true)
                .attachments(true, false, vec![1])
                .copies(true, true)
                .build(),
        )
        .texture_format(
            TextureFormatCapabilities::builder(TextureFormat::Rgba8UnormSrgb)
                .sampled(true, true)
                .storage(true, true)
                .copies(true, true)
                .build(),
        )
        .texture_format(
            TextureFormatCapabilities::builder(TextureFormat::Rgba16Float)
                .storage(true, true)
                .copies(true, true)
                .build(),
        )
        .texture_format(
            TextureFormatCapabilities::builder(TextureFormat::Depth32Float)
                .attachments(false, true, vec![1])
                .build(),
        )
        .build()
}

fn executor() -> FrameExecutor<TestRhi> {
    FrameExecutor::new(TestRhi::new(caps(), device()))
}

struct WrongDeviceProvider(TestRegistry);

impl FrameResourceProvider<TestRhi> for WrongDeviceProvider {
    fn texture(
        &self,
        id: TextureBindingId,
    ) -> Result<BoundTexture<TestTexture, <TestRhi as ExecutionBackend>::Lease>, FrameBindingError>
    {
        let mut value = self.0.texture(id)?;
        value.device = DeviceIdentity::new(999);
        Ok(value)
    }
    fn buffer(
        &self,
        id: BufferBindingId,
    ) -> Result<BoundBuffer<TestBuffer, <TestRhi as ExecutionBackend>::Lease>, FrameBindingError>
    {
        let mut value = self.0.buffer(id)?;
        value.device = DeviceIdentity::new(999);
        Ok(value)
    }
}

impl RenderObjectProvider<TestRhi> for WrongDeviceProvider {
    fn raster_pipeline(
        &self,
        id: RasterPipelineId,
    ) -> Result<
        BoundRasterPipeline<
            <TestRhi as ExecutionBackend>::RasterPipeline,
            <TestRhi as ExecutionBackend>::Lease,
        >,
        RecordingError,
    > {
        self.0.raster_pipeline(id)
    }
    fn compute_pipeline(
        &self,
        id: ComputePipelineId,
    ) -> Result<
        BoundComputePipeline<
            <TestRhi as ExecutionBackend>::ComputePipeline,
            <TestRhi as ExecutionBackend>::Lease,
        >,
        RecordingError,
    > {
        self.0.compute_pipeline(id)
    }
    fn bindings(
        &self,
        id: BindingSetId,
        resources: &[ResolvedBindingResource<'_, TestTexture, TestBuffer>],
        offsets: &[u32],
    ) -> Result<
        BoundBindings<
            <TestRhi as ExecutionBackend>::Bindings,
            <TestRhi as ExecutionBackend>::Lease,
        >,
        RecordingError,
    > {
        self.0.bindings(id, resources, offsets)
    }
}

struct WrongBindingProvider(TestRegistry);

impl FrameResourceProvider<TestRhi> for WrongBindingProvider {
    fn texture(
        &self,
        id: TextureBindingId,
    ) -> Result<BoundTexture<TestTexture, <TestRhi as ExecutionBackend>::Lease>, FrameBindingError>
    {
        self.0.texture(id)
    }
    fn buffer(
        &self,
        id: BufferBindingId,
    ) -> Result<BoundBuffer<TestBuffer, <TestRhi as ExecutionBackend>::Lease>, FrameBindingError>
    {
        self.0.buffer(id)
    }
}

impl RenderObjectProvider<TestRhi> for WrongBindingProvider {
    fn raster_pipeline(
        &self,
        id: RasterPipelineId,
    ) -> Result<
        BoundRasterPipeline<
            <TestRhi as ExecutionBackend>::RasterPipeline,
            <TestRhi as ExecutionBackend>::Lease,
        >,
        RecordingError,
    > {
        self.0.raster_pipeline(id)
    }
    fn compute_pipeline(
        &self,
        id: ComputePipelineId,
    ) -> Result<
        BoundComputePipeline<
            <TestRhi as ExecutionBackend>::ComputePipeline,
            <TestRhi as ExecutionBackend>::Lease,
        >,
        RecordingError,
    > {
        self.0.compute_pipeline(id)
    }
    fn bindings(
        &self,
        id: BindingSetId,
        resources: &[ResolvedBindingResource<'_, TestTexture, TestBuffer>],
        offsets: &[u32],
    ) -> Result<
        BoundBindings<
            <TestRhi as ExecutionBackend>::Bindings,
            <TestRhi as ExecutionBackend>::Lease,
        >,
        RecordingError,
    > {
        let mut binding = self.0.bindings(id, resources, offsets)?;
        binding.device = DeviceIdentity::new(999);
        Ok(binding)
    }
}

mod completion_and_provider;
mod portable;
mod transient_reuse;
