//! Backend device, recording, submission, and retirement contract.

use std::ops::Range;

use crate::{
    access::{BufferCopyRegion, BufferRange, TextureCopyRegion, TextureRange},
    pass::{ScissorRect, Viewport},
    plan::{BufferUsage, TextureUsage},
    rhi::{BufferDesc, DeviceCapabilities, IndexFormat, QueueId, ResourceAccessState, TextureDesc},
};

use super::{BoundBuffer, BoundTexture, PresentationSubmission, RasterPassDescriptor};

/// Stable identity of one backend device instance.
///
/// Values are opaque and need only be unique among simultaneously live device
/// instances. They are used to reject accidentally mixed frame bindings.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DeviceIdentity(u64);

impl DeviceIdentity {
    /// Creates an opaque identity for one backend device instance.
    pub fn new(raw: u64) -> Self {
        Self(raw)
    }
}

/// Observable completion state of one submitted command buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CompletionStatus {
    /// The submission has not completed.
    Pending,
    /// The backend cannot presently determine whether the submission completed.
    ///
    /// This is deliberately distinct from a terminal failure. Callers must
    /// retain every lease and must not recycle any physical resource while the
    /// status is unknown.
    Unknown,
    /// The submission completed successfully.
    Complete,
    /// The accepted submission reached a terminal failure.
    Failed(CompletionFailure),
}

/// Why an accepted submission reached a terminal failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CompletionFailure {
    /// The native device was lost while work was pending.
    DeviceLost,
    /// The backend reported another terminal execution failure.
    ExecutionFailed,
}

/// Minimal single-queue backend contract used by future graph execution.
///
/// The graph owns dependency analysis, semantic state planning, pass order, and
/// transient lifetimes. A backend creates physical transients, records the
/// plan's commands, and submits exactly the completed command buffer supplied
/// by the execution adapter.
///
/// A backend must retain all leases passed to [`Self::retire`] until their
/// completion is terminal. Dropping a backend with pending retirements must
/// either wait for them or perform native device teardown that makes it safe to
/// release every referenced object; it must never simply discard pending leases
/// while submitted GPU work could still reference them.
///
/// Recording errors never submit their encoder. The executor still calls the
/// matching `end_*` operation after a callback error; if scope closure itself
/// fails, dropping the unfinished encoder must safely discard its native
/// recording state without executing it.
pub trait ExecutionBackend {
    /// Backend-native physical texture object.
    type Texture: Clone;
    /// Backend-native physical buffer object.
    type Buffer: Clone;
    /// Backend-native raster pipeline object.
    type RasterPipeline;
    /// Backend-native compute pipeline object.
    type ComputePipeline;
    /// Backend-native binding object.
    type Bindings;
    /// Mutable encoder used to record one ordered submission.
    type Encoder;
    /// Finished command buffer accepted by [`Self::submit`].
    type CommandBuffer;
    /// Submission-completion object, such as a fence or timeline value.
    type Completion: Clone;
    /// One-shot token that allows an acquired image to be presented after its
    /// graph submission. It deliberately has no graph-visible surface data.
    /// Dropping an unsubmitted token must safely cancel or abandon its native
    /// acquisition, including after any pre-submit execution error.
    type PresentationToken;
    /// Lease retained by newly created transient resources until completion.
    type Lease: Clone;
    /// Backend-specific failure type.
    type Error;

    /// Returns the capabilities used to compile execution plans for this backend.
    fn capabilities(&self) -> &DeviceCapabilities;

    /// Returns the identity of the device receiving commands.
    fn device_identity(&self) -> DeviceIdentity;

    /// Creates one physical transient texture for the current frame with every
    /// operation required by the compiled plan.
    fn create_transient_texture(
        &mut self,
        descriptor: TextureDesc,
        usage: TextureUsage,
    ) -> Result<BoundTexture<Self::Texture, Self::Lease>, Self::Error>;

    /// Creates one physical transient buffer for the current frame with every
    /// operation required by the compiled plan.
    fn create_transient_buffer(
        &mut self,
        descriptor: BufferDesc,
        usage: BufferUsage,
    ) -> Result<BoundBuffer<Self::Buffer, Self::Lease>, Self::Error>;

    /// Begins recording one ordered single-queue submission.
    fn begin_encoder(&mut self, queue: QueueId) -> Result<Self::Encoder, Self::Error>;

    /// Applies one semantic texture transition selected by the execution plan.
    ///
    /// `before == after` is a same-state memory barrier: implementations must
    /// preserve ordering and visibility for the declared range rather than
    /// treating it as a no-op.
    fn transition_texture(
        &mut self,
        encoder: &mut Self::Encoder,
        texture: &Self::Texture,
        range: TextureRange,
        before: ResourceAccessState,
        after: ResourceAccessState,
    ) -> Result<(), Self::Error>;

    /// Applies one semantic buffer transition selected by the execution plan.
    ///
    /// `before == after` is a same-state memory barrier: implementations must
    /// preserve ordering and visibility for the declared range rather than
    /// treating it as a no-op.
    fn transition_buffer(
        &mut self,
        encoder: &mut Self::Encoder,
        buffer: &Self::Buffer,
        range: BufferRange,
        before: ResourceAccessState,
        after: ResourceAccessState,
    ) -> Result<(), Self::Error>;

    /// Begins one raster pass.
    fn begin_raster(
        &mut self,
        encoder: &mut Self::Encoder,
        descriptor: &RasterPassDescriptor<'_, Self::Texture>,
    ) -> Result<(), Self::Error>;

    /// Ends the currently open raster pass.
    fn end_raster(&mut self, encoder: &mut Self::Encoder) -> Result<(), Self::Error>;

    /// Begins one compute pass with an optional diagnostic label.
    fn begin_compute(
        &mut self,
        encoder: &mut Self::Encoder,
        label: &str,
    ) -> Result<(), Self::Error>;

    /// Ends the currently open compute pass.
    fn end_compute(&mut self, encoder: &mut Self::Encoder) -> Result<(), Self::Error>;

    /// Begins one copy pass with an optional diagnostic label.
    fn begin_copy(&mut self, encoder: &mut Self::Encoder, label: &str) -> Result<(), Self::Error>;

    /// Ends the currently open copy pass.
    fn end_copy(&mut self, encoder: &mut Self::Encoder) -> Result<(), Self::Error>;

    /// Selects the raster pipeline for the currently open raster pass.
    fn set_raster_pipeline(
        &mut self,
        encoder: &mut Self::Encoder,
        pipeline: &Self::RasterPipeline,
    ) -> Result<(), Self::Error>;

    /// Selects the compute pipeline for the currently open compute pass.
    fn set_compute_pipeline(
        &mut self,
        encoder: &mut Self::Encoder,
        pipeline: &Self::ComputePipeline,
    ) -> Result<(), Self::Error>;

    /// Applies bindings to the currently open raster or compute pass.
    fn set_bindings(
        &mut self,
        encoder: &mut Self::Encoder,
        bindings: &Self::Bindings,
    ) -> Result<(), Self::Error>;

    /// Selects one vertex buffer for the currently open raster pass.
    fn set_vertex_buffer(
        &mut self,
        encoder: &mut Self::Encoder,
        slot: u32,
        buffer: &Self::Buffer,
        offset: u64,
    ) -> Result<(), Self::Error>;

    /// Selects the index buffer for the currently open raster pass.
    fn set_index_buffer(
        &mut self,
        encoder: &mut Self::Encoder,
        buffer: &Self::Buffer,
        offset: u64,
        format: IndexFormat,
    ) -> Result<(), Self::Error>;

    /// Sets the viewport for the currently open raster pass.
    fn set_viewport(
        &mut self,
        encoder: &mut Self::Encoder,
        viewport: Viewport,
    ) -> Result<(), Self::Error>;

    /// Sets the scissor rectangle for the currently open raster pass.
    fn set_scissor(
        &mut self,
        encoder: &mut Self::Encoder,
        scissor: ScissorRect,
    ) -> Result<(), Self::Error>;

    /// Records one non-indexed draw in the currently open raster pass.
    fn draw(
        &mut self,
        encoder: &mut Self::Encoder,
        vertices: Range<u32>,
        instances: Range<u32>,
    ) -> Result<(), Self::Error>;

    /// Records one indexed draw.
    fn draw_indexed(
        &mut self,
        encoder: &mut Self::Encoder,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) -> Result<(), Self::Error>;

    /// Records one compute dispatch in the currently open compute pass.
    fn dispatch(
        &mut self,
        encoder: &mut Self::Encoder,
        groups: [u32; 3],
    ) -> Result<(), Self::Error>;

    /// Records one texture copy in the currently open copy pass.
    fn copy_texture(
        &mut self,
        encoder: &mut Self::Encoder,
        source: &Self::Texture,
        destination: &Self::Texture,
        region: TextureCopyRegion,
    ) -> Result<(), Self::Error>;

    /// Records one buffer copy in the currently open copy pass.
    fn copy_buffer(
        &mut self,
        encoder: &mut Self::Encoder,
        source: &Self::Buffer,
        destination: &Self::Buffer,
        region: BufferCopyRegion,
    ) -> Result<(), Self::Error>;

    /// Finishes recording and returns a command buffer ready for submission.
    fn finish_encoder(
        &mut self,
        encoder: Self::Encoder,
    ) -> Result<Self::CommandBuffer, Self::Error>;

    /// Submits one finished command buffer to the backend's ordered queue and
    /// presents the acquired images associated with the supplied roots.
    ///
    /// The backend must consume each token only after accepting the command
    /// buffer, and must retain any native image lifetime required by that
    /// presentation until the returned completion becomes terminal. On an
    /// `Err`, it must leave every supplied token unconsumed so token `Drop`
    /// performs the required cancellation.
    ///
    /// Returning `Err` guarantees that this call did not transfer any GPU work
    /// or presentation request to native queues. After command acceptance, a
    /// backend must return `Ok` even if presentation later fails or acceptance
    /// is uncertain; its completion must become terminal (including
    /// [`CompletionStatus::Failed`]) so the executor can retire every lease safely.
    fn submit(
        &mut self,
        queue: QueueId,
        command_buffer: Self::CommandBuffer,
        presentations: Vec<PresentationSubmission<Self::PresentationToken>>,
    ) -> Result<Self::Completion, Self::Error>;

    /// Returns the current completion state of one prior submission.
    fn completion_status(&self, completion: &Self::Completion) -> CompletionStatus;

    /// Transfers an in-flight completion and all strong leases to the backend's
    /// non-blocking retirement queue.
    fn retire(&mut self, completion: Self::Completion, leases: Vec<Self::Lease>);

    /// Polls pending retirement entries and releases leases that reached a
    /// terminal completion state.
    fn collect_retired(&mut self) -> Result<usize, Self::Error>;
}
