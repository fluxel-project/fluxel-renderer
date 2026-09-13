//! Deterministic CPU-only backend fixture implementation.
//!
//! `TestRhi` records semantic backend operations without requiring a window or
//! GPU. It is deliberately a test double, rather than an implementation model
//! for a native graphics API.

use std::{collections::HashMap, fmt, ops::Range};

use super::{
    lease::TestLease,
    trace::{
        TestBindings, TestBuffer, TestColorAttachment, TestCompletion, TestComputePipeline,
        TestDepthStencilAttachment, TestPresentationToken, TestRasterPipeline, TestTexture,
        TestTraceEvent,
    },
};

use crate::{
    access::{BufferCopyRegion, BufferRange, TextureCopyRegion, TextureRange},
    backend::{
        BoundBuffer, BoundTexture, CompletionStatus, DeviceIdentity, ExecutionBackend,
        PresentationSubmission, RasterPassDescriptor,
    },
    pass::{ScissorRect, Viewport},
    plan::{BufferUsage, TextureUsage},
    rhi::{BufferDesc, DeviceCapabilities, IndexFormat, QueueId, ResourceAccessState, TextureDesc},
};

// Test registries commonly use compact caller-chosen handles. Reserve the
// upper half for backend-created transients so the two fixture sources do not
// manufacture false physical aliases.
const FIRST_TRANSIENT_HANDLE: u64 = 1 << 63;

/// Failure injected into one subsequent [`TestRhi`] backend operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TestRhiError {
    detail: String,
}

impl TestRhiError {
    /// Creates one injectable backend failure.
    pub fn new(detail: impl Into<String>) -> Self {
        Self {
            detail: detail.into(),
        }
    }
}

impl fmt::Display for TestRhiError {
    fn fmt(&self, _formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        _formatter.write_str(&self.detail)
    }
}

impl std::error::Error for TestRhiError {}

/// An encoder recording one test queue submission.
#[derive(Debug)]
pub struct TestEncoder {
    queue: QueueId,
    events: Vec<TestTraceEvent>,
    texture_states: HashMap<TestTexture, ResourceAccessState>,
    buffer_states: HashMap<TestBuffer, ResourceAccessState>,
}

/// A completed test command buffer.
#[derive(Debug)]
pub struct TestCommandBuffer {
    queue: QueueId,
    events: Vec<TestTraceEvent>,
}

#[derive(Debug)]
struct Retired {
    completion: TestCompletion,
    leases: Vec<TestLease>,
}

/// A deterministic CPU-only implementation of [`ExecutionBackend`].
#[derive(Debug)]
pub struct TestRhi {
    capabilities: DeviceCapabilities,
    identity: DeviceIdentity,
    next_handle: u64,
    next_completion: u64,
    trace: Vec<TestTraceEvent>,
    trace_enabled: bool,
    completion: HashMap<TestCompletion, CompletionStatus>,
    retired: Vec<Retired>,
    ended_passes: usize,
    fail_next: Option<TestRhiError>,
    fail_finish_encoder: Option<TestRhiError>,
    fail_submit: Option<TestRhiError>,
    transient_texture_usage: Option<TextureUsage>,
    transient_buffer_usage: Option<BufferUsage>,
    texture_states: HashMap<TestTexture, ResourceAccessState>,
    buffer_states: HashMap<TestBuffer, ResourceAccessState>,
}

impl TestRhi {
    /// Creates a test backend with the supplied immutable capability snapshot.
    pub fn new(capabilities: DeviceCapabilities, identity: DeviceIdentity) -> Self {
        Self {
            capabilities,
            identity,
            next_handle: FIRST_TRANSIENT_HANDLE,
            next_completion: 1,
            trace: Vec::new(),
            trace_enabled: true,
            completion: HashMap::new(),
            retired: Vec::new(),
            ended_passes: 0,
            fail_next: None,
            fail_finish_encoder: None,
            fail_submit: None,
            transient_texture_usage: None,
            transient_buffer_usage: None,
            texture_states: HashMap::new(),
            buffer_states: HashMap::new(),
        }
    }

    /// Returns all trace events recorded so far.
    pub fn trace(&self) -> &[TestTraceEvent] {
        &self.trace
    }

    /// Removes and returns all trace events recorded so far.
    pub fn take_trace(&mut self) -> Vec<TestTraceEvent> {
        std::mem::take(&mut self.trace)
    }

    /// Enables or disables structured trace collection for later executions.
    pub fn set_trace_enabled(&mut self, enabled: bool) {
        self.trace_enabled = enabled;
    }

    /// Causes the next fallible backend operation to return `error`.
    pub fn fail_next(&mut self, error: TestRhiError) {
        self.fail_next = Some(error);
    }

    /// Causes the next encoder finish to fail before submission acceptance.
    pub fn fail_finish_encoder(&mut self, error: TestRhiError) {
        self.fail_finish_encoder = Some(error);
    }

    /// Causes the next submission to fail before command acceptance.
    pub fn fail_submit(&mut self, error: TestRhiError) {
        self.fail_submit = Some(error);
    }

    /// Makes subsequent transient texture allocations report this physical
    /// allowed-operation set. This supports backend-contract tests.
    pub fn set_transient_texture_usage(&mut self, usage: TextureUsage) {
        self.transient_texture_usage = Some(usage);
    }

    /// Makes subsequent transient buffer allocations report this physical
    /// allowed-operation set. This supports backend-contract tests.
    pub fn set_transient_buffer_usage(&mut self, usage: BufferUsage) {
        self.transient_buffer_usage = Some(usage);
    }

    /// Marks a submitted token as successfully complete.
    pub fn complete(&mut self, completion: TestCompletion) {
        self.completion
            .insert(completion, CompletionStatus::Complete);
    }

    /// Marks a submitted token as failed.
    pub fn fail(&mut self, completion: TestCompletion) {
        self.completion.insert(
            completion,
            CompletionStatus::Failed(crate::backend::CompletionFailure::ExecutionFailed),
        );
    }

    /// Makes a submitted token temporarily unobservable to completion polling.
    pub fn unknown(&mut self, completion: TestCompletion) {
        self.completion
            .insert(completion, CompletionStatus::Unknown);
    }

    /// Replaces the device identity for generation-isolation contract tests.
    pub fn set_device_identity(&mut self, identity: DeviceIdentity) {
        self.identity = identity;
    }

    /// Returns the number of retirement entries that still retain leases.
    pub fn retired_count(&self) -> usize {
        self.retired.len()
    }

    /// Returns how many pass scopes were closed, including discarded encoders.
    pub fn ended_pass_count(&self) -> usize {
        self.ended_passes
    }

    fn allocate(&mut self) -> u64 {
        let handle = self.next_handle;
        self.next_handle = self
            .next_handle
            .checked_add(1)
            .expect("TestRhi transient handle space exhausted");
        handle
    }

    fn failure(&mut self) -> Result<(), TestRhiError> {
        match self.fail_next.take() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn record(&mut self, _encoder: &mut TestEncoder, _event: TestTraceEvent) {
        if self.trace_enabled {
            _encoder.events.push(_event);
        }
    }
}

impl ExecutionBackend for TestRhi {
    type Texture = TestTexture;
    type Buffer = TestBuffer;
    type RasterPipeline = TestRasterPipeline;
    type ComputePipeline = TestComputePipeline;
    type Bindings = TestBindings;
    type Encoder = TestEncoder;
    type CommandBuffer = TestCommandBuffer;
    type Completion = TestCompletion;
    type PresentationToken = TestPresentationToken;
    type Lease = TestLease;
    type Error = TestRhiError;

    fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }
    fn device_identity(&self) -> DeviceIdentity {
        self.identity
    }

    fn create_transient_texture(
        &mut self,
        _descriptor: TextureDesc,
        _usage: TextureUsage,
    ) -> Result<BoundTexture<TestTexture, TestLease>, TestRhiError> {
        self.failure()?;
        let physical = TestTexture::new(self.allocate());
        self.texture_states
            .insert(physical, ResourceAccessState::Undefined);
        let (lease, _) = TestLease::fresh();
        if self.trace_enabled {
            self.trace.push(TestTraceEvent::CreateTexture {
                texture: physical,
                descriptor: _descriptor,
                usage: _usage,
            });
        }
        Ok(BoundTexture {
            device: self.identity,
            identity: physical.identity(),
            physical,
            descriptor: _descriptor,
            usage: self.transient_texture_usage.unwrap_or(_usage),
            initial_state: ResourceAccessState::Undefined,
            lease,
        })
    }

    fn create_transient_buffer(
        &mut self,
        _descriptor: BufferDesc,
        _usage: BufferUsage,
    ) -> Result<BoundBuffer<TestBuffer, TestLease>, TestRhiError> {
        self.failure()?;
        let physical = TestBuffer::new(self.allocate());
        self.buffer_states
            .insert(physical, ResourceAccessState::Undefined);
        let (lease, _) = TestLease::fresh();
        if self.trace_enabled {
            self.trace.push(TestTraceEvent::CreateBuffer {
                buffer: physical,
                descriptor: _descriptor,
                usage: _usage,
            });
        }
        Ok(BoundBuffer {
            device: self.identity,
            identity: physical.identity(),
            physical,
            descriptor: _descriptor,
            usage: self.transient_buffer_usage.unwrap_or(_usage),
            initial_state: ResourceAccessState::Undefined,
            lease,
        })
    }

    fn begin_encoder(&mut self, _queue: QueueId) -> Result<TestEncoder, TestRhiError> {
        self.failure()?;
        Ok(TestEncoder {
            queue: _queue,
            events: vec![TestTraceEvent::BeginEncoder { queue: _queue }],
            texture_states: self.texture_states.clone(),
            buffer_states: self.buffer_states.clone(),
        })
    }

    fn transition_texture(
        &mut self,
        _encoder: &mut TestEncoder,
        _texture: &TestTexture,
        _range: TextureRange,
        _before: ResourceAccessState,
        _after: ResourceAccessState,
    ) -> Result<(), TestRhiError> {
        self.failure()?;
        let actual = _encoder.texture_states.entry(*_texture).or_insert(_before);
        if *actual != _before {
            return Err(TestRhiError::new(
                "texture transition before state differs from physical state",
            ));
        }
        *actual = _after;
        self.record(
            _encoder,
            TestTraceEvent::TransitionTexture {
                texture: *_texture,
                range: _range,
                before: _before,
                after: _after,
            },
        );
        Ok(())
    }
    fn transition_buffer(
        &mut self,
        _encoder: &mut TestEncoder,
        _buffer: &TestBuffer,
        _range: BufferRange,
        _before: ResourceAccessState,
        _after: ResourceAccessState,
    ) -> Result<(), TestRhiError> {
        self.failure()?;
        if matches!(_range, BufferRange::Whole) {
            let actual = _encoder.buffer_states.entry(*_buffer).or_insert(_before);
            if *actual != _before {
                return Err(TestRhiError::new(
                    "buffer transition before state differs from physical state",
                ));
            }
            *actual = _after;
        }
        self.record(
            _encoder,
            TestTraceEvent::TransitionBuffer {
                buffer: *_buffer,
                range: _range,
                before: _before,
                after: _after,
            },
        );
        Ok(())
    }
    fn begin_raster(
        &mut self,
        _encoder: &mut TestEncoder,
        _descriptor: &RasterPassDescriptor<'_, TestTexture>,
    ) -> Result<(), TestRhiError> {
        self.failure()?;
        self.record(
            _encoder,
            TestTraceEvent::BeginRaster {
                label: _descriptor.label.to_owned(),
                color_attachments: _descriptor.colors.len(),
                has_depth_stencil: _descriptor.depth_stencil.is_some(),
                colors: _descriptor
                    .colors
                    .iter()
                    .map(|attachment| TestColorAttachment {
                        index: attachment.index,
                        texture: *attachment.texture,
                        range: attachment.range,
                        operations: attachment.operations,
                    })
                    .collect(),
                depth_stencil: _descriptor.depth_stencil.as_ref().map(|attachment| {
                    TestDepthStencilAttachment {
                        texture: *attachment.texture,
                        range: attachment.range,
                        depth: attachment.depth,
                        stencil: attachment.stencil,
                    }
                }),
            },
        );
        Ok(())
    }
    fn end_raster(&mut self, _encoder: &mut TestEncoder) -> Result<(), TestRhiError> {
        self.failure()?;
        self.ended_passes += 1;
        self.record(_encoder, TestTraceEvent::EndRaster);
        Ok(())
    }
    fn begin_compute(
        &mut self,
        _encoder: &mut TestEncoder,
        _label: &str,
    ) -> Result<(), TestRhiError> {
        self.failure()?;
        self.record(
            _encoder,
            TestTraceEvent::BeginCompute {
                label: _label.to_owned(),
            },
        );
        Ok(())
    }
    fn end_compute(&mut self, _encoder: &mut TestEncoder) -> Result<(), TestRhiError> {
        self.failure()?;
        self.ended_passes += 1;
        self.record(_encoder, TestTraceEvent::EndCompute);
        Ok(())
    }
    fn begin_copy(&mut self, _encoder: &mut TestEncoder, _label: &str) -> Result<(), TestRhiError> {
        self.failure()?;
        self.record(
            _encoder,
            TestTraceEvent::BeginCopy {
                label: _label.to_owned(),
            },
        );
        Ok(())
    }
    fn end_copy(&mut self, _encoder: &mut TestEncoder) -> Result<(), TestRhiError> {
        self.failure()?;
        self.ended_passes += 1;
        self.record(_encoder, TestTraceEvent::EndCopy);
        Ok(())
    }
    fn set_raster_pipeline(
        &mut self,
        _encoder: &mut TestEncoder,
        _pipeline: &TestRasterPipeline,
    ) -> Result<(), TestRhiError> {
        self.failure()?;
        self.record(
            _encoder,
            TestTraceEvent::SetRasterPipeline {
                pipeline: *_pipeline,
            },
        );
        Ok(())
    }
    fn set_compute_pipeline(
        &mut self,
        _encoder: &mut TestEncoder,
        _pipeline: &TestComputePipeline,
    ) -> Result<(), TestRhiError> {
        self.failure()?;
        self.record(
            _encoder,
            TestTraceEvent::SetComputePipeline {
                pipeline: *_pipeline,
            },
        );
        Ok(())
    }
    fn set_bindings(
        &mut self,
        _encoder: &mut TestEncoder,
        _bindings: &TestBindings,
    ) -> Result<(), TestRhiError> {
        self.failure()?;
        self.record(
            _encoder,
            TestTraceEvent::SetBindings {
                bindings: *_bindings,
            },
        );
        Ok(())
    }
    fn set_vertex_buffer(
        &mut self,
        _encoder: &mut TestEncoder,
        _slot: u32,
        _buffer: &TestBuffer,
        _offset: u64,
    ) -> Result<(), TestRhiError> {
        self.failure()?;
        self.record(
            _encoder,
            TestTraceEvent::SetVertexBuffer {
                slot: _slot,
                buffer: *_buffer,
                offset: _offset,
            },
        );
        Ok(())
    }
    fn set_index_buffer(
        &mut self,
        _encoder: &mut TestEncoder,
        _buffer: &TestBuffer,
        _offset: u64,
        _format: IndexFormat,
    ) -> Result<(), TestRhiError> {
        self.failure()?;
        self.record(
            _encoder,
            TestTraceEvent::SetIndexBuffer {
                buffer: *_buffer,
                offset: _offset,
                format: _format,
            },
        );
        Ok(())
    }
    fn set_viewport(
        &mut self,
        _encoder: &mut TestEncoder,
        _viewport: Viewport,
    ) -> Result<(), TestRhiError> {
        self.failure()?;
        self.record(
            _encoder,
            TestTraceEvent::SetViewport {
                viewport: _viewport,
            },
        );
        Ok(())
    }
    fn set_scissor(
        &mut self,
        _encoder: &mut TestEncoder,
        _scissor: ScissorRect,
    ) -> Result<(), TestRhiError> {
        self.failure()?;
        self.record(_encoder, TestTraceEvent::SetScissor { scissor: _scissor });
        Ok(())
    }
    fn draw(
        &mut self,
        _encoder: &mut TestEncoder,
        _vertices: Range<u32>,
        _instances: Range<u32>,
    ) -> Result<(), TestRhiError> {
        self.failure()?;
        self.record(
            _encoder,
            TestTraceEvent::Draw {
                vertices: _vertices,
                instances: _instances,
            },
        );
        Ok(())
    }
    fn draw_indexed(
        &mut self,
        _encoder: &mut TestEncoder,
        _indices: Range<u32>,
        _base_vertex: i32,
        _instances: Range<u32>,
    ) -> Result<(), TestRhiError> {
        self.failure()?;
        self.record(
            _encoder,
            TestTraceEvent::DrawIndexed {
                indices: _indices,
                base_vertex: _base_vertex,
                instances: _instances,
            },
        );
        Ok(())
    }
    fn dispatch(
        &mut self,
        _encoder: &mut TestEncoder,
        _groups: [u32; 3],
    ) -> Result<(), TestRhiError> {
        self.failure()?;
        self.record(_encoder, TestTraceEvent::Dispatch { groups: _groups });
        Ok(())
    }
    fn copy_texture(
        &mut self,
        _encoder: &mut TestEncoder,
        _source: &TestTexture,
        _destination: &TestTexture,
        _region: TextureCopyRegion,
    ) -> Result<(), TestRhiError> {
        self.failure()?;
        self.record(
            _encoder,
            TestTraceEvent::CopyTexture {
                source: *_source,
                destination: *_destination,
                region: _region,
            },
        );
        Ok(())
    }
    fn copy_buffer(
        &mut self,
        _encoder: &mut TestEncoder,
        _source: &TestBuffer,
        _destination: &TestBuffer,
        _region: BufferCopyRegion,
    ) -> Result<(), TestRhiError> {
        self.failure()?;
        self.record(
            _encoder,
            TestTraceEvent::CopyBuffer {
                source: *_source,
                destination: *_destination,
                region: _region,
            },
        );
        Ok(())
    }
    fn finish_encoder(
        &mut self,
        mut _encoder: TestEncoder,
    ) -> Result<TestCommandBuffer, TestRhiError> {
        if let Some(error) = self.fail_finish_encoder.take() {
            return Err(error);
        }
        self.failure()?;
        if self.trace_enabled {
            _encoder.events.push(TestTraceEvent::Finish {
                queue: _encoder.queue,
            });
        }
        Ok(TestCommandBuffer {
            queue: _encoder.queue,
            events: _encoder.events,
        })
    }
    fn submit(
        &mut self,
        _queue: QueueId,
        _command_buffer: TestCommandBuffer,
        _presentations: Vec<PresentationSubmission<TestPresentationToken>>,
    ) -> Result<TestCompletion, TestRhiError> {
        if let Some(error) = self.fail_submit.take() {
            return Err(error);
        }
        self.failure()?;
        if _command_buffer.queue != _queue {
            return Err(TestRhiError::new(
                "test command buffer was submitted to a different queue",
            ));
        }
        for event in &_command_buffer.events {
            match event {
                TestTraceEvent::TransitionTexture {
                    texture,
                    before,
                    after,
                    ..
                } => {
                    let actual = self.texture_states.entry(*texture).or_insert(*before);
                    if *actual != *before {
                        return Err(TestRhiError::new(
                            "submitted texture transition before state differs from physical state",
                        ));
                    }
                    *actual = *after;
                }
                TestTraceEvent::TransitionBuffer {
                    buffer,
                    range,
                    before,
                    after,
                } => {
                    if matches!(range, BufferRange::Whole) {
                        let actual = self.buffer_states.entry(*buffer).or_insert(*before);
                        if *actual != *before {
                            return Err(TestRhiError::new(
                                "submitted buffer transition before state differs from physical state",
                            ));
                        }
                        *actual = *after;
                    }
                }
                _ => {}
            }
        }
        let completion = TestCompletion::new(self.next_completion);
        self.next_completion += 1;
        let presentations: Vec<_> = _presentations
            .into_iter()
            .map(|presentation| {
                let value = presentation.token.value();
                presentation.token.mark_presented();
                (presentation.target, value)
            })
            .collect();
        if self.trace_enabled {
            self.trace.extend(_command_buffer.events);
            self.trace.push(TestTraceEvent::Submit {
                queue: _queue,
                completion,
                presentations,
            });
        }
        self.completion
            .insert(completion, CompletionStatus::Pending);
        Ok(completion)
    }
    fn completion_status(&self, _completion: &TestCompletion) -> CompletionStatus {
        self.completion
            .get(_completion)
            .copied()
            .unwrap_or(CompletionStatus::Failed(
                crate::backend::CompletionFailure::ExecutionFailed,
            ))
    }
    fn retire(&mut self, _completion: TestCompletion, _leases: Vec<TestLease>) {
        if self.trace_enabled {
            self.trace.push(TestTraceEvent::Retire {
                completion: _completion,
                lease_count: _leases.len(),
            });
        }
        self.retired.push(Retired {
            completion: _completion,
            leases: _leases,
        });
    }
    fn collect_retired(&mut self) -> Result<usize, TestRhiError> {
        self.failure()?;
        let before = self.retired.len();
        let completion = &self.completion;
        self.retired.retain(|entry| {
            let _ = entry.leases.len();
            matches!(
                completion.get(&entry.completion).copied(),
                Some(CompletionStatus::Pending | CompletionStatus::Unknown)
            )
        });
        Ok(before - self.retired.len())
    }
}
