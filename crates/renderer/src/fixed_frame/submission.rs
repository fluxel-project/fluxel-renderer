//! Drives fixed-frame upload and raster completion through one terminal state machine.
//!
//! Work that has not reached native submission may release its snapshot reservations,
//! while abandoning accepted raster work must poison every participating generation.
//! The state machine hands completed readback images to callers and never exposes RHI
//! completion or reservation internals.

use fluxel_rhi::NativeExecutionError;

/// A non-blocking two-phase fixed frame operation.
pub struct FixedFrameSubmission {
    pub(super) phase: CameraPhase,
    pub(in crate::fixed_frame) executor: Arc<fluxel_rendergraph::FrameExecutor<RasterBackend>>,
    pub(in crate::fixed_frame) snapshot: FrameMeshSnapshot,
    pub(super) graph: Option<Arc<CameraGraph>>,
    pub(in crate::fixed_frame) objects: Option<RasterObjectProvider>,
    pub(in crate::fixed_frame) reservation: Option<SnapshotDrawReservation>,
    pub(in crate::fixed_frame) texture_snapshot: Option<FrameTextureSnapshot>,
    pub(in crate::fixed_frame) texture_reservation: Option<SnapshotDrawReservation>,
    pub(super) target_export: Option<fluxel_rendergraph::ExportTextureSlot>,
    pub(in crate::fixed_frame) image: Option<FrameImage>,
    pub(in crate::fixed_frame) failure: Option<FixedFrameFailure>,
    #[cfg(all(test, windows))]
    pub(super) completed: Option<fluxel_rendergraph::ExecutedFrame<RasterBackend>>,
}

#[derive(Clone)]
pub(super) enum FrameMeshSnapshot {
    Indexed(IndexedMeshSnapshot),
    Normal(NormalIndexedMeshSnapshot),
    TexturedUv(TexturedIndexedMeshSnapshot),
    VertexColor(VertexColorIndexedMeshSnapshot),
}

impl FrameMeshSnapshot {
    pub(super) fn positions(&self) -> &UploadedBuffer {
        match self {
            Self::Indexed(snapshot) => snapshot.positions(),
            Self::Normal(snapshot) => snapshot.positions(),
            Self::TexturedUv(snapshot) => snapshot.positions(),
            Self::VertexColor(snapshot) => snapshot.positions(),
        }
    }

    pub(super) fn indices(&self) -> &UploadedBuffer {
        match self {
            Self::Indexed(snapshot) => snapshot.indices(),
            Self::Normal(snapshot) => snapshot.indices(),
            Self::TexturedUv(snapshot) => snapshot.indices(),
            Self::VertexColor(snapshot) => snapshot.indices(),
        }
    }

    pub(super) fn texture_coordinates(&self) -> Option<&UploadedBuffer> {
        match self {
            Self::Indexed(_) | Self::Normal(_) | Self::VertexColor(_) => None,
            Self::TexturedUv(snapshot) => Some(snapshot.texture_coordinates()),
        }
    }

    pub(super) fn normals(&self) -> Option<&UploadedBuffer> {
        match self {
            Self::Normal(snapshot) => Some(snapshot.normals()),
            Self::Indexed(_) | Self::TexturedUv(_) | Self::VertexColor(_) => None,
        }
    }

    pub(super) fn colors(&self) -> Option<&UploadedBuffer> {
        match self {
            Self::VertexColor(snapshot) => Some(snapshot.colors()),
            Self::Indexed(_) | Self::Normal(_) | Self::TexturedUv(_) => None,
        }
    }

    pub(super) fn position_count(&self) -> u32 {
        match self {
            Self::Indexed(snapshot) => snapshot.position_count(),
            Self::Normal(snapshot) => snapshot.position_count(),
            Self::TexturedUv(snapshot) => snapshot.position_count(),
            Self::VertexColor(snapshot) => snapshot.position_count(),
        }
    }
}

pub(super) enum CameraPhase {
    Uploading(PendingBufferUpload),
    RasterReady(UploadedBuffer),
    RasterAccepted(fluxel_rendergraph::ExecutedFrame<RasterBackend>),
    Terminal,
}

impl FixedFrameSubmission {
    /// Polls GPU completion without waiting.
    pub fn poll(&mut self) -> FixedFrameStatus {
        if let Some(image) = &self.image {
            return FixedFrameStatus::Complete(image.clone());
        }
        if let Some(failure) = &self.failure {
            return FixedFrameStatus::Failed(failure.clone());
        }
        // Temporarily install Terminal so every match arm must explicitly restore
        // a retryable phase or publish a terminal result; no early return can
        // accidentally leave an already-consumed operation reusable.
        let phase = core::mem::replace(&mut self.phase, CameraPhase::Terminal);
        match phase {
            CameraPhase::Uploading(upload) => self.poll_upload(upload),
            CameraPhase::RasterReady(uniform) => self.submit_raster(uniform),
            CameraPhase::RasterAccepted(frame) => self.poll_raster(frame),
            CameraPhase::Terminal => unreachable!("terminal operation has image or failure"),
        }
    }

    fn poll_upload(&mut self, upload: PendingBufferUpload) -> FixedFrameStatus {
        match upload.status() {
            Ok(status) if completion_requires_retention(status) => {
                self.phase = CameraPhase::Uploading(upload);
                FixedFrameStatus::Pending
            }
            Ok(CompletionStatus::Complete) => match upload.finalize() {
                Ok(uniform) => self.submit_raster(uniform),
                Err(incomplete) => {
                    self.phase = CameraPhase::Uploading(incomplete.into_pending());
                    FixedFrameStatus::Pending
                }
            },
            Ok(CompletionStatus::Failed(error)) => {
                self.finish_pre_accept(FixedFrameFailure::UniformCompletion(error));
                FixedFrameStatus::Failed(self.failure.clone().unwrap())
            }
            Ok(_) => {
                self.finish_pre_accept(FixedFrameFailure::UniformObservation(
                    FixedFrameUniformObservationError::UnknownCompletionStatus,
                ));
                FixedFrameStatus::Failed(self.failure.clone().unwrap())
            }
            Err(error) => {
                self.finish_pre_accept(FixedFrameFailure::UniformObservation(
                    FixedFrameUniformObservationError::Status(error),
                ));
                FixedFrameStatus::Failed(self.failure.clone().unwrap())
            }
        }
    }

    fn submit_raster(&mut self, uniform: UploadedBuffer) -> FixedFrameStatus {
        let graph = self.graph.as_ref().expect("pre-accept retains graph");
        let objects = self.objects.as_ref().expect("pre-accept retains objects");
        let resources = CameraResources {
            device: self.snapshot.positions().buffer().device_identity(),
            positions: self.snapshot.positions().buffer().clone(),
            indices: self.snapshot.indices().buffer().clone(),
            texture_coordinates: self
                .snapshot
                .texture_coordinates()
                .map(|coordinates| coordinates.buffer().clone()),
            normals: self
                .snapshot
                .normals()
                .map(|normals| normals.buffer().clone()),
            colors: self.snapshot.colors().map(|colors| colors.buffer().clone()),
            uniform: uniform.buffer().clone(),
            texture: self
                .texture_snapshot
                .as_ref()
                .map(|texture| texture.texture().texture().clone()),
            position_state: self.snapshot.positions().outgoing_state(),
            index_state: self.snapshot.indices().outgoing_state(),
            texture_coordinate_state: self
                .snapshot
                .texture_coordinates()
                .map(UploadedBuffer::outgoing_state),
            normal_state: self.snapshot.normals().map(UploadedBuffer::outgoing_state),
            color_state: self.snapshot.colors().map(UploadedBuffer::outgoing_state),
            texture_state: self
                .texture_snapshot
                .as_ref()
                .map(|texture| texture.texture().outgoing_state()),
        };
        let mut inputs = FrameInputs::new(());
        inputs
            .bind_buffer(graph.position_slot, position_binding())
            .bind_buffer(graph.index_slot, index_binding())
            .bind_buffer(graph.uniform_slot, uniform_binding());
        if let Some(slot) = graph.texture_coordinate_slot {
            inputs.bind_buffer(slot, texture_coordinate_binding());
        }
        if let Some(slot) = graph.normal_slot {
            inputs.bind_buffer(slot, normal_binding());
        }
        if let Some(slot) = graph.vertex_color_slot {
            inputs.bind_buffer(slot, vertex_color_binding());
        }
        if let Some(slot) = graph.texture_slot {
            inputs.bind_texture(slot, texture_binding());
        }
        match self.executor.execute(
            &graph.compiled,
            graph.compiled.instantiate_local(inputs),
            &resources,
            objects,
        ) {
            Ok(frame) => {
                self.target_export = graph.target_export;
                self.phase = CameraPhase::RasterAccepted(frame);
                FixedFrameStatus::Pending
            }
            Err(ExecutionError::ExecutorBusy) => {
                self.phase = CameraPhase::RasterReady(uniform);
                FixedFrameStatus::Busy
            }
            Err(error) => {
                self.finish_pre_accept(FixedFrameFailure::RasterStart(execution_error(error)));
                FixedFrameStatus::Failed(self.failure.clone().unwrap())
            }
        }
    }

    fn poll_raster(
        &mut self,
        mut frame: fluxel_rendergraph::ExecutedFrame<RasterBackend>,
    ) -> FixedFrameStatus {
        match frame.submission.status() {
            Err(ExecutionError::ExecutorBusy) => {
                self.phase = CameraPhase::RasterAccepted(frame);
                FixedFrameStatus::Busy
            }
            Err(error) => self.finish_accepted(FixedFrameFailure::RasterObservation(
                FixedFrameRasterObservationError::Execution(execution_error(error)),
            )),
            Ok(status) if completion_requires_retention(status) => {
                self.phase = CameraPhase::RasterAccepted(frame);
                FixedFrameStatus::Pending
            }
            Ok(CompletionStatus::Complete) => {
                let Some(exported) = frame.exports.texture(
                    self.target_export
                        .expect("accepted frame has target export"),
                ) else {
                    return self.finish_accepted(FixedFrameFailure::RasterObservation(
                        FixedFrameRasterObservationError::MissingTargetExport,
                    ));
                };
                let image = FrameImage {
                    extent: [
                        exported.descriptor.extent.width,
                        exported.descriptor.extent.height,
                    ],
                    format: exported.descriptor.format,
                    _texture: exported.physical.clone(),
                };
                self.reservation
                    .take()
                    .expect("accepted fixed frame owns its reservation")
                    .release_complete();
                if let Some(mut reservation) = self.texture_reservation.take() {
                    reservation.release_complete();
                }
                self.image = Some(image.clone());
                #[cfg(all(test, windows))]
                {
                    self.completed = Some(frame);
                }
                #[cfg(not(all(test, windows)))]
                {
                    // Completion has copied the only application-visible
                    // ownership into FrameImage; graph/provider artifacts
                    // are no longer needed by the public operation.
                    self.graph.take();
                    self.objects.take();
                }
                self.phase = CameraPhase::Terminal;
                FixedFrameStatus::Complete(image)
            }
            Ok(CompletionStatus::Failed(failure)) => {
                self.finish_accepted(FixedFrameFailure::RasterCompletion(failure))
            }
            Ok(_) => self.finish_accepted(FixedFrameFailure::RasterObservation(
                FixedFrameRasterObservationError::UnknownCompletionStatus,
            )),
        }
    }

    fn finish_pre_accept(&mut self, failure: FixedFrameFailure) {
        self.graph.take();
        self.objects.take();
        if let Some(reservation) = self.reservation.take() {
            reservation.release_before_submit();
        }
        if let Some(reservation) = self.texture_reservation.take() {
            reservation.release_before_submit();
        }
        self.phase = CameraPhase::Terminal;
        self.failure = Some(failure);
    }
    fn finish_accepted(&mut self, failure: FixedFrameFailure) -> FixedFrameStatus {
        if let Some(mut reservation) = self.reservation.take() {
            reservation.poison();
        }
        if let Some(mut reservation) = self.texture_reservation.take() {
            reservation.poison();
        }
        self.phase = CameraPhase::Terminal;
        self.failure = Some(failure.clone());
        FixedFrameStatus::Failed(failure)
    }
}

impl Drop for FixedFrameSubmission {
    fn drop(&mut self) {
        // Before submit, both generation gates are safe to release. Once raster
        // work is accepted, completion is unknown, so abandoning either gate
        // poisons the whole mesh/texture reservation set monotonically.
        if let Some(reservation) = self.reservation.take() {
            if matches!(self.phase, CameraPhase::RasterAccepted(_)) {
                let mut reservation = reservation;
                reservation.poison();
            } else {
                reservation.release_before_submit();
            }
        }
        if let Some(reservation) = self.texture_reservation.take() {
            if matches!(self.phase, CameraPhase::RasterAccepted(_)) {
                let mut reservation = reservation;
                reservation.poison();
            } else {
                reservation.release_before_submit();
            }
        }
    }
}

pub(super) fn execution_error(
    error: ExecutionError<NativeExecutionError>,
) -> FixedFrameExecutionError {
    match error {
        ExecutionError::FrameBinding(error) => FixedFrameExecutionError::FrameBinding(error),
        ExecutionError::Recording(error) => FixedFrameExecutionError::Recording(error),
        ExecutionError::CapabilityMismatch => FixedFrameExecutionError::CapabilityMismatch,
        ExecutionError::WrongCompiledGraph => FixedFrameExecutionError::WrongCompiledGraph,
        ExecutionError::UnsupportedExecutionFeature(feature) => {
            FixedFrameExecutionError::UnsupportedExecutionFeature(feature)
        }
        ExecutionError::Backend(error) => FixedFrameExecutionError::Backend(error),
        ExecutionError::ExecutorBusy => unreachable!("busy is handled before failure mapping"),
        _ => FixedFrameExecutionError::Unknown,
    }
}

/// The observed state of a [`FixedFrameSubmission`].
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum FixedFrameStatus {
    /// Uniform upload or an accepted raster submission is still incomplete.
    Pending,
    /// The executor is momentarily held by another safe operation; retry poll.
    Busy,
    /// The fixed frame completed and its opaque output is available.
    Complete(FrameImage),
    /// The GPU did not establish the promised terminal state.
    Failed(FixedFrameFailure),
}

/// Opaque ownership plus metadata for one completed fixed target.
#[derive(Clone, Debug)]
pub struct FrameImage {
    pub(in crate::fixed_frame) extent: [u32; 2],
    pub(in crate::fixed_frame) format: TextureFormat,
    // Retained privately so application code cannot observe or use a native object.
    pub(in crate::fixed_frame) _texture: Texture,
}

impl FrameImage {
    /// Returns the pixel dimensions of this headless image.
    #[must_use]
    pub const fn extent(&self) -> [u32; 2] {
        self.extent
    }

    /// Returns the fixed image format.
    #[must_use]
    pub const fn format(&self) -> TextureFormat {
        self.format
    }
}

#[cfg(all(test, windows))]
pub(super) struct SnapshotResources {
    pub(super) device: DeviceIdentity,
    pub(super) positions: Buffer,
    pub(super) indices: Buffer,
    pub(super) position_state: ResourceAccessState,
    pub(super) index_state: ResourceAccessState,
}

#[cfg(all(test, windows))]
impl FrameResourceProvider<RasterBackend> for SnapshotResources {
    fn texture(
        &self,
        _: fluxel_rendergraph::TextureBindingId,
    ) -> Result<fluxel_rendergraph::BoundTexture<Texture, ResourceLease>, FrameBindingError> {
        Err(missing_binding(
            FrameBindingErrorKind::MissingTexture,
            "fixed frame has no texture imports",
        ))
    }

    fn buffer(
        &self,
        id: BufferBindingId,
    ) -> Result<BoundBuffer<Buffer, ResourceLease>, FrameBindingError> {
        let (buffer, initial_state) = match id {
            id if id == position_binding() => (&self.positions, self.position_state),
            id if id == index_binding() => (&self.indices, self.index_state),
            _ => {
                return Err(missing_binding(
                    FrameBindingErrorKind::MissingBuffer,
                    "unknown fixed frame buffer",
                ));
            }
        };
        Ok(BoundBuffer {
            device: self.device,
            identity: buffer.identity(),
            physical: buffer.clone(),
            descriptor: buffer.descriptor().buffer,
            usage: buffer.allowed_usage(),
            initial_state,
            lease: buffer.lease().into(),
        })
    }
}

pub(super) fn missing_binding(kind: FrameBindingErrorKind, detail: &str) -> FrameBindingError {
    FrameBindingError {
        kind,
        texture_slot: None,
        buffer_slot: None,
        resource: None,
        surface_binding: None,
        detail: detail.into(),
    }
}
use super::*;
