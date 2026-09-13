//! Owns the fixed visible-frame transaction from an acquired image to present completion.
//!
//! This module is intentionally the sole renderer path that turns an RHI
//! acquired frame into graph input. It does not expose a surface, native handle,
//! or presentation token to callers. Before queue acceptance a failure releases
//! the mesh reservation and dropping the token discards the image; after
//! acceptance a failure poisons that reservation because completion is unknown.

use core::fmt;
use std::{cell::RefCell, sync::Arc};

use super::*;

/// Why a visible fixed frame could not start.
#[derive(Debug)]
#[non_exhaustive]
pub enum VisibleFrameStartError {
    /// The closed mesh/camera/material recipe was not admissible.
    Draw(DrawStartError),
    /// The acquired presentation image belongs to another device.
    ForeignSurfaceDevice,
}

impl fmt::Display for VisibleFrameStartError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Draw(error) => write!(f, "visible fixed frame did not start: {error}"),
            Self::ForeignSurfaceDevice => {
                f.write_str("acquired presentation image belongs to another device")
            }
        }
    }
}

impl std::error::Error for VisibleFrameStartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Draw(error) => Some(error),
            Self::ForeignSurfaceDevice => None,
        }
    }
}

/// Observable state of one visible fixed-frame transaction.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum VisibleFrameStatus {
    /// Uniform upload or accepted present work remains incomplete.
    Pending,
    /// Another operation temporarily holds the renderer executor.
    Busy,
    /// Raster submission was accepted and the acquired image was consumed by
    /// the native presentation path; GPU completion is not yet known.
    ///
    /// This does not promise that a later native present operation succeeded.
    /// The milestone permits orchestration to acquire a later surface image;
    /// it does not permit this submission's resources or frame slot to retire.
    Submitted,
    /// The image was submitted and presented successfully.
    Complete,
    /// The work did not establish its promised terminal state.
    Failed(FixedFrameFailure),
}

/// A non-blocking renderer-owned submission of one acquired presentation image.
pub struct VisibleFrameSubmission {
    phase: VisiblePhase,
    executor: Arc<fluxel_rendergraph::FrameExecutor<RasterBackend>>,
    graph: Option<Arc<CameraGraph>>,
    objects: Option<RasterObjectProvider>,
    resources: Option<VisibleCameraResources>,
    reservation: Option<SnapshotDrawReservation>,
    failure: Option<FixedFrameFailure>,
}

enum VisiblePhase {
    Uploading(PendingBufferUpload),
    RasterReady(UploadedBuffer),
    RasterAccepted(fluxel_rendergraph::ExecutedFrame<RasterBackend>),
    Terminal,
}

impl FixedFrameRenderer {
    pub(in crate::fixed_frame) fn start_visible_camera(
        &self,
        snapshot: &IndexedMeshSnapshot,
        graph: Arc<CameraGraph>,
        uniform: FrameUniform,
        reservation: SnapshotDrawReservation,
        surface: fluxel_rendergraph::BoundSurfaceTexture<Texture, TextureLease, PresentationToken>,
    ) -> Result<VisibleFrameSubmission, VisibleFrameStartError> {
        let pipeline = match self
            .device
            .create_raster_pipeline(visible_camera_recipe().kernel())
        {
            Ok(pipeline) => pipeline,
            Err(error) => {
                return Err(release_visible_start(
                    reservation,
                    surface,
                    VisibleFrameStartError::Draw(DrawStartError::Pipeline(error)),
                ));
            }
        };
        let mut objects = RasterObjectProvider::new(&self.device);
        if let Err(error) = objects.register_raster_pipeline(graph.pipeline, pipeline) {
            return Err(release_visible_start(
                reservation,
                surface,
                VisibleFrameStartError::Draw(DrawStartError::Provider(error)),
            ));
        }
        if let Err(error) = objects.register_raster_uniform_bindings(graph.bindings, graph.pipeline)
        {
            return Err(release_visible_start(
                reservation,
                surface,
                VisibleFrameStartError::Draw(DrawStartError::Provider(error)),
            ));
        }
        let pending = match self.device.upload_immutable_buffer(
            BufferDescriptor {
                buffer: fluxel_rendergraph::BufferDesc {
                    size: FRAME_UNIFORM_BYTES as u64,
                },
                usage: fluxel_rendergraph::BufferUsage::from_kinds([
                    fluxel_rendergraph::BufferUsageKind::CopyDestination,
                    fluxel_rendergraph::BufferUsageKind::Uniform,
                ]),
                memory: MemoryPolicy::DeviceOnly,
            },
            uniform.bytes(),
        ) {
            Ok(pending) => pending,
            Err(error) => {
                return Err(release_visible_start(
                    reservation,
                    surface,
                    VisibleFrameStartError::Draw(DrawStartError::UniformStart(error)),
                ));
            }
        };
        Ok(VisibleFrameSubmission {
            phase: VisiblePhase::Uploading(pending),
            executor: Arc::clone(&self.executor),
            graph: Some(graph),
            objects: Some(objects),
            resources: Some(VisibleCameraResources::new(
                self.device.identity(),
                snapshot,
                surface,
            )),
            reservation: Some(reservation),
            failure: None,
        })
    }
}

/// The visible target changes only output ownership, never the fixed camera
/// material pipeline or its uniform binding ABI.
const fn visible_camera_recipe() -> RasterRecipe {
    RasterRecipe::LEGACY_UNLIT
}

fn release_visible_start(
    reservation: SnapshotDrawReservation,
    surface: fluxel_rendergraph::BoundSurfaceTexture<Texture, TextureLease, PresentationToken>,
    error: VisibleFrameStartError,
) -> VisibleFrameStartError {
    reservation.release_before_submit();
    drop(surface);
    error
}

impl VisibleFrameSubmission {
    /// Polls upload, raster, and presentation completion without waiting.
    pub fn poll(&mut self) -> VisibleFrameStatus {
        if let Some(failure) = &self.failure {
            return VisibleFrameStatus::Failed(failure.clone());
        }
        let phase = core::mem::replace(&mut self.phase, VisiblePhase::Terminal);
        match phase {
            VisiblePhase::Uploading(upload) => self.poll_upload(upload),
            VisiblePhase::RasterReady(uniform) => self.submit_raster(uniform),
            VisiblePhase::RasterAccepted(frame) => self.poll_raster(frame),
            VisiblePhase::Terminal => {
                unreachable!("terminal visible frame has a failure or completion")
            }
        }
    }

    fn poll_upload(&mut self, upload: PendingBufferUpload) -> VisibleFrameStatus {
        match upload.status() {
            Ok(status) if completion_requires_retention(status) => {
                self.phase = VisiblePhase::Uploading(upload);
                VisibleFrameStatus::Pending
            }
            Ok(CompletionStatus::Complete) => match upload.finalize() {
                Ok(uniform) => self.submit_raster(uniform),
                Err(incomplete) => {
                    self.phase = VisiblePhase::Uploading(incomplete.into_pending());
                    VisibleFrameStatus::Pending
                }
            },
            Ok(CompletionStatus::Failed(error)) => {
                self.finish_pre_accept(FixedFrameFailure::UniformCompletion(error))
            }
            Ok(_) => self.finish_pre_accept(FixedFrameFailure::UniformObservation(
                FixedFrameUniformObservationError::UnknownCompletionStatus,
            )),
            Err(error) => self.finish_pre_accept(FixedFrameFailure::UniformObservation(
                FixedFrameUniformObservationError::Status(error),
            )),
        }
    }

    fn submit_raster(&mut self, uniform: UploadedBuffer) -> VisibleFrameStatus {
        let graph = self.graph.as_ref().expect("pre-accept retains graph");
        let resources = self
            .resources
            .as_ref()
            .expect("pre-accept retains acquired image");
        let objects = self.objects.as_ref().expect("pre-accept retains objects");
        let mut inputs = FrameInputs::new(());
        inputs
            .bind_buffer(graph.position_slot, position_binding())
            .bind_buffer(graph.index_slot, index_binding())
            .bind_buffer(graph.uniform_slot, uniform_binding())
            .bind_surface(
                graph
                    .surface_slot
                    .expect("visible graph has a surface slot"),
                surface_binding(),
            );
        resources.set_uniform(uniform.buffer().clone());
        match self.executor.execute(
            &graph.compiled,
            graph.compiled.instantiate_local(inputs),
            resources,
            objects,
        ) {
            Ok(frame) => {
                self.phase = VisiblePhase::RasterAccepted(frame);
                VisibleFrameStatus::Submitted
            }
            Err(ExecutionError::ExecutorBusy) => {
                self.phase = VisiblePhase::RasterReady(uniform);
                VisibleFrameStatus::Busy
            }
            Err(error) => self.finish_pre_accept(FixedFrameFailure::RasterStart(
                super::submission::execution_error(error),
            )),
        }
    }

    fn poll_raster(
        &mut self,
        mut frame: fluxel_rendergraph::ExecutedFrame<RasterBackend>,
    ) -> VisibleFrameStatus {
        match frame.submission.status() {
            Err(ExecutionError::ExecutorBusy) => {
                self.phase = VisiblePhase::RasterAccepted(frame);
                VisibleFrameStatus::Busy
            }
            Err(error) => self.finish_accepted(FixedFrameFailure::RasterObservation(
                FixedFrameRasterObservationError::Execution(super::submission::execution_error(
                    error,
                )),
            )),
            Ok(status) if completion_requires_retention(status) => {
                self.phase = VisiblePhase::RasterAccepted(frame);
                VisibleFrameStatus::Pending
            }
            Ok(CompletionStatus::Complete) => {
                self.reservation
                    .take()
                    .expect("accepted visible frame owns its reservation")
                    .release_complete();
                self.graph.take();
                self.objects.take();
                self.resources.take();
                self.phase = VisiblePhase::Terminal;
                VisibleFrameStatus::Complete
            }
            Ok(CompletionStatus::Failed(failure)) => {
                self.finish_accepted(FixedFrameFailure::RasterCompletion(failure))
            }
            Ok(_) => self.finish_accepted(FixedFrameFailure::RasterObservation(
                FixedFrameRasterObservationError::UnknownCompletionStatus,
            )),
        }
    }

    fn finish_pre_accept(&mut self, failure: FixedFrameFailure) -> VisibleFrameStatus {
        self.graph.take();
        self.objects.take();
        self.resources.take();
        if let Some(reservation) = self.reservation.take() {
            reservation.release_before_submit();
        }
        self.phase = VisiblePhase::Terminal;
        self.failure = Some(failure.clone());
        VisibleFrameStatus::Failed(failure)
    }

    fn finish_accepted(&mut self, failure: FixedFrameFailure) -> VisibleFrameStatus {
        if let Some(mut reservation) = self.reservation.take() {
            reservation.poison();
        }
        self.graph.take();
        self.objects.take();
        self.resources.take();
        self.phase = VisiblePhase::Terminal;
        self.failure = Some(failure.clone());
        VisibleFrameStatus::Failed(failure)
    }
}

impl Drop for VisibleFrameSubmission {
    fn drop(&mut self) {
        if let Some(reservation) = self.reservation.take() {
            if matches!(self.phase, VisiblePhase::RasterAccepted(_)) {
                let mut reservation = reservation;
                reservation.poison();
            } else {
                reservation.release_before_submit();
            }
        }
    }
}

impl fmt::Debug for VisibleFrameSubmission {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VisibleFrameSubmission")
            .finish_non_exhaustive()
    }
}

struct VisibleCameraResources {
    device: DeviceIdentity,
    positions: Buffer,
    indices: Buffer,
    position_state: ResourceAccessState,
    index_state: ResourceAccessState,
    uniform: RefCell<Option<Buffer>>,
    surface: OneShotSurface<
        fluxel_rendergraph::BoundSurfaceTexture<Texture, ResourceLease, PresentationToken>,
    >,
}

/// Holds the acquired image permission until graph resolution consumes it.
///
/// A present token is linear even though `FrameResourceProvider` is shared by
/// reference, so the provider must make a second resolution fail closed rather
/// than duplicate or silently drop the image permission.
struct OneShotSurface<T>(RefCell<Option<T>>);

impl<T> OneShotSurface<T> {
    fn new(value: T) -> Self {
        Self(RefCell::new(Some(value)))
    }

    fn take(&self) -> Option<T> {
        self.0.borrow_mut().take()
    }
}

impl VisibleCameraResources {
    fn new(
        device: DeviceIdentity,
        snapshot: &IndexedMeshSnapshot,
        surface: fluxel_rendergraph::BoundSurfaceTexture<Texture, TextureLease, PresentationToken>,
    ) -> Self {
        Self {
            device,
            positions: snapshot.positions().buffer().clone(),
            indices: snapshot.indices().buffer().clone(),
            position_state: snapshot.positions().outgoing_state(),
            index_state: snapshot.indices().outgoing_state(),
            uniform: RefCell::new(None),
            surface: OneShotSurface::new(fluxel_rendergraph::BoundSurfaceTexture {
                texture: fluxel_rendergraph::BoundTexture {
                    device: surface.texture.device,
                    identity: surface.texture.identity,
                    physical: surface.texture.physical,
                    descriptor: surface.texture.descriptor,
                    usage: surface.texture.usage,
                    initial_state: surface.texture.initial_state,
                    lease: surface.texture.lease.into(),
                },
                presentation: surface.presentation,
            }),
        }
    }

    fn set_uniform(&self, uniform: Buffer) {
        *self.uniform.borrow_mut() = Some(uniform);
    }
}

impl FrameResourceProvider<RasterBackend> for VisibleCameraResources {
    fn texture(
        &self,
        _: TextureBindingId,
    ) -> Result<fluxel_rendergraph::BoundTexture<Texture, ResourceLease>, FrameBindingError> {
        Err(missing_binding(
            FrameBindingErrorKind::MissingTexture,
            "visible fixed frame has no caller-owned texture imports",
        ))
    }

    fn buffer(
        &self,
        id: BufferBindingId,
    ) -> Result<BoundBuffer<Buffer, ResourceLease>, FrameBindingError> {
        let (buffer, initial_state) = if id == position_binding() {
            (&self.positions, self.position_state)
        } else if id == index_binding() {
            (&self.indices, self.index_state)
        } else if id == uniform_binding() {
            let uniform = self.uniform.borrow();
            let buffer = uniform.as_ref().ok_or_else(|| {
                missing_binding(
                    FrameBindingErrorKind::MissingBuffer,
                    "visible frame uniform is unavailable",
                )
            })?;
            return Ok(bound_buffer(
                self.device,
                buffer,
                ResourceAccessState::CopyDestination,
            ));
        } else {
            return Err(missing_binding(
                FrameBindingErrorKind::MissingBuffer,
                "unknown visible fixed frame buffer",
            ));
        };
        Ok(bound_buffer(self.device, buffer, initial_state))
    }

    fn surface(
        &self,
        id: fluxel_rendergraph::SurfaceBindingId,
    ) -> fluxel_rendergraph::SurfaceBindingResult<RasterBackend> {
        if id != surface_binding() {
            return Err(missing_binding(
                FrameBindingErrorKind::MissingSurface,
                "unknown visible fixed frame surface",
            ));
        }
        self.surface.take().ok_or_else(|| {
            missing_binding(
                FrameBindingErrorKind::MissingSurface,
                "visible fixed frame surface was already consumed",
            )
        })
    }
}

#[cfg(test)]
#[path = "visible/tests/mod.rs"]
mod tests;

fn bound_buffer(
    device: DeviceIdentity,
    buffer: &Buffer,
    initial_state: ResourceAccessState,
) -> BoundBuffer<Buffer, ResourceLease> {
    BoundBuffer {
        device,
        identity: buffer.identity(),
        physical: buffer.clone(),
        descriptor: buffer.descriptor().buffer,
        usage: buffer.allowed_usage(),
        initial_state,
        lease: buffer.lease().into(),
    }
}
