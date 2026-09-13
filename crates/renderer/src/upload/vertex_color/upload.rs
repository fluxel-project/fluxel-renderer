//! Closed RGBA8 vertex-color upload coordination.
//!
//! This module owns the CPU-side shape and three-stream publication contract
//! for the vertex-color recipe.  It deliberately does not know how a raster
//! pass binds these buffers; that remains the fixed-frame/RHI boundary.

use core::fmt;
use std::sync::{Arc, atomic::Ordering};

use fluxel_rendergraph::{BufferUsageKind, CompletionFailure, CompletionStatus};
use fluxel_rhi::{BufferUploadError, Device, PendingBufferUpload, UploadedBuffer};

use super::super::{
    indexed::buffer_descriptor,
    shared::{
        NEXT_GENERATION, RetainedUploadState, SnapshotDrawReservation, SnapshotUseError,
        SnapshotUseGate, completion_requires_retention,
    },
};
use super::domain::{VertexColorGeometry, VertexColorGeometryStream, VertexColorMeshPayload};

/// A terminal or observed failure of a three-stream vertex-color upload.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum VertexColorIndexedMeshUploadFailure {
    /// Position submission completed with a failure.
    PositionCompletion(CompletionFailure),
    /// Color submission completed with a failure.
    ColorCompletion(CompletionFailure),
    /// Index submission completed with a failure.
    IndexCompletion(CompletionFailure),
    /// Position completion could no longer be observed.
    PositionObservation(BufferUploadError),
    /// Color completion could no longer be observed.
    ColorObservation(BufferUploadError),
    /// Index completion could no longer be observed.
    IndexObservation(BufferUploadError),
    /// Position acceptance was followed by a color-start failure.
    ColorStartAfterPositionAccepted(BufferUploadError),
    /// Position and color acceptance were followed by an index-start failure.
    IndexStartAfterPositionAndColorAccepted(BufferUploadError),
    /// Position completion was outside this closed contract.
    PositionUnknownCompletion,
    /// Color completion was outside this closed contract.
    ColorUnknownCompletion,
    /// Index completion was outside this closed contract.
    IndexUnknownCompletion,
}
/// Why no vertex-color upload work was accepted.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum VertexColorIndexedMeshUploadStartError {
    /// Position upload was rejected before native acceptance.
    PositionUpload(BufferUploadError),
}
impl fmt::Display for VertexColorIndexedMeshUploadStartError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PositionUpload(error) => {
                write!(f, "vertex-color position upload did not start: {error}")
            }
        }
    }
}
impl std::error::Error for VertexColorIndexedMeshUploadStartError {}

/// Non-blocking state of a [`VertexColorIndexedMeshUpload`].
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum VertexColorIndexedMeshUploadStatus {
    /// At least one accepted stream remains incomplete.
    Pending,
    /// All three streams completed and one immutable snapshot was published.
    Ready,
    /// This operation cannot publish a snapshot but retains accepted siblings.
    Failed(VertexColorIndexedMeshUploadFailure),
}

/// One opaque ready generation containing position, color, and index streams.
#[derive(Clone)]
pub struct VertexColorIndexedMeshSnapshot {
    generation: u64,
    position_count: u32,
    index_count: u32,
    positions: UploadedBuffer,
    colors: UploadedBuffer,
    indices: UploadedBuffer,
    position_metadata: Arc<[[f32; 3]]>,
    color_metadata: Arc<[[u8; 4]]>,
    index_metadata: Arc<[u32]>,
    use_gate: Arc<SnapshotUseGate>,
}
impl fmt::Debug for VertexColorIndexedMeshSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VertexColorIndexedMeshSnapshot")
            .field("generation", &self.generation)
            .field("position_count", &self.position_count)
            .field("index_count", &self.index_count)
            .finish()
    }
}
impl VertexColorIndexedMeshSnapshot {
    /// Returns this generation's opaque identity.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }
    /// Returns the number of positions.
    #[must_use]
    pub const fn position_count(&self) -> u32 {
        self.position_count
    }
    /// Returns the number of indices.
    #[must_use]
    pub const fn index_count(&self) -> u32 {
        self.index_count
    }
    #[must_use]
    pub(crate) fn positions(&self) -> &UploadedBuffer {
        &self.positions
    }
    #[must_use]
    pub(crate) fn colors(&self) -> &UploadedBuffer {
        &self.colors
    }
    #[must_use]
    pub(crate) fn indices(&self) -> &UploadedBuffer {
        &self.indices
    }
    #[must_use]
    pub(crate) fn position_metadata(&self) -> &[[f32; 3]] {
        &self.position_metadata
    }
    #[must_use]
    pub(crate) fn color_metadata(&self) -> &[[u8; 4]] {
        &self.color_metadata
    }
    #[must_use]
    pub(crate) fn index_metadata(&self) -> &[u32] {
        &self.index_metadata
    }
    pub(crate) fn reserve_for_draw(&self) -> Result<SnapshotDrawReservation, SnapshotUseError> {
        self.use_gate.reserve()
    }
}

enum UploadSlot {
    Absent,
    Pending(PendingBufferUpload),
    Ready(UploadedBuffer),
}

/// Coordinates three accepted immutable uploads without blocking the CPU.
pub struct VertexColorIndexedMeshUpload {
    generation: u64,
    position_count: u32,
    index_count: u32,
    position_metadata: Arc<[[f32; 3]]>,
    color_metadata: Arc<[[u8; 4]]>,
    index_metadata: Arc<[u32]>,
    positions: UploadSlot,
    colors: UploadSlot,
    indices: UploadSlot,
    failure: Option<VertexColorIndexedMeshUploadFailure>,
    snapshot: Option<VertexColorIndexedMeshSnapshot>,
}
impl VertexColorIndexedMeshUpload {
    /// Starts position, color, then index uploads; partial acceptance remains owned by this operation.
    pub fn begin(
        device: &Device,
        geometry: &VertexColorGeometry,
    ) -> Result<Self, VertexColorIndexedMeshUploadStartError> {
        let payload = VertexColorMeshPayload::from_geometry(geometry)
            .expect("VertexColorGeometry validates its closed payload ABI");
        let positions = device
            .upload_immutable_buffer(
                buffer_descriptor(payload.positions.len(), BufferUsageKind::Vertex),
                &payload.positions,
            )
            .map_err(VertexColorIndexedMeshUploadStartError::PositionUpload)?;
        let mut operation = Self {
            generation: NEXT_GENERATION.fetch_add(1, Ordering::Relaxed),
            position_count: payload.position_count,
            index_count: payload.index_count,
            position_metadata: Arc::from(geometry.geometry().positions()),
            color_metadata: Arc::from(geometry.colors()),
            index_metadata: Arc::from(geometry.geometry().indices()),
            positions: UploadSlot::Pending(positions),
            colors: UploadSlot::Absent,
            indices: UploadSlot::Absent,
            failure: None,
            snapshot: None,
        };
        match device.upload_immutable_buffer(
            buffer_descriptor(payload.colors.len(), BufferUsageKind::Vertex),
            &payload.colors,
        ) {
            Err(error) => {
                operation.failure = Some(
                    VertexColorIndexedMeshUploadFailure::ColorStartAfterPositionAccepted(error),
                )
            }
            Ok(colors) => {
                operation.colors = UploadSlot::Pending(colors);
                match device.upload_immutable_buffer(
                    buffer_descriptor(payload.indices.len(), BufferUsageKind::Index),
                    &payload.indices,
                ) {
                    Ok(indices) => operation.indices = UploadSlot::Pending(indices),
                    Err(error) => operation.failure = Some(
                        VertexColorIndexedMeshUploadFailure::IndexStartAfterPositionAndColorAccepted(error),
                    ),
                }
            }
        }
        Ok(operation)
    }
    /// Polls every accepted stream and publishes only a fully completed generation.
    pub fn poll(&mut self) -> VertexColorIndexedMeshUploadStatus {
        poll_slot(
            &mut self.positions,
            VertexColorGeometryStream::Position,
            &mut self.failure,
        );
        poll_slot(
            &mut self.indices,
            VertexColorGeometryStream::Index,
            &mut self.failure,
        );
        poll_slot(
            &mut self.colors,
            VertexColorGeometryStream::Color,
            &mut self.failure,
        );
        if let Some(failure) = &self.failure {
            return VertexColorIndexedMeshUploadStatus::Failed(failure.clone());
        }
        if self.snapshot.is_none()
            && VertexColorSnapshotPublication::new(
                slot_state(&self.positions),
                slot_state(&self.colors),
                slot_state(&self.indices),
                false,
            )
            .can_publish()
        {
            self.snapshot = Some(VertexColorIndexedMeshSnapshot {
                generation: self.generation,
                position_count: self.position_count,
                index_count: self.index_count,
                positions: take_ready(&mut self.positions),
                colors: take_ready(&mut self.colors),
                indices: take_ready(&mut self.indices),
                position_metadata: Arc::clone(&self.position_metadata),
                color_metadata: Arc::clone(&self.color_metadata),
                index_metadata: Arc::clone(&self.index_metadata),
                use_gate: Arc::new(SnapshotUseGate::new()),
            });
        }
        if self.snapshot.is_some() {
            VertexColorIndexedMeshUploadStatus::Ready
        } else {
            VertexColorIndexedMeshUploadStatus::Pending
        }
    }
    /// Returns the completed immutable snapshot, if all streams are ready.
    #[must_use]
    pub fn ready_snapshot(&self) -> Option<VertexColorIndexedMeshSnapshot> {
        self.snapshot.clone()
    }
}
impl fmt::Debug for VertexColorIndexedMeshUpload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VertexColorIndexedMeshUpload")
            .field("generation", &self.generation)
            .field("position_count", &self.position_count)
            .field("index_count", &self.index_count)
            .field("failure", &self.failure)
            .field("ready", &self.snapshot.is_some())
            .finish_non_exhaustive()
    }
}

/// State-only publication policy for the three immutable streams.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::upload) struct VertexColorSnapshotPublication {
    positions: RetainedUploadState,
    colors: RetainedUploadState,
    indices: RetainedUploadState,
    failed: bool,
}
impl VertexColorSnapshotPublication {
    pub(in crate::upload) const fn new(
        positions: RetainedUploadState,
        colors: RetainedUploadState,
        indices: RetainedUploadState,
        failed: bool,
    ) -> Self {
        Self {
            positions,
            colors,
            indices,
            failed,
        }
    }
    pub(in crate::upload) const fn can_publish(self) -> bool {
        !self.failed
            && matches!(self.positions, RetainedUploadState::Ready)
            && matches!(self.colors, RetainedUploadState::Ready)
            && matches!(self.indices, RetainedUploadState::Ready)
    }
}

fn slot_state(slot: &UploadSlot) -> RetainedUploadState {
    match slot {
        UploadSlot::Absent => RetainedUploadState::Missing,
        UploadSlot::Pending(_) => RetainedUploadState::Pending,
        UploadSlot::Ready(_) => RetainedUploadState::Ready,
    }
}
fn take_ready(slot: &mut UploadSlot) -> UploadedBuffer {
    match core::mem::replace(slot, UploadSlot::Absent) {
        UploadSlot::Ready(buffer) => buffer,
        _ => unreachable!("publication requires a ready vertex-color upload"),
    }
}
fn poll_slot(
    slot: &mut UploadSlot,
    stream: VertexColorGeometryStream,
    failure: &mut Option<VertexColorIndexedMeshUploadFailure>,
) {
    let UploadSlot::Pending(upload) = slot else {
        return;
    };
    match upload.status() {
        Ok(status) if completion_requires_retention(status) => {}
        Ok(CompletionStatus::Complete) => {
            let UploadSlot::Pending(upload) = core::mem::replace(slot, UploadSlot::Absent) else {
                unreachable!()
            };
            match upload.finalize() {
                Ok(buffer) => *slot = UploadSlot::Ready(buffer),
                Err(incomplete) => {
                    let status = incomplete.status();
                    *slot = UploadSlot::Pending(incomplete.into_pending());
                    record_completion(status, stream, failure);
                }
            }
        }
        Ok(status) => record_completion(status, stream, failure),
        Err(error) => {
            failure.get_or_insert(observation_failure(stream, error));
        }
    }
}
fn record_completion(
    status: CompletionStatus,
    stream: VertexColorGeometryStream,
    failure: &mut Option<VertexColorIndexedMeshUploadFailure>,
) {
    match status {
        CompletionStatus::Pending | CompletionStatus::Unknown | CompletionStatus::Complete => {}
        CompletionStatus::Failed(reason) => {
            failure.get_or_insert(completion_failure(stream, reason));
        }
        _ => {
            failure.get_or_insert(unknown_completion_failure(stream));
        }
    }
}
pub(crate) fn completion_failure(
    stream: VertexColorGeometryStream,
    reason: CompletionFailure,
) -> VertexColorIndexedMeshUploadFailure {
    match stream {
        VertexColorGeometryStream::Position => {
            VertexColorIndexedMeshUploadFailure::PositionCompletion(reason)
        }
        VertexColorGeometryStream::Color => {
            VertexColorIndexedMeshUploadFailure::ColorCompletion(reason)
        }
        VertexColorGeometryStream::Index => {
            VertexColorIndexedMeshUploadFailure::IndexCompletion(reason)
        }
    }
}
pub(crate) fn observation_failure(
    stream: VertexColorGeometryStream,
    error: BufferUploadError,
) -> VertexColorIndexedMeshUploadFailure {
    match stream {
        VertexColorGeometryStream::Position => {
            VertexColorIndexedMeshUploadFailure::PositionObservation(error)
        }
        VertexColorGeometryStream::Color => {
            VertexColorIndexedMeshUploadFailure::ColorObservation(error)
        }
        VertexColorGeometryStream::Index => {
            VertexColorIndexedMeshUploadFailure::IndexObservation(error)
        }
    }
}
pub(crate) fn unknown_completion_failure(
    stream: VertexColorGeometryStream,
) -> VertexColorIndexedMeshUploadFailure {
    match stream {
        VertexColorGeometryStream::Position => {
            VertexColorIndexedMeshUploadFailure::PositionUnknownCompletion
        }
        VertexColorGeometryStream::Color => {
            VertexColorIndexedMeshUploadFailure::ColorUnknownCompletion
        }
        VertexColorGeometryStream::Index => {
            VertexColorIndexedMeshUploadFailure::IndexUnknownCompletion
        }
    }
}
