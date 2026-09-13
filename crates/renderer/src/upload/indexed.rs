//! Immutable indexed-mesh upload and snapshot types.

use core::fmt;
use std::sync::{Arc, atomic::Ordering};

use fluxel_rendergraph::{
    BufferDesc, BufferUsage, BufferUsageKind, CompletionFailure, CompletionStatus,
};
use fluxel_rhi::{
    BufferDescriptor, BufferUploadError, Device, MemoryPolicy, PendingBufferUpload, UploadedBuffer,
};

use crate::Geometry;

use super::shared::{
    NEXT_GENERATION, RetainedUploadState, SnapshotDrawReservation, SnapshotUseError,
    SnapshotUseGate, completion_requires_retention,
};

/// An immutable, GPU-ready indexed-mesh generation.
///
/// This type deliberately exposes no native buffers, raw bytes, or resource
/// states to applications. The renderer retains those facts for later graph
/// lowering. Cloning the snapshot retains its complete native generation.
#[derive(Clone)]
pub struct IndexedMeshSnapshot {
    generation: u64,
    position_count: u32,
    index_count: u32,
    positions: UploadedBuffer,
    indices: UploadedBuffer,
    #[allow(
        dead_code,
        reason = "the next fixed renderer consumes CPU clip metadata"
    )]
    position_metadata: Arc<[[f32; 3]]>,
    #[allow(
        dead_code,
        reason = "the next fixed renderer consumes CPU clip metadata"
    )]
    index_metadata: Arc<[u32]>,
    use_gate: Arc<SnapshotUseGate>,
}

/// Narrow debug view which keeps renderer-internal GPU resources, CPU copies,
/// and synchronization state outside the public observation surface.
pub(super) struct IndexedMeshSnapshotDebug {
    pub(super) generation: u64,
    pub(super) position_count: u32,
    pub(super) index_count: u32,
}

impl fmt::Debug for IndexedMeshSnapshotDebug {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IndexedMeshSnapshot")
            .field("generation", &self.generation)
            .field("position_count", &self.position_count)
            .field("index_count", &self.index_count)
            .finish()
    }
}

impl fmt::Debug for IndexedMeshSnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        IndexedMeshSnapshotDebug {
            generation: self.generation,
            position_count: self.position_count,
            index_count: self.index_count,
        }
        .fmt(formatter)
    }
}

impl IndexedMeshSnapshot {
    /// Returns the opaque identity of this immutable generation.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Returns the number of `f32x3` vertex positions in this generation.
    #[must_use]
    pub const fn position_count(&self) -> u32 {
        self.position_count
    }

    /// Returns the number of `u32` indices in this generation.
    #[must_use]
    pub const fn index_count(&self) -> u32 {
        self.index_count
    }

    /// Returns the uploaded position resource for renderer-internal lowering.
    #[must_use]
    #[allow(
        dead_code,
        reason = "the next renderer-lowering slice consumes this private ready resource"
    )]
    pub(crate) fn positions(&self) -> &UploadedBuffer {
        &self.positions
    }

    /// Returns the uploaded index resource for renderer-internal lowering.
    #[must_use]
    #[allow(
        dead_code,
        reason = "the next renderer-lowering slice consumes this private ready resource"
    )]
    pub(crate) fn indices(&self) -> &UploadedBuffer {
        &self.indices
    }

    /// Returns the original positions for renderer-internal CPU-side lowering.
    #[must_use]
    #[allow(
        dead_code,
        reason = "the next fixed renderer consumes CPU clip metadata"
    )]
    pub(crate) fn position_metadata(&self) -> &[[f32; 3]] {
        &self.position_metadata
    }

    /// Returns the original indices for renderer-internal CPU-side lowering.
    #[must_use]
    #[allow(
        dead_code,
        reason = "the next fixed renderer consumes CPU clip metadata"
    )]
    pub(crate) fn index_metadata(&self) -> &[u32] {
        &self.index_metadata
    }

    /// Reserves this immutable generation for one renderer submission.
    ///
    /// A clone is another owner of the same native buffers, not another
    /// independently state-tracked generation.  The fixed renderer therefore
    /// serializes use until it has observed a terminal completion.
    pub(crate) fn reserve_for_draw(&self) -> Result<SnapshotDrawReservation, SnapshotUseError> {
        self.use_gate.reserve()
    }
}

/// A terminal or observed failure of one indexed-mesh upload generation.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum IndexedMeshUploadFailure {
    /// The position-buffer submission reached a terminal GPU failure.
    Positions(CompletionFailure),
    /// The index-buffer submission reached a terminal GPU failure.
    Indices(CompletionFailure),
    /// A position submission could no longer be observed safely.
    PositionCompletion(BufferUploadError),
    /// An index submission could no longer be observed safely.
    IndexCompletion(BufferUploadError),
    /// Positions were accepted, but the index submission could not be started.
    ///
    /// The returned upload still owns and retires the already accepted position
    /// submission. This failure is therefore intentionally not a start error.
    IndexStartAfterPositionsAccepted(BufferUploadError),
}

/// Why an indexed-mesh upload could not be started before any GPU work was accepted.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum IndexedMeshUploadStartError {
    /// Indexed snapshots require at least one vertex position.
    EmptyPositions,
    /// Indexed snapshots require at least one index.
    NonIndexedGeometry,
    /// The vertex count cannot be represented by this slice's public metadata.
    PositionCountTooLarge,
    /// The index count cannot be represented by this slice's public metadata.
    IndexCountTooLarge,
    /// The position upload was rejected before the native queue accepted work.
    PositionUpload(BufferUploadError),
}

impl fmt::Display for IndexedMeshUploadStartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPositions => formatter.write_str("indexed mesh positions are empty"),
            Self::NonIndexedGeometry => formatter.write_str("indexed mesh geometry has no indices"),
            Self::PositionCountTooLarge => {
                formatter.write_str("indexed mesh position count exceeds u32")
            }
            Self::IndexCountTooLarge => formatter.write_str("indexed mesh index count exceeds u32"),
            Self::PositionUpload(error) => {
                write!(formatter, "position upload did not start: {error}")
            }
        }
    }
}

impl std::error::Error for IndexedMeshUploadStartError {}

/// The non-blocking observable state of an [`IndexedMeshUpload`].
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum IndexedMeshUploadStatus {
    /// At least one accepted upload is still incomplete.
    Pending,
    /// Both accepted uploads completed and an immutable snapshot is available.
    Ready,
    /// This generation cannot become ready.
    ///
    /// The upload object may still retain a sibling accepted submission until
    /// that submission becomes terminal. Calling [`IndexedMeshUpload::poll`]
    /// remains useful and never turns this status back into `Ready`.
    Failed(IndexedMeshUploadFailure),
}

/// One opt-in, non-blocking indexed-mesh upload operation.
///
/// The operation is intentionally not cloneable: it is the unique coordinator
/// that advances both accepted submissions. It never waits for the CPU; call
/// [`Self::poll`] from an application's normal progress loop.
pub struct IndexedMeshUpload {
    generation: u64,
    position_count: u32,
    index_count: u32,
    position_metadata: Arc<[[f32; 3]]>,
    index_metadata: Arc<[u32]>,
    positions: UploadSlot,
    indices: UploadSlot,
    failure: Option<IndexedMeshUploadFailure>,
    snapshot: Option<IndexedMeshSnapshot>,
}

enum UploadSlot {
    Absent,
    Pending(PendingBufferUpload),
    Ready(UploadedBuffer),
}

impl IndexedMeshUpload {
    /// Starts the two immutable uploads required for `geometry`.
    ///
    /// Position and index payloads are serialized exactly as tightly packed
    /// little-endian IEEE-754 `f32x3` and little-endian `u32` values. If the
    /// position request fails, no queue work has been accepted and this returns
    /// an error. Once positions have been accepted, every outcome returns an
    /// owning operation, including an index-start failure, so accepted storage
    /// remains alive until it is safe to retire.
    pub fn begin(
        device: &Device,
        geometry: &Geometry,
    ) -> Result<Self, IndexedMeshUploadStartError> {
        let payload = IndexedMeshPayload::from_geometry(geometry)?;
        let position_metadata: Arc<[[f32; 3]]> = Arc::from(geometry.positions());
        let index_metadata: Arc<[u32]> = Arc::from(geometry.indices());
        let generation = NEXT_GENERATION.fetch_add(1, Ordering::Relaxed);
        let positions = device
            .upload_immutable_buffer(
                buffer_descriptor(payload.positions.len(), BufferUsageKind::Vertex),
                &payload.positions,
            )
            .map_err(IndexedMeshUploadStartError::PositionUpload)?;

        match device.upload_immutable_buffer(
            buffer_descriptor(payload.indices.len(), BufferUsageKind::Index),
            &payload.indices,
        ) {
            Ok(indices) => Ok(Self {
                generation,
                position_count: payload.position_count,
                index_count: payload.index_count,
                position_metadata,
                index_metadata,
                positions: UploadSlot::Pending(positions),
                indices: UploadSlot::Pending(indices),
                failure: None,
                snapshot: None,
            }),
            Err(error) => Ok(Self {
                generation,
                position_count: payload.position_count,
                index_count: payload.index_count,
                position_metadata,
                index_metadata,
                positions: UploadSlot::Pending(positions),
                indices: UploadSlot::Absent,
                failure: Some(IndexedMeshUploadFailure::IndexStartAfterPositionsAccepted(
                    error,
                )),
                snapshot: None,
            }),
        }
    }

    /// Polls both accepted uploads without blocking the calling thread.
    ///
    /// A terminal failure is monotonic. Even after reporting it, this method
    /// continues polling any accepted sibling so its keepalive storage can be
    /// retained through a terminal completion.
    pub fn poll(&mut self) -> IndexedMeshUploadStatus {
        poll_slot(&mut self.positions, true, &mut self.failure);
        poll_slot(&mut self.indices, false, &mut self.failure);

        if let Some(failure) = &self.failure {
            return IndexedMeshUploadStatus::Failed(failure.clone());
        }
        let publication = SnapshotPublication::new(
            upload_slot_state(&self.positions),
            upload_slot_state(&self.indices),
            self.failure.is_some(),
        );
        if self.snapshot.is_none() && publication.can_publish() {
            let positions = take_ready(&mut self.positions);
            let indices = take_ready(&mut self.indices);
            self.snapshot = Some(IndexedMeshSnapshot {
                generation: self.generation,
                position_count: self.position_count,
                index_count: self.index_count,
                positions,
                indices,
                position_metadata: Arc::clone(&self.position_metadata),
                index_metadata: Arc::clone(&self.index_metadata),
                use_gate: Arc::new(SnapshotUseGate::new()),
            });
        }
        if self.snapshot.is_some() {
            IndexedMeshUploadStatus::Ready
        } else {
            IndexedMeshUploadStatus::Pending
        }
    }

    /// Returns a strong, immutable ready snapshot after [`Self::poll`] reports `Ready`.
    #[must_use]
    pub fn ready_snapshot(&self) -> Option<IndexedMeshSnapshot> {
        self.snapshot.clone()
    }

    /// Reports whether a failed coordinator still owns accepted work whose
    /// completion is pending, unknown, or temporarily unobservable.
    #[cfg(feature = "gpu-residency")]
    pub(crate) fn retirement_pending(&self) -> bool {
        [&self.positions, &self.indices]
            .into_iter()
            .any(upload_slot_requires_retention)
    }
}

/// Pure publication policy kept separate from the RHI completion adapter.
///
/// The small state-only coordinator makes the atomic renderer contract testable
/// without a native device: a partial pair never publishes, and an observed
/// failure remains terminal even while another accepted upload is retained.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SnapshotPublication {
    pub(super) positions: RetainedUploadState,
    pub(super) indices: RetainedUploadState,
    pub(super) failed: bool,
}

impl SnapshotPublication {
    pub(super) const fn new(
        positions: RetainedUploadState,
        indices: RetainedUploadState,
        failed: bool,
    ) -> Self {
        Self {
            positions,
            indices,
            failed,
        }
    }

    pub(super) const fn can_publish(self) -> bool {
        !self.failed
            && matches!(self.positions, RetainedUploadState::Ready)
            && matches!(self.indices, RetainedUploadState::Ready)
    }
}

impl fmt::Debug for IndexedMeshUpload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IndexedMeshUpload")
            .field("generation", &self.generation)
            .field("position_count", &self.position_count)
            .field("index_count", &self.index_count)
            .field("failure", &self.failure)
            .field("ready", &self.snapshot.is_some())
            .finish_non_exhaustive()
    }
}

pub(super) fn buffer_descriptor(
    byte_len: usize,
    terminal_usage: BufferUsageKind,
) -> BufferDescriptor {
    let usage = BufferUsage::from_kinds([BufferUsageKind::CopyDestination, terminal_usage]);
    #[cfg(test)]
    let usage = usage.with(BufferUsageKind::CopySource);
    BufferDescriptor {
        buffer: BufferDesc {
            size: u64::try_from(byte_len).expect("validated renderer payload length fits u64"),
        },
        // The conformance-only CopySource addition permits the doc-hidden
        // readback oracle. Production snapshots authorize only their upload
        // and future vertex/index consumption states.
        usage,
        memory: MemoryPolicy::DeviceOnly,
    }
}

fn poll_slot(
    slot: &mut UploadSlot,
    positions: bool,
    failure: &mut Option<IndexedMeshUploadFailure>,
) {
    let UploadSlot::Pending(upload) = slot else {
        return;
    };
    match upload.status() {
        Ok(status) if completion_requires_retention(status) => {}
        Ok(CompletionStatus::Complete) => {
            let UploadSlot::Pending(upload) = core::mem::replace(slot, UploadSlot::Absent) else {
                unreachable!("slot was pending while completing it");
            };
            match upload.finalize() {
                Ok(buffer) => *slot = UploadSlot::Ready(buffer),
                Err(incomplete) => {
                    let observed = incomplete.status();
                    *slot = UploadSlot::Pending(incomplete.into_pending());
                    record_status_failure(observed, positions, failure);
                }
            }
        }
        Ok(status) => record_status_failure(status, positions, failure),
        Err(error) => {
            failure.get_or_insert_with(|| {
                if positions {
                    IndexedMeshUploadFailure::PositionCompletion(error)
                } else {
                    IndexedMeshUploadFailure::IndexCompletion(error)
                }
            });
        }
    }
}

fn record_status_failure(
    status: CompletionStatus,
    positions: bool,
    failure: &mut Option<IndexedMeshUploadFailure>,
) {
    if let CompletionStatus::Failed(reason) = status {
        failure.get_or_insert(if positions {
            IndexedMeshUploadFailure::Positions(reason)
        } else {
            IndexedMeshUploadFailure::Indices(reason)
        });
    }
}

fn take_ready(slot: &mut UploadSlot) -> UploadedBuffer {
    match core::mem::replace(slot, UploadSlot::Absent) {
        UploadSlot::Ready(buffer) => buffer,
        _ => unreachable!("snapshot construction requires ready uploads"),
    }
}

fn upload_slot_state(slot: &UploadSlot) -> RetainedUploadState {
    match slot {
        UploadSlot::Absent => RetainedUploadState::Missing,
        UploadSlot::Pending(_) => RetainedUploadState::Pending,
        UploadSlot::Ready(_) => RetainedUploadState::Ready,
    }
}

#[cfg(feature = "gpu-residency")]
fn upload_slot_requires_retention(slot: &UploadSlot) -> bool {
    let UploadSlot::Pending(upload) = slot else {
        return false;
    };
    match upload.status() {
        Ok(CompletionStatus::Complete | CompletionStatus::Failed(_)) => false,
        Ok(_) | Err(_) => true,
    }
}

#[derive(Debug)]
pub(super) struct IndexedMeshPayload {
    pub(super) positions: Vec<u8>,
    pub(super) indices: Vec<u8>,
    pub(super) position_count: u32,
    pub(super) index_count: u32,
}

impl IndexedMeshPayload {
    pub(super) fn from_geometry(geometry: &Geometry) -> Result<Self, IndexedMeshUploadStartError> {
        if geometry.positions().is_empty() {
            return Err(IndexedMeshUploadStartError::EmptyPositions);
        }
        if !geometry.is_indexed() {
            return Err(IndexedMeshUploadStartError::NonIndexedGeometry);
        }
        let position_count = u32::try_from(geometry.positions().len())
            .map_err(|_| IndexedMeshUploadStartError::PositionCountTooLarge)?;
        let index_count = u32::try_from(geometry.indices().len())
            .map_err(|_| IndexedMeshUploadStartError::IndexCountTooLarge)?;
        let position_bytes = geometry
            .positions()
            .len()
            .checked_mul(12)
            .ok_or(IndexedMeshUploadStartError::PositionCountTooLarge)?;
        let index_bytes = geometry
            .indices()
            .len()
            .checked_mul(4)
            .ok_or(IndexedMeshUploadStartError::IndexCountTooLarge)?;
        let mut positions = Vec::with_capacity(position_bytes);
        for position in geometry.positions() {
            for component in position {
                positions.extend_from_slice(&component.to_bits().to_le_bytes());
            }
        }
        let mut indices = Vec::with_capacity(index_bytes);
        for index in geometry.indices() {
            indices.extend_from_slice(&index.to_le_bytes());
        }
        Ok(Self {
            positions,
            indices,
            position_count,
            index_count,
        })
    }
}
