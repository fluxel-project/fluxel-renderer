//! Immutable position, index, and canonical-normal mesh upload types.

use core::fmt;
use std::sync::{Arc, atomic::Ordering};

use fluxel_rendergraph::{BufferUsageKind, CompletionFailure, CompletionStatus};
use fluxel_rhi::{BufferUploadError, Device, PendingBufferUpload, UploadedBuffer};

use crate::Geometry;

use super::indexed::buffer_descriptor;
use super::shared::{
    NEXT_GENERATION, RetainedUploadState, SnapshotDrawReservation, SnapshotUseError,
    SnapshotUseGate, completion_requires_retention,
};

/// The logical stream represented by the closed normal-geometry upload ABI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum NormalGeometryStream {
    /// Tightly packed `f32x3` positions.
    Position,
    /// Tightly packed `u32` indices.
    Index,
    /// Tightly packed canonical unit `f32x3` normals.
    Normal,
}

/// Why [`NormalGeometry`] cannot be constructed.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum NormalGeometryError {
    /// There are no vertex positions.
    EmptyPositions,
    /// The geometry has no index stream.
    MissingIndices,
    /// The index stream cannot describe a whole number of triangles.
    NonTriangleIndexCount {
        /// Number of supplied indices.
        index_count: usize,
    },
    /// The normal and position streams do not have one entry per vertex.
    NormalCountMismatch {
        /// Number of positions.
        positions: usize,
        /// Number of supplied normals.
        normals: usize,
    },
    /// A normal is non-finite, zero, or cannot be represented as a finite unit vector.
    NonNormalizableNormal {
        /// Vertex containing the normal.
        vertex: usize,
    },
    /// A stream count does not fit the closed RHI ABI.
    CountOutOfRange {
        /// Stream whose count failed conversion.
        stream: NormalGeometryStream,
        /// Original count.
        count: usize,
    },
    /// A stream byte length cannot be represented exactly.
    ByteLengthOverflow {
        /// Stream whose payload length overflowed.
        stream: NormalGeometryStream,
    },
}

impl fmt::Display for NormalGeometryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPositions => f.write_str("normal geometry positions are empty"),
            Self::MissingIndices => f.write_str("normal geometry has no indices"),
            Self::NonTriangleIndexCount { index_count } => write!(
                f,
                "normal geometry has {index_count} indices, not a whole number of triangles"
            ),
            Self::NormalCountMismatch { positions, normals } => write!(
                f,
                "normal geometry has {normals} normals for {positions} positions"
            ),
            Self::NonNormalizableNormal { vertex } => {
                write!(
                    f,
                    "normal geometry normal at vertex {vertex} is not normalizable"
                )
            }
            Self::CountOutOfRange { stream, count } => write!(
                f,
                "normal geometry {stream:?} count {count} exceeds the closed upload ABI"
            ),
            Self::ByteLengthOverflow { stream } => {
                write!(f, "normal geometry {stream:?} payload length overflows")
            }
        }
    }
}

impl std::error::Error for NormalGeometryError {}

/// Indexed geometry with one canonical finite unit `f32x3` normal per position.
///
/// Construction owns the only normalization step for this slice. Consequently,
/// public normal metadata and the immutable GPU payload observe identical
/// canonical `f32` bytes, including `+0.0` for every zero component.
#[derive(Clone, Debug, PartialEq)]
pub struct NormalGeometry {
    geometry: Geometry,
    normals: Vec<[f32; 3]>,
}

impl NormalGeometry {
    /// Creates closed normal geometry after validating and canonicalizing all streams.
    pub fn new(
        geometry: Geometry,
        normals: impl Into<Vec<[f32; 3]>>,
    ) -> Result<Self, NormalGeometryError> {
        let normals = normals.into();
        let positions = geometry.positions();
        if positions.is_empty() {
            return Err(NormalGeometryError::EmptyPositions);
        }
        if !geometry.is_indexed() {
            return Err(NormalGeometryError::MissingIndices);
        }
        if !geometry.indices().len().is_multiple_of(3) {
            return Err(NormalGeometryError::NonTriangleIndexCount {
                index_count: geometry.indices().len(),
            });
        }
        if normals.len() != positions.len() {
            return Err(NormalGeometryError::NormalCountMismatch {
                positions: positions.len(),
                normals: normals.len(),
            });
        }
        let normals = normals
            .into_iter()
            .enumerate()
            .map(|(vertex, normal)| {
                canonicalize_normal(normal)
                    .ok_or(NormalGeometryError::NonNormalizableNormal { vertex })
            })
            .collect::<Result<Vec<_>, _>>()?;
        NormalMeshPayload::validate_lengths(&geometry, &normals)?;
        Ok(Self { geometry, normals })
    }

    /// Returns the owned indexed geometry.
    #[must_use]
    pub const fn geometry(&self) -> &Geometry {
        &self.geometry
    }

    /// Returns the canonical finite unit normal for every vertex position.
    #[must_use]
    pub fn normals(&self) -> &[[f32; 3]] {
        &self.normals
    }
}

/// Normalizes one input with the slice's fixed f64 arithmetic recipe.
///
/// The named additions intentionally freeze the dot-product evaluation order:
/// `x² + y²`, followed by `+ z²`.  The final cast is then canonicalized so a
/// mathematical zero never leaks an input's negative-zero bit into metadata or
/// the native vertex payload.
pub(super) fn canonicalize_normal(normal: [f32; 3]) -> Option<[f32; 3]> {
    let x = f64::from(normal[0]);
    let y = f64::from(normal[1]);
    let z = f64::from(normal[2]);
    if !(x.is_finite() && y.is_finite() && z.is_finite()) {
        return None;
    }
    let scale = x.abs().max(y.abs()).max(z.abs());
    if !scale.is_finite() || scale == 0.0 {
        return None;
    }
    let scaled_x = x / scale;
    let scaled_y = y / scale;
    let scaled_z = z / scale;
    if !(scaled_x.is_finite() && scaled_y.is_finite() && scaled_z.is_finite()) {
        return None;
    }
    let x_squared = scaled_x * scaled_x;
    let y_squared = scaled_y * scaled_y;
    let z_squared = scaled_z * scaled_z;
    let xy_sum = x_squared + y_squared;
    let length_squared = xy_sum + z_squared;
    if !length_squared.is_finite() {
        return None;
    }
    let length = length_squared.sqrt();
    if !length.is_finite() || length == 0.0 {
        return None;
    }
    let unit_x = (scaled_x / length) as f32;
    let unit_y = (scaled_y / length) as f32;
    let unit_z = (scaled_z / length) as f32;
    if !(unit_x.is_finite() && unit_y.is_finite() && unit_z.is_finite()) {
        return None;
    }
    Some([
        if unit_x == 0.0 { 0.0 } else { unit_x },
        if unit_y == 0.0 { 0.0 } else { unit_y },
        if unit_z == 0.0 { 0.0 } else { unit_z },
    ])
}

/// A failure after at least one normal-geometry upload was accepted.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum NormalIndexedMeshUploadFailure {
    /// Position submission completed with a failure.
    PositionCompletion(CompletionFailure),
    /// Index submission completed with a failure.
    IndexCompletion(CompletionFailure),
    /// Normal submission completed with a failure.
    NormalCompletion(CompletionFailure),
    /// Position completion could no longer be observed.
    PositionObservation(BufferUploadError),
    /// Index completion could no longer be observed.
    IndexObservation(BufferUploadError),
    /// Normal completion could no longer be observed.
    NormalObservation(BufferUploadError),
    /// Position acceptance was followed by an index start failure.
    IndexStartAfterPositionAccepted(BufferUploadError),
    /// Position and index acceptance were followed by a normal start failure.
    NormalStartAfterPositionAndIndexAccepted(BufferUploadError),
    /// Position completion was outside this closed contract.
    PositionUnknownCompletion,
    /// Index completion was outside this closed contract.
    IndexUnknownCompletion,
    /// Normal completion was outside this closed contract.
    NormalUnknownCompletion,
}

/// Why no normal-geometry upload was accepted.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum NormalIndexedMeshUploadStartError {
    /// Position upload was rejected before native acceptance.
    PositionUpload(BufferUploadError),
}

impl fmt::Display for NormalIndexedMeshUploadStartError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PositionUpload(error) => {
                write!(f, "normal position upload did not start: {error}")
            }
        }
    }
}

impl std::error::Error for NormalIndexedMeshUploadStartError {}

/// Non-blocking state of a [`NormalIndexedMeshUpload`].
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum NormalIndexedMeshUploadStatus {
    /// At least one accepted stream remains incomplete.
    Pending,
    /// All three streams completed and a single immutable snapshot was published.
    Ready,
    /// This operation cannot publish a snapshot, but retains accepted siblings.
    Failed(NormalIndexedMeshUploadFailure),
}

/// One opaque ready generation containing position, index, and normal streams.
///
/// Clones share the same generation and use gate; cloning does not create an
/// independently reservable GPU resource set.
#[derive(Clone)]
pub struct NormalIndexedMeshSnapshot {
    generation: u64,
    position_count: u32,
    index_count: u32,
    positions: UploadedBuffer,
    indices: UploadedBuffer,
    normals: UploadedBuffer,
    position_metadata: Arc<[[f32; 3]]>,
    index_metadata: Arc<[u32]>,
    normal_metadata: Arc<[[f32; 3]]>,
    use_gate: Arc<SnapshotUseGate>,
}

pub(super) struct NormalIndexedMeshSnapshotDebugProbe {
    pub(super) generation: u64,
    pub(super) position_count: u32,
    pub(super) index_count: u32,
}

impl fmt::Debug for NormalIndexedMeshSnapshotDebugProbe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NormalIndexedMeshSnapshot")
            .field("generation", &self.generation)
            .field("position_count", &self.position_count)
            .field("index_count", &self.index_count)
            .finish()
    }
}

impl fmt::Debug for NormalIndexedMeshSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        NormalIndexedMeshSnapshotDebugProbe {
            generation: self.generation,
            position_count: self.position_count,
            index_count: self.index_count,
        }
        .fmt(f)
    }
}

impl NormalIndexedMeshSnapshot {
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
    pub(crate) fn indices(&self) -> &UploadedBuffer {
        &self.indices
    }

    #[must_use]
    pub(crate) fn normals(&self) -> &UploadedBuffer {
        &self.normals
    }

    #[must_use]
    pub(crate) fn position_metadata(&self) -> &[[f32; 3]] {
        &self.position_metadata
    }

    #[must_use]
    pub(crate) fn index_metadata(&self) -> &[u32] {
        &self.index_metadata
    }

    #[must_use]
    pub(crate) fn normal_metadata(&self) -> &[[f32; 3]] {
        &self.normal_metadata
    }

    pub(crate) fn reserve_for_draw(&self) -> Result<SnapshotDrawReservation, SnapshotUseError> {
        self.use_gate.reserve()
    }
}

enum NormalUploadSlot {
    Absent,
    Pending(PendingBufferUpload),
    Ready(UploadedBuffer),
}

/// Coordinates three accepted immutable normal-mesh uploads without blocking.
pub struct NormalIndexedMeshUpload {
    generation: u64,
    position_count: u32,
    index_count: u32,
    position_metadata: Arc<[[f32; 3]]>,
    index_metadata: Arc<[u32]>,
    normal_metadata: Arc<[[f32; 3]]>,
    positions: NormalUploadSlot,
    indices: NormalUploadSlot,
    normals: NormalUploadSlot,
    failure: Option<NormalIndexedMeshUploadFailure>,
    snapshot: Option<NormalIndexedMeshSnapshot>,
}

impl NormalIndexedMeshUpload {
    /// Atomically prevalidates and starts position, index, and normal uploads in order.
    ///
    /// Native starts are sequential. If a later stream is rejected after an
    /// earlier one was accepted, the returned operation owns that partial work
    /// and must keep polling every accepted sibling to a terminal state.
    pub fn begin(
        device: &Device,
        geometry: &NormalGeometry,
    ) -> Result<Self, NormalIndexedMeshUploadStartError> {
        let payload = NormalMeshPayload::from_geometry(geometry)
            .expect("NormalGeometry validates its closed payload ABI");
        let positions = device
            .upload_immutable_buffer(
                buffer_descriptor(payload.positions.len(), BufferUsageKind::Vertex),
                &payload.positions,
            )
            .map_err(normal_position_start_error)?;
        let base = Self {
            generation: NEXT_GENERATION.fetch_add(1, Ordering::Relaxed),
            position_count: payload.position_count,
            index_count: payload.index_count,
            position_metadata: Arc::from(geometry.geometry.positions()),
            index_metadata: Arc::from(geometry.geometry.indices()),
            normal_metadata: Arc::from(geometry.normals()),
            positions: NormalUploadSlot::Pending(positions),
            indices: NormalUploadSlot::Absent,
            normals: NormalUploadSlot::Absent,
            failure: None,
            snapshot: None,
        };
        let indices = device.upload_immutable_buffer(
            buffer_descriptor(payload.indices.len(), BufferUsageKind::Index),
            &payload.indices,
        );
        let mut operation = base;
        match indices {
            Err(error) => {
                operation.failure = Some(normal_start_failure(NormalGeometryStream::Index, error));
                Ok(operation)
            }
            Ok(indices) => {
                operation.indices = NormalUploadSlot::Pending(indices);
                match device.upload_immutable_buffer(
                    buffer_descriptor(payload.normals.len(), BufferUsageKind::Vertex),
                    &payload.normals,
                ) {
                    Ok(normals) => {
                        operation.normals = NormalUploadSlot::Pending(normals);
                        Ok(operation)
                    }
                    Err(error) => {
                        operation.failure =
                            Some(normal_start_failure(NormalGeometryStream::Normal, error));
                        Ok(operation)
                    }
                }
            }
        }
    }

    /// Polls every accepted sibling and publishes only after all three complete.
    /// A recorded failure is monotonic, but it never permits already-accepted
    /// sibling uploads or their resources to be abandoned early.
    pub fn poll(&mut self) -> NormalIndexedMeshUploadStatus {
        poll_normal_slot(
            &mut self.positions,
            NormalGeometryStream::Position,
            &mut self.failure,
        );
        poll_normal_slot(
            &mut self.indices,
            NormalGeometryStream::Index,
            &mut self.failure,
        );
        poll_normal_slot(
            &mut self.normals,
            NormalGeometryStream::Normal,
            &mut self.failure,
        );
        if let Some(failure) = &self.failure {
            return NormalIndexedMeshUploadStatus::Failed(failure.clone());
        }
        if self.snapshot.is_none()
            && NormalSnapshotPublication::new(
                normal_upload_slot_state(&self.positions),
                normal_upload_slot_state(&self.indices),
                normal_upload_slot_state(&self.normals),
                self.failure.is_some(),
            )
            .can_publish()
        {
            self.snapshot = Some(NormalIndexedMeshSnapshot {
                generation: self.generation,
                position_count: self.position_count,
                index_count: self.index_count,
                positions: take_normal_ready(&mut self.positions),
                indices: take_normal_ready(&mut self.indices),
                normals: take_normal_ready(&mut self.normals),
                position_metadata: Arc::clone(&self.position_metadata),
                index_metadata: Arc::clone(&self.index_metadata),
                normal_metadata: Arc::clone(&self.normal_metadata),
                use_gate: Arc::new(SnapshotUseGate::new()),
            });
        }
        if self.snapshot.is_some() {
            NormalIndexedMeshUploadStatus::Ready
        } else {
            NormalIndexedMeshUploadStatus::Pending
        }
    }

    /// Returns the complete opaque snapshot once polling reports ready.
    #[must_use]
    pub fn ready_snapshot(&self) -> Option<NormalIndexedMeshSnapshot> {
        self.snapshot.clone()
    }
}

impl fmt::Debug for NormalIndexedMeshUpload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NormalIndexedMeshUpload")
            .field("generation", &self.generation)
            .field("position_count", &self.position_count)
            .field("index_count", &self.index_count)
            .field("failure", &self.failure)
            .field("ready", &self.snapshot.is_some())
            .finish_non_exhaustive()
    }
}

/// Pure normal three-stream publication policy, independent of native handles.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct NormalSnapshotPublication {
    positions: RetainedUploadState,
    indices: RetainedUploadState,
    normals: RetainedUploadState,
    failed: bool,
}

impl NormalSnapshotPublication {
    pub(super) const fn new(
        positions: RetainedUploadState,
        indices: RetainedUploadState,
        normals: RetainedUploadState,
        failed: bool,
    ) -> Self {
        Self {
            positions,
            indices,
            normals,
            failed,
        }
    }

    pub(super) const fn can_publish(self) -> bool {
        !self.failed
            && matches!(self.positions, RetainedUploadState::Ready)
            && matches!(self.indices, RetainedUploadState::Ready)
            && matches!(self.normals, RetainedUploadState::Ready)
    }
}

fn normal_upload_slot_state(slot: &NormalUploadSlot) -> RetainedUploadState {
    match slot {
        NormalUploadSlot::Absent => RetainedUploadState::Missing,
        NormalUploadSlot::Pending(_) => RetainedUploadState::Pending,
        NormalUploadSlot::Ready(_) => RetainedUploadState::Ready,
    }
}

pub(super) fn normal_start_failure(
    rejected_stream: NormalGeometryStream,
    error: BufferUploadError,
) -> NormalIndexedMeshUploadFailure {
    match rejected_stream {
        NormalGeometryStream::Index => {
            NormalIndexedMeshUploadFailure::IndexStartAfterPositionAccepted(error)
        }
        NormalGeometryStream::Normal => {
            NormalIndexedMeshUploadFailure::NormalStartAfterPositionAndIndexAccepted(error)
        }
        NormalGeometryStream::Position => {
            unreachable!("position rejection occurs before a normal upload operation exists")
        }
    }
}

pub(super) fn normal_position_start_error(
    error: BufferUploadError,
) -> NormalIndexedMeshUploadStartError {
    NormalIndexedMeshUploadStartError::PositionUpload(error)
}

fn poll_normal_slot(
    slot: &mut NormalUploadSlot,
    stream: NormalGeometryStream,
    failure: &mut Option<NormalIndexedMeshUploadFailure>,
) {
    let NormalUploadSlot::Pending(upload) = slot else {
        return;
    };
    match upload.status() {
        Ok(status) if completion_requires_retention(status) => {}
        Ok(CompletionStatus::Complete) => {
            let NormalUploadSlot::Pending(upload) =
                core::mem::replace(slot, NormalUploadSlot::Absent)
            else {
                unreachable!("slot was pending while completing it")
            };
            match upload.finalize() {
                Ok(buffer) => *slot = NormalUploadSlot::Ready(buffer),
                Err(incomplete) => {
                    let status = incomplete.status();
                    *slot = NormalUploadSlot::Pending(incomplete.into_pending());
                    record_normal_completion(status, stream, failure);
                }
            }
        }
        Ok(status) => record_normal_completion(status, stream, failure),
        Err(error) => {
            failure.get_or_insert(normal_observation_failure(stream, error));
        }
    }
}

fn record_normal_completion(
    status: CompletionStatus,
    stream: NormalGeometryStream,
    failure: &mut Option<NormalIndexedMeshUploadFailure>,
) {
    match status {
        CompletionStatus::Pending | CompletionStatus::Unknown | CompletionStatus::Complete => {}
        CompletionStatus::Failed(reason) => {
            failure.get_or_insert(normal_completion_failure(stream, reason));
        }
        _ => {
            failure.get_or_insert(normal_unknown_completion_failure(stream));
        }
    }
}

pub(super) fn normal_completion_failure(
    stream: NormalGeometryStream,
    reason: CompletionFailure,
) -> NormalIndexedMeshUploadFailure {
    match stream {
        NormalGeometryStream::Position => {
            NormalIndexedMeshUploadFailure::PositionCompletion(reason)
        }
        NormalGeometryStream::Index => NormalIndexedMeshUploadFailure::IndexCompletion(reason),
        NormalGeometryStream::Normal => NormalIndexedMeshUploadFailure::NormalCompletion(reason),
    }
}

pub(super) fn normal_observation_failure(
    stream: NormalGeometryStream,
    error: BufferUploadError,
) -> NormalIndexedMeshUploadFailure {
    match stream {
        NormalGeometryStream::Position => {
            NormalIndexedMeshUploadFailure::PositionObservation(error)
        }
        NormalGeometryStream::Index => NormalIndexedMeshUploadFailure::IndexObservation(error),
        NormalGeometryStream::Normal => NormalIndexedMeshUploadFailure::NormalObservation(error),
    }
}

pub(super) fn normal_unknown_completion_failure(
    stream: NormalGeometryStream,
) -> NormalIndexedMeshUploadFailure {
    match stream {
        NormalGeometryStream::Position => NormalIndexedMeshUploadFailure::PositionUnknownCompletion,
        NormalGeometryStream::Index => NormalIndexedMeshUploadFailure::IndexUnknownCompletion,
        NormalGeometryStream::Normal => NormalIndexedMeshUploadFailure::NormalUnknownCompletion,
    }
}

fn take_normal_ready(slot: &mut NormalUploadSlot) -> UploadedBuffer {
    match core::mem::replace(slot, NormalUploadSlot::Absent) {
        NormalUploadSlot::Ready(buffer) => buffer,
        _ => unreachable!("publication requires a ready normal upload"),
    }
}

pub(super) struct NormalMeshPayload {
    pub(super) positions: Vec<u8>,
    pub(super) indices: Vec<u8>,
    pub(super) normals: Vec<u8>,
    pub(super) position_count: u32,
    pub(super) index_count: u32,
}

impl NormalMeshPayload {
    pub(super) fn validate_lengths(
        geometry: &Geometry,
        normals: &[[f32; 3]],
    ) -> Result<(), NormalGeometryError> {
        for (stream, count, stride) in [
            (
                NormalGeometryStream::Position,
                geometry.positions().len(),
                12usize,
            ),
            (NormalGeometryStream::Index, geometry.indices().len(), 4),
            (NormalGeometryStream::Normal, normals.len(), 12),
        ] {
            u32::try_from(count)
                .map_err(|_| NormalGeometryError::CountOutOfRange { stream, count })?;
            let bytes = count
                .checked_mul(stride)
                .ok_or(NormalGeometryError::ByteLengthOverflow { stream })?;
            u64::try_from(bytes).map_err(|_| NormalGeometryError::ByteLengthOverflow { stream })?;
        }
        Ok(())
    }

    pub(super) fn from_geometry(geometry: &NormalGeometry) -> Result<Self, NormalGeometryError> {
        Self::validate_lengths(&geometry.geometry, &geometry.normals)?;
        let mut positions = Vec::with_capacity(geometry.geometry.positions().len() * 12);
        for position in geometry.geometry.positions() {
            for value in position {
                positions.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        let mut indices = Vec::with_capacity(geometry.geometry.indices().len() * 4);
        for index in geometry.geometry.indices() {
            indices.extend_from_slice(&index.to_le_bytes());
        }
        let mut normals = Vec::with_capacity(geometry.normals.len() * 12);
        for normal in &geometry.normals {
            for value in normal {
                normals.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        Ok(Self {
            positions,
            indices,
            normals,
            position_count: u32::try_from(geometry.geometry.positions().len()).map_err(|_| {
                NormalGeometryError::CountOutOfRange {
                    stream: NormalGeometryStream::Position,
                    count: geometry.geometry.positions().len(),
                }
            })?,
            index_count: u32::try_from(geometry.geometry.indices().len()).map_err(|_| {
                NormalGeometryError::CountOutOfRange {
                    stream: NormalGeometryStream::Index,
                    count: geometry.geometry.indices().len(),
                }
            })?,
        })
    }
}
