//! Immutable position, index, and UV mesh upload types.

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

/// The logical stream whose size could not be represented by the textured
/// geometry upload ABI.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TexturedGeometryStream {
    /// Tightly packed `f32x3` positions.
    Position,
    /// Tightly packed `u32` indices.
    Index,
    /// Tightly packed `f32x2` texture coordinates.
    TextureCoordinate,
}

/// Why [`TexturedGeometry`] cannot be constructed.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TexturedGeometryError {
    /// There are no vertex positions.
    EmptyPositions,
    /// The geometry has no index stream.
    MissingIndices,
    /// An indexed geometry supplied an empty index stream.
    EmptyIndices,
    /// The UV and position streams do not have one entry per vertex.
    TextureCoordinateCountMismatch {
        /// Number of positions.
        positions: usize,
        /// Number of texture coordinates.
        texture_coordinates: usize,
    },
    /// A texture-coordinate component is not finite.
    NonFiniteTextureCoordinate {
        /// Vertex containing the component.
        vertex: usize,
        /// Zero for U and one for V.
        component: usize,
    },
    /// A stream count does not fit the closed RHI ABI.
    CountOutOfRange {
        /// Stream whose count failed conversion.
        stream: TexturedGeometryStream,
        /// Original count.
        count: usize,
    },
    /// A stream byte length cannot be represented exactly.
    ByteLengthOverflow {
        /// Stream whose payload length overflowed.
        stream: TexturedGeometryStream,
    },
}

impl fmt::Display for TexturedGeometryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPositions => f.write_str("textured geometry positions are empty"),
            Self::MissingIndices => f.write_str("textured geometry has no indices"),
            Self::EmptyIndices => f.write_str("textured geometry indices are empty"),
            Self::TextureCoordinateCountMismatch {
                positions,
                texture_coordinates,
            } => write!(
                f,
                "textured geometry has {texture_coordinates} texture coordinates for {positions} positions"
            ),
            Self::NonFiniteTextureCoordinate { vertex, component } => write!(
                f,
                "textured geometry texture coordinate {component} at vertex {vertex} is non-finite"
            ),
            Self::CountOutOfRange { stream, count } => write!(
                f,
                "textured geometry {stream:?} count {count} exceeds the closed upload ABI"
            ),
            Self::ByteLengthOverflow { stream } => {
                write!(f, "textured geometry {stream:?} payload length overflows")
            }
        }
    }
}

impl std::error::Error for TexturedGeometryError {}

/// Indexed geometry with one finite `f32x2` texture coordinate per position.
#[derive(Clone, Debug, PartialEq)]
pub struct TexturedGeometry {
    geometry: Geometry,
    texture_coordinates: Vec<[f32; 2]>,
}

impl TexturedGeometry {
    /// Creates closed textured geometry after validating the three upload streams.
    pub fn new(
        geometry: Geometry,
        texture_coordinates: impl Into<Vec<[f32; 2]>>,
    ) -> Result<Self, TexturedGeometryError> {
        let texture_coordinates = texture_coordinates.into();
        let positions = geometry.positions();
        if positions.is_empty() {
            return Err(TexturedGeometryError::EmptyPositions);
        }
        if !geometry.is_indexed() {
            return Err(TexturedGeometryError::MissingIndices);
        }
        if geometry.indices().is_empty() {
            return Err(TexturedGeometryError::EmptyIndices);
        }
        if texture_coordinates.len() != positions.len() {
            return Err(TexturedGeometryError::TextureCoordinateCountMismatch {
                positions: positions.len(),
                texture_coordinates: texture_coordinates.len(),
            });
        }
        for (vertex, coordinate) in texture_coordinates.iter().enumerate() {
            for (component, value) in coordinate.iter().enumerate() {
                if !value.is_finite() {
                    return Err(TexturedGeometryError::NonFiniteTextureCoordinate {
                        vertex,
                        component,
                    });
                }
            }
        }
        TexturedMeshPayload::validate_lengths(&geometry, &texture_coordinates)?;
        Ok(Self {
            geometry,
            texture_coordinates,
        })
    }

    /// Returns the owned indexed geometry.
    #[must_use]
    pub const fn geometry(&self) -> &Geometry {
        &self.geometry
    }

    /// Returns the finite per-vertex texture coordinates.
    #[must_use]
    pub fn texture_coordinates(&self) -> &[[f32; 2]] {
        &self.texture_coordinates
    }
}

/// A failure after at least one textured geometry upload was accepted.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TexturedIndexedMeshUploadFailure {
    /// Position submission completed with a failure.
    PositionCompletion(CompletionFailure),
    /// Index submission completed with a failure.
    IndexCompletion(CompletionFailure),
    /// Texture-coordinate submission completed with a failure.
    TextureCoordinateCompletion(CompletionFailure),
    /// Position completion could no longer be observed.
    PositionObservation(BufferUploadError),
    /// Index completion could no longer be observed.
    IndexObservation(BufferUploadError),
    /// Texture-coordinate completion could no longer be observed.
    TextureCoordinateObservation(BufferUploadError),
    /// Position acceptance was followed by an index start failure.
    IndexStartAfterPositionAccepted(BufferUploadError),
    /// Position and index acceptance were followed by a UV start failure.
    TextureCoordinateStartAfterPositionAndIndexAccepted(BufferUploadError),
    /// Position completion was outside this closed contract.
    PositionUnknownCompletion,
    /// Index completion was outside this closed contract.
    IndexUnknownCompletion,
    /// Texture-coordinate completion was outside this closed contract.
    TextureCoordinateUnknownCompletion,
}

/// Why no textured geometry upload was accepted.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TexturedIndexedMeshUploadStartError {
    /// Position upload was rejected before native acceptance.
    PositionUpload(BufferUploadError),
}

impl fmt::Display for TexturedIndexedMeshUploadStartError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PositionUpload(error) => {
                write!(f, "textured position upload did not start: {error}")
            }
        }
    }
}
impl std::error::Error for TexturedIndexedMeshUploadStartError {}

/// Non-blocking state of a [`TexturedIndexedMeshUpload`].
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TexturedIndexedMeshUploadStatus {
    /// At least one accepted stream remains incomplete.
    Pending,
    /// All three streams completed and a single immutable snapshot was published.
    Ready,
    /// This operation cannot publish a snapshot, but retains accepted siblings.
    Failed(TexturedIndexedMeshUploadFailure),
}

/// One opaque ready generation containing all textured mesh streams.
///
/// Clones share the same generation and use gate; cloning does not create an
/// independently reservable GPU resource set.
#[derive(Clone)]
pub struct TexturedIndexedMeshSnapshot {
    generation: u64,
    position_count: u32,
    index_count: u32,
    positions: UploadedBuffer,
    indices: UploadedBuffer,
    texture_coordinates: UploadedBuffer,
    position_metadata: Arc<[[f32; 3]]>,
    index_metadata: Arc<[u32]>,
    texture_coordinate_metadata: Arc<[[f32; 2]]>,
    use_gate: Arc<SnapshotUseGate>,
}

pub(super) struct TexturedIndexedMeshSnapshotDebugProbe {
    pub(super) generation: u64,
    pub(super) position_count: u32,
    pub(super) index_count: u32,
}

impl fmt::Debug for TexturedIndexedMeshSnapshotDebugProbe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TexturedIndexedMeshSnapshot")
            .field("generation", &self.generation)
            .field("position_count", &self.position_count)
            .field("index_count", &self.index_count)
            .finish()
    }
}

impl fmt::Debug for TexturedIndexedMeshSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        TexturedIndexedMeshSnapshotDebugProbe {
            generation: self.generation,
            position_count: self.position_count,
            index_count: self.index_count,
        }
        .fmt(f)
    }
}

impl TexturedIndexedMeshSnapshot {
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
    pub(crate) fn texture_coordinates(&self) -> &UploadedBuffer {
        &self.texture_coordinates
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
    pub(crate) fn texture_coordinate_metadata(&self) -> &[[f32; 2]] {
        &self.texture_coordinate_metadata
    }
    pub(crate) fn reserve_for_draw(&self) -> Result<SnapshotDrawReservation, SnapshotUseError> {
        self.use_gate.reserve()
    }
}

enum TexturedUploadSlot {
    Absent,
    Pending(PendingBufferUpload),
    Ready(UploadedBuffer),
}

/// Coordinates the three accepted immutable buffer uploads without blocking.
pub struct TexturedIndexedMeshUpload {
    generation: u64,
    position_count: u32,
    index_count: u32,
    position_metadata: Arc<[[f32; 3]]>,
    index_metadata: Arc<[u32]>,
    texture_coordinate_metadata: Arc<[[f32; 2]]>,
    positions: TexturedUploadSlot,
    indices: TexturedUploadSlot,
    texture_coordinates: TexturedUploadSlot,
    failure: Option<TexturedIndexedMeshUploadFailure>,
    snapshot: Option<TexturedIndexedMeshSnapshot>,
}

impl TexturedIndexedMeshUpload {
    /// Atomically prevalidates and then starts position, index, and UV uploads in order.
    ///
    /// Native starts are sequential. If a later stream is rejected after an
    /// earlier one was accepted, the returned operation owns that partial work
    /// and must keep polling every accepted sibling to a terminal state.
    pub fn begin(
        device: &Device,
        geometry: &TexturedGeometry,
    ) -> Result<Self, TexturedIndexedMeshUploadStartError> {
        let payload = TexturedMeshPayload::from_geometry(geometry)
            .expect("TexturedGeometry validates its closed payload ABI");
        let positions = device
            .upload_immutable_buffer(
                buffer_descriptor(payload.positions.len(), BufferUsageKind::Vertex),
                &payload.positions,
            )
            .map_err(textured_position_start_error)?;
        let base = Self {
            generation: NEXT_GENERATION.fetch_add(1, Ordering::Relaxed),
            position_count: payload.position_count,
            index_count: payload.index_count,
            position_metadata: Arc::from(geometry.geometry.positions()),
            index_metadata: Arc::from(geometry.geometry.indices()),
            texture_coordinate_metadata: Arc::from(geometry.texture_coordinates()),
            positions: TexturedUploadSlot::Pending(positions),
            indices: TexturedUploadSlot::Absent,
            texture_coordinates: TexturedUploadSlot::Absent,
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
                operation.failure =
                    Some(textured_start_failure(TexturedGeometryStream::Index, error));
                Ok(operation)
            }
            Ok(indices) => {
                operation.indices = TexturedUploadSlot::Pending(indices);
                match device.upload_immutable_buffer(
                    buffer_descriptor(payload.texture_coordinates.len(), BufferUsageKind::Vertex),
                    &payload.texture_coordinates,
                ) {
                    Ok(uvs) => {
                        operation.texture_coordinates = TexturedUploadSlot::Pending(uvs);
                        Ok(operation)
                    }
                    Err(error) => {
                        operation.failure = Some(textured_start_failure(
                            TexturedGeometryStream::TextureCoordinate,
                            error,
                        ));
                        Ok(operation)
                    }
                }
            }
        }
    }

    /// Polls every accepted sibling and publishes only if all three complete.
    /// A recorded failure is monotonic, but it never permits already-accepted
    /// sibling uploads or their resources to be abandoned early.
    pub fn poll(&mut self) -> TexturedIndexedMeshUploadStatus {
        poll_textured_slot(
            &mut self.positions,
            TexturedGeometryStream::Position,
            &mut self.failure,
        );
        poll_textured_slot(
            &mut self.indices,
            TexturedGeometryStream::Index,
            &mut self.failure,
        );
        poll_textured_slot(
            &mut self.texture_coordinates,
            TexturedGeometryStream::TextureCoordinate,
            &mut self.failure,
        );
        if let Some(failure) = &self.failure {
            return TexturedIndexedMeshUploadStatus::Failed(failure.clone());
        }
        if self.snapshot.is_none()
            && TexturedSnapshotPublication::new(
                textured_upload_slot_state(&self.positions),
                textured_upload_slot_state(&self.indices),
                textured_upload_slot_state(&self.texture_coordinates),
                self.failure.is_some(),
            )
            .can_publish()
        {
            self.snapshot = Some(TexturedIndexedMeshSnapshot {
                generation: self.generation,
                position_count: self.position_count,
                index_count: self.index_count,
                positions: take_textured_ready(&mut self.positions),
                indices: take_textured_ready(&mut self.indices),
                texture_coordinates: take_textured_ready(&mut self.texture_coordinates),
                position_metadata: Arc::clone(&self.position_metadata),
                index_metadata: Arc::clone(&self.index_metadata),
                texture_coordinate_metadata: Arc::clone(&self.texture_coordinate_metadata),
                use_gate: Arc::new(SnapshotUseGate::new()),
            });
        }
        if self.snapshot.is_some() {
            TexturedIndexedMeshUploadStatus::Ready
        } else {
            TexturedIndexedMeshUploadStatus::Pending
        }
    }
    /// Returns the complete opaque snapshot once polling reports ready.
    #[must_use]
    pub fn ready_snapshot(&self) -> Option<TexturedIndexedMeshSnapshot> {
        self.snapshot.clone()
    }
}

/// Pure three-stream publication policy.  It deliberately does not inspect
/// native handles, so all partial-acceptance paths remain regression-testable
/// without pretending that a CPU test proves a GPU completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct TexturedSnapshotPublication {
    positions: RetainedUploadState,
    indices: RetainedUploadState,
    texture_coordinates: RetainedUploadState,
    failed: bool,
}

impl TexturedSnapshotPublication {
    pub(super) const fn new(
        positions: RetainedUploadState,
        indices: RetainedUploadState,
        texture_coordinates: RetainedUploadState,
        failed: bool,
    ) -> Self {
        Self {
            positions,
            indices,
            texture_coordinates,
            failed,
        }
    }

    pub(super) const fn can_publish(self) -> bool {
        !self.failed
            && matches!(self.positions, RetainedUploadState::Ready)
            && matches!(self.indices, RetainedUploadState::Ready)
            && matches!(self.texture_coordinates, RetainedUploadState::Ready)
    }
}

fn textured_upload_slot_state(slot: &TexturedUploadSlot) -> RetainedUploadState {
    match slot {
        TexturedUploadSlot::Absent => RetainedUploadState::Missing,
        TexturedUploadSlot::Pending(_) => RetainedUploadState::Pending,
        TexturedUploadSlot::Ready(_) => RetainedUploadState::Ready,
    }
}

pub(super) fn textured_start_failure(
    rejected_stream: TexturedGeometryStream,
    error: BufferUploadError,
) -> TexturedIndexedMeshUploadFailure {
    match rejected_stream {
        TexturedGeometryStream::Index => {
            TexturedIndexedMeshUploadFailure::IndexStartAfterPositionAccepted(error)
        }
        TexturedGeometryStream::TextureCoordinate => {
            TexturedIndexedMeshUploadFailure::TextureCoordinateStartAfterPositionAndIndexAccepted(
                error,
            )
        }
        TexturedGeometryStream::Position => {
            unreachable!("position rejection occurs before a textured upload operation exists")
        }
    }
}

pub(super) fn textured_position_start_error(
    error: BufferUploadError,
) -> TexturedIndexedMeshUploadStartError {
    TexturedIndexedMeshUploadStartError::PositionUpload(error)
}

fn poll_textured_slot(
    slot: &mut TexturedUploadSlot,
    stream: TexturedGeometryStream,
    failure: &mut Option<TexturedIndexedMeshUploadFailure>,
) {
    let TexturedUploadSlot::Pending(upload) = slot else {
        return;
    };
    match upload.status() {
        Ok(status) if completion_requires_retention(status) => {}
        Ok(CompletionStatus::Complete) => {
            let TexturedUploadSlot::Pending(upload) =
                core::mem::replace(slot, TexturedUploadSlot::Absent)
            else {
                unreachable!()
            };
            match upload.finalize() {
                Ok(buffer) => *slot = TexturedUploadSlot::Ready(buffer),
                Err(incomplete) => {
                    let status = incomplete.status();
                    *slot = TexturedUploadSlot::Pending(incomplete.into_pending());
                    record_textured_completion(status, stream, failure);
                }
            }
        }
        Ok(status) => record_textured_completion(status, stream, failure),
        Err(error) => {
            failure.get_or_insert(textured_observation_failure(stream, error));
        }
    }
}
fn record_textured_completion(
    status: CompletionStatus,
    stream: TexturedGeometryStream,
    failure: &mut Option<TexturedIndexedMeshUploadFailure>,
) {
    match status {
        CompletionStatus::Pending | CompletionStatus::Unknown | CompletionStatus::Complete => {}
        CompletionStatus::Failed(reason) => {
            failure.get_or_insert(textured_completion_failure(stream, reason));
        }
        _ => {
            failure.get_or_insert(textured_unknown_completion_failure(stream));
        }
    }
}

pub(super) fn textured_completion_failure(
    stream: TexturedGeometryStream,
    reason: CompletionFailure,
) -> TexturedIndexedMeshUploadFailure {
    match stream {
        TexturedGeometryStream::Position => {
            TexturedIndexedMeshUploadFailure::PositionCompletion(reason)
        }
        TexturedGeometryStream::Index => TexturedIndexedMeshUploadFailure::IndexCompletion(reason),
        TexturedGeometryStream::TextureCoordinate => {
            TexturedIndexedMeshUploadFailure::TextureCoordinateCompletion(reason)
        }
    }
}

pub(super) fn textured_observation_failure(
    stream: TexturedGeometryStream,
    error: BufferUploadError,
) -> TexturedIndexedMeshUploadFailure {
    match stream {
        TexturedGeometryStream::Position => {
            TexturedIndexedMeshUploadFailure::PositionObservation(error)
        }
        TexturedGeometryStream::Index => TexturedIndexedMeshUploadFailure::IndexObservation(error),
        TexturedGeometryStream::TextureCoordinate => {
            TexturedIndexedMeshUploadFailure::TextureCoordinateObservation(error)
        }
    }
}

pub(super) fn textured_unknown_completion_failure(
    stream: TexturedGeometryStream,
) -> TexturedIndexedMeshUploadFailure {
    match stream {
        TexturedGeometryStream::Position => {
            TexturedIndexedMeshUploadFailure::PositionUnknownCompletion
        }
        TexturedGeometryStream::Index => TexturedIndexedMeshUploadFailure::IndexUnknownCompletion,
        TexturedGeometryStream::TextureCoordinate => {
            TexturedIndexedMeshUploadFailure::TextureCoordinateUnknownCompletion
        }
    }
}
fn take_textured_ready(slot: &mut TexturedUploadSlot) -> UploadedBuffer {
    match core::mem::replace(slot, TexturedUploadSlot::Absent) {
        TexturedUploadSlot::Ready(buffer) => buffer,
        _ => unreachable!("publication requires a ready textured upload"),
    }
}

pub(super) struct TexturedMeshPayload {
    pub(super) positions: Vec<u8>,
    pub(super) indices: Vec<u8>,
    pub(super) texture_coordinates: Vec<u8>,
    pub(super) position_count: u32,
    pub(super) index_count: u32,
}
impl TexturedMeshPayload {
    pub(super) fn validate_lengths(
        geometry: &Geometry,
        coordinates: &[[f32; 2]],
    ) -> Result<(), TexturedGeometryError> {
        for (stream, count, stride) in [
            (
                TexturedGeometryStream::Position,
                geometry.positions().len(),
                12usize,
            ),
            (TexturedGeometryStream::Index, geometry.indices().len(), 4),
            (
                TexturedGeometryStream::TextureCoordinate,
                coordinates.len(),
                8,
            ),
        ] {
            u32::try_from(count)
                .map_err(|_| TexturedGeometryError::CountOutOfRange { stream, count })?;
            let bytes = count
                .checked_mul(stride)
                .ok_or(TexturedGeometryError::ByteLengthOverflow { stream })?;
            u64::try_from(bytes)
                .map_err(|_| TexturedGeometryError::ByteLengthOverflow { stream })?;
        }
        Ok(())
    }
    pub(super) fn from_geometry(
        geometry: &TexturedGeometry,
    ) -> Result<Self, TexturedGeometryError> {
        Self::validate_lengths(&geometry.geometry, &geometry.texture_coordinates)?;
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
        let mut texture_coordinates = Vec::with_capacity(geometry.texture_coordinates.len() * 8);
        for coordinate in &geometry.texture_coordinates {
            for value in coordinate {
                texture_coordinates.extend_from_slice(&value.to_bits().to_le_bytes());
            }
        }
        Ok(Self {
            positions,
            indices,
            texture_coordinates,
            position_count: u32::try_from(geometry.geometry.positions().len()).map_err(|_| {
                TexturedGeometryError::CountOutOfRange {
                    stream: TexturedGeometryStream::Position,
                    count: geometry.geometry.positions().len(),
                }
            })?,
            index_count: u32::try_from(geometry.geometry.indices().len()).map_err(|_| {
                TexturedGeometryError::CountOutOfRange {
                    stream: TexturedGeometryStream::Index,
                    count: geometry.geometry.indices().len(),
                }
            })?,
        })
    }
}
