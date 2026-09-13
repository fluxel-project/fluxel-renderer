//! Linear RGBA8 texture upload and material types.

use core::fmt;
use std::sync::{Arc, atomic::Ordering};

use fluxel_rendergraph::{
    CompletionFailure, CompletionStatus, Extent3d, TextureDesc, TextureDimension, TextureFormat,
    TextureUsage, TextureUsageKind,
};
use fluxel_rhi::{
    Device, MemoryPolicy, PendingTextureUpload, TextureDescriptor, TextureUploadError,
    UploadedTexture,
};

use crate::BasicMaterial;

use super::super::shared::{
    NEXT_GENERATION, SnapshotDrawReservation, SnapshotUseError, SnapshotUseGate,
    completion_requires_retention,
};

/// A tightly packed, immutable RGBA8 image for the fixed base-color path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Rgba8Image {
    extent: [u32; 2],
    pixels: Vec<u8>,
}

impl Rgba8Image {
    /// Creates a full-image, tightly packed RGBA8 payload.
    pub fn new(extent: [u32; 2], pixels: Vec<u8>) -> Result<Self, Rgba8ImageError> {
        let expected_len = rgba8_byte_len(extent)?;
        if pixels.len() != expected_len {
            return Err(Rgba8ImageError::IncorrectByteLength {
                expected: expected_len,
                actual: pixels.len(),
            });
        }
        Ok(Self { extent, pixels })
    }

    /// Returns the two-dimensional texel extent.
    #[must_use]
    pub const fn extent(&self) -> [u32; 2] {
        self.extent
    }

    /// Returns the tightly packed RGBA8 texels in row-major order.
    #[must_use]
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }
}

/// Why an [`Rgba8Image`] could not be constructed.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Rgba8ImageError {
    /// The image width is zero.
    ZeroWidth,
    /// The image height is zero.
    ZeroHeight,
    /// The requested tightly packed byte length cannot be represented locally.
    ByteLengthOverflow,
    /// The supplied payload is not exactly one full tightly packed image.
    IncorrectByteLength {
        /// Required byte count.
        expected: usize,
        /// Supplied byte count.
        actual: usize,
    },
}

impl fmt::Display for Rgba8ImageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroWidth => formatter.write_str("RGBA8 image width is zero"),
            Self::ZeroHeight => formatter.write_str("RGBA8 image height is zero"),
            Self::ByteLengthOverflow => formatter.write_str("RGBA8 image byte length overflows"),
            Self::IncorrectByteLength { expected, actual } => {
                write!(
                    formatter,
                    "RGBA8 image has {actual} bytes; expected {expected}"
                )
            }
        }
    }
}

impl std::error::Error for Rgba8ImageError {}

pub(in crate::upload) fn rgba8_byte_len(extent: [u32; 2]) -> Result<usize, Rgba8ImageError> {
    let [width, height] = extent;
    if width == 0 {
        return Err(Rgba8ImageError::ZeroWidth);
    }
    if height == 0 {
        return Err(Rgba8ImageError::ZeroHeight);
    }
    let bytes = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(Rgba8ImageError::ByteLengthOverflow)?;
    usize::try_from(bytes).map_err(|_| Rgba8ImageError::ByteLengthOverflow)
}

/// A completed immutable base-color texture generation.
///
/// The native texture, its outgoing state, and its lease remain opaque. A
/// clone retains the same generation and shares one use gate.
#[derive(Clone)]
pub struct BaseColorTextureSnapshot {
    generation: u64,
    extent: [u32; 2],
    #[allow(
        dead_code,
        reason = "sampled raster lowering is a later vertical slice"
    )]
    texture: UploadedTexture,
    #[allow(
        dead_code,
        reason = "sampled raster lowering is a later vertical slice"
    )]
    use_gate: Arc<SnapshotUseGate>,
}

/// Narrow debug view which deliberately excludes renderer-internal native
/// resource, state, descriptor, usage, identity, and synchronization facts.
pub(in crate::upload) struct BaseColorTextureSnapshotDebug {
    pub(in crate::upload) generation: u64,
    pub(in crate::upload) extent: [u32; 2],
}

impl fmt::Debug for BaseColorTextureSnapshotDebug {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BaseColorTextureSnapshot")
            .field("generation", &self.generation)
            .field("extent", &self.extent)
            .finish()
    }
}

impl fmt::Debug for BaseColorTextureSnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        BaseColorTextureSnapshotDebug {
            generation: self.generation,
            extent: self.extent,
        }
        .fmt(formatter)
    }
}

impl BaseColorTextureSnapshot {
    /// Returns the opaque identity of this immutable generation.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Returns the uploaded texture extent in texels.
    #[must_use]
    pub const fn extent(&self) -> [u32; 2] {
        self.extent
    }

    /// Returns the uploaded texture for renderer-internal lowering.
    #[must_use]
    #[allow(
        dead_code,
        reason = "sampled raster lowering is a later vertical slice"
    )]
    pub(crate) fn texture(&self) -> &UploadedTexture {
        &self.texture
    }

    /// Reserves this texture generation for one renderer submission.
    #[allow(
        dead_code,
        reason = "sampled raster lowering is a later vertical slice"
    )]
    pub(crate) fn reserve_for_draw(&self) -> Result<SnapshotDrawReservation, SnapshotUseError> {
        self.use_gate.reserve()
    }
}

/// Why a base-color texture upload could not start before queue acceptance.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum BaseColorTextureUploadStartError {
    /// The native immutable texture request was rejected before acceptance.
    Upload(TextureUploadError),
}

impl fmt::Display for BaseColorTextureUploadStartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Upload(error) => write!(
                formatter,
                "base-color texture upload did not start: {error}"
            ),
        }
    }
}

impl std::error::Error for BaseColorTextureUploadStartError {}

/// A terminal or observed failure of an accepted base-color texture upload.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum BaseColorTextureUploadFailure {
    /// The texture submission reached a terminal GPU failure.
    Completion(CompletionFailure),
    /// Completion could no longer be observed safely.
    Observation(TextureUploadError),
    /// The backend reported a completion state outside this closed upload contract.
    UnknownCompletionStatus,
}

/// The non-blocking observable state of a base-color texture upload.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum BaseColorTextureUploadStatus {
    /// The accepted submission has not completed.
    Pending,
    /// Completion proved the texture is ready for immutable use.
    Ready,
    /// This accepted generation cannot become ready.
    Failed(BaseColorTextureUploadFailure),
}

/// One owning, non-blocking immutable base-color texture upload.
///
/// Before `begin` returns successfully the source image is merely borrowed and
/// can be retried. After native acceptance this operation retains all state
/// required to observe or safely retire the submission.
pub struct BaseColorTextureUpload {
    generation: u64,
    extent: [u32; 2],
    pending: Option<PendingTextureUpload>,
    failure: Option<BaseColorTextureUploadFailure>,
    snapshot: Option<BaseColorTextureSnapshot>,
}

impl BaseColorTextureUpload {
    /// Starts one immutable RGBA8 texture upload.
    pub fn begin(
        device: &Device,
        image: &Rgba8Image,
    ) -> Result<Self, BaseColorTextureUploadStartError> {
        let pending = device
            .upload_immutable_texture(base_color_texture_descriptor(image.extent), image.pixels())
            .map_err(BaseColorTextureUploadStartError::Upload)?;
        Ok(Self {
            generation: NEXT_GENERATION.fetch_add(1, Ordering::Relaxed),
            extent: image.extent,
            pending: Some(pending),
            failure: None,
            snapshot: None,
        })
    }

    /// Polls the accepted upload without waiting for the CPU.
    pub fn poll(&mut self) -> BaseColorTextureUploadStatus {
        if let Some(pending) = self.pending.as_ref() {
            match pending.status() {
                Ok(status) if completion_requires_retention(status) => {}
                Ok(CompletionStatus::Complete) => {
                    let pending = self.pending.take().expect("pending upload was observed");
                    match pending.finalize() {
                        Ok(texture) => {
                            self.snapshot = Some(BaseColorTextureSnapshot {
                                generation: self.generation,
                                extent: self.extent,
                                texture,
                                use_gate: Arc::new(SnapshotUseGate::new()),
                            });
                        }
                        Err(incomplete) => {
                            let status = incomplete.status();
                            self.pending = Some(incomplete.into_pending());
                            self.record_status(status);
                        }
                    }
                }
                Ok(status) => self.record_status(status),
                Err(error) => {
                    self.failure
                        .get_or_insert(BaseColorTextureUploadFailure::Observation(error));
                }
            }
        }
        if let Some(failure) = &self.failure {
            BaseColorTextureUploadStatus::Failed(failure.clone())
        } else if self.snapshot.is_some() {
            BaseColorTextureUploadStatus::Ready
        } else {
            BaseColorTextureUploadStatus::Pending
        }
    }

    fn record_status(&mut self, status: CompletionStatus) {
        match status {
            CompletionStatus::Pending | CompletionStatus::Unknown | CompletionStatus::Complete => {}
            CompletionStatus::Failed(failure) => {
                self.failure
                    .get_or_insert(BaseColorTextureUploadFailure::Completion(failure));
            }
            _ => {
                self.failure
                    .get_or_insert(BaseColorTextureUploadFailure::UnknownCompletionStatus);
            }
        }
    }

    /// Returns a strong immutable snapshot after [`Self::poll`] reports ready.
    #[must_use]
    pub fn ready_snapshot(&self) -> Option<BaseColorTextureSnapshot> {
        self.snapshot.clone()
    }

    /// Reports whether a failed upload still owns accepted work whose
    /// completion is pending, unknown, or temporarily unobservable.
    #[cfg(feature = "gpu-residency")]
    pub(crate) fn retirement_pending(&self) -> bool {
        self.pending
            .as_ref()
            .is_some_and(|pending| match pending.status() {
                Ok(CompletionStatus::Complete | CompletionStatus::Failed(_)) => false,
                Ok(_) | Err(_) => true,
            })
    }
}

impl fmt::Debug for BaseColorTextureUpload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BaseColorTextureUpload")
            .field("generation", &self.generation)
            .field("extent", &self.extent)
            .field("failure", &self.failure)
            .field("ready", &self.snapshot.is_some())
            .finish_non_exhaustive()
    }
}

fn base_color_texture_descriptor(extent: [u32; 2]) -> TextureDescriptor {
    let usage =
        TextureUsage::from_kinds([TextureUsageKind::CopyDestination, TextureUsageKind::Sampled]);
    // This widening exists only in renderer unit/conformance builds, where the
    // RHI's doc-hidden readback oracle must observe the completed upload.  The
    // production descriptor remains exactly CopyDestination + Sampled.
    #[cfg(test)]
    let usage = usage.with(TextureUsageKind::CopySource);
    TextureDescriptor {
        texture: TextureDesc {
            dimension: TextureDimension::D2,
            extent: Extent3d {
                width: extent[0],
                height: extent[1],
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            sample_count: 1,
            format: TextureFormat::Rgba8Unorm,
        },
        usage,
        memory: MemoryPolicy::DeviceOnly,
    }
}

/// A basic material coupled to one immutable base-color texture generation.
#[derive(Clone, Debug)]
pub struct TexturedBasicMaterial {
    material: BasicMaterial,
    base_color_texture: BaseColorTextureSnapshot,
}

impl TexturedBasicMaterial {
    /// Couples a basic material to one ready immutable base-color texture.
    #[must_use]
    pub fn new(material: BasicMaterial, base_color_texture: BaseColorTextureSnapshot) -> Self {
        Self {
            material,
            base_color_texture,
        }
    }

    /// Returns the scalar basic-material properties.
    #[must_use]
    pub const fn material(&self) -> &BasicMaterial {
        &self.material
    }

    /// Returns the immutable base-color texture generation.
    #[must_use]
    pub const fn base_color_texture(&self) -> &BaseColorTextureSnapshot {
        &self.base_color_texture
    }
}
