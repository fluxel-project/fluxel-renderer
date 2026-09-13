//! sRGB RGBA8 texture upload and material types.

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

/// A tightly packed, immutable RGBA8 image whose RGB bytes are IEC sRGB
/// encoded. Alpha remains an uninterpreted linear byte.
///
/// This distinct type prevents the fixed sRGB path from accidentally accepting
/// a linear [`crate::Rgba8Image`]. Uploading never transforms these bytes: decode is a
/// property of the native sRGB texture format at sampling time.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Srgba8Image {
    extent: [u32; 2],
    pixels: Vec<u8>,
}

impl Srgba8Image {
    /// Creates a full-image, tightly packed sRGB-encoded RGBA8 payload.
    pub fn new(extent: [u32; 2], pixels: Vec<u8>) -> Result<Self, Srgba8ImageError> {
        let expected_len = srgba8_byte_len(extent)?;
        if pixels.len() != expected_len {
            return Err(Srgba8ImageError::IncorrectByteLength {
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

    /// Returns the original tightly packed encoded texels in row-major order.
    #[must_use]
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }
}

/// Why an [`Srgba8Image`] could not be constructed.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Srgba8ImageError {
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

impl fmt::Display for Srgba8ImageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroWidth => formatter.write_str("sRGBA8 image width is zero"),
            Self::ZeroHeight => formatter.write_str("sRGBA8 image height is zero"),
            Self::ByteLengthOverflow => formatter.write_str("sRGBA8 image byte length overflows"),
            Self::IncorrectByteLength { expected, actual } => {
                write!(
                    formatter,
                    "sRGBA8 image has {actual} bytes; expected {expected}"
                )
            }
        }
    }
}

impl std::error::Error for Srgba8ImageError {}

fn srgba8_byte_len(extent: [u32; 2]) -> Result<usize, Srgba8ImageError> {
    let [width, height] = extent;
    if width == 0 {
        return Err(Srgba8ImageError::ZeroWidth);
    }
    if height == 0 {
        return Err(Srgba8ImageError::ZeroHeight);
    }
    let bytes = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(Srgba8ImageError::ByteLengthOverflow)?;
    usize::try_from(bytes).map_err(|_| Srgba8ImageError::ByteLengthOverflow)
}

/// A completed immutable sRGB base-color texture generation.
///
/// The resource and synchronization facts are deliberately opaque; clones
/// retain the same generation and share its single-submission gate.
#[derive(Clone)]
pub struct SrgbBaseColorTextureSnapshot {
    generation: u64,
    extent: [u32; 2],
    texture: UploadedTexture,
    use_gate: Arc<SnapshotUseGate>,
}

struct SrgbBaseColorTextureSnapshotDebug {
    generation: u64,
    extent: [u32; 2],
}

impl fmt::Debug for SrgbBaseColorTextureSnapshotDebug {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SrgbBaseColorTextureSnapshot")
            .field("generation", &self.generation)
            .field("extent", &self.extent)
            .finish()
    }
}

impl fmt::Debug for SrgbBaseColorTextureSnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        SrgbBaseColorTextureSnapshotDebug {
            generation: self.generation,
            extent: self.extent,
        }
        .fmt(formatter)
    }
}

impl SrgbBaseColorTextureSnapshot {
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

    /// Returns the texture for renderer-internal lowering only.
    #[must_use]
    pub(crate) fn texture(&self) -> &UploadedTexture {
        &self.texture
    }

    /// Reserves this generation for one renderer submission.
    pub(crate) fn reserve_for_draw(&self) -> Result<SnapshotDrawReservation, SnapshotUseError> {
        self.use_gate.reserve()
    }
}

/// Why an sRGB base-color texture upload could not start before queue acceptance.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SrgbBaseColorTextureUploadStartError {
    /// The native immutable texture request was rejected before acceptance.
    Upload(TextureUploadError),
}

impl fmt::Display for SrgbBaseColorTextureUploadStartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Upload(error) => write!(
                formatter,
                "sRGB base-color texture upload did not start: {error}"
            ),
        }
    }
}

impl std::error::Error for SrgbBaseColorTextureUploadStartError {}

/// A terminal or observed failure of an accepted sRGB base-color texture upload.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SrgbBaseColorTextureUploadFailure {
    /// The texture submission reached a terminal GPU failure.
    Completion(CompletionFailure),
    /// Completion could no longer be observed safely.
    Observation(TextureUploadError),
    /// The backend reported a completion state outside this closed upload contract.
    UnknownCompletionStatus,
}

/// The non-blocking observable state of an sRGB base-color texture upload.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SrgbBaseColorTextureUploadStatus {
    /// The accepted submission has not completed.
    Pending,
    /// Completion proved the texture is ready for immutable use.
    Ready,
    /// This accepted generation cannot become ready.
    Failed(SrgbBaseColorTextureUploadFailure),
}

/// One owning, non-blocking immutable sRGB base-color texture upload.
pub struct SrgbBaseColorTextureUpload {
    generation: u64,
    extent: [u32; 2],
    pending: Option<PendingTextureUpload>,
    failure: Option<SrgbBaseColorTextureUploadFailure>,
    snapshot: Option<SrgbBaseColorTextureSnapshot>,
}

impl SrgbBaseColorTextureUpload {
    /// Starts one immutable sRGB RGBA8 texture upload without decoding its bytes.
    pub fn begin(
        device: &Device,
        image: &Srgba8Image,
    ) -> Result<Self, SrgbBaseColorTextureUploadStartError> {
        let pending = device
            .upload_immutable_texture(
                srgb_base_color_texture_descriptor(image.extent),
                image.pixels(),
            )
            .map_err(SrgbBaseColorTextureUploadStartError::Upload)?;
        Ok(Self {
            generation: NEXT_GENERATION.fetch_add(1, Ordering::Relaxed),
            extent: image.extent,
            pending: Some(pending),
            failure: None,
            snapshot: None,
        })
    }

    /// Polls the accepted upload without waiting for the CPU.
    pub fn poll(&mut self) -> SrgbBaseColorTextureUploadStatus {
        if let Some(pending) = self.pending.as_ref() {
            match pending.status() {
                Ok(status) if completion_requires_retention(status) => {}
                Ok(CompletionStatus::Complete) => {
                    let pending = self.pending.take().expect("pending upload was observed");
                    match pending.finalize() {
                        Ok(texture) => {
                            self.snapshot = Some(SrgbBaseColorTextureSnapshot {
                                generation: self.generation,
                                extent: self.extent,
                                texture,
                                use_gate: Arc::new(SnapshotUseGate::new()),
                            })
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
                        .get_or_insert(SrgbBaseColorTextureUploadFailure::Observation(error));
                }
            }
        }
        if let Some(failure) = &self.failure {
            SrgbBaseColorTextureUploadStatus::Failed(failure.clone())
        } else if self.snapshot.is_some() {
            SrgbBaseColorTextureUploadStatus::Ready
        } else {
            SrgbBaseColorTextureUploadStatus::Pending
        }
    }

    fn record_status(&mut self, status: CompletionStatus) {
        match status {
            CompletionStatus::Pending | CompletionStatus::Unknown | CompletionStatus::Complete => {}
            CompletionStatus::Failed(failure) => {
                self.failure
                    .get_or_insert(SrgbBaseColorTextureUploadFailure::Completion(failure));
            }
            _ => {
                self.failure
                    .get_or_insert(SrgbBaseColorTextureUploadFailure::UnknownCompletionStatus);
            }
        }
    }

    /// Returns a strong immutable snapshot after [`Self::poll`] reports ready.
    #[must_use]
    pub fn ready_snapshot(&self) -> Option<SrgbBaseColorTextureSnapshot> {
        self.snapshot.clone()
    }
}

impl fmt::Debug for SrgbBaseColorTextureUpload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SrgbBaseColorTextureUpload")
            .field("generation", &self.generation)
            .field("extent", &self.extent)
            .field("failure", &self.failure)
            .field("ready", &self.snapshot.is_some())
            .finish_non_exhaustive()
    }
}

fn srgb_base_color_texture_descriptor(extent: [u32; 2]) -> TextureDescriptor {
    let usage =
        TextureUsage::from_kinds([TextureUsageKind::CopyDestination, TextureUsageKind::Sampled]);
    // Test-only readback observes the uploaded encoded bytes. Production keeps
    // the closed immutable contract at CopyDestination + Sampled.
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
            format: TextureFormat::Rgba8UnormSrgb,
        },
        usage,
        memory: MemoryPolicy::DeviceOnly,
    }
}

/// A basic material coupled to one immutable sRGB base-color texture generation.
#[derive(Clone, Debug)]
pub struct SrgbTexturedBasicMaterial {
    material: BasicMaterial,
    base_color_texture: SrgbBaseColorTextureSnapshot,
}

impl SrgbTexturedBasicMaterial {
    /// Couples a basic material to one ready immutable sRGB base-color texture.
    #[must_use]
    pub fn new(material: BasicMaterial, base_color_texture: SrgbBaseColorTextureSnapshot) -> Self {
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

    /// Returns the immutable sRGB base-color texture generation.
    #[must_use]
    pub const fn base_color_texture(&self) -> &SrgbBaseColorTextureSnapshot {
        &self.base_color_texture
    }
}
