//! Fixed-frame draw admission and closed raster transaction startup.
//!
//! This module is the pre-accept renderer boundary. Its public API selects an
//! audited recipe and validates caller-owned snapshots; its internal start
//! path acquires reservations, constructs the matching graph bindings, and
//! hands lifecycle ownership to `submission`. It neither exposes configurable
//! pipelines nor interprets terminal GPU completion.

use super::*;

/// Coordinates the fixed headless `f32x3/u32` indexed draw slice.
///
/// This type owns no application-visible native resource handles. It accepts
/// only completed snapshots and produces opaque image metadata after
/// non-blocking completion observation.
pub struct FixedFrameRenderer {
    pub(in crate::fixed_frame) device: Device,
    pub(in crate::fixed_frame) capabilities: DeviceCapabilities,
    pub(in crate::fixed_frame) executor: Arc<fluxel_rendergraph::FrameExecutor<RasterBackend>>,
    #[cfg(feature = "gpu-residency")]
    pub(crate) residency: crate::residency::NativeResidency,
}

/// The deliberately disjoint texture domains accepted by fixed-frame lowering.
#[derive(Clone)]
pub(super) enum FrameTextureSnapshot {
    Linear(BaseColorTextureSnapshot),
    Srgb(SrgbBaseColorTextureSnapshot),
}

impl FrameTextureSnapshot {
    pub(super) fn texture(&self) -> &fluxel_rhi::UploadedTexture {
        match self {
            Self::Linear(snapshot) => snapshot.texture(),
            Self::Srgb(snapshot) => snapshot.texture(),
        }
    }

    pub(super) fn reserve_for_draw(&self) -> Result<SnapshotDrawReservation, SnapshotUseError> {
        match self {
            Self::Linear(snapshot) => snapshot.reserve_for_draw(),
            Self::Srgb(snapshot) => snapshot.reserve_for_draw(),
        }
    }
}

/// Private imported-resource view; accepted submissions retain the domain above.
pub(super) trait FrameTexture {
    fn uploaded_texture(&self) -> &fluxel_rhi::UploadedTexture;
}

impl FrameTexture for FrameTextureSnapshot {
    fn uploaded_texture(&self) -> &fluxel_rhi::UploadedTexture {
        self.texture()
    }
}

impl FrameTexture for BaseColorTextureSnapshot {
    fn uploaded_texture(&self) -> &fluxel_rhi::UploadedTexture {
        self.texture()
    }
}

/// Lifecycle-bearing inputs which must remain coupled after graph acceptance.
struct StartResources {
    snapshot: FrameMeshSnapshot,
    texture: Option<FrameTextureSnapshot>,
    graph: Arc<CameraGraph>,
    uniform: FrameUniform,
    reservation: SnapshotDrawReservation,
    texture_reservation: Option<SnapshotDrawReservation>,
    recipe: RasterRecipe,
}

/// UV-specific front-end inputs normalized into `StartResources` by `start`.
pub(super) struct UvStartRequest {
    pub(in crate::fixed_frame) graph: Arc<CameraGraph>,
    pub(in crate::fixed_frame) uniform: FrameUniform,
    pub(in crate::fixed_frame) reservation: SnapshotDrawReservation,
    pub(in crate::fixed_frame) texture_reservation: SnapshotDrawReservation,
    pub(in crate::fixed_frame) recipe: RasterRecipe,
}

mod api;
mod error;
mod start;
#[cfg(test)]
mod tests;
mod validation;

pub use error::{
    DrawStartError, FixedFrameExecutionError, FixedFrameFailure, FixedFrameRasterObservationError,
    FixedFrameUniformObservationError,
};
#[cfg(test)]
pub(in crate::fixed_frame) use validation::PairReservationError;
pub(in crate::fixed_frame) use validation::reserve_pair;
pub(crate) use validation::validate_clip;
use validation::{
    map_pair_error, map_snapshot_use, rgba8_unorm_filterable, rgba8_unorm_srgb_filterable,
    validate_textured_clip,
};
