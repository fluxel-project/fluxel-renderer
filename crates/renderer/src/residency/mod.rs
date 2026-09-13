//! Closed logical-asset to immutable GPU residency integration.
//!
//! The public types expose only typed logical markers, preparation progress,
//! and opaque ready assets. The residency table, physical identities, upload
//! operations, and RenderGraph imports remain renderer-private.

mod cache;
mod native;

pub use native::{
    AssetResidencyError, AssetResidencyFailure, ResidencyRecreation, ResidentAssetPair,
    ResidentAssetStatus,
};
pub(crate) use native::{NativeResidency, new_native_residency};

#[cfg(test)]
mod tests {
    mod cache;

    #[cfg(windows)]
    mod native;
}

/// Logical kind used by the fixed indexed-geometry residency path.
#[derive(Debug)]
pub struct MeshAsset;

impl fluxel_assets::AssetKind for MeshAsset {}

/// Logical kind used by the fixed linear-RGBA8 image residency path.
#[derive(Debug)]
pub struct ImageAsset;

impl fluxel_assets::AssetKind for ImageAsset {}
