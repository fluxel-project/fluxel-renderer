//! Native fixed mesh/image residency adapter and public preparation facade.

use core::fmt;
use std::sync::Mutex;

use fluxel_assets::{AssetId, AssetSnapshot, ContentGeneration};
use fluxel_rendergraph::DeviceIdentity;
use fluxel_rhi::Device;

use crate::{
    BaseColorTextureSnapshot, BaseColorTextureUpload, BaseColorTextureUploadFailure,
    BaseColorTextureUploadStartError, BaseColorTextureUploadStatus, BasicMaterial, Camera,
    DrawStartError, FixedFrameRenderer, FixedFrameSubmission, Geometry, IndexedMeshSnapshot,
    IndexedMeshUpload, IndexedMeshUploadFailure, IndexedMeshUploadStartError,
    IndexedMeshUploadStatus, Rgba8Image, TexturedBasicMaterial,
};

use super::cache::{
    PrepareFailure, PrepareStatus, ResidencyBackend, StartError, Table, UploadPoll,
};
use super::{ImageAsset, MeshAsset};

pub(super) struct NativeBackend {
    device: Device,
}

impl NativeBackend {
    pub(super) fn new(device: Device) -> Self {
        Self { device }
    }
}

impl ResidencyBackend for NativeBackend {
    type MeshPending = IndexedMeshUpload;
    type MeshResident = IndexedMeshSnapshot;
    type MeshStartError = IndexedMeshUploadStartError;
    type MeshFailure = IndexedMeshUploadFailure;
    type ImagePending = BaseColorTextureUpload;
    type ImageResident = BaseColorTextureSnapshot;
    type ImageStartError = BaseColorTextureUploadStartError;
    type ImageFailure = BaseColorTextureUploadFailure;

    fn device(&self) -> DeviceIdentity {
        self.device.identity()
    }

    fn begin_mesh(
        &mut self,
        geometry: &Geometry,
    ) -> Result<Self::MeshPending, Self::MeshStartError> {
        IndexedMeshUpload::begin(&self.device, geometry)
    }

    fn poll_mesh(
        &mut self,
        pending: &mut Self::MeshPending,
    ) -> UploadPoll<Self::MeshResident, Self::MeshFailure> {
        match pending.poll() {
            IndexedMeshUploadStatus::Pending => UploadPoll::Pending,
            IndexedMeshUploadStatus::Ready => UploadPoll::Ready(
                pending
                    .ready_snapshot()
                    .expect("ready mesh upload publishes its immutable snapshot"),
            ),
            IndexedMeshUploadStatus::Failed(failure) => UploadPoll::Failed {
                failure,
                retirement_pending: pending.retirement_pending(),
            },
        }
    }

    fn begin_image(
        &mut self,
        image: &Rgba8Image,
    ) -> Result<Self::ImagePending, Self::ImageStartError> {
        BaseColorTextureUpload::begin(&self.device, image)
    }

    fn poll_image(
        &mut self,
        pending: &mut Self::ImagePending,
    ) -> UploadPoll<Self::ImageResident, Self::ImageFailure> {
        match pending.poll() {
            BaseColorTextureUploadStatus::Pending => UploadPoll::Pending,
            BaseColorTextureUploadStatus::Ready => UploadPoll::Ready(
                pending
                    .ready_snapshot()
                    .expect("ready image upload publishes its immutable snapshot"),
            ),
            BaseColorTextureUploadStatus::Failed(failure) => UploadPoll::Failed {
                failure,
                retirement_pending: pending.retirement_pending(),
            },
        }
    }
}

pub(crate) struct NativeResidency(Mutex<Table<NativeBackend>>);

pub(crate) fn new_native_residency(device: &Device) -> NativeResidency {
    NativeResidency(Mutex::new(Table::new(NativeBackend::new(device.clone()))))
}

/// An opaque ready mesh/image realization selected during frame preparation.
///
/// It retains the exact immutable GPU snapshots but exposes only logical and
/// device-generation diagnostics. RenderGraph and RHI resources remain private.
#[derive(Clone)]
pub struct ResidentAssetPair {
    mesh_id: AssetId<MeshAsset>,
    mesh_generation: ContentGeneration,
    image_id: AssetId<ImageAsset>,
    image_generation: ContentGeneration,
    device: DeviceIdentity,
    mesh: IndexedMeshSnapshot,
    image: BaseColorTextureSnapshot,
}

impl ResidentAssetPair {
    /// Returns the logical mesh identity resolved by preparation.
    #[must_use]
    pub const fn mesh_id(&self) -> AssetId<MeshAsset> {
        self.mesh_id
    }

    /// Returns the immutable mesh content generation resolved by preparation.
    #[must_use]
    pub const fn mesh_generation(&self) -> ContentGeneration {
        self.mesh_generation
    }

    /// Returns the logical image identity resolved by preparation.
    #[must_use]
    pub const fn image_id(&self) -> AssetId<ImageAsset> {
        self.image_id
    }

    /// Returns the immutable image content generation resolved by preparation.
    #[must_use]
    pub const fn image_generation(&self) -> ContentGeneration {
        self.image_generation
    }

    /// Returns the opaque device instance for this realization.
    #[must_use]
    pub const fn device(&self) -> DeviceIdentity {
        self.device
    }
}

impl fmt::Debug for ResidentAssetPair {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResidentAssetPair")
            .field("mesh_id", &self.mesh_id)
            .field("mesh_generation", &self.mesh_generation)
            .field("image_id", &self.image_id)
            .field("image_generation", &self.image_generation)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

/// Non-blocking result of resolving one fixed mesh/image pair.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum ResidentAssetStatus {
    /// At least one immutable GPU upload remains pending or unknown.
    Pending,
    /// Both exact logical generations are committed for this device.
    Ready(ResidentAssetPair),
    /// One accepted upload reached a terminal failure.
    Failed(AssetResidencyFailure),
}

/// A terminal failure observed while resolving a fixed resident asset.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum AssetResidencyFailure {
    /// The indexed-mesh realization could not complete.
    Mesh(IndexedMeshUploadFailure),
    /// The RGBA8 image realization could not complete.
    Image(BaseColorTextureUploadFailure),
}

/// Why resident-asset preparation could not start or access its private table.
#[derive(Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum AssetResidencyError {
    /// The indexed-mesh upload was rejected before queue acceptance.
    MeshStart(IndexedMeshUploadStartError),
    /// The RGBA8 image upload was rejected before queue acceptance.
    ImageStart(BaseColorTextureUploadStartError),
    /// The mesh snapshot predates the newest generation already observed for
    /// the same logical asset.
    StaleMeshGeneration,
    /// The image snapshot predates the newest generation already observed for
    /// the same logical asset.
    StaleImageGeneration,
    /// A prior panic poisoned renderer-private residency bookkeeping.
    SynchronizationPoisoned,
    /// Device recreation was requested using the same device instance.
    SameDevice,
}

impl fmt::Display for AssetResidencyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MeshStart(error) => write!(formatter, "mesh residency did not start: {error}"),
            Self::ImageStart(error) => write!(formatter, "image residency did not start: {error}"),
            Self::StaleMeshGeneration => {
                formatter.write_str("mesh residency rejected a stale content generation")
            }
            Self::StaleImageGeneration => {
                formatter.write_str("image residency rejected a stale content generation")
            }
            Self::SynchronizationPoisoned => {
                formatter.write_str("renderer residency synchronization was poisoned")
            }
            Self::SameDevice => {
                formatter.write_str("residency recreation requires a distinct device generation")
            }
        }
    }
}

impl std::error::Error for AssetResidencyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::MeshStart(error) => Some(error),
            Self::ImageStart(error) => Some(error),
            Self::StaleMeshGeneration
            | Self::StaleImageGeneration
            | Self::SynchronizationPoisoned
            | Self::SameDevice => None,
        }
    }
}

/// Counts immutable logical generations seeded onto a replacement device.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidencyRecreation {
    /// Mesh generations whose new-device uploads were started.
    pub meshes: usize,
    /// Image generations whose new-device uploads were started.
    pub images: usize,
}

impl FixedFrameRenderer {
    /// Resolves or starts the exact mesh/image generations during frame preparation.
    ///
    /// Repeated calls for the same logical/content/device key share one upload.
    /// No asset store or logical identity enters graph recording or pass execution.
    pub fn prepare_resident_assets(
        &self,
        mesh: &AssetSnapshot<MeshAsset, Geometry>,
        image: &AssetSnapshot<ImageAsset, Rgba8Image>,
    ) -> Result<ResidentAssetStatus, AssetResidencyError> {
        let mut residency = self
            .residency
            .0
            .lock()
            .map_err(|_| AssetResidencyError::SynchronizationPoisoned)?;
        match residency
            .prepare(mesh.clone(), image.clone())
            .map_err(map_start_error)?
        {
            PrepareStatus::Pending => Ok(ResidentAssetStatus::Pending),
            PrepareStatus::Ready {
                mesh: resident_mesh,
                image: resident_image,
            } => Ok(ResidentAssetStatus::Ready(ResidentAssetPair {
                mesh_id: mesh.id(),
                mesh_generation: mesh.generation(),
                image_id: image.id(),
                image_generation: image.generation(),
                device: residency.device(),
                mesh: resident_mesh,
                image: resident_image,
            })),
            PrepareStatus::Failed(PrepareFailure::Mesh(error)) => Ok(ResidentAssetStatus::Failed(
                AssetResidencyFailure::Mesh(error),
            )),
            PrepareStatus::Failed(PrepareFailure::Image(error)) => Ok(ResidentAssetStatus::Failed(
                AssetResidencyFailure::Image(error),
            )),
        }
    }

    /// Requests retirement of every cached generation for one logical mesh.
    ///
    /// Collection removes renderer lookup ownership only. An accepted frame's
    /// RHI leases continue retaining its physical resources until completion.
    pub fn request_mesh_retirement(
        &self,
        id: AssetId<MeshAsset>,
    ) -> Result<(), AssetResidencyError> {
        self.residency
            .0
            .lock()
            .map_err(|_| AssetResidencyError::SynchronizationPoisoned)?
            .request_mesh_retire(id);
        Ok(())
    }

    /// Requests retirement of every cached generation for one logical image.
    pub fn request_image_retirement(
        &self,
        id: AssetId<ImageAsset>,
    ) -> Result<(), AssetResidencyError> {
        self.residency
            .0
            .lock()
            .map_err(|_| AssetResidencyError::SynchronizationPoisoned)?
            .request_image_retire(id);
        Ok(())
    }

    /// Advances pending retirement and drops renderer lookup ownership when safe.
    pub fn collect_retired_assets(&self) -> Result<(), AssetResidencyError> {
        self.residency
            .0
            .lock()
            .map_err(|_| AssetResidencyError::SynchronizationPoisoned)?
            .collect_retired();
        Ok(())
    }

    /// Seeds this renderer's new device from the prior renderer's retained CPU snapshots.
    ///
    /// Old-device entries become retirement candidates only after every
    /// new-device upload was accepted. Existing submitted frames remain safe
    /// because their RHI leases are independent of residency lookup ownership.
    pub fn recreate_asset_residency_from(
        &self,
        prior: &Self,
    ) -> Result<ResidencyRecreation, AssetResidencyError> {
        let replacement_mutex = &self.residency.0;
        let prior_mutex = &prior.residency.0;
        if std::ptr::eq(replacement_mutex, prior_mutex) {
            return Err(AssetResidencyError::SameDevice);
        }

        // A stable address order makes snapshot, seeding, and exact old-key
        // retirement one linearizable transaction without permitting inverse
        // A<-B / B<-A recreation calls to deadlock.
        if std::ptr::from_ref(replacement_mutex).addr() < std::ptr::from_ref(prior_mutex).addr() {
            let mut replacement = self
                .residency
                .0
                .lock()
                .map_err(|_| AssetResidencyError::SynchronizationPoisoned)?;
            let mut old = prior
                .residency
                .0
                .lock()
                .map_err(|_| AssetResidencyError::SynchronizationPoisoned)?;
            recreate_locked(&mut replacement, &mut old)
        } else {
            let mut old = prior
                .residency
                .0
                .lock()
                .map_err(|_| AssetResidencyError::SynchronizationPoisoned)?;
            let mut replacement = self
                .residency
                .0
                .lock()
                .map_err(|_| AssetResidencyError::SynchronizationPoisoned)?;
            recreate_locked(&mut replacement, &mut old)
        }
    }

    /// Draws a prepared resident pair through the existing closed textured recipe.
    ///
    /// Asset lookup has already finished: this call receives only the opaque
    /// device-local realization selected by [`Self::prepare_resident_assets`].
    pub fn draw_resident_textured(
        &self,
        resident: &ResidentAssetPair,
        camera: &Camera,
        material: &BasicMaterial,
        extent: [u32; 2],
    ) -> Result<FixedFrameSubmission, DrawStartError> {
        let material = TexturedBasicMaterial::new(material.clone(), resident.image.clone());
        self.draw_textured(&resident.mesh, camera, &material, extent)
    }
}

fn recreate_locked(
    replacement: &mut Table<NativeBackend>,
    old: &mut Table<NativeBackend>,
) -> Result<ResidencyRecreation, AssetResidencyError> {
    if replacement.device() == old.device() {
        return Err(AssetResidencyError::SameDevice);
    }
    let (meshes, images) = old.recreate_sources();
    for mesh in &meshes {
        replacement
            .seed_mesh(mesh.clone())
            .map_err(map_start_error)?;
    }
    for image in &images {
        replacement
            .seed_image(image.clone())
            .map_err(map_start_error)?;
    }
    for mesh in &meshes {
        old.request_mesh_generation_retire(mesh.id(), mesh.generation());
    }
    for image in &images {
        old.request_image_generation_retire(image.id(), image.generation());
    }
    Ok(ResidencyRecreation {
        meshes: meshes.len(),
        images: images.len(),
    })
}

fn map_start_error(
    error: StartError<IndexedMeshUploadStartError, BaseColorTextureUploadStartError>,
) -> AssetResidencyError {
    match error {
        StartError::Mesh(error) => AssetResidencyError::MeshStart(error),
        StartError::Image(error) => AssetResidencyError::ImageStart(error),
        StartError::StaleMesh => AssetResidencyError::StaleMeshGeneration,
        StartError::StaleImage => AssetResidencyError::StaleImageGeneration,
    }
}
