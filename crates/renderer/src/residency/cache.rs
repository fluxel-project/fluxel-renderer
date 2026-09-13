//! Renderer-private logical-asset to device-resident cache.
//!
//! The table deliberately owns no render-graph resources.  It only makes the
//! lifetime and replacement decisions around backend-owned upload tokens.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use fluxel_assets::{AssetId, AssetSnapshot, ContentGeneration};
use fluxel_rendergraph::DeviceIdentity;

use crate::{Geometry, Rgba8Image};

use super::{ImageAsset, MeshAsset};

/// A non-blocking upload observation.
pub(super) enum UploadPoll<R, F> {
    /// The upload still owns backend work.
    Pending,
    /// The upload completed and produced a resident value.
    Ready(R),
    /// The upload reached a terminal failure.
    Failed {
        failure: F,
        /// Whether the backend still requires the upload token to be retained
        /// and polled after surfacing this failure.
        retirement_pending: bool,
    },
}

/// The renderer-private bridge to a native upload implementation.
pub(super) trait ResidencyBackend {
    type MeshPending;
    type MeshResident: Clone;
    type MeshStartError;
    type MeshFailure: Clone;
    type ImagePending;
    type ImageResident: Clone;
    type ImageStartError;
    type ImageFailure: Clone;

    fn device(&self) -> DeviceIdentity;
    fn begin_mesh(
        &mut self,
        geometry: &Geometry,
    ) -> Result<Self::MeshPending, Self::MeshStartError>;
    fn poll_mesh(
        &mut self,
        pending: &mut Self::MeshPending,
    ) -> UploadPoll<Self::MeshResident, Self::MeshFailure>;
    fn begin_image(
        &mut self,
        image: &Rgba8Image,
    ) -> Result<Self::ImagePending, Self::ImageStartError>;
    fn poll_image(
        &mut self,
        pending: &mut Self::ImagePending,
    ) -> UploadPoll<Self::ImageResident, Self::ImageFailure>;
}

/// A cache key deliberately excludes diagnostic slot fields: `AssetId` is the
/// logical identity, and content generation plus device select the realization.
pub(super) struct AssetKey<K: fluxel_assets::AssetKind> {
    id: AssetId<K>,
    generation: ContentGeneration,
    device: DeviceIdentity,
}

impl<K: fluxel_assets::AssetKind> Copy for AssetKey<K> {}

impl<K: fluxel_assets::AssetKind> Clone for AssetKey<K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K: fluxel_assets::AssetKind> PartialEq for AssetKey<K> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.generation == other.generation && self.device == other.device
    }
}

impl<K: fluxel_assets::AssetKind> Eq for AssetKey<K> {}

impl<K: fluxel_assets::AssetKind> AssetKey<K> {
    fn new(id: AssetId<K>, generation: ContentGeneration, device: DeviceIdentity) -> Self {
        Self {
            id,
            generation,
            device,
        }
    }
}

impl<K: fluxel_assets::AssetKind> Hash for AssetKey<K> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.generation.get().hash(state);
        self.device.hash(state);
    }
}

/// Result of starting an upload.
#[derive(Debug)]
pub(super) enum StartError<ME, IE> {
    Mesh(ME),
    Image(IE),
    /// A caller tried to realize an older CPU generation than the table's
    /// current source for this logical mesh.
    StaleMesh,
    /// A caller tried to realize an older CPU generation than the table's
    /// current source for this logical image.
    StaleImage,
}

/// Terminal upload failure observed while preparing a pair.
pub(super) enum PrepareFailure<MF, IF> {
    Mesh(MF),
    Image(IF),
}

/// Result of preparing the mesh/image pair required by one draw.
pub(super) enum PrepareStatus<MR, IR, MF, IF> {
    Pending,
    Ready {
        #[allow(dead_code)]
        mesh: MR,
        #[allow(dead_code)]
        image: IR,
    },
    Failed(PrepareFailure<MF, IF>),
}

enum EntryState<P, R, F> {
    PendingUpload(P),
    Committed(R),
    /// A failure was reported once, but its backend token must be polled until
    /// that backend stops retaining accepted work.
    Quarantined {
        pending: P,
        failure: F,
    },
    Failed(F),
    RetireCandidate(Retained<P, R, F>),
}

enum Retained<P, R, F> {
    PendingUpload(P),
    Committed(R),
    Quarantined { pending: P, failure: F },
    Failed(F),
}

struct Entry<P, R, F> {
    state: EntryState<P, R, F>,
}

/// Renderer-private residency table for exactly one backend object.
pub(super) struct Table<B: ResidencyBackend> {
    backend: B,
    mesh_entries: MeshEntries<B>,
    image_entries: ImageEntries<B>,
    latest_meshes: HashMap<AssetId<MeshAsset>, AssetSnapshot<MeshAsset, Geometry>>,
    latest_images: HashMap<AssetId<ImageAsset>, AssetSnapshot<ImageAsset, Rgba8Image>>,
}

type MeshEntries<B> = HashMap<
    AssetKey<MeshAsset>,
    Entry<
        <B as ResidencyBackend>::MeshPending,
        <B as ResidencyBackend>::MeshResident,
        <B as ResidencyBackend>::MeshFailure,
    >,
>;

type ImageEntries<B> = HashMap<
    AssetKey<ImageAsset>,
    Entry<
        <B as ResidencyBackend>::ImagePending,
        <B as ResidencyBackend>::ImageResident,
        <B as ResidencyBackend>::ImageFailure,
    >,
>;

type Preparation<B> = Result<
    PrepareStatus<
        <B as ResidencyBackend>::MeshResident,
        <B as ResidencyBackend>::ImageResident,
        <B as ResidencyBackend>::MeshFailure,
        <B as ResidencyBackend>::ImageFailure,
    >,
    StartError<<B as ResidencyBackend>::MeshStartError, <B as ResidencyBackend>::ImageStartError>,
>;

type RecreationSources = (
    Vec<AssetSnapshot<MeshAsset, Geometry>>,
    Vec<AssetSnapshot<ImageAsset, Rgba8Image>>,
);

impl<B: ResidencyBackend> Table<B> {
    pub(super) fn new(backend: B) -> Self {
        Self {
            backend,
            mesh_entries: HashMap::new(),
            image_entries: HashMap::new(),
            latest_meshes: HashMap::new(),
            latest_images: HashMap::new(),
        }
    }

    #[cfg(test)]
    pub(super) fn backend(&self) -> &B {
        &self.backend
    }

    #[cfg(test)]
    pub(super) fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    pub(super) fn device(&self) -> DeviceIdentity {
        self.backend.device()
    }

    /// Starts missing uploads, advances existing ones, and returns a pair only
    /// after both realizations are committed.
    pub(super) fn prepare(
        &mut self,
        mesh: AssetSnapshot<MeshAsset, Geometry>,
        image: AssetSnapshot<ImageAsset, Rgba8Image>,
    ) -> Preparation<B> {
        let device = self.backend.device();
        let mesh_key = AssetKey::new(mesh.id(), mesh.generation(), device);
        let image_key = AssetKey::new(image.id(), image.generation(), device);
        self.accept_mesh(mesh, mesh_key)?;
        self.accept_image(image, image_key)?;

        let mesh = self.advance_mesh(mesh_key);
        let image = self.advance_image(image_key);
        match (mesh, image) {
            (ResourceState::Failed(error), _) => {
                Ok(PrepareStatus::Failed(PrepareFailure::Mesh(error)))
            }
            (_, ResourceState::Failed(error)) => {
                Ok(PrepareStatus::Failed(PrepareFailure::Image(error)))
            }
            (ResourceState::Committed(mesh), ResourceState::Committed(image)) => {
                Ok(PrepareStatus::Ready { mesh, image })
            }
            _ => Ok(PrepareStatus::Pending),
        }
    }

    pub(super) fn request_mesh_retire(&mut self, id: AssetId<MeshAsset>) {
        let device = self.backend.device();
        self.latest_meshes.remove(&id);
        retire_matching(&mut self.mesh_entries, id, device);
    }

    pub(super) fn request_image_retire(&mut self, id: AssetId<ImageAsset>) {
        let device = self.backend.device();
        self.latest_images.remove(&id);
        retire_matching(&mut self.image_entries, id, device);
    }

    /// Retires one exact mesh realization without affecting a newer generation.
    pub(super) fn request_mesh_generation_retire(
        &mut self,
        id: AssetId<MeshAsset>,
        generation: ContentGeneration,
    ) {
        if self
            .latest_meshes
            .get(&id)
            .is_some_and(|source| source.generation() == generation)
        {
            self.latest_meshes.remove(&id);
        }
        retire_exact(
            &mut self.mesh_entries,
            AssetKey::new(id, generation, self.backend.device()),
        );
    }

    /// Retires one exact image realization without affecting a newer generation.
    pub(super) fn request_image_generation_retire(
        &mut self,
        id: AssetId<ImageAsset>,
        generation: ContentGeneration,
    ) {
        if self
            .latest_images
            .get(&id)
            .is_some_and(|source| source.generation() == generation)
        {
            self.latest_images.remove(&id);
        }
        retire_exact(
            &mut self.image_entries,
            AssetKey::new(id, generation, self.backend.device()),
        );
    }

    /// Recreates a mesh upload on this table's device without observing it.
    pub(super) fn seed_mesh(
        &mut self,
        mesh: AssetSnapshot<MeshAsset, Geometry>,
    ) -> Result<(), StartError<B::MeshStartError, B::ImageStartError>> {
        let key = AssetKey::new(mesh.id(), mesh.generation(), self.backend.device());
        self.accept_mesh(mesh, key)
    }

    /// Recreates an image upload on this table's device without observing it.
    pub(super) fn seed_image(
        &mut self,
        image: AssetSnapshot<ImageAsset, Rgba8Image>,
    ) -> Result<(), StartError<B::MeshStartError, B::ImageStartError>> {
        let key = AssetKey::new(image.id(), image.generation(), self.backend.device());
        self.accept_image(image, key)
    }

    /// Retires committed entries immediately. Pending entries are polled until
    /// terminal, so their backend tokens remain alive for the required period.
    pub(super) fn collect_retired(&mut self) {
        collect_mesh_retired(&mut self.backend, &mut self.mesh_entries);
        collect_image_retired(&mut self.backend, &mut self.image_entries);
    }

    /// Returns CPU snapshots needed to reconstruct this table for a new device.
    pub(super) fn recreate_sources(&self) -> RecreationSources {
        let mut meshes: Vec<_> = self.latest_meshes.values().cloned().collect();
        let mut images: Vec<_> = self.latest_images.values().cloned().collect();
        meshes.sort_by_key(AssetSnapshot::id);
        images.sort_by_key(AssetSnapshot::id);
        (meshes, images)
    }

    fn accept_mesh(
        &mut self,
        snapshot: AssetSnapshot<MeshAsset, Geometry>,
        key: AssetKey<MeshAsset>,
    ) -> Result<(), StartError<B::MeshStartError, B::ImageStartError>> {
        if is_stale(&self.latest_meshes, &snapshot) {
            return Err(StartError::StaleMesh);
        }
        if let Some(entry) = self.mesh_entries.remove(&key) {
            self.latest_meshes.insert(snapshot.id(), snapshot);
            self.mesh_entries.insert(
                key,
                Entry {
                    state: revive(entry.state),
                },
            );
            return Ok(());
        }
        let pending = self
            .backend
            .begin_mesh(snapshot.value())
            .map_err(StartError::Mesh)?;
        // Do not disturb the old source or realization until the replacement
        // has been accepted by the backend.
        self.latest_meshes.insert(snapshot.id(), snapshot);
        retire_replaced(&mut self.mesh_entries, key);
        self.mesh_entries.insert(
            key,
            Entry {
                state: EntryState::PendingUpload(pending),
            },
        );
        Ok(())
    }

    fn accept_image(
        &mut self,
        snapshot: AssetSnapshot<ImageAsset, Rgba8Image>,
        key: AssetKey<ImageAsset>,
    ) -> Result<(), StartError<B::MeshStartError, B::ImageStartError>> {
        if is_stale(&self.latest_images, &snapshot) {
            return Err(StartError::StaleImage);
        }
        if let Some(entry) = self.image_entries.remove(&key) {
            self.latest_images.insert(snapshot.id(), snapshot);
            self.image_entries.insert(
                key,
                Entry {
                    state: revive(entry.state),
                },
            );
            return Ok(());
        }
        let pending = self
            .backend
            .begin_image(snapshot.value())
            .map_err(StartError::Image)?;
        self.latest_images.insert(snapshot.id(), snapshot);
        retire_replaced(&mut self.image_entries, key);
        self.image_entries.insert(
            key,
            Entry {
                state: EntryState::PendingUpload(pending),
            },
        );
        Ok(())
    }

    fn advance_mesh(
        &mut self,
        key: AssetKey<MeshAsset>,
    ) -> ResourceState<B::MeshResident, B::MeshFailure> {
        advance_mesh(&mut self.backend, &mut self.mesh_entries, key)
    }

    fn advance_image(
        &mut self,
        key: AssetKey<ImageAsset>,
    ) -> ResourceState<B::ImageResident, B::ImageFailure> {
        advance_image(&mut self.backend, &mut self.image_entries, key)
    }
}

enum ResourceState<R, F> {
    Pending,
    Committed(R),
    Failed(F),
}

fn is_stale<K: fluxel_assets::AssetKind, V>(
    latest: &HashMap<AssetId<K>, AssetSnapshot<K, V>>,
    snapshot: &AssetSnapshot<K, V>,
) -> bool {
    latest
        .get(&snapshot.id())
        .is_some_and(|current| snapshot.generation() < current.generation())
}

fn revive<P, R, F>(state: EntryState<P, R, F>) -> EntryState<P, R, F> {
    match state {
        EntryState::RetireCandidate(Retained::PendingUpload(pending)) => {
            EntryState::PendingUpload(pending)
        }
        EntryState::RetireCandidate(Retained::Committed(resident)) => {
            EntryState::Committed(resident)
        }
        EntryState::RetireCandidate(Retained::Quarantined { pending, failure }) => {
            EntryState::Quarantined { pending, failure }
        }
        EntryState::RetireCandidate(Retained::Failed(failure)) => EntryState::Failed(failure),
        active => active,
    }
}

fn retire_replaced<K: fluxel_assets::AssetKind, P, R, F>(
    entries: &mut HashMap<AssetKey<K>, Entry<P, R, F>>,
    replacement: AssetKey<K>,
) {
    let keys: Vec<_> = entries
        .keys()
        .copied()
        .filter(|key| {
            key.id == replacement.id && key.device == replacement.device && *key != replacement
        })
        .collect();
    for key in keys {
        let entry = entries.remove(&key).expect("key came from this table");
        let state = match entry.state {
            EntryState::PendingUpload(pending) => {
                EntryState::RetireCandidate(Retained::PendingUpload(pending))
            }
            EntryState::Committed(resident) => {
                EntryState::RetireCandidate(Retained::Committed(resident))
            }
            EntryState::Quarantined { pending, failure } => {
                EntryState::RetireCandidate(Retained::Quarantined { pending, failure })
            }
            EntryState::Failed(failure) => EntryState::RetireCandidate(Retained::Failed(failure)),
            retained @ EntryState::RetireCandidate(_) => retained,
        };
        entries.insert(key, Entry { state });
    }
}

fn retire_matching<K: fluxel_assets::AssetKind, P, R, F>(
    entries: &mut HashMap<AssetKey<K>, Entry<P, R, F>>,
    id: AssetId<K>,
    device: DeviceIdentity,
) {
    let keys: Vec<_> = entries
        .keys()
        .copied()
        .filter(|key| key.id == id && key.device == device)
        .collect();
    for key in keys {
        let entry = entries.remove(&key).expect("key came from this table");
        let state = match entry.state {
            EntryState::PendingUpload(pending) => {
                EntryState::RetireCandidate(Retained::PendingUpload(pending))
            }
            EntryState::Committed(resident) => {
                EntryState::RetireCandidate(Retained::Committed(resident))
            }
            EntryState::Quarantined { pending, failure } => {
                EntryState::RetireCandidate(Retained::Quarantined { pending, failure })
            }
            EntryState::Failed(failure) => EntryState::RetireCandidate(Retained::Failed(failure)),
            retained @ EntryState::RetireCandidate(_) => retained,
        };
        entries.insert(key, Entry { state });
    }
}

fn retire_exact<K: fluxel_assets::AssetKind, P, R, F>(
    entries: &mut HashMap<AssetKey<K>, Entry<P, R, F>>,
    key: AssetKey<K>,
) {
    let Some(entry) = entries.remove(&key) else {
        return;
    };
    let state = match entry.state {
        EntryState::PendingUpload(pending) => {
            EntryState::RetireCandidate(Retained::PendingUpload(pending))
        }
        EntryState::Committed(resident) => {
            EntryState::RetireCandidate(Retained::Committed(resident))
        }
        EntryState::Quarantined { pending, failure } => {
            EntryState::RetireCandidate(Retained::Quarantined { pending, failure })
        }
        EntryState::Failed(failure) => EntryState::RetireCandidate(Retained::Failed(failure)),
        retained @ EntryState::RetireCandidate(_) => retained,
    };
    entries.insert(key, Entry { state });
}

fn advance_mesh<B: ResidencyBackend>(
    backend: &mut B,
    entries: &mut MeshEntries<B>,
    key: AssetKey<MeshAsset>,
) -> ResourceState<B::MeshResident, B::MeshFailure> {
    let Some(entry) = entries.remove(&key) else {
        return ResourceState::Pending;
    };
    match entry.state {
        EntryState::Committed(resident) => {
            entries.insert(
                key,
                Entry {
                    state: EntryState::Committed(resident.clone()),
                },
            );
            ResourceState::Committed(resident)
        }
        EntryState::PendingUpload(mut pending) => match backend.poll_mesh(&mut pending) {
            UploadPoll::Pending => {
                entries.insert(
                    key,
                    Entry {
                        state: EntryState::PendingUpload(pending),
                    },
                );
                ResourceState::Pending
            }
            UploadPoll::Ready(resident) => {
                entries.insert(
                    key,
                    Entry {
                        state: EntryState::Committed(resident.clone()),
                    },
                );
                ResourceState::Committed(resident)
            }
            UploadPoll::Failed { failure, .. } => {
                entries.insert(
                    key,
                    Entry {
                        state: EntryState::Quarantined {
                            pending,
                            failure: failure.clone(),
                        },
                    },
                );
                ResourceState::Failed(failure)
            }
        },
        EntryState::Quarantined {
            mut pending,
            failure,
        } => match backend.poll_mesh(&mut pending) {
            UploadPoll::Pending => {
                entries.insert(
                    key,
                    Entry {
                        state: EntryState::Quarantined {
                            pending,
                            failure: failure.clone(),
                        },
                    },
                );
                ResourceState::Failed(failure)
            }
            UploadPoll::Ready(_)
            | UploadPoll::Failed {
                retirement_pending: false,
                ..
            } => {
                entries.insert(
                    key,
                    Entry {
                        state: EntryState::Failed(failure.clone()),
                    },
                );
                ResourceState::Failed(failure)
            }
            UploadPoll::Failed {
                retirement_pending: true,
                ..
            } => {
                entries.insert(
                    key,
                    Entry {
                        state: EntryState::Quarantined {
                            pending,
                            failure: failure.clone(),
                        },
                    },
                );
                ResourceState::Failed(failure)
            }
        },
        EntryState::RetireCandidate(_) => ResourceState::Pending,
        EntryState::Failed(failure) => {
            entries.insert(
                key,
                Entry {
                    state: EntryState::Failed(failure.clone()),
                },
            );
            ResourceState::Failed(failure)
        }
    }
}

fn advance_image<B: ResidencyBackend>(
    backend: &mut B,
    entries: &mut ImageEntries<B>,
    key: AssetKey<ImageAsset>,
) -> ResourceState<B::ImageResident, B::ImageFailure> {
    let Some(entry) = entries.remove(&key) else {
        return ResourceState::Pending;
    };
    match entry.state {
        EntryState::Committed(resident) => {
            entries.insert(
                key,
                Entry {
                    state: EntryState::Committed(resident.clone()),
                },
            );
            ResourceState::Committed(resident)
        }
        EntryState::PendingUpload(mut pending) => match backend.poll_image(&mut pending) {
            UploadPoll::Pending => {
                entries.insert(
                    key,
                    Entry {
                        state: EntryState::PendingUpload(pending),
                    },
                );
                ResourceState::Pending
            }
            UploadPoll::Ready(resident) => {
                entries.insert(
                    key,
                    Entry {
                        state: EntryState::Committed(resident.clone()),
                    },
                );
                ResourceState::Committed(resident)
            }
            UploadPoll::Failed { failure, .. } => {
                entries.insert(
                    key,
                    Entry {
                        state: EntryState::Quarantined {
                            pending,
                            failure: failure.clone(),
                        },
                    },
                );
                ResourceState::Failed(failure)
            }
        },
        EntryState::Quarantined {
            mut pending,
            failure,
        } => match backend.poll_image(&mut pending) {
            UploadPoll::Pending => {
                entries.insert(
                    key,
                    Entry {
                        state: EntryState::Quarantined {
                            pending,
                            failure: failure.clone(),
                        },
                    },
                );
                ResourceState::Failed(failure)
            }
            UploadPoll::Ready(_)
            | UploadPoll::Failed {
                retirement_pending: false,
                ..
            } => {
                entries.insert(
                    key,
                    Entry {
                        state: EntryState::Failed(failure.clone()),
                    },
                );
                ResourceState::Failed(failure)
            }
            UploadPoll::Failed {
                retirement_pending: true,
                ..
            } => {
                entries.insert(
                    key,
                    Entry {
                        state: EntryState::Quarantined {
                            pending,
                            failure: failure.clone(),
                        },
                    },
                );
                ResourceState::Failed(failure)
            }
        },
        EntryState::RetireCandidate(_) => ResourceState::Pending,
        EntryState::Failed(failure) => {
            entries.insert(
                key,
                Entry {
                    state: EntryState::Failed(failure.clone()),
                },
            );
            ResourceState::Failed(failure)
        }
    }
}

fn collect_mesh_retired<B: ResidencyBackend>(backend: &mut B, entries: &mut MeshEntries<B>) {
    entries.retain(|_, entry| match &mut entry.state {
        EntryState::RetireCandidate(Retained::Committed(_)) => false,
        EntryState::RetireCandidate(Retained::PendingUpload(pending)) => {
            retains_upload(backend.poll_mesh(pending))
        }
        EntryState::RetireCandidate(Retained::Quarantined { pending, .. }) => {
            retains_upload(backend.poll_mesh(pending))
        }
        EntryState::Quarantined { pending, failure } => {
            let failure = failure.clone();
            if retains_upload(backend.poll_mesh(pending)) {
                true
            } else {
                entry.state = EntryState::Failed(failure);
                true
            }
        }
        EntryState::RetireCandidate(Retained::Failed(_)) => false,
        _ => true,
    });
}

fn collect_image_retired<B: ResidencyBackend>(backend: &mut B, entries: &mut ImageEntries<B>) {
    entries.retain(|_, entry| match &mut entry.state {
        EntryState::RetireCandidate(Retained::Committed(_)) => false,
        EntryState::RetireCandidate(Retained::PendingUpload(pending)) => {
            retains_upload(backend.poll_image(pending))
        }
        EntryState::RetireCandidate(Retained::Quarantined { pending, .. }) => {
            retains_upload(backend.poll_image(pending))
        }
        EntryState::Quarantined { pending, failure } => {
            let failure = failure.clone();
            if retains_upload(backend.poll_image(pending)) {
                true
            } else {
                entry.state = EntryState::Failed(failure);
                true
            }
        }
        EntryState::RetireCandidate(Retained::Failed(_)) => false,
        _ => true,
    });
}

fn retains_upload<R, F>(poll: UploadPoll<R, F>) -> bool {
    matches!(
        poll,
        UploadPoll::Pending
            | UploadPoll::Failed {
                retirement_pending: true,
                ..
            }
    )
}
