use fluxel_assets::{Acquire, AssetSnapshot, AssetStore, Production, ResidentBytes};
use fluxel_rendergraph::DeviceIdentity;

use crate::{Geometry, Rgba8Image};

use super::super::cache::{
    PrepareFailure, PrepareStatus, ResidencyBackend, StartError, Table, UploadPoll,
};
use super::super::{ImageAsset, MeshAsset};

type Sources = (
    AssetStore<MeshAsset, Geometry, ()>,
    AssetSnapshot<MeshAsset, Geometry>,
    AssetSnapshot<ImageAsset, Rgba8Image>,
);

#[derive(Default)]
struct FakeBackend {
    device: u64,
    mesh_begins: usize,
    image_begins: usize,
    mesh_fail: bool,
    image_fail: bool,
    mesh_failure_retirement_pending: bool,
    image_failure_retirement_pending: bool,
    mesh_start_fail: bool,
    image_start_fail: bool,
    initial_polls: u8,
}

struct Pending {
    polls_left: u8,
}

impl FakeBackend {
    fn new(device: u64) -> Self {
        Self {
            device,
            ..Self::default()
        }
    }
}

impl ResidencyBackend for FakeBackend {
    type MeshPending = Pending;
    type MeshResident = usize;
    type MeshStartError = &'static str;
    type MeshFailure = &'static str;
    type ImagePending = Pending;
    type ImageResident = usize;
    type ImageStartError = &'static str;
    type ImageFailure = &'static str;

    fn device(&self) -> DeviceIdentity {
        DeviceIdentity::new(self.device)
    }
    fn begin_mesh(&mut self, _: &Geometry) -> Result<Pending, &'static str> {
        self.mesh_begins += 1;
        if self.mesh_start_fail {
            Err("mesh start")
        } else {
            Ok(Pending {
                polls_left: self.initial_polls.max(1),
            })
        }
    }
    fn poll_mesh(&mut self, pending: &mut Pending) -> UploadPoll<usize, &'static str> {
        if self.mesh_fail {
            return UploadPoll::Failed {
                failure: "mesh failure",
                retirement_pending: self.mesh_failure_retirement_pending,
            };
        }
        if pending.polls_left > 0 {
            pending.polls_left -= 1;
            UploadPoll::Pending
        } else {
            UploadPoll::Ready(self.mesh_begins)
        }
    }
    fn begin_image(&mut self, _: &Rgba8Image) -> Result<Pending, &'static str> {
        self.image_begins += 1;
        if self.image_start_fail {
            Err("image start")
        } else {
            Ok(Pending {
                polls_left: self.initial_polls.max(1),
            })
        }
    }
    fn poll_image(&mut self, pending: &mut Pending) -> UploadPoll<usize, &'static str> {
        if self.image_fail {
            return UploadPoll::Failed {
                failure: "image failure",
                retirement_pending: self.image_failure_retirement_pending,
            };
        }
        if pending.polls_left > 0 {
            pending.polls_left -= 1;
            UploadPoll::Pending
        } else {
            UploadPoll::Ready(self.image_begins)
        }
    }
}

fn geometry(x: f32) -> Geometry {
    Geometry::from_positions(vec![[x, 0.0, 0.0], [x + 1.0, 0.0, 0.0], [x, 1.0, 0.0]])
}

fn image(byte: u8) -> Rgba8Image {
    Rgba8Image::new([1, 1], vec![byte, byte, byte, 255]).unwrap()
}

fn mesh_snapshot(
    value: Geometry,
) -> (
    AssetStore<MeshAsset, Geometry, ()>,
    AssetSnapshot<MeshAsset, Geometry>,
) {
    let store = AssetStore::new();
    let handle = store.create().unwrap();
    let Acquire::Producer(producer) = store.acquire(&handle).unwrap() else {
        panic!("new asset must produce")
    };
    let snapshot = producer.commit(value, ResidentBytes::new(1)).unwrap();
    (store, snapshot)
}

fn image_snapshot(
    value: Rgba8Image,
) -> (
    AssetStore<ImageAsset, Rgba8Image, ()>,
    AssetSnapshot<ImageAsset, Rgba8Image>,
) {
    let store = AssetStore::new();
    let handle = store.create().unwrap();
    let Acquire::Producer(producer) = store.acquire(&handle).unwrap() else {
        panic!("new asset must produce")
    };
    let snapshot = producer.commit(value, ResidentBytes::new(1)).unwrap();
    (store, snapshot)
}

fn replacement(
    store: &AssetStore<MeshAsset, Geometry, ()>,
    snapshot: &AssetSnapshot<MeshAsset, Geometry>,
) -> AssetSnapshot<MeshAsset, Geometry> {
    let Production::Producer(producer) = store.request_replacement(snapshot.handle()).unwrap()
    else {
        panic!("uncontended replacement must produce")
    };
    producer
        .commit(geometry(4.0), ResidentBytes::new(1))
        .unwrap()
}

fn image_replacement(
    store: &AssetStore<ImageAsset, Rgba8Image, ()>,
    snapshot: &AssetSnapshot<ImageAsset, Rgba8Image>,
) -> AssetSnapshot<ImageAsset, Rgba8Image> {
    let Production::Producer(producer) = store.request_replacement(snapshot.handle()).unwrap()
    else {
        panic!("uncontended replacement must produce")
    };
    producer.commit(image(9), ResidentBytes::new(1)).unwrap()
}

fn sources() -> Sources {
    let (mesh_store, mesh) = mesh_snapshot(geometry(0.0));
    let (_, image) = image_snapshot(image(5));
    (mesh_store, mesh, image)
}

#[test]
fn same_key_starts_each_upload_once() {
    let (_, mesh, image) = sources();
    let mut table = Table::new(FakeBackend::new(1));
    assert!(matches!(
        table.prepare(mesh.clone(), image.clone()),
        Ok(PrepareStatus::Pending)
    ));
    assert!(matches!(
        table.prepare(mesh, image),
        Ok(PrepareStatus::Ready { .. })
    ));
    assert_eq!(table.backend().mesh_begins, 1);
    assert_eq!(table.backend().image_begins, 1);
}

#[test]
fn pending_pair_is_not_ready() {
    let (_, mesh, image) = sources();
    let mut table = Table::new(FakeBackend::new(1));
    assert!(matches!(
        table.prepare(mesh, image),
        Ok(PrepareStatus::Pending)
    ));
}

#[test]
fn replacement_retires_old_generation_and_starts_new_one() {
    let (store, mesh, image) = sources();
    let replacement = replacement(&store, &mesh);
    let mut table = Table::new(FakeBackend::new(1));
    let _ = table.prepare(mesh.clone(), image.clone());
    let _ = table.prepare(mesh, image.clone());
    assert!(matches!(
        table.prepare(replacement.clone(), image.clone()),
        Ok(PrepareStatus::Pending)
    ));
    table.collect_retired();
    assert!(matches!(
        table.prepare(replacement, image),
        Ok(PrepareStatus::Ready { .. })
    ));
    assert_eq!(table.backend().mesh_begins, 2);
}

#[test]
fn committed_retire_is_collected_and_can_be_started_again() {
    let (_, mesh, image) = sources();
    let mut table = Table::new(FakeBackend::new(1));
    let _ = table.prepare(mesh.clone(), image.clone());
    let _ = table.prepare(mesh.clone(), image.clone());
    table.request_mesh_retire(mesh.id());
    table.collect_retired();
    assert!(matches!(
        table.prepare(mesh, image),
        Ok(PrepareStatus::Pending)
    ));
    assert_eq!(table.backend().mesh_begins, 2);
}

#[test]
fn explicit_reuse_revives_a_retired_committed_entry() {
    let (_, mesh, image) = sources();
    let mut table = Table::new(FakeBackend::new(1));
    let _ = table.prepare(mesh.clone(), image.clone());
    let _ = table.prepare(mesh.clone(), image.clone());
    table.request_mesh_retire(mesh.id());
    assert!(matches!(
        table.prepare(mesh, image),
        Ok(PrepareStatus::Ready { .. })
    ));
    assert_eq!(table.backend().mesh_begins, 1);
}

#[test]
fn explicit_reuse_revives_a_retired_pending_upload() {
    let (_, mesh, image) = sources();
    let mut table = Table::new(FakeBackend::new(1));
    table.backend_mut().initial_polls = 2;
    let _ = table.prepare(mesh.clone(), image.clone());
    table.request_mesh_retire(mesh.id());
    assert!(matches!(
        table.prepare(mesh.clone(), image.clone()),
        Ok(PrepareStatus::Pending)
    ));
    assert!(matches!(
        table.prepare(mesh, image),
        Ok(PrepareStatus::Ready { .. })
    ));
    assert_eq!(table.backend().mesh_begins, 1);
}

#[test]
fn pending_retire_is_retained_until_terminal_poll() {
    let (_, mesh, image) = sources();
    let mut table = Table::new(FakeBackend::new(1));
    let _ = table.prepare(mesh.clone(), image);
    table.request_mesh_retire(mesh.id());
    assert!(
        !table
            .recreate_sources()
            .0
            .iter()
            .any(|source| source.id() == mesh.id())
    );
    table.collect_retired();
    assert_eq!(table.backend().mesh_begins, 1);
    table.collect_retired();
    assert!(
        !table
            .recreate_sources()
            .0
            .iter()
            .any(|source| source.id() == mesh.id())
    );
}

#[test]
fn retired_pending_uploads_survive_a_failed_poll_that_still_requires_retention() {
    let (_, mesh, image) = sources();
    let mut table = Table::new(FakeBackend::new(1));
    table.seed_mesh(mesh.clone()).unwrap();
    table.seed_image(image.clone()).unwrap();
    table.backend_mut().mesh_fail = true;
    table.backend_mut().image_fail = true;
    table.backend_mut().mesh_failure_retirement_pending = true;
    table.backend_mut().image_failure_retirement_pending = true;
    table.request_mesh_generation_retire(mesh.id(), mesh.generation());
    table.request_image_generation_retire(image.id(), image.generation());

    table.collect_retired();

    table.backend_mut().mesh_fail = false;
    table.backend_mut().image_fail = false;
    table.seed_mesh(mesh).unwrap();
    table.seed_image(image).unwrap();
    assert_eq!(table.backend().mesh_begins, 1);
    assert_eq!(table.backend().image_begins, 1);
}

#[test]
fn image_retire_removes_its_recreation_source() {
    let (_, mesh, image) = sources();
    let mut table = Table::new(FakeBackend::new(1));
    let _ = table.prepare(mesh, image.clone());
    table.request_image_retire(image.id());
    assert!(
        !table
            .recreate_sources()
            .1
            .iter()
            .any(|source| source.id() == image.id())
    );
}

#[test]
fn exact_generation_retire_keeps_newer_image_generation_available() {
    let (_, mesh, _) = sources();
    let (image_store, first) = image_snapshot(image(5));
    let second = image_replacement(&image_store, &first);
    let mut table = Table::new(FakeBackend::new(1));
    let _ = table.prepare(mesh.clone(), first.clone());
    let _ = table.prepare(mesh.clone(), first.clone());
    let _ = table.prepare(mesh.clone(), second.clone());
    table.request_image_generation_retire(first.id(), first.generation());
    table.collect_retired();
    assert_eq!(
        table.recreate_sources().1[0].generation(),
        second.generation()
    );
    assert!(matches!(
        table.prepare(mesh, second),
        Ok(PrepareStatus::Ready { .. })
    ));
}

#[test]
fn recreation_sources_seed_a_new_device_once_each() {
    let (_, mesh, image) = sources();
    let mut old = Table::new(FakeBackend::new(1));
    let _ = old.prepare(mesh, image);
    let (meshes, images) = old.recreate_sources();
    let mut recreated = Table::new(FakeBackend::new(2));
    assert_eq!(recreated.device(), DeviceIdentity::new(2));
    for source in meshes {
        recreated.seed_mesh(source).unwrap();
    }
    for source in images {
        recreated.seed_image(source).unwrap();
    }
    assert_eq!(recreated.backend().mesh_begins, 1);
    assert_eq!(recreated.backend().image_begins, 1);
}

#[test]
fn failed_upload_is_sticky_until_explicit_retire_then_can_retry() {
    let (_, mesh, image) = sources();
    let mut table = Table::new(FakeBackend::new(1));
    table.backend_mut().mesh_fail = true;
    assert!(matches!(
        table.prepare(mesh.clone(), image.clone()),
        Ok(PrepareStatus::Failed(PrepareFailure::Mesh("mesh failure")))
    ));
    table.backend_mut().mesh_fail = false;
    assert!(matches!(
        table.prepare(mesh.clone(), image.clone()),
        Ok(PrepareStatus::Failed(PrepareFailure::Mesh("mesh failure")))
    ));
    table.request_mesh_generation_retire(mesh.id(), mesh.generation());
    assert!(table.recreate_sources().0.is_empty());
    table.collect_retired();
    table.collect_retired();
    assert!(matches!(
        table.prepare(mesh, image),
        Ok(PrepareStatus::Pending)
    ));
    assert_eq!(table.backend().mesh_begins, 2);
}

#[test]
fn failed_mesh_quarantines_while_accepted_image_keeps_progressing() {
    let (_, mesh, image) = sources();
    let mut table = Table::new(FakeBackend::new(1));
    table.backend_mut().initial_polls = 2;
    table.backend_mut().mesh_fail = true;
    table.backend_mut().mesh_failure_retirement_pending = true;
    assert!(matches!(
        table.prepare(mesh.clone(), image.clone()),
        Ok(PrepareStatus::Failed(PrepareFailure::Mesh("mesh failure")))
    ));
    assert_eq!(table.backend().image_begins, 1);
    assert!(matches!(
        table.prepare(mesh.clone(), image.clone()),
        Ok(PrepareStatus::Failed(PrepareFailure::Mesh("mesh failure")))
    ));
    table.backend_mut().mesh_fail = false;
    assert!(matches!(
        table.prepare(mesh.clone(), image.clone()),
        Ok(PrepareStatus::Failed(PrepareFailure::Mesh("mesh failure")))
    ));
    assert_eq!(table.backend().image_begins, 1);
    table.request_mesh_generation_retire(mesh.id(), mesh.generation());
    table.collect_retired();
    table.collect_retired();
    assert!(matches!(
        table.prepare(mesh, image),
        Ok(PrepareStatus::Pending)
    ));
    assert_eq!(table.backend().mesh_begins, 2);
}

#[test]
fn stale_generation_cannot_replace_latest_source_or_retire_newer_entry() {
    let (store, first, image) = sources();
    let second = replacement(&store, &first);
    let mut table = Table::new(FakeBackend::new(1));
    let _ = table.prepare(first.clone(), image.clone());
    let _ = table.prepare(first.clone(), image.clone());
    let _ = table.prepare(second.clone(), image.clone());
    let _ = table.prepare(second.clone(), image.clone());
    assert!(matches!(
        table.prepare(first.clone(), image),
        Err(StartError::StaleMesh)
    ));
    let (mut meshes, _) = table.recreate_sources();
    assert_eq!(meshes.len(), 1);
    assert_eq!(meshes[0].generation(), second.generation());
    let mut recreated = Table::new(FakeBackend::new(2));
    recreated.seed_mesh(meshes.remove(0)).unwrap();
    assert!(matches!(
        recreated.seed_mesh(first),
        Err(StartError::StaleMesh)
    ));
}

#[test]
fn rejected_replacement_preserves_previous_active_generation_and_source() {
    let (store, first, image) = sources();
    let second = replacement(&store, &first);
    let mut table = Table::new(FakeBackend::new(1));
    let _ = table.prepare(first.clone(), image.clone());
    let _ = table.prepare(first.clone(), image.clone());
    table.backend_mut().mesh_start_fail = true;
    assert!(matches!(
        table.prepare(second, image.clone()),
        Err(StartError::Mesh("mesh start"))
    ));
    table.backend_mut().mesh_start_fail = false;
    assert!(matches!(
        table.prepare(first.clone(), image),
        Ok(PrepareStatus::Ready { .. })
    ));
    assert_eq!(table.backend().mesh_begins, 2);
    let (meshes, _) = table.recreate_sources();
    assert_eq!(meshes[0].generation(), first.generation());
}

#[test]
fn device_change_selects_an_independent_key() {
    let (_, mesh, image) = sources();
    let mut table = Table::new(FakeBackend::new(1));
    let _ = table.prepare(mesh.clone(), image.clone());
    let _ = table.prepare(mesh.clone(), image.clone());
    table.backend_mut().device = 2;
    assert!(matches!(
        table.prepare(mesh.clone(), image.clone()),
        Ok(PrepareStatus::Pending)
    ));
    assert!(matches!(
        table.prepare(mesh, image),
        Ok(PrepareStatus::Ready { .. })
    ));
    assert_eq!(table.backend().mesh_begins, 2);
    assert_eq!(table.backend().image_begins, 2);
}

#[test]
fn start_errors_are_typed() {
    let (_, mesh, image) = sources();
    let mut table = Table::new(FakeBackend::new(1));
    table.backend_mut().mesh_start_fail = true;
    assert!(matches!(
        table.prepare(mesh, image),
        Err(StartError::Mesh("mesh start"))
    ));
}
