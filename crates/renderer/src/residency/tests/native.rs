//! Windows native conformance fixtures for the public asset-residency facade.
//!
//! These fixtures intentionally construct CPU content through `fluxel-assets`
//! and interact with residency solely through the renderer's public surface.

use std::{
    thread,
    time::{Duration, Instant},
};

use fluxel_assets::{Acquire, AssetSnapshot, AssetStore, Production, ResidentBytes};
use fluxel_rhi::{Backend, Device, DeviceOptions, Validation};

use crate::{
    AssetResidencyError, BasicMaterial, Camera, FixedFrameRenderer, FixedFrameStatus, Geometry,
    ImageAsset, MeshAsset, ResidentAssetPair, ResidentAssetStatus, Rgba8Image,
};

type Sources = (
    AssetStore<MeshAsset, Geometry, ()>,
    AssetSnapshot<MeshAsset, Geometry>,
    AssetSnapshot<ImageAsset, Rgba8Image>,
);

// These functions are compile-time contracts: preparation accepts CPU asset
// snapshots, while drawing has no asset-store or snapshot input at all.
fn prepare_contract(
    renderer: &FixedFrameRenderer,
    mesh: &AssetSnapshot<MeshAsset, Geometry>,
    image: &AssetSnapshot<ImageAsset, Rgba8Image>,
) -> Result<ResidentAssetStatus, AssetResidencyError> {
    renderer.prepare_resident_assets(mesh, image)
}

fn draw_contract(
    renderer: &FixedFrameRenderer,
    resident: &ResidentAssetPair,
) -> Result<crate::FixedFrameSubmission, crate::DrawStartError> {
    renderer.draw_resident_textured(
        resident,
        &Camera::default(),
        &BasicMaterial::default(),
        [8, 8],
    )
}

#[test]
#[ignore = "requires a Windows DX12 device with required validation"]
fn public_residency_dx12() {
    run(Backend::Dx12);
}

#[test]
#[ignore = "requires a Windows Vulkan device with required validation"]
fn public_residency_vulkan() {
    run(Backend::Vulkan);
}

fn run(backend: Backend) {
    let _guard = crate::native_fixture_guard();
    let device = open(backend);
    let renderer = FixedFrameRenderer::new(device.clone());
    let (mesh_store, mesh, image) = sources();

    // Same logical/content/device inputs eventually produce the same public
    // identity tuple.  The facade deliberately does not expose physical IDs;
    // repeated readiness is therefore the observable sharing contract.
    let first = ready(&renderer, &mesh, &image);
    let reused = ready(&renderer, &mesh, &image);
    assert_same_public_realization(&first, &reused);

    let mut first_draw = draw_contract(&renderer, &first).expect("resident draw starts");
    let mut shared_draw = draw_contract(&renderer, &reused).expect("shared resident draw starts");
    complete(&mut first_draw);
    complete(&mut shared_draw);

    let replacement = replace(&mesh_store, &mesh, geometry(0.15));
    let replacement_pair = ready(&renderer, &replacement, &image);
    assert_eq!(replacement_pair.mesh_id(), first.mesh_id());
    assert_ne!(replacement_pair.mesh_generation(), first.mesh_generation());
    assert_eq!(
        replacement_pair.image_generation(),
        first.image_generation()
    );
    assert!(matches!(
        renderer.prepare_resident_assets(&mesh, &image),
        Err(AssetResidencyError::StaleMeshGeneration)
    ));
    assert_eq!(
        renderer.recreate_asset_residency_from(&renderer),
        Err(AssetResidencyError::SameDevice)
    );

    // A distinct Device must receive new uploads from retained CPU snapshots,
    // preserving logical generations while changing the opaque device identity.
    let replacement_device = open(backend);
    let replacement_renderer = FixedFrameRenderer::new(replacement_device);
    let recreation = replacement_renderer
        .recreate_asset_residency_from(&renderer)
        .expect("distinct device recreates residency");
    assert!(recreation.meshes >= 1 && recreation.images >= 1);
    let reuploaded = ready(&replacement_renderer, &replacement, &image);
    assert_eq!(
        reuploaded.mesh_generation(),
        replacement_pair.mesh_generation()
    );
    assert_eq!(
        reuploaded.image_generation(),
        replacement_pair.image_generation()
    );
    assert_ne!(reuploaded.device(), replacement_pair.device());
    let mut reuploaded_draw =
        draw_contract(&replacement_renderer, &reuploaded).expect("reuploaded resident pair draws");
    complete(&mut reuploaded_draw);

    // Lookup ownership can disappear without invalidating an already accepted
    // opaque pair: its retained snapshots still draw and complete safely.
    renderer.request_mesh_retirement(mesh.id()).unwrap();
    renderer.request_image_retirement(image.id()).unwrap();
    renderer.collect_retired_assets().unwrap();
    let mut retired_draw = draw_contract(&renderer, &first).expect("retired pair remains drawable");
    complete(&mut retired_draw);
}

fn open(backend: Backend) -> Device {
    Device::open(
        backend,
        DeviceOptions {
            validation: Validation::Required,
            ..DeviceOptions::default()
        },
    )
    .unwrap()
}

fn sources() -> Sources {
    let mesh_store = AssetStore::new();
    let mesh_handle = mesh_store.create().unwrap();
    let Acquire::Producer(mesh_producer) = mesh_store.acquire(&mesh_handle).unwrap() else {
        panic!("new mesh must acquire its producer")
    };
    let mesh = mesh_producer
        .commit(geometry(-0.2), ResidentBytes::new(1))
        .unwrap();

    let image_store: AssetStore<ImageAsset, Rgba8Image, ()> = AssetStore::new();
    let image_handle = image_store.create().unwrap();
    let Acquire::Producer(image_producer) = image_store.acquire(&image_handle).unwrap() else {
        panic!("new image must acquire its producer")
    };
    let image = image_producer
        .commit(
            Rgba8Image::new([1, 1], vec![64, 128, 255, 255]).unwrap(),
            ResidentBytes::new(4),
        )
        .unwrap();
    (mesh_store, mesh, image)
}

fn replace(
    store: &AssetStore<MeshAsset, Geometry, ()>,
    previous: &AssetSnapshot<MeshAsset, Geometry>,
    value: Geometry,
) -> AssetSnapshot<MeshAsset, Geometry> {
    let Production::Producer(producer) = store.request_replacement(previous.handle()).unwrap()
    else {
        panic!("uncontended replacement must produce")
    };
    producer.commit(value, ResidentBytes::new(1)).unwrap()
}

fn geometry(x: f32) -> Geometry {
    Geometry::from_positions(vec![
        [x - 0.4, -0.4, 0.0],
        [x + 0.4, -0.4, 0.0],
        [x, 0.4, 0.0],
    ])
    .with_indices(vec![0, 1, 2])
    .unwrap()
}

fn ready(
    renderer: &FixedFrameRenderer,
    mesh: &AssetSnapshot<MeshAsset, Geometry>,
    image: &AssetSnapshot<ImageAsset, Rgba8Image>,
) -> ResidentAssetPair {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match prepare_contract(renderer, mesh, image).expect("residency preparation starts") {
            ResidentAssetStatus::Ready(pair) => return pair,
            ResidentAssetStatus::Pending => {
                assert!(Instant::now() < deadline, "residency timed out")
            }
            ResidentAssetStatus::Failed(error) => panic!("residency failed: {error:?}"),
        }
        thread::sleep(Duration::from_millis(1));
    }
}

fn complete(submission: &mut crate::FixedFrameSubmission) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match submission.poll() {
            FixedFrameStatus::Complete(_) => return,
            FixedFrameStatus::Pending | FixedFrameStatus::Busy => {
                assert!(Instant::now() < deadline, "resident draw timed out")
            }
            FixedFrameStatus::Failed(error) => panic!("resident draw failed: {error:?}"),
        }
        thread::sleep(Duration::from_millis(1));
    }
}

fn assert_same_public_realization(left: &ResidentAssetPair, right: &ResidentAssetPair) {
    assert_eq!(left.mesh_id(), right.mesh_id());
    assert_eq!(left.mesh_generation(), right.mesh_generation());
    assert_eq!(left.image_id(), right.image_id());
    assert_eq!(left.image_generation(), right.image_generation());
    assert_eq!(left.device(), right.device());
}
