//! Snapshot-use gate and native fault-contract tests.

use super::*;

pub(crate) fn generation_use_gate_allows_concurrent_immutable_readers_and_poison_is_monotonic() {
    let gate = Arc::new(SnapshotUseGate::new());
    let mut first = gate.reserve().unwrap();
    let mut second = gate.reserve().unwrap();

    // Finishing one reader must not serialize or invalidate another reader of
    // the same immutable generation.
    second.release_complete();
    let mut third = gate.reserve().unwrap();
    first.release_complete();
    third.release_complete();
    let poisoned = gate.reserve().unwrap();
    let mut sibling = gate.reserve().unwrap();
    drop(poisoned);
    // A known completion after a sibling has become unknown cannot revive it.
    sibling.release_complete();
    assert_eq!(gate.reserve().unwrap_err(), SnapshotUseError::Poisoned);
}

pub(crate) fn unknown_completion_keeps_the_reservation_until_a_terminal_observation() {
    let gate = Arc::new(SnapshotUseGate::new());
    let mut reservation = gate.reserve().expect("reserve immutable generation");

    // An unknown completion is retryable: the operation must not release the
    // accepted generation merely because the backend cannot observe it yet.
    let status = fluxel_rendergraph::CompletionStatus::Unknown;
    if !completion_requires_retention(status) {
        reservation.release_complete();
    }
    assert_eq!(gate.active_readers(), 1, "unknown retains the lease");

    // Only a terminal successful observation advances the lifecycle and
    // releases the immutable-generation lease.
    assert!(!completion_requires_retention(
        fluxel_rendergraph::CompletionStatus::Complete
    ));
    reservation.release_complete();
    assert_eq!(gate.active_readers(), 0, "complete releases the lease");
}

#[cfg(windows)]
pub(crate) fn u05_textured_three_upload_fault_contract_dx12() {
    run_textured_three_upload_fault_contract(fluxel_rhi::Backend::Dx12);
}

#[cfg(windows)]
pub(crate) fn u05_textured_three_upload_fault_contract_vulkan() {
    run_textured_three_upload_fault_contract(fluxel_rhi::Backend::Vulkan);
}

#[cfg(windows)]
fn run_textured_three_upload_fault_contract(backend: fluxel_rhi::Backend) {
    use fluxel_rhi::{DeviceOptions, Validation, test_support};

    let guard = crate::native_fixture_guard();
    let device = Device::open(
        backend,
        DeviceOptions {
            validation: Validation::Required,
            ..DeviceOptions::default()
        },
    )
    .unwrap_or_else(|error| panic!("U05 textured upload {backend:?} device open failed: {error}"));
    test_support::clear_validation_diagnostics(&device);
    let geometry = super::geometry::textured_geometry();

    // No submit was accepted: begin returns the retryable start error.
    test_support::inject_submit_rejected_after(0);
    assert!(matches!(
        TexturedIndexedMeshUpload::begin(&device, &geometry),
        Err(TexturedIndexedMeshUploadStartError::PositionUpload(_))
    ));

    // The next two injections distinguish both owning partial-acceptance
    // paths: no snapshot may escape although earlier leases remain owned.
    test_support::inject_submit_rejected_after(1);
    let mut index_rejected = TexturedIndexedMeshUpload::begin(&device, &geometry).unwrap();
    assert!(matches!(
        index_rejected.poll(),
        TexturedIndexedMeshUploadStatus::Failed(
            TexturedIndexedMeshUploadFailure::IndexStartAfterPositionAccepted(_)
        )
    ));
    assert!(index_rejected.ready_snapshot().is_none());
    drop(index_rejected);

    test_support::inject_submit_rejected_after(2);
    let mut uv_rejected = TexturedIndexedMeshUpload::begin(&device, &geometry).unwrap();
    assert!(matches!(
        uv_rejected.poll(),
        TexturedIndexedMeshUploadStatus::Failed(
            TexturedIndexedMeshUploadFailure::TextureCoordinateStartAfterPositionAndIndexAccepted(
                _
            )
        )
    ));
    assert!(uv_rejected.ready_snapshot().is_none());
    drop(uv_rejected);

    // Each accepted-unknown injection is observed on precisely the stream
    // whose submit was selected; it remains an owning failed operation.
    for (after, stream) in [
        (0, TexturedGeometryStream::Position),
        (1, TexturedGeometryStream::Index),
        (2, TexturedGeometryStream::TextureCoordinate),
    ] {
        test_support::inject_submit_accepted_unknown_after(after);
        let mut upload = TexturedIndexedMeshUpload::begin(&device, &geometry).unwrap();
        let status = poll_textured_upload_until_terminal(&mut upload, backend);
        assert!(matches!(
            (stream, status),
            (
                TexturedGeometryStream::Position,
                TexturedIndexedMeshUploadStatus::Failed(
                    TexturedIndexedMeshUploadFailure::PositionCompletion(
                        CompletionFailure::DeviceLost
                    )
                )
            ) | (
                TexturedGeometryStream::Index,
                TexturedIndexedMeshUploadStatus::Failed(
                    TexturedIndexedMeshUploadFailure::IndexCompletion(
                        CompletionFailure::DeviceLost
                    )
                )
            ) | (
                TexturedGeometryStream::TextureCoordinate,
                TexturedIndexedMeshUploadStatus::Failed(
                    TexturedIndexedMeshUploadFailure::TextureCoordinateCompletion(
                        CompletionFailure::DeviceLost
                    )
                )
            )
        ));
        assert!(upload.ready_snapshot().is_none());
        drop(upload);
    }

    // Dropping a partially accepted operation neither publishes a snapshot
    // nor blocks. The subsequent normal operation proves its RHI-owned
    // quarantine did not make the shared device unusable.
    test_support::inject_submit_accepted_unknown_after(1);
    let dropped_partial = TexturedIndexedMeshUpload::begin(&device, &geometry).unwrap();
    assert!(dropped_partial.ready_snapshot().is_none());
    drop(dropped_partial);

    let mut normal = TexturedIndexedMeshUpload::begin(&device, &geometry).unwrap();
    assert_eq!(
        poll_textured_upload_until_terminal(&mut normal, backend),
        TexturedIndexedMeshUploadStatus::Ready
    );
    let snapshot = normal
        .ready_snapshot()
        .expect("normal triple upload publishes once");
    assert_eq!(snapshot.position_count(), 2);
    assert_eq!(snapshot.index_count(), 3);
    let diagnostics = test_support::validation_diagnostics(&device);
    assert!(diagnostics.is_empty(), "U05 {backend:?}: {diagnostics:?}");
    let commit = std::env::var("FLUXEL_TEST_COMMIT").expect(
        "U05 three-upload fault evidence requires FLUXEL_TEST_COMMIT at the exact tested SHA",
    );
    assert!(commit.len() == 40 && commit.bytes().all(|byte| byte.is_ascii_hexdigit()));
    eprintln!(
        "artifact case=U05-three-upload-fault backend={backend:?} commit={commit} hardware={:?} rejected_after=[0,1,2] accepted_unknown_after=[0,1,2] partial_snapshot=false drop_partial_nonblocking=true normal=Ready generation={} outgoing=[{:?},{:?},{:?}] diagnostics={diagnostics:?}",
        device.hardware(),
        snapshot.generation(),
        snapshot.positions().outgoing_state(),
        snapshot.indices().outgoing_state(),
        snapshot.texture_coordinates().outgoing_state(),
    );
    drop(guard);
}

#[cfg(windows)]
pub(crate) fn u08_normal_three_upload_fault_contract_dx12() {
    run_normal_three_upload_fault_contract(fluxel_rhi::Backend::Dx12);
}

#[cfg(windows)]
pub(crate) fn u08_normal_three_upload_fault_contract_vulkan() {
    run_normal_three_upload_fault_contract(fluxel_rhi::Backend::Vulkan);
}

#[cfg(windows)]
fn run_normal_three_upload_fault_contract(backend: fluxel_rhi::Backend) {
    use fluxel_rhi::{DeviceOptions, Validation, test_support};

    let guard = crate::native_fixture_guard();
    let device = Device::open(
        backend,
        DeviceOptions {
            validation: Validation::Required,
            ..DeviceOptions::default()
        },
    )
    .unwrap_or_else(|error| panic!("U08 normal upload {backend:?} device open failed: {error}"));
    test_support::clear_validation_diagnostics(&device);
    let geometry = super::geometry::normal_geometry();

    // Rejection before position acceptance remains retryable: no partial
    // owner exists to publish or quarantine.
    test_support::inject_submit_rejected_after(0);
    assert!(matches!(
        NormalIndexedMeshUpload::begin(&device, &geometry),
        Err(NormalIndexedMeshUploadStartError::PositionUpload(_))
    ));

    // The two later rejects retain their accepted siblings, while keeping
    // the generation opaque: a partial normal mesh must never escape.
    test_support::inject_submit_rejected_after(1);
    let mut index_rejected = NormalIndexedMeshUpload::begin(&device, &geometry).unwrap();
    assert!(matches!(
        index_rejected.poll(),
        NormalIndexedMeshUploadStatus::Failed(
            NormalIndexedMeshUploadFailure::IndexStartAfterPositionAccepted(_)
        )
    ));
    assert!(index_rejected.ready_snapshot().is_none());
    drop(index_rejected);

    test_support::inject_submit_rejected_after(2);
    let mut normal_rejected = NormalIndexedMeshUpload::begin(&device, &geometry).unwrap();
    assert!(matches!(
        normal_rejected.poll(),
        NormalIndexedMeshUploadStatus::Failed(
            NormalIndexedMeshUploadFailure::NormalStartAfterPositionAndIndexAccepted(_)
        )
    ));
    assert!(normal_rejected.ready_snapshot().is_none());
    drop(normal_rejected);

    // Accepted-unknown is terminally unsafe for the selected stream. Each
    // operation still owns every accepted sibling through its retirement.
    for (after, stream) in [
        (0, NormalGeometryStream::Position),
        (1, NormalGeometryStream::Index),
        (2, NormalGeometryStream::Normal),
    ] {
        test_support::inject_submit_accepted_unknown_after(after);
        let mut upload = NormalIndexedMeshUpload::begin(&device, &geometry).unwrap();
        let status = poll_normal_upload_until_terminal(&mut upload, backend);
        assert!(matches!(
            (stream, status),
            (
                NormalGeometryStream::Position,
                NormalIndexedMeshUploadStatus::Failed(
                    NormalIndexedMeshUploadFailure::PositionCompletion(
                        CompletionFailure::DeviceLost
                    )
                )
            ) | (
                NormalGeometryStream::Index,
                NormalIndexedMeshUploadStatus::Failed(
                    NormalIndexedMeshUploadFailure::IndexCompletion(CompletionFailure::DeviceLost)
                )
            ) | (
                NormalGeometryStream::Normal,
                NormalIndexedMeshUploadStatus::Failed(
                    NormalIndexedMeshUploadFailure::NormalCompletion(CompletionFailure::DeviceLost)
                )
            )
        ));
        assert!(upload.ready_snapshot().is_none());
        drop(upload);
    }

    // Dropping a partially accepted operation is non-blocking and never
    // publishes a generation. A later upload proves the RHI quarantine is
    // scoped to retained resources, not the shared device.
    test_support::inject_submit_accepted_unknown_after(1);
    let dropped_partial = NormalIndexedMeshUpload::begin(&device, &geometry).unwrap();
    assert!(dropped_partial.ready_snapshot().is_none());
    drop(dropped_partial);

    let mut normal = NormalIndexedMeshUpload::begin(&device, &geometry).unwrap();
    assert_eq!(
        poll_normal_upload_until_terminal(&mut normal, backend),
        NormalIndexedMeshUploadStatus::Ready
    );
    let snapshot = normal
        .ready_snapshot()
        .expect("normal triple upload publishes once");
    assert_eq!(snapshot.position_count(), 2);
    assert_eq!(snapshot.index_count(), 3);
    let diagnostics = test_support::validation_diagnostics(&device);
    assert!(diagnostics.is_empty(), "U08 {backend:?}: {diagnostics:?}");
    let commit = normal_fault_commit();
    eprintln!(
        "artifact case=U08-normal-three-upload-fault backend={backend:?} commit={commit} hardware={:?} rejected_after=[0,1,2] accepted_unknown_after=[0,1,2] partial_snapshot=false drop_partial_nonblocking=true normal=Ready generation={} outgoing=[{:?},{:?},{:?}] diagnostics={diagnostics:?}",
        device.hardware(),
        snapshot.generation(),
        snapshot.positions().outgoing_state(),
        snapshot.indices().outgoing_state(),
        snapshot.normals().outgoing_state(),
    );
    drop(guard);
}

#[cfg(windows)]
fn poll_normal_upload_until_terminal(
    upload: &mut NormalIndexedMeshUpload,
    backend: fluxel_rhi::Backend,
) -> NormalIndexedMeshUploadStatus {
    use std::{
        thread,
        time::{Duration, Instant},
    };
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match upload.poll() {
            status @ NormalIndexedMeshUploadStatus::Ready
            | status @ NormalIndexedMeshUploadStatus::Failed(_) => return status,
            NormalIndexedMeshUploadStatus::Pending if Instant::now() < deadline => {
                thread::sleep(Duration::from_millis(1));
            }
            NormalIndexedMeshUploadStatus::Pending => {
                panic!("U08 normal upload {backend:?} timed out")
            }
        }
    }
}

#[cfg(windows)]
fn normal_fault_commit() -> String {
    let commit = std::env::var("FLUXEL_TEST_COMMIT")
        .expect("U08 normal three-upload fault evidence requires FLUXEL_TEST_COMMIT");
    assert!(
        normal_fault_commit_is_valid(&commit),
        "U08 normal three-upload fault evidence requires a 40-character hexadecimal SHA"
    );
    if commit.bytes().all(|byte| byte == b'0') {
        return "working-tree".into();
    }

    let head = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .expect("U08 normal three-upload fault evidence needs git rev-parse HEAD");
    assert!(
        head.status.success(),
        "U08 normal three-upload fault evidence git rev-parse HEAD failed"
    );
    assert_eq!(
        String::from_utf8(head.stdout)
            .expect("git rev-parse HEAD emitted non-UTF-8 output")
            .trim(),
        commit,
        "U08 normal three-upload fault evidence commit differs from HEAD"
    );

    let status = std::process::Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=normal"])
        .output()
        .expect("U08 normal three-upload fault evidence needs git status");
    assert!(
        status.status.success(),
        "U08 normal three-upload fault evidence git status failed"
    );
    assert!(
        status.stdout.is_empty(),
        "U08 normal three-upload exact-SHA evidence requires a clean worktree: {}",
        String::from_utf8_lossy(&status.stdout)
    );
    commit
}

#[cfg(windows)]
fn normal_fault_commit_is_valid(commit: &str) -> bool {
    commit.len() == 40 && commit.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(windows)]
fn poll_textured_upload_until_terminal(
    upload: &mut TexturedIndexedMeshUpload,
    backend: fluxel_rhi::Backend,
) -> TexturedIndexedMeshUploadStatus {
    use std::{
        thread,
        time::{Duration, Instant},
    };
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match upload.poll() {
            status @ TexturedIndexedMeshUploadStatus::Ready
            | status @ TexturedIndexedMeshUploadStatus::Failed(_) => return status,
            TexturedIndexedMeshUploadStatus::Pending if Instant::now() < deadline => {
                thread::sleep(Duration::from_millis(1));
            }
            TexturedIndexedMeshUploadStatus::Pending => {
                panic!("U05 textured upload {backend:?} timed out")
            }
        }
    }
}

#[cfg(windows)]
pub(crate) fn u01_indexed_mesh_upload_dx12() {
    run_u01(fluxel_rhi::Backend::Dx12);
}

#[cfg(windows)]
pub(crate) fn u01_indexed_mesh_upload_vulkan() {
    run_u01(fluxel_rhi::Backend::Vulkan);
}

#[cfg(windows)]
fn run_u01(backend: fluxel_rhi::Backend) {
    use std::{
        thread,
        time::{Duration, Instant},
    };

    use fluxel_rendergraph::ResourceAccessState;
    use fluxel_rhi::{DeviceOptions, Validation, test_support};

    let guard = crate::native_fixture_guard();
    let device = Device::open(
        backend,
        DeviceOptions {
            validation: Validation::Required,
            ..DeviceOptions::default()
        },
    )
    .unwrap_or_else(|error| panic!("U01 {backend:?} device open failed: {error}"));
    test_support::clear_validation_diagnostics(&device);

    let geometry = Geometry::from_positions(vec![
        [-1.0, -0.5, 0.0],
        [0.0, 1.0, -0.0],
        [0.75, -0.25, f32::from_bits(0x7fc0_0011)],
    ])
    .with_indices(vec![2, 1, 0, 2, 0, 1])
    .unwrap();
    let expected = IndexedMeshPayload::from_geometry(&geometry).unwrap();
    let mut upload = IndexedMeshUpload::begin(&device, &geometry)
        .unwrap_or_else(|error| panic!("U01 {backend:?} upload start failed: {error}"));
    let deadline = Instant::now() + Duration::from_secs(10);
    let snapshot = loop {
        match upload.poll() {
            IndexedMeshUploadStatus::Ready => {
                break upload
                    .ready_snapshot()
                    .expect("Ready upload must publish its complete snapshot");
            }
            IndexedMeshUploadStatus::Failed(error) => {
                panic!("U01 {backend:?} upload failed: {error:?}")
            }
            IndexedMeshUploadStatus::Pending if Instant::now() < deadline => {
                thread::sleep(Duration::from_millis(1));
            }
            IndexedMeshUploadStatus::Pending => panic!("U01 {backend:?} timed out"),
        }
    };
    let actual_positions = test_support::readback_uploaded_buffer(&device, snapshot.positions())
        .unwrap_or_else(|error| panic!("U01 {backend:?} position readback failed: {error}"));
    let actual_indices = test_support::readback_uploaded_buffer(&device, snapshot.indices())
        .unwrap_or_else(|error| panic!("U01 {backend:?} index readback failed: {error}"));
    assert_eq!(
        actual_positions, expected.positions,
        "U01 {backend:?} positions"
    );
    assert_eq!(actual_indices, expected.indices, "U01 {backend:?} indices");
    assert_eq!(
        snapshot.positions().outgoing_state(),
        ResourceAccessState::CopyDestination
    );
    assert_eq!(
        snapshot.indices().outgoing_state(),
        ResourceAccessState::CopyDestination
    );
    let diagnostics = test_support::validation_diagnostics(&device);
    assert!(diagnostics.is_empty(), "U01 {backend:?}: {diagnostics:?}");

    let commit = std::env::var("FLUXEL_TEST_COMMIT").unwrap_or_else(|_| "working-tree".into());
    eprintln!(
        "artifact case=U01 backend={backend:?} commit={commit} os={}; hardware={:?}; input=indexed f32x3 positions={} u32 indices={}; position_descriptor={:?}; index_descriptor={:?}; expected_positions={:?}; actual_positions={actual_positions:?}; expected_indices={:?}; actual_indices={actual_indices:?}; first_difference_positions={:?}; first_difference_indices={:?}; outgoing_position={:?}; outgoing_index={:?}; completion=Complete; diagnostics={diagnostics:?}",
        std::env::consts::OS,
        device.hardware(),
        snapshot.position_count(),
        snapshot.index_count(),
        snapshot.positions().buffer().descriptor(),
        snapshot.indices().buffer().descriptor(),
        expected.positions,
        expected.indices,
        first_difference(&expected.positions, &actual_positions),
        first_difference(&expected.indices, &actual_indices),
        snapshot.positions().outgoing_state(),
        snapshot.indices().outgoing_state(),
    );
    drop(guard);
}

#[cfg(windows)]
fn first_difference(expected: &[u8], actual: &[u8]) -> Option<usize> {
    expected
        .iter()
        .zip(actual)
        .position(|(expected, actual)| expected != actual)
        .or_else(|| (expected.len() != actual.len()).then_some(expected.len().min(actual.len())))
}
