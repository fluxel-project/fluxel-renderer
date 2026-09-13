//! Native compute witnesses.

use super::native_common::{NoObjects, Resources, TextureCopyData, add_copy};
use super::native_raster::r01_oracle;
use super::native_raster::{ComputeCase, RasterBufferCase, add_compute};
use crate::*;
use fluxel_rendergraph::*;
struct X01ComputeData {
    pipeline: ComputePipelineId,
    bindings: BindingSetId,
    texture: TextureRead,
    output: BufferWrite,
}
struct X01CopyData {
    source: BufferRead,
    destination: BufferWrite,
}

/// Native proof for executor-private cross-frame transient reuse. The middle
/// buffer remains graph-local; imports make its results observable without
/// turning it into an uncacheable exported resource.
pub(super) fn run_t01_transient_reuse(backend: crate::Backend) {
    const SIZE: u64 = 64;
    let mut graph = RenderGraph::new();
    let source = graph.import_buffer_slot(
        "t01-source",
        ImportBufferContract {
            descriptor: BufferDesc { size: SIZE },
            initial_state: ResourceAccessState::CopyDestination,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let destination = graph.import_buffer_slot(
        "t01-destination",
        ImportBufferContract {
            descriptor: BufferDesc { size: SIZE },
            initial_state: ResourceAccessState::CopyDestination,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let transient = graph.create_buffer("t01-graph-transient", BufferDesc { size: SIZE });
    let transient = add_copy(
        &mut graph,
        "t01-copy-in",
        &source.version,
        transient,
        0,
        0,
        SIZE,
    );
    let output = add_copy(
        &mut graph,
        "t01-copy-out",
        &transient,
        destination.version,
        0,
        0,
        SIZE,
    );
    // Keep the external source's declared incoming state true on every frame.
    // The observation is unused, but its final-state root records the required
    // CopySource -> CopyDestination restoration after the copy pass.
    let _source_export = graph.export_buffer(
        source.version,
        ExportBufferContract {
            final_state: ResourceAccessState::CopyDestination,
        },
    );
    let export = graph.export_buffer(
        output,
        ExportBufferContract {
            final_state: ResourceAccessState::CopyDestination,
        },
    );
    let device = Device::open(
        backend,
        DeviceOptions {
            validation: Validation::Required,
            ..DeviceOptions::default()
        },
    )
    .unwrap();
    let backend_instance = ComputeBackend::new(device.clone());
    let compiled = graph
        .compile(backend_instance.capabilities())
        .expect("T01 graph compiles against the selected native compute device")
        .graph;
    crate::imp::clear_validation_diagnostics(&device.inner);
    let usage = BufferUsage::from_kinds([
        BufferUsageKind::CopySource,
        BufferUsageKind::CopyDestination,
    ]);
    let source_bytes: Vec<u8> = (0..SIZE as u8).map(|byte| byte ^ 0xA5).collect();
    let mut resources = Resources {
        device: device.identity(),
        buffers: HashMap::new(),
        textures: HashMap::new(),
    };
    for (id, bytes) in [
        (BufferBindingId::new(1), source_bytes.clone()),
        (BufferBindingId::new(2), vec![0xCD; SIZE as usize]),
    ] {
        let buffer = device
            .create_buffer(BufferDescriptor {
                buffer: BufferDesc { size: SIZE },
                usage,
                memory: MemoryPolicy::DeviceOnly,
            })
            .unwrap();
        let state = crate::imp::upload_buffer_for_test(
            &device.inner,
            buffer.native(),
            buffer.lease().into(),
            &bytes,
        )
        .unwrap();
        resources.buffers.insert(id, (buffer, state));
    }
    let executor = FrameExecutor::new(backend_instance);
    let execute = |resources: &Resources| {
        let mut inputs = FrameInputs::new(());
        inputs.bind_buffer(source.slot, BufferBindingId::new(1));
        inputs.bind_buffer(destination.slot, BufferBindingId::new(2));
        executor
            .execute(
                &compiled,
                compiled.instantiate_local(inputs),
                resources,
                &NoObjects,
            )
            .unwrap()
    };
    let mut first = execute(&resources);
    let first_completion = first.submission.completion().clone();
    executor
        .try_backend()
        .unwrap()
        .wait(&first_completion, std::time::Duration::from_secs(10))
        .unwrap();
    assert_eq!(
        first.submission.status().unwrap(),
        CompletionStatus::Complete
    );
    drop(first);
    let mut second = execute(&resources);
    let second_completion = second.submission.completion().clone();
    executor
        .try_backend()
        .unwrap()
        .wait(&second_completion, std::time::Duration::from_secs(10))
        .unwrap();
    assert_eq!(
        second.submission.status().unwrap(),
        CompletionStatus::Complete
    );
    let output = second.exports.buffer(export).unwrap();
    let actual = crate::imp::readback_buffer_for_test(
        &device.inner,
        output.physical.native(),
        output.lease.clone(),
        output.outgoing_state,
        SIZE,
    )
    .unwrap();
    assert_eq!(
        actual, source_bytes,
        "T01/{backend:?} two-frame byte oracle"
    );
    drop(second);
    let observations = executor
        .try_backend()
        .unwrap()
        .test_transient_observations();
    assert_eq!(observations.allocations.len(), 1);
    let identity = observations.allocations[0];
    let transitions: Vec<_> = observations
        .transitions
        .iter()
        .copied()
        .filter(|transition| transition.identity == identity)
        .collect();
    assert_eq!(transitions.len(), 4, "T01/{backend:?} two barriers/frame");
    assert_eq!(transitions[2].before, transitions[1].after);
    assert_eq!(transitions[1].after, ResourceAccessState::CopySource);
    executor.invalidate_graph(&compiled).unwrap();
    let mut third = execute(&resources);
    let third_completion = third.submission.completion().clone();
    executor
        .try_backend()
        .unwrap()
        .wait(&third_completion, std::time::Duration::from_secs(10))
        .unwrap();
    assert_eq!(
        third.submission.status().unwrap(),
        CompletionStatus::Complete
    );
    drop(third);
    let after_invalidate = executor
        .try_backend()
        .unwrap()
        .test_transient_observations();
    assert_eq!(after_invalidate.allocations.len(), 2);
    assert_ne!(
        after_invalidate.allocations[0],
        after_invalidate.allocations[1]
    );
    let diagnostics = crate::imp::validation_diagnostics(&device.inner);
    assert!(diagnostics.is_empty(), "T01/{backend:?}: {diagnostics:#?}");
    eprintln!(
        "artifact case=T01 backend={backend:?} hardware={:?}; transient_allocations={:?}; frame2_first_before={:?}; terminal={:?}; exact_readback={actual:?}; invalidate_new_identity=true; diagnostics=empty",
        device.hardware(),
        after_invalidate.allocations,
        transitions[2].before,
        transitions[1].after,
    );
}

pub(super) fn run_x01(backend: crate::Backend) {
    run_x01_case(backend, &build_x01(), "independent");
}

/// Hardware witness for the two closed RGBA8 storage-texture recipes.
/// The staging observers are deliberately outside the graph and consume the
/// graph-recorded terminal state; they cannot repair a missing transition.
pub(super) fn run_s01(backend: crate::Backend) {
    let device = Device::open(
        backend,
        crate::DeviceOptions {
            validation: crate::Validation::Required,
            ..crate::DeviceOptions::default()
        },
    )
    .unwrap();
    assert!(
        device.capabilities().rgba8_unorm_storage_write,
        "S01/{backend:?} selected support matrix requires RGBA8 storage write; hardware={:?}",
        device.hardware()
    );
    match backend {
        crate::Backend::Dx12 => {
            assert!(
                !device.capabilities().rgba8_unorm_storage_read_enabled,
                "S01/DX12 contract must not advertise the unproved typed storage-read path"
            );
            return run_s01_store_only(backend, device);
        }
        crate::Backend::Vulkan => assert!(
            device.capabilities().rgba8_unorm_storage_read_enabled,
            "S01/Vulkan selected support matrix requires RGBA8 storage read; hardware={:?}",
            device.hardware()
        ),
    }
    crate::imp::clear_validation_diagnostics(&device.inner);
    let descriptor = TextureDesc {
        dimension: TextureDimension::D2,
        extent: Extent3d {
            width: 4,
            height: 3,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
    };
    let mut graph = RenderGraph::new();
    let texture_slot = graph.import_texture_slot(
        "s01-rgba8",
        ImportTextureContract {
            descriptor,
            initial_state: ResourceAccessState::Undefined,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Undefined,
        },
    );
    let buffer_slot = graph.import_buffer_slot(
        "s01-packed",
        ImportBufferContract {
            descriptor: BufferDesc { size: 256 },
            initial_state: ResourceAccessState::CopyDestination,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let store_pipeline = ComputePipelineId::new(801);
    let store_bindings = BindingSetId::new(802);
    let store = graph.add_compute_pass(
        "s01-store",
        |pass| {
            let (out, texture) = pass.write_texture(
                texture_slot.version,
                TextureWriteUse::Storage,
                TextureRange::Whole,
                WriteCoverage::Full,
            );
            (out, (texture, store_pipeline, store_bindings))
        },
        |commands, resolver, data, _| {
            commands.set_pipeline(data.1)?;
            let bound = resolver.resolve_bindings(
                data.2,
                &[BindingResource::TextureWrite(&data.0)],
                &[],
            )?;
            commands.set_bindings(&bound)?;
            commands.dispatch([1, 1, 1])
        },
    );
    let load_pipeline = ComputePipelineId::new(803);
    let load_bindings = BindingSetId::new(804);
    let load = graph.add_compute_pass(
        "s01-load",
        |pass| {
            let source =
                pass.read_texture(&store.output, TextureReadUse::Storage, TextureRange::Whole);
            let (out, destination) = pass.read_write_buffer(
                buffer_slot.version,
                BufferReadWriteUse::Storage,
                BufferRange::Whole,
            );
            (out, (source, destination, load_pipeline, load_bindings))
        },
        |commands, resolver, data, _| {
            commands.set_pipeline(data.2)?;
            let bound = resolver.resolve_bindings(
                data.3,
                &[
                    BindingResource::TextureRead(&data.0),
                    BindingResource::BufferReadWrite(&data.1),
                ],
                &[],
            )?;
            commands.set_bindings(&bound)?;
            commands.dispatch([1, 1, 1])
        },
    );
    let texture_export = graph.export_texture(
        store.output,
        ExportTextureContract {
            final_state: ResourceAccessState::CopySource,
        },
    );
    let buffer_export = graph.export_buffer(
        load.output,
        ExportBufferContract {
            final_state: ResourceAccessState::CopySource,
        },
    );
    let backend_instance = ComputeBackend::new(device.clone());
    let compiled = graph
        .compile(backend_instance.capabilities())
        .unwrap()
        .graph;
    let texture = device
        .create_texture(TextureDescriptor {
            texture: descriptor,
            usage: TextureUsage::from_kinds([
                TextureUsageKind::StorageWrite,
                TextureUsageKind::StorageRead,
                TextureUsageKind::CopySource,
            ]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let buffer = device
        .create_buffer(BufferDescriptor {
            buffer: BufferDesc { size: 256 },
            usage: BufferUsage::from_kinds([
                BufferUsageKind::StorageRead,
                BufferUsageKind::StorageWrite,
                BufferUsageKind::CopySource,
                BufferUsageKind::CopyDestination,
            ]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let buffer_state = crate::imp::upload_buffer_for_test(
        &device.inner,
        buffer.native(),
        buffer.lease().into(),
        &vec![0; 256],
    )
    .unwrap();
    let resources = Resources {
        device: device.identity(),
        buffers: HashMap::from([(BufferBindingId::new(2), (buffer, buffer_state))]),
        textures: HashMap::from([(
            TextureBindingId::new(1),
            (texture, ResourceAccessState::Undefined),
        )]),
    };
    let mut inputs = FrameInputs::new(());
    inputs.bind_texture(texture_slot.slot, TextureBindingId::new(1));
    inputs.bind_buffer(buffer_slot.slot, BufferBindingId::new(2));
    let mut provider = ComputeObjectProvider::new(&device);
    for (id, kernel, bindings) in [
        (
            store_pipeline,
            ComputeKernel::TextureStoreRgba8,
            store_bindings,
        ),
        (
            load_pipeline,
            ComputeKernel::TextureLoadRgba8,
            load_bindings,
        ),
    ] {
        provider
            .register_pipeline(id, device.create_compute_pipeline(kernel).unwrap())
            .unwrap();
        provider.register_bindings(bindings, id).unwrap();
    }
    let executor = FrameExecutor::new(backend_instance);
    let mut frame = executor
        .execute(
            &compiled,
            compiled.instantiate_local(inputs),
            &resources,
            &provider,
        )
        .unwrap();
    let completion = frame.submission.completion().clone();
    executor
        .try_backend()
        .unwrap()
        .wait(&completion, Duration::from_secs(10))
        .unwrap();
    assert_eq!(
        frame.submission.status().unwrap(),
        CompletionStatus::Complete
    );
    let exported_texture = frame.exports.texture(texture_export).unwrap();
    let pixels = crate::imp::readback_texture_for_test(
        &device.inner,
        exported_texture.physical.native(),
        exported_texture.lease.clone(),
        exported_texture.descriptor,
        exported_texture.outgoing_state,
    )
    .unwrap()
    .tight;
    let expected_pixel = [64, 128, 191, 255];
    assert_eq!(
        pixels,
        expected_pixel.repeat(12),
        "S01/{backend:?} texture storage store exact oracle"
    );
    let exported_buffer = frame.exports.buffer(buffer_export).unwrap();
    let values = readback_exported_buffer_for_test(&device, exported_buffer).unwrap();
    let packed = 0xFFBF_8040u32.to_le_bytes();
    eprintln!(
        "S01/{backend:?} observed storage-load bytes={:?}",
        &values[..48]
    );
    assert_eq!(
        &values[..48],
        packed.repeat(12).as_slice(),
        "S01/{backend:?} texture storage load exact oracle"
    );
    assert!(
        values[48..].iter().all(|b| *b == 0),
        "S01/{backend:?} untouched storage buffer tail"
    );
    let diagnostics = crate::imp::validation_diagnostics(&device.inner);
    assert!(
        diagnostics.is_empty(),
        "S01/{backend:?} diagnostics: {diagnostics:#?}"
    );
    let commit = std::env::var("FLUXEL_TEST_COMMIT").unwrap_or_else(|_| "working-tree".into());
    eprintln!(
        "artifact case=S01 backend={backend:?} commit={commit} os={}; hardware={:?}; canonical_plan_label=S01-store-load-readback-v1; execution_plan={:?}; store={:?}; load={:?}; texture_descriptor={descriptor:?}; texture_oracle={expected_pixel:?}x12; buffer_oracle=0xFFBF8040x12; completion=Complete; diagnostics={diagnostics:?}",
        std::env::consts::OS,
        device.hardware(),
        compiled.execution_plan(),
        ComputeKernel::TextureStoreRgba8.portable_identity(),
        ComputeKernel::TextureLoadRgba8.portable_identity()
    );
}

fn run_s01_store_only(backend: crate::Backend, device: Device) {
    crate::imp::clear_validation_diagnostics(&device.inner);
    let descriptor = TextureDesc {
        dimension: TextureDimension::D2,
        extent: Extent3d {
            width: 4,
            height: 3,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
    };
    // Compile-only proof: the rejected read performs no native work.
    let backend_instance = ComputeBackend::new(device.clone());
    let mut denied = RenderGraph::new();
    let denied_slot = denied.import_texture_slot(
        "s01-denied-read",
        ImportTextureContract {
            descriptor,
            initial_state: ResourceAccessState::CopyDestination,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let denied_buffer = denied.create_buffer("s01-denied-output", BufferDesc { size: 4 });
    let denied_pass = denied.add_compute_pass(
        "s01-denied-read",
        |pass| {
            let read = pass.read_texture(
                &denied_slot.version,
                TextureReadUse::Storage,
                TextureRange::Whole,
            );
            let (out, _) = pass.write_buffer(
                denied_buffer,
                BufferWriteUse::Storage,
                BufferRange::Whole,
                WriteCoverage::Full,
            );
            (out, read)
        },
        |commands, _, _, _: &()| commands.dispatch([1, 1, 1]),
    );
    denied.export_buffer(
        denied_pass.output,
        ExportBufferContract {
            final_state: ResourceAccessState::CopySource,
        },
    );
    let denied_error = match denied.compile(backend_instance.capabilities()) {
        Err(error) => error,
        Ok(_) => panic!("DX12 storage read unexpectedly compiled"),
    };
    assert_eq!(
        denied_error.kind,
        CompileErrorKind::UnsupportedSemanticRequirement
    );
    assert!(matches!(
        denied_error
            .context
            .unsupported
            .as_deref()
            .map(|value| &value.requirement),
        Some(CapabilityRequirement::TextureState {
            format: TextureFormat::Rgba8Unorm,
            sample_count: 1,
            state: ResourceAccessState::ShaderStorageRead
        })
    ));
    let mut graph = RenderGraph::new();
    let slot = graph.import_texture_slot(
        "s01-store-rgba8",
        ImportTextureContract {
            descriptor,
            initial_state: ResourceAccessState::Undefined,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Undefined,
        },
    );
    let pipeline_id = ComputePipelineId::new(811);
    let bindings_id = BindingSetId::new(812);
    let store = graph.add_compute_pass(
        "s01-store",
        |pass| {
            let (out, texture) = pass.write_texture(
                slot.version,
                TextureWriteUse::Storage,
                TextureRange::Whole,
                WriteCoverage::Full,
            );
            (out, (texture, pipeline_id, bindings_id))
        },
        |commands, resolver, data, _| {
            commands.set_pipeline(data.1)?;
            let bound = resolver.resolve_bindings(
                data.2,
                &[BindingResource::TextureWrite(&data.0)],
                &[],
            )?;
            commands.set_bindings(&bound)?;
            commands.dispatch([1, 1, 1])
        },
    );
    let export = graph.export_texture(
        store.output,
        ExportTextureContract {
            final_state: ResourceAccessState::CopySource,
        },
    );
    let compiled = graph
        .compile(backend_instance.capabilities())
        .unwrap()
        .graph;
    let texture = device
        .create_texture(TextureDescriptor {
            texture: descriptor,
            usage: TextureUsage::from_kinds([
                TextureUsageKind::StorageWrite,
                TextureUsageKind::CopySource,
            ]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let resources = Resources {
        device: device.identity(),
        buffers: HashMap::new(),
        textures: HashMap::from([(
            TextureBindingId::new(1),
            (texture, ResourceAccessState::Undefined),
        )]),
    };
    let mut inputs = FrameInputs::new(());
    inputs.bind_texture(slot.slot, TextureBindingId::new(1));
    let mut provider = ComputeObjectProvider::new(&device);
    provider
        .register_pipeline(
            pipeline_id,
            device
                .create_compute_pipeline(ComputeKernel::TextureStoreRgba8)
                .unwrap(),
        )
        .unwrap();
    provider
        .register_bindings(bindings_id, pipeline_id)
        .unwrap();
    let executor = FrameExecutor::new(backend_instance);
    let frame = executor
        .execute(
            &compiled,
            compiled.instantiate_local(inputs),
            &resources,
            &provider,
        )
        .unwrap();
    let completion = frame.submission.completion().clone();
    executor
        .try_backend()
        .unwrap()
        .wait(&completion, Duration::from_secs(10))
        .unwrap();
    let exported = frame.exports.texture(export).unwrap();
    let pixels = crate::imp::readback_texture_for_test(
        &device.inner,
        exported.physical.native(),
        exported.lease.clone(),
        exported.descriptor,
        exported.outgoing_state,
    )
    .unwrap()
    .tight;
    assert_eq!(
        pixels,
        [64, 128, 191, 255].repeat(12),
        "S01/{backend:?} storage-store exact oracle"
    );
    let diagnostics = crate::imp::validation_diagnostics(&device.inner);
    assert!(
        diagnostics.is_empty(),
        "S01/{backend:?} diagnostics: {diagnostics:#?}"
    );
    let commit = std::env::var("FLUXEL_TEST_COMMIT").unwrap_or_else(|_| "working-tree".into());
    eprintln!(
        "artifact case=S01 backend={backend:?} commit={commit} os={}; hardware={:?}; supported_access=StorageWrite only; canonical_plan_label=S01-store-readback-v1; execution_plan={:?}; store={:?}; texture_oracle=[64,128,191,255]x12; completion=Complete; diagnostics={diagnostics:?}",
        std::env::consts::OS,
        device.hardware(),
        compiled.execution_plan(),
        ComputeKernel::TextureStoreRgba8.portable_identity()
    );
}

pub(super) fn build_x01() -> RasterBufferCase {
    let texture = TextureDesc {
        dimension: fluxel_rendergraph::TextureDimension::D2,
        extent: Extent3d {
            width: 8,
            height: 8,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
    };
    let bytes = BufferDesc { size: 256 };
    let raster_id = RasterPipelineId::new(403);
    let compute_id = ComputePipelineId::new(404);
    let bindings_id = BindingSetId::new(405);
    let mut graph = RenderGraph::new();
    let image = graph.create_texture("x01-raster-target", texture);
    let storage = graph.create_buffer("x01-packed", bytes);
    let copied = graph.create_buffer("x01-copy-destination", bytes);
    let raster = graph.add_raster_pass(
        "x01-raster",
        |pass| {
            let out = pass.color_attachment(
                image,
                ColorAttachmentDesc {
                    index: 0,
                    range: TextureRange::Whole,
                    operations: AttachmentOps {
                        load: LoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
                        store: StoreOp::Store,
                        write_coverage: WriteCoverage::Full,
                    },
                },
            );
            (out, raster_id)
        },
        |commands, _, pipeline, _| {
            commands.set_pipeline(*pipeline)?;
            commands.draw(0..3, 0..1)
        },
    );
    let compute = graph.add_compute_pass(
        "x01-pack",
        |pass| {
            let sampled = pass.read_texture(
                &raster.output,
                fluxel_rendergraph::TextureReadUse::Sampled,
                TextureRange::Whole,
            );
            let (out, storage) = pass.write_buffer(
                storage,
                fluxel_rendergraph::BufferWriteUse::Storage,
                BufferRange::Whole,
                WriteCoverage::Full,
            );
            (
                out,
                X01ComputeData {
                    pipeline: compute_id,
                    bindings: bindings_id,
                    texture: sampled,
                    output: storage,
                },
            )
        },
        |commands, resolver, data, _| {
            commands.set_pipeline(data.pipeline)?;
            let bindings = resolver.resolve_bindings(
                data.bindings,
                &[
                    BindingResource::TextureRead(&data.texture),
                    BindingResource::BufferWrite(&data.output),
                ],
                &[],
            )?;
            commands.set_bindings(&bindings)?;
            commands.dispatch([1, 1, 1])
        },
    );
    let copy = graph.add_copy_pass(
        "x01-copy",
        |pass| {
            let source = pass.read_buffer(&compute.output, BufferRange::Whole);
            let (out, destination) =
                pass.write_buffer(copied, BufferRange::Whole, WriteCoverage::Full);
            (
                out,
                X01CopyData {
                    source,
                    destination,
                },
            )
        },
        |commands, _, data, _| {
            commands.copy_buffer(
                &data.source,
                &data.destination,
                BufferCopyRegion {
                    source_offset: 0,
                    destination_offset: 0,
                    size: 256,
                },
            )
        },
    );
    let export = graph.export_buffer(
        copy.output,
        ExportBufferContract {
            final_state: ResourceAccessState::CopySource,
        },
    );
    let compiled = graph
        .compile(&RasterBackend::portable_capabilities())
        .unwrap()
        .graph;
    let expected: Vec<u8> = r01_oracle()
        .chunks_exact(4)
        .flat_map(|pixel| u32::from_le_bytes(pixel.try_into().unwrap()).to_le_bytes())
        .collect();
    RasterBufferCase {
        compiled,
        export,
        raster_pipeline: raster_id,
        compute_pipeline: compute_id,
        bindings: bindings_id,
        expected,
    }
}

pub(super) fn run_x01_case(backend: crate::Backend, case: &RasterBufferCase, mode: &str) {
    let device = Device::open(
        backend,
        crate::DeviceOptions {
            validation: crate::Validation::Required,
            ..crate::DeviceOptions::default()
        },
    )
    .unwrap();
    crate::imp::clear_validation_diagnostics(&device.inner);
    let mut provider = RasterObjectProvider::new(&device);
    provider
        .register_raster_pipeline(
            case.raster_pipeline,
            device
                .create_raster_pipeline(crate::RasterKernel::Triangle)
                .unwrap(),
        )
        .unwrap();
    provider
        .register_compute_pipeline(
            case.compute_pipeline,
            device
                .create_compute_pipeline(crate::ComputeKernel::TexturePackRgba8)
                .unwrap(),
        )
        .unwrap();
    provider
        .register_bindings(case.bindings, case.compute_pipeline)
        .unwrap();
    let resources = Resources {
        device: device.identity(),
        buffers: HashMap::new(),
        textures: HashMap::new(),
    };
    let executor = fluxel_rendergraph::FrameExecutor::new(RasterBackend::new(device.clone()));
    let mut frame = executor
        .execute(
            &case.compiled,
            case.compiled.instantiate_local(FrameInputs::new(())),
            &resources,
            &provider,
        )
        .unwrap();
    let completion = frame.submission.completion().clone();
    executor
        .try_backend()
        .unwrap()
        .wait(&completion, Duration::from_secs(10))
        .unwrap();
    assert_eq!(
        frame.submission.status().unwrap(),
        CompletionStatus::Complete
    );
    let exported = frame.exports.buffer(case.export).unwrap();
    let outgoing_state = exported.outgoing_state;
    let descriptor = exported.descriptor;
    let actual = readback_raster_exported_buffer_for_test(&device, exported).unwrap();
    assert_eq!(
        actual,
        case.expected,
        "{backend:?} first difference: {:?}",
        actual.iter().zip(&case.expected).position(|(a, b)| a != b)
    );
    let diagnostics = crate::imp::validation_diagnostics(&device.inner);
    assert!(
        diagnostics.is_empty(),
        "X01/{backend:?} diagnostics: {diagnostics:#?}"
    );
    let commit = std::env::var("FLUXEL_TEST_COMMIT").unwrap_or_else(|_| "working-tree".into());
    let first_difference = actual
        .iter()
        .zip(&case.expected)
        .position(|(left, right)| left != right);
    eprintln!(
        "artifact case=X01 mode={mode} backend={backend:?} commit={commit} os={}; hardware={:?}; canonical_plan_label=X01-raster-compute-copy-v1; execution_plan={:?}; raster_artifact={:?}; compute_artifact={:?}; binding_layout=sampled rgba8 texture + one storage buffer; dispatch=[1,1,1]; input=Raster triangle 8x8 -> textureLoad row-major rgba8 pack -> buffer copy; resource=buffer descriptor={descriptor:?}; expected={:?}; actual={actual:?}; first_difference={first_difference:?}; outgoing_state={outgoing_state:?}; completion=Complete; diagnostics={diagnostics:?}",
        std::env::consts::OS,
        device.hardware(),
        case.compiled.execution_plan(),
        crate::RasterKernel::Triangle.portable_identity(),
        crate::ComputeKernel::TexturePackRgba8.portable_identity(),
        case.expected,
    );
}

pub(super) fn build_k01() -> ComputeCase {
    let (mut graph, input_version, input_slot) = compute_input_graph("k01-input");
    let pipeline = ComputePipelineId::new(101);
    let bindings = BindingSetId::new(201);
    let output = add_compute(
        &mut graph,
        "k01-wrapping-add",
        input_version,
        pipeline,
        bindings,
        compute_range(),
        [1, 1, 1],
    );
    let export = graph.export_buffer(
        output,
        ExportBufferContract {
            final_state: ResourceAccessState::ShaderStorageReadWrite,
        },
    );
    let compiled = graph
        .compile(&ComputeBackend::portable_capabilities())
        .unwrap()
        .graph;
    let input = compute_input_bytes();
    let mut expected = input.clone();
    for bytes in expected[..256].chunks_exact_mut(4) {
        let value = u32::from_le_bytes(bytes.try_into().unwrap()).wrapping_add(1);
        bytes.copy_from_slice(&value.to_le_bytes());
    }
    ComputeCase {
        compiled,
        export,
        slot: input_slot,
        input,
        expected,
        artifacts: vec![(pipeline, crate::ComputeKernel::WrappingAdd, bindings)],
        input_description: "one RW storage buffer bytes[0..256]; wrapping add; bytes[256..320] sentinel=0xCD",
    }
}

pub(super) fn run_k02(backend: crate::Backend) {
    let case = build_k02();
    run_compute_case_bundle("K02", backend, &case);
}

pub(super) fn build_k02() -> ComputeCase {
    let (mut graph, input_version, input_slot) = compute_input_graph("k02-input");
    let add_pipeline = ComputePipelineId::new(102);
    let multiply_pipeline = ComputePipelineId::new(103);
    let add_bindings = BindingSetId::new(202);
    let multiply_bindings = BindingSetId::new(203);
    let after_add = add_compute(
        &mut graph,
        "k02-wrapping-add",
        input_version,
        add_pipeline,
        add_bindings,
        compute_range(),
        [1, 1, 1],
    );
    let output = add_compute(
        &mut graph,
        "k02-wrapping-multiply",
        after_add,
        multiply_pipeline,
        multiply_bindings,
        compute_range(),
        [1, 1, 1],
    );
    let export = graph.export_buffer(
        output,
        ExportBufferContract {
            final_state: ResourceAccessState::ShaderStorageReadWrite,
        },
    );
    let compiled = graph
        .compile(&ComputeBackend::portable_capabilities())
        .unwrap()
        .graph;
    let input = compute_input_bytes();
    let mut expected = input.clone();
    for bytes in expected[..256].chunks_exact_mut(4) {
        let value = u32::from_le_bytes(bytes.try_into().unwrap());
        bytes.copy_from_slice(&value.wrapping_add(1).wrapping_mul(3).to_le_bytes());
    }
    ComputeCase {
        compiled,
        export,
        slot: input_slot,
        input,
        expected,
        artifacts: vec![
            (
                add_pipeline,
                crate::ComputeKernel::WrappingAdd,
                add_bindings,
            ),
            (
                multiply_pipeline,
                crate::ComputeKernel::WrappingMultiply,
                multiply_bindings,
            ),
        ],
        input_description: "two ordered RW storage passes bytes[0..256]: wrapping add then wrapping multiply; bytes[256..320] sentinel=0xCD",
    }
}

pub(super) fn run_compute_case_bundle(case: &str, backend: crate::Backend, bundle: &ComputeCase) {
    run_compute_case(
        case,
        backend,
        &bundle.compiled,
        bundle.export,
        bundle.slot,
        bundle.input.clone(),
        &bundle.expected,
        &bundle.artifacts,
        bundle.input_description,
    );
}

fn compute_input_graph(
    name: &str,
) -> (
    RenderGraph,
    fluxel_rendergraph::BufferVersion,
    fluxel_rendergraph::ImportBufferSlot,
) {
    let mut graph = RenderGraph::new();
    let slot = graph.import_buffer_slot(
        name,
        ImportBufferContract {
            descriptor: BufferDesc { size: 320 },
            initial_state: ResourceAccessState::CopyDestination,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    (graph, slot.version, slot.slot)
}

fn compute_range() -> BufferRange {
    BufferRange::Bytes {
        offset: 0,
        size: 256,
    }
}

fn compute_input_bytes() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(320);
    for index in 0..64_u32 {
        bytes.extend_from_slice(&(u32::MAX - index * 17).to_le_bytes());
    }
    bytes.extend(std::iter::repeat_n(0xCD, 64));
    bytes
}

#[allow(
    clippy::too_many_arguments,
    reason = "hardware artifact fields are intentionally explicit"
)]
fn run_compute_case(
    case: &str,
    backend: crate::Backend,
    compiled: &fluxel_rendergraph::CompiledGraph,
    export: ExportBufferSlot,
    slot: fluxel_rendergraph::ImportBufferSlot,
    input: Vec<u8>,
    expected: &[u8],
    artifacts: &[(ComputePipelineId, crate::ComputeKernel, BindingSetId)],
    input_description: &str,
) {
    let device = Device::open(
        backend,
        crate::DeviceOptions {
            validation: crate::Validation::Required,
            ..crate::DeviceOptions::default()
        },
    )
    .unwrap();
    crate::imp::clear_validation_diagnostics(&device.inner);
    let usage = BufferUsage::from_kinds([
        fluxel_rendergraph::BufferUsageKind::CopySource,
        fluxel_rendergraph::BufferUsageKind::CopyDestination,
        fluxel_rendergraph::BufferUsageKind::StorageRead,
        fluxel_rendergraph::BufferUsageKind::StorageWrite,
    ]);
    let buffer = device
        .create_buffer(BufferDescriptor {
            buffer: BufferDesc {
                size: input.len() as u64,
            },
            usage,
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let initial_state = crate::imp::upload_buffer_for_test(
        &device.inner,
        buffer.native(),
        buffer.lease().into(),
        &input,
    )
    .unwrap();
    let mut resources = Resources {
        device: device.identity(),
        buffers: HashMap::from([(BufferBindingId::new(1), (buffer, initial_state))]),
        textures: HashMap::new(),
    };
    let mut frame_inputs = FrameInputs::new(());
    frame_inputs.bind_buffer(slot, BufferBindingId::new(1));
    let mut provider = ComputeObjectProvider::new(&device);
    for (pipeline_id, kernel, bindings) in artifacts {
        let pipeline = device.create_compute_pipeline(*kernel).unwrap();
        provider.register_pipeline(*pipeline_id, pipeline).unwrap();
        provider.register_bindings(*bindings, *pipeline_id).unwrap();
    }
    let executor = fluxel_rendergraph::FrameExecutor::new(ComputeBackend::new(device.clone()));
    let mut frame = executor
        .execute(
            compiled,
            compiled.instantiate_local(frame_inputs),
            &resources,
            &provider,
        )
        .unwrap();
    let completion = frame.submission.completion().clone();
    executor
        .try_backend()
        .unwrap()
        .wait(&completion, Duration::from_secs(10))
        .unwrap();
    assert_eq!(
        frame.submission.status().unwrap(),
        CompletionStatus::Complete
    );
    let exported = frame.exports.buffer(export).unwrap();
    assert_eq!(
        exported.outgoing_state,
        ResourceAccessState::ShaderStorageReadWrite
    );
    let actual = readback_exported_buffer_for_test(&device, exported).unwrap();
    assert_eq!(
        actual,
        expected,
        "{backend:?} first difference: {:?}",
        actual
            .iter()
            .zip(expected)
            .position(|(left, right)| left != right)
    );
    let diagnostics = crate::imp::validation_diagnostics(&device.inner);
    assert!(
        diagnostics.is_empty(),
        "{case}/{backend:?} validation diagnostics: {diagnostics:#?}"
    );
    let commit = std::env::var("FLUXEL_TEST_COMMIT").unwrap_or_else(|_| "working-tree".into());
    let artifacts: Vec<_> = artifacts
        .iter()
        .map(|(_, kernel, _)| kernel.portable_identity())
        .collect();
    let plan_summary = format!("{:?}", compiled.execution_plan());
    eprintln!(
        "artifact case={case} backend={backend:?} commit={commit} os={}; hardware={:?}; canonical_plan_label={case}-compute-v1; execution_plan={plan_summary}; shader={artifacts:?}; binding_layout=one RW storage buffer range=0..256; dispatch=[1,1,1]; input={input_description}; expected={expected:?}; actual={actual:?}; first_difference={:?}; outgoing_state={:?}; completion=Complete; diagnostics={diagnostics:?}",
        std::env::consts::OS,
        device.hardware(),
        actual
            .iter()
            .zip(expected)
            .position(|(left, right)| left != right),
        exported.outgoing_state,
    );
    // Keep the provider/resource binding path explicit in this fixture.
    resources.buffers.clear();
}

pub(super) fn run_buffer_case(
    case: &str,
    backend: crate::Backend,
    compiled: &fluxel_rendergraph::CompiledGraph,
    export: ExportBufferSlot,
    inputs: &[(fluxel_rendergraph::ImportBufferSlot, Vec<u8>)],
    expected: &[u8],
) {
    let device = Device::open(
        backend,
        crate::DeviceOptions {
            validation: crate::Validation::Required,
            ..crate::DeviceOptions::default()
        },
    )
    .unwrap();
    crate::imp::clear_validation_diagnostics(&device.inner);
    let usage = BufferUsage::from_kinds([
        fluxel_rendergraph::BufferUsageKind::CopySource,
        fluxel_rendergraph::BufferUsageKind::CopyDestination,
    ]);
    let mut resources = Resources {
        device: device.identity(),
        buffers: HashMap::new(),
        textures: HashMap::new(),
    };
    let mut frame_inputs = FrameInputs::new(());
    for (index, (slot, bytes)) in inputs.iter().enumerate() {
        let buffer = device
            .create_buffer(BufferDescriptor {
                buffer: BufferDesc {
                    size: bytes.len() as u64,
                },
                usage,
                memory: MemoryPolicy::DeviceOnly,
            })
            .unwrap();
        let state = crate::imp::upload_buffer_for_test(
            &device.inner,
            buffer.native(),
            buffer.lease().into(),
            bytes,
        )
        .unwrap();
        let id = BufferBindingId::new(index as u64 + 1);
        resources.buffers.insert(id, (buffer, state));
        frame_inputs.bind_buffer(*slot, id);
    }
    let executor = fluxel_rendergraph::FrameExecutor::new(CopyBackend::new(device.clone()));
    let mut frame = executor
        .execute(
            compiled,
            compiled.instantiate_local(frame_inputs),
            &resources,
            &NoObjects,
        )
        .unwrap();
    let completion = frame.submission.completion().clone();
    executor
        .try_backend()
        .unwrap()
        .wait(&completion, Duration::from_secs(10))
        .unwrap();
    assert_eq!(
        frame.submission.status().unwrap(),
        CompletionStatus::Complete
    );
    let exported = frame.exports.buffer(export).unwrap();
    let actual = crate::imp::readback_buffer_for_test(
        &device.inner,
        exported.physical.native(),
        exported.lease.clone(),
        exported.outgoing_state,
        exported.descriptor.size,
    )
    .unwrap();
    assert_eq!(
        actual,
        expected,
        "{backend:?} first difference: {:?}",
        actual.iter().zip(expected).position(|(a, b)| a != b)
    );
    let diagnostics = crate::imp::validation_diagnostics(&device.inner);
    assert!(
        diagnostics.is_empty(),
        "{case}/{backend:?} validation diagnostics: {diagnostics:#?}"
    );
    let commit = std::env::var("FLUXEL_TEST_COMMIT").unwrap_or_else(|_| "working-tree".into());
    let input = match case {
        "C01" => "src[8..32] -> dst[16..40], dst sentinel=0xCD",
        "C03" => "srcA[0..32] -> dst[0..32], then srcB[8..32] -> dst[16..40]",
        _ => "unknown",
    };
    eprintln!(
        "artifact case={case} backend={backend:?} commit={commit} os={}; hardware={:?}; shader=N/A; input={input}; resource=buffer size={}; expected={expected:?}; actual={actual:?}; first_difference={:?}; completion=Complete; diagnostics={diagnostics:?}",
        std::env::consts::OS,
        device.hardware(),
        exported.descriptor.size,
        actual.iter().zip(expected).position(|(a, b)| a != b)
    );
}

pub(super) fn run_c02(backend: crate::Backend) {
    let source_desc = TextureDesc {
        dimension: fluxel_rendergraph::TextureDimension::D2,
        extent: fluxel_rendergraph::Extent3d {
            width: 7,
            height: 5,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
    };
    let destination_desc = TextureDesc {
        extent: fluxel_rendergraph::Extent3d {
            width: 11,
            height: 9,
            depth: 1,
        },
        ..source_desc
    };
    let region = TextureCopyRegion {
        source_origin: [1, 1, 0],
        destination_origin: [3, 2, 0],
        extent: [5, 3, 1],
        source_mip_level: 0,
        destination_mip_level: 0,
    };
    let mut graph = RenderGraph::new();
    let source_slot = graph.import_texture_slot(
        "c02-source",
        fluxel_rendergraph::ImportTextureContract {
            descriptor: source_desc,
            initial_state: ResourceAccessState::CopyDestination,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let destination_slot = graph.import_texture_slot(
        "c02-destination",
        fluxel_rendergraph::ImportTextureContract {
            descriptor: destination_desc,
            initial_state: ResourceAccessState::CopyDestination,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let copied = graph.add_copy_pass(
        "c02-copy",
        |pass| {
            let source = pass.read_texture(&source_slot.version, TextureRange::Whole);
            let (output, destination) = pass.write_texture(
                destination_slot.version,
                TextureRange::Whole,
                WriteCoverage::Unknown,
            );
            (
                output,
                TextureCopyData {
                    source,
                    destination,
                    region,
                },
            )
        },
        |commands, _, data, _| commands.copy_texture(&data.source, &data.destination, data.region),
    );
    let export = graph.export_texture(
        copied.output,
        fluxel_rendergraph::ExportTextureContract {
            final_state: ResourceAccessState::CopyDestination,
        },
    );
    let compiled = graph
        .compile(&CopyBackend::portable_capabilities())
        .unwrap()
        .graph;
    let mut source = Vec::new();
    for y in 0..source_desc.extent.height {
        for x in 0..source_desc.extent.width {
            source.extend_from_slice(&[x as u8, y as u8, (x + y) as u8, 0xFF]);
        }
    }
    let destination =
        vec![0xCD; (destination_desc.extent.width * destination_desc.extent.height * 4) as usize];
    let mut expected = destination.clone();
    for y in 0..region.extent[1] as usize {
        let src_start = ((region.source_origin[1] as usize + y)
            * source_desc.extent.width as usize
            + region.source_origin[0] as usize)
            * 4;
        let dst_start = ((region.destination_origin[1] as usize + y)
            * destination_desc.extent.width as usize
            + region.destination_origin[0] as usize)
            * 4;
        let count = region.extent[0] as usize * 4;
        expected[dst_start..dst_start + count]
            .copy_from_slice(&source[src_start..src_start + count]);
    }
    run_texture_case(
        backend,
        &compiled,
        export,
        (source_slot.slot, source, source_desc),
        (destination_slot.slot, destination, destination_desc),
        &expected,
    );
}

fn run_texture_case(
    backend: crate::Backend,
    compiled: &fluxel_rendergraph::CompiledGraph,
    export: fluxel_rendergraph::ExportTextureSlot,
    source: (fluxel_rendergraph::ImportTextureSlot, Vec<u8>, TextureDesc),
    destination: (fluxel_rendergraph::ImportTextureSlot, Vec<u8>, TextureDesc),
    expected: &[u8],
) {
    let device = Device::open(
        backend,
        crate::DeviceOptions {
            validation: crate::Validation::Required,
            ..crate::DeviceOptions::default()
        },
    )
    .unwrap();
    crate::imp::clear_validation_diagnostics(&device.inner);
    let usage = TextureUsage::from_kinds([
        fluxel_rendergraph::TextureUsageKind::CopySource,
        fluxel_rendergraph::TextureUsageKind::CopyDestination,
    ]);
    let mut resources = Resources {
        device: device.identity(),
        buffers: HashMap::new(),
        textures: HashMap::new(),
    };
    let mut frame_inputs = FrameInputs::new(());
    for (index, (slot, bytes, descriptor)) in [source, destination].into_iter().enumerate() {
        let texture = device
            .create_texture(TextureDescriptor {
                texture: descriptor,
                usage,
                memory: MemoryPolicy::DeviceOnly,
            })
            .unwrap();
        let state = crate::imp::upload_texture_for_test(
            &device.inner,
            texture.native(),
            texture.lease().into(),
            descriptor,
            &bytes,
        )
        .unwrap();
        let id = TextureBindingId::new(index as u64 + 1);
        resources.textures.insert(id, (texture, state));
        frame_inputs.bind_texture(slot, id);
    }
    let executor = fluxel_rendergraph::FrameExecutor::new(CopyBackend::new(device.clone()));
    let mut frame = executor
        .execute(
            compiled,
            compiled.instantiate_local(frame_inputs),
            &resources,
            &NoObjects,
        )
        .unwrap();
    let completion = frame.submission.completion().clone();
    executor
        .try_backend()
        .unwrap()
        .wait(&completion, Duration::from_secs(10))
        .unwrap();
    assert_eq!(
        frame.submission.status().unwrap(),
        CompletionStatus::Complete
    );
    let exported = frame.exports.texture(export).unwrap();
    let readback = crate::imp::readback_texture_for_test(
        &device.inner,
        exported.physical.native(),
        exported.lease.clone(),
        exported.descriptor,
        exported.outgoing_state,
    )
    .unwrap();
    assert_eq!(
        readback.tight,
        expected,
        "{backend:?} first difference: {:?}",
        readback
            .tight
            .iter()
            .zip(expected)
            .position(|(a, b)| a != b)
    );
    let row_bytes = exported.descriptor.extent.width as usize * 4;
    for row in 0..exported.descriptor.extent.height as usize {
        assert!(
            readback.padded[row * readback.bytes_per_row as usize + row_bytes
                ..(row + 1) * readback.bytes_per_row as usize]
                .iter()
                .all(|byte| *byte == 0),
            "{backend:?} row padding modified"
        );
    }
    let diagnostics = crate::imp::validation_diagnostics(&device.inner);
    assert!(
        diagnostics.is_empty(),
        "C02/{backend:?} validation diagnostics: {diagnostics:#?}"
    );
    let commit = std::env::var("FLUXEL_TEST_COMMIT").unwrap_or_else(|_| "working-tree".into());
    eprintln!(
        "artifact case=C02 backend={backend:?} commit={commit} os={}; hardware={:?}; shader=N/A; input=src 7x5 origin(1,1) -> dst 11x9 origin(3,2), extent 5x3; resource=texture descriptor={:?} row_pitch={} padding=zero; expected={expected:?}; actual={:?}; first_difference={:?}; completion=Complete; diagnostics={diagnostics:?}",
        std::env::consts::OS,
        device.hardware(),
        exported.descriptor,
        readback.bytes_per_row,
        readback.tight,
        readback
            .tight
            .iter()
            .zip(expected)
            .position(|(a, b)| a != b)
    );
}

pub(super) fn hardware_guard() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}
