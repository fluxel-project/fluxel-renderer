//! Recording and command-resolution execution contracts.

use super::*;

#[test]
fn independent_frames_use_dynamic_data_and_distinct_bindings() {
    let (graph, slot) = imported_compute_graph();
    let registry = registry_with_buffers();
    let executor = executor();
    let mut first = FrameInputs::new(2);
    first.bind_buffer(slot, BufferBindingId::new(1));
    let mut second = FrameInputs::new(5);
    second.bind_buffer(slot, BufferBindingId::new(2));

    executor
        .execute(&graph, graph.instantiate_local(first), &registry, &registry)
        .unwrap();
    executor
        .execute(
            &graph,
            graph.instantiate_local(second),
            &registry,
            &registry,
        )
        .unwrap();
    let trace = executor.try_backend().unwrap().trace().to_vec();
    assert!(
        trace
            .iter()
            .any(|event| matches!(event, TestTraceEvent::Dispatch { groups: [2, 1, 1] }))
    );
    assert!(
        trace
            .iter()
            .any(|event| matches!(event, TestTraceEvent::Dispatch { groups: [5, 1, 1] }))
    );
    assert_eq!(
        trace
            .iter()
            .filter(|event| matches!(event, TestTraceEvent::Submit { .. }))
            .count(),
        2
    );
}

#[test]
fn culled_callback_never_runs_and_empty_execution_submits() {
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&calls);
    let mut graph = RenderGraph::new();
    let transient = graph.create_buffer("dead", buffer());
    graph.add_compute_pass(
        "dead",
        |pass| {
            let (out, _) = pass.write_buffer(
                transient,
                BufferWriteUse::Storage,
                BufferRange::whole(),
                WriteCoverage::Full,
            );
            (out, ())
        },
        move |_, _, _, _| {
            observed.fetch_add(1, Ordering::SeqCst);
            Ok(())
        },
    );
    let compiled = graph.compile(&capabilities()).unwrap().graph;
    let registry = TestRegistry::new(device());
    let executor = executor();
    executor
        .execute(
            &compiled,
            compiled.instantiate_local(FrameInputs::new(())),
            &registry,
            &registry,
        )
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        executor
            .try_backend()
            .unwrap()
            .trace()
            .iter()
            .filter(|event| matches!(event, TestTraceEvent::Submit { .. }))
            .count(),
        1,
        "an empty plan still submits its empty ordered command buffer"
    );
}

#[test]
fn transitions_preserve_subresources_and_partial_buffer_ranges() {
    let mut graph = RenderGraph::new();
    let image = graph.create_texture("image", texture());
    let data = graph.create_buffer("data", buffer());
    let range = TextureRange::Subresources {
        base_mip_level: 0,
        mip_level_count: 1,
        base_array_layer: 0,
        array_layer_count: 1,
        aspect: TextureAspect::Color,
    };
    let bytes = BufferRange::Bytes {
        offset: 8,
        size: 16,
    };
    let initialized = graph.add_compute_pass(
        "initialize",
        |pass| {
            let (image, _) = pass.write_texture(
                image,
                TextureWriteUse::Storage,
                TextureRange::whole(),
                WriteCoverage::Full,
            );
            let (data, _) = pass.write_buffer(
                data,
                BufferWriteUse::Storage,
                BufferRange::whole(),
                WriteCoverage::Full,
            );
            ((image, data), ())
        },
        |_, _, _, _| Ok(()),
    );
    let pass = graph.add_compute_pass(
        "partial",
        |pass| {
            pass.read_texture(&initialized.output.0, TextureReadUse::Sampled, range);
            pass.read_buffer(&initialized.output.1, BufferReadUse::Storage, bytes);
            ((), ())
        },
        |_, _, _, _| Ok(()),
    );
    graph.mark_side_effect(
        pass.id,
        SideEffectReason::Diagnostic("retain partial".into()),
    );
    graph.export_texture(
        initialized.output.0,
        ExportTextureContract {
            final_state: ResourceAccessState::ShaderSampledRead,
        },
    );
    graph.export_buffer(
        initialized.output.1,
        ExportBufferContract {
            final_state: ResourceAccessState::ShaderStorageRead,
        },
    );
    let compiled = graph.compile(&capabilities()).unwrap().graph;
    let executor = executor();
    executor
        .execute(
            &compiled,
            compiled.instantiate_local(FrameInputs::new(())),
            &TestRegistry::new(device()),
            &TestRegistry::new(device()),
        )
        .unwrap();
    let trace = executor.try_backend().unwrap().trace().to_vec();
    assert!(trace.iter().any(|event| matches!(
        event,
        TestTraceEvent::TransitionTexture { range: observed, .. } if *observed == range
    )));
    assert!(trace.iter().any(|event| matches!(
        event,
        TestTraceEvent::TransitionBuffer { range: observed, .. } if *observed == bytes
    )));
}

#[test]
fn binding_resolver_and_compute_pipeline_are_recorded() {
    let pipeline = ComputePipelineId::new(4);
    let bindings = BindingSetId::new(5);
    let mut graph = RenderGraph::new();
    let input = graph.import_buffer_slot(
        "input",
        ImportBufferContract {
            descriptor: buffer(),
            initial_state: ResourceAccessState::ShaderStorageRead,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let pass = graph.add_compute_pass(
        "bound",
        |pass| {
            let read =
                pass.read_buffer(&input.version, BufferReadUse::Storage, BufferRange::whole());
            ((), read)
        },
        move |commands, resolver, read, _| {
            commands.set_pipeline(pipeline)?;
            let resolved =
                resolver.resolve_bindings(bindings, &[BindingResource::BufferRead(read)], &[])?;
            commands.set_bindings(&resolved)?;
            commands.dispatch([1, 1, 1])
        },
    );
    graph.mark_side_effect(pass.id, SideEffectReason::Diagnostic("retain".into()));
    let compiled = graph.compile(&capabilities()).unwrap().graph;
    let mut registry = registry_with_buffers();
    registry.register_compute_pipeline(pipeline, TestComputePipeline::new(4));
    registry.register_bindings(bindings, TestBindings::new(5));
    let mut inputs = FrameInputs::new(());
    inputs.bind_buffer(input.slot, BufferBindingId::new(1));
    let executor = executor();
    executor
        .execute(
            &compiled,
            compiled.instantiate_local(inputs),
            &registry,
            &registry,
        )
        .unwrap();
    let trace = executor.try_backend().unwrap().trace().to_vec();
    assert!(
        trace
            .iter()
            .any(|event| matches!(event, TestTraceEvent::SetComputePipeline { .. }))
    );
    assert!(
        trace
            .iter()
            .any(|event| matches!(event, TestTraceEvent::SetBindings { .. }))
    );
}

#[test]
fn binding_resolution_preserves_partial_ranges_and_semantics() {
    let recipe = BindingSetId::new(62);
    let pipeline = ComputePipelineId::new(63);
    let texture_binding = TextureBindingId::new(64);
    let buffer_binding = BufferBindingId::new(65);
    let texture_range = TextureRange::Subresources {
        base_mip_level: 0,
        mip_level_count: 1,
        base_array_layer: 0,
        array_layer_count: 1,
        aspect: TextureAspect::Color,
    };
    let buffer_range = BufferRange::Bytes {
        offset: 16,
        size: 32,
    };
    let mut graph = RenderGraph::new();
    let sampled_texture = graph.import_texture_slot(
        "sampled",
        ImportTextureContract {
            descriptor: texture(),
            initial_state: ResourceAccessState::ShaderSampledRead,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let storage_buffer = graph.import_buffer_slot(
        "storage",
        ImportBufferContract {
            descriptor: buffer(),
            initial_state: ResourceAccessState::ShaderStorageRead,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let pass = graph.add_compute_pass(
        "bind",
        |pass| {
            let sampled = pass.read_texture(
                &sampled_texture.version,
                TextureReadUse::Sampled,
                texture_range,
            );
            let storage = pass.read_buffer(
                &storage_buffer.version,
                BufferReadUse::Storage,
                buffer_range,
            );
            ((), (sampled, storage))
        },
        move |commands, resolver, (sampled, storage), _| {
            commands.set_pipeline(pipeline)?;
            let bindings = resolver.resolve_bindings(
                recipe,
                &[
                    BindingResource::TextureRead(sampled),
                    BindingResource::BufferRead(storage),
                ],
                &[256],
            )?;
            commands.set_bindings(&bindings)?;
            commands.dispatch([1, 1, 1])
        },
    );
    graph.mark_side_effect(pass.id, SideEffectReason::Diagnostic("retain".into()));
    let compiled = graph.compile(&capabilities()).unwrap().graph;
    let mut registry = TestRegistry::new(device());
    registry.register_texture(
        texture_binding,
        TestTexture::new(64),
        texture(),
        TextureUsage::from_kinds([TextureUsageKind::Sampled]),
        ResourceAccessState::ShaderSampledRead,
    );
    registry.register_buffer(
        buffer_binding,
        TestBuffer::new(65),
        buffer(),
        BufferUsage::from_kinds([BufferUsageKind::StorageRead]),
        ResourceAccessState::ShaderStorageRead,
    );
    registry.register_compute_pipeline(pipeline, TestComputePipeline::new(63));
    registry.register_bindings_with_contract(
        recipe,
        TestBindings::new(62),
        vec![
            TestBindingResource::Texture {
                physical: TestTexture::new(64),
                range: texture_range,
                semantic: BindingResourceSemantic::TextureRead(TextureReadUse::Sampled),
            },
            TestBindingResource::Buffer {
                physical: TestBuffer::new(65),
                range: buffer_range,
                semantic: BindingResourceSemantic::BufferRead(BufferReadUse::Storage),
            },
        ],
        vec![256],
    );
    let mut inputs = FrameInputs::new(());
    inputs.bind_texture(sampled_texture.slot, texture_binding);
    inputs.bind_buffer(storage_buffer.slot, buffer_binding);
    executor()
        .execute(
            &compiled,
            compiled.instantiate_local(inputs),
            &registry,
            &registry,
        )
        .unwrap();
    assert_eq!(
        registry.binding_resolutions(),
        vec![fluxel_rendergraph::test_rhi::TestBindingResolution {
            recipe,
            resources: vec![
                TestBindingResource::Texture {
                    physical: TestTexture::new(64),
                    range: texture_range,
                    semantic: BindingResourceSemantic::TextureRead(TextureReadUse::Sampled),
                },
                TestBindingResource::Buffer {
                    physical: TestBuffer::new(65),
                    range: buffer_range,
                    semantic: BindingResourceSemantic::BufferRead(BufferReadUse::Storage),
                },
            ],
            dynamic_offsets: vec![256],
        }]
    );
}

#[test]
fn binding_recipe_rejects_a_resource_contract_mismatch() {
    let recipe = BindingSetId::new(72);
    let pipeline = ComputePipelineId::new(73);
    let binding = BufferBindingId::new(74);
    let mut graph = RenderGraph::new();
    let input = graph.import_buffer_slot(
        "input",
        ImportBufferContract {
            descriptor: buffer(),
            initial_state: ResourceAccessState::ShaderStorageRead,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let pass = graph.add_compute_pass(
        "bind",
        |pass| {
            let read = pass.read_buffer(
                &input.version,
                BufferReadUse::Storage,
                BufferRange::Bytes {
                    offset: 8,
                    size: 16,
                },
            );
            ((), read)
        },
        move |commands, resolver, read, _| {
            commands.set_pipeline(pipeline)?;
            resolver.resolve_bindings(recipe, &[BindingResource::BufferRead(read)], &[])?;
            Ok(())
        },
    );
    graph.mark_side_effect(pass.id, SideEffectReason::Diagnostic("retain".into()));
    let compiled = graph.compile(&capabilities()).unwrap().graph;
    let mut registry = TestRegistry::new(device());
    registry.register_buffer(
        binding,
        TestBuffer::new(74),
        buffer(),
        BufferUsage::from_kinds([BufferUsageKind::StorageRead]),
        ResourceAccessState::ShaderStorageRead,
    );
    registry.register_compute_pipeline(pipeline, TestComputePipeline::new(73));
    registry.register_bindings_with_contract(
        recipe,
        TestBindings::new(72),
        vec![TestBindingResource::Buffer {
            physical: TestBuffer::new(74),
            range: BufferRange::whole(),
            semantic: BindingResourceSemantic::BufferRead(BufferReadUse::Storage),
        }],
        vec![],
    );
    let mut inputs = FrameInputs::new(());
    inputs.bind_buffer(input.slot, binding);
    assert!(matches!(
        executor().execute(
            &compiled,
            compiled.instantiate_local(inputs),
            &registry,
            &registry,
        ),
        Err(ExecutionError::Recording(error))
            if error.kind == RecordingErrorKind::IncompatibleBindingRecipe
    ));
}

#[test]
fn copy_ranges_are_checked_and_dead_transients_are_not_created() {
    let mut graph = RenderGraph::new();
    let source = graph.create_buffer("source", buffer());
    let destination = graph.create_buffer("destination", buffer());
    let _dead = graph.create_buffer("dead", buffer());
    let write = graph.add_compute_pass(
        "write",
        |pass| {
            let (source, _) = pass.write_buffer(
                source,
                BufferWriteUse::Storage,
                BufferRange::whole(),
                WriteCoverage::Full,
            );
            let (destination, _) = pass.write_buffer(
                destination,
                BufferWriteUse::Storage,
                BufferRange::whole(),
                WriteCoverage::Full,
            );
            ((source, destination), ())
        },
        |_, _, _, _| Ok(()),
    );
    let copy = graph.add_copy_pass(
        "copy",
        |pass| {
            let read = pass.read_buffer(
                &write.output.0,
                BufferRange::Bytes {
                    offset: 8,
                    size: 16,
                },
            );
            let (out, written) = pass.write_buffer(
                write.output.1,
                BufferRange::Bytes {
                    offset: 24,
                    size: 16,
                },
                WriteCoverage::Full,
            );
            (out, (read, written))
        },
        |commands, _, handles, _| {
            commands.copy_buffer(
                &handles.0,
                &handles.1,
                BufferCopyRegion {
                    source_offset: 8,
                    destination_offset: 24,
                    size: 16,
                },
            )
        },
    );
    graph.export_buffer(
        copy.output,
        ExportBufferContract {
            final_state: ResourceAccessState::CopyDestination,
        },
    );
    let compiled = graph.compile(&capabilities()).unwrap().graph;
    let executor = executor();
    executor
        .execute(
            &compiled,
            compiled.instantiate_local(FrameInputs::new(())),
            &TestRegistry::new(device()),
            &TestRegistry::new(device()),
        )
        .unwrap();
    let trace = executor.try_backend().unwrap().trace().to_vec();
    assert_eq!(
        trace
            .iter()
            .filter(|event| matches!(event, TestTraceEvent::CreateBuffer { .. }))
            .count(),
        2
    );
    assert!(trace.iter().any(|event| matches!(
        event,
        TestTraceEvent::CopyBuffer { region, .. }
            if *region == BufferCopyRegion { source_offset: 8, destination_offset: 24, size: 16 }
    )));
}

#[test]
fn callback_and_backend_failures_do_not_submit() {
    let mut callback_graph = RenderGraph::new();
    let callback = callback_graph.add_compute_pass(
        "callback-error",
        |_| ((), ()),
        |_, _, _, _| {
            Err(RecordingError {
                kind: RecordingErrorKind::InvalidCommandArgument,
                context: DiagnosticContext {
                    passes: Vec::new(),
                    resource: None,
                    texture_slot: None,
                    buffer_slot: None,
                    detail: "intentional callback failure".into(),
                    unsupported: None,
                },
            })
        },
    );
    callback_graph.mark_side_effect(
        callback.id,
        SideEffectReason::Diagnostic("retain callback".into()),
    );
    let callback_graph = callback_graph.compile(&capabilities()).unwrap().graph;
    let registry = TestRegistry::new(device());
    let executor = executor();
    assert!(matches!(
        executor.execute(
            &callback_graph,
            callback_graph.instantiate_local(FrameInputs::new(())),
            &registry,
            &registry
        ),
        Err(ExecutionError::Recording(error)) if error.kind == RecordingErrorKind::InvalidCommandArgument
    ));
    assert_eq!(executor.try_backend().unwrap().ended_pass_count(), 1);

    let (graph, slot) = imported_compute_graph();
    let registry = registry_with_buffers();
    executor
        .try_backend()
        .unwrap()
        .fail_next(TestRhiError::new("injected"));
    let mut inputs = FrameInputs::new(1);
    inputs.bind_buffer(slot, BufferBindingId::new(1));
    assert!(matches!(
        executor.execute(
            &graph,
            graph.instantiate_local(inputs),
            &registry,
            &registry
        ),
        Err(ExecutionError::Backend(_))
    ));
    assert!(
        !executor
            .try_backend()
            .unwrap()
            .trace()
            .iter()
            .any(|event| matches!(event, TestTraceEvent::Submit { .. }))
    );
}

#[test]
fn compute_dispatch_rejects_zero_and_over_limit_dimensions_before_backend_recording() {
    for groups in [[0, 1, 1], [5, 1, 1]] {
        let mut caps = capabilities();
        caps.limits = DeviceLimits::new(4, 256).with_max_compute_workgroups_per_dimension([4; 3]);
        let mut graph = RenderGraph::new();
        let pass = graph.add_compute_pass(
            "limited-dispatch",
            |_| ((), ()),
            move |commands, _, _, _| commands.dispatch(groups),
        );
        graph.mark_side_effect(pass.id, SideEffectReason::Diagnostic("retain".into()));
        let graph = graph.compile(&caps).unwrap().graph;
        let executor = FrameExecutor::new(TestRhi::new(caps, device()));
        let registry = TestRegistry::new(device());

        assert!(matches!(
            executor.execute(
                &graph,
                graph.instantiate_local(FrameInputs::new(())),
                &registry,
                &registry,
            ),
            Err(ExecutionError::Recording(error))
                if error.kind == RecordingErrorKind::InvalidCommandArgument
        ));
        assert!(
            !executor
                .try_backend()
                .unwrap()
                .trace()
                .iter()
                .any(|event| matches!(event, TestTraceEvent::Dispatch { .. }))
        );
    }
}

#[test]
fn panicking_callback_still_closes_the_native_pass() {
    let mut graph = RenderGraph::new();
    let pass = graph.add_compute_pass(
        "panic",
        |_| ((), ()),
        |_, _, _, _| -> RecordResult { panic!("intentional callback panic") },
    );
    graph.mark_side_effect(pass.id, SideEffectReason::Diagnostic("retain".into()));
    let graph = graph.compile(&capabilities()).unwrap().graph;
    let registry = TestRegistry::new(device());
    let executor = executor();

    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = executor.execute(
            &graph,
            graph.instantiate_local(FrameInputs::new(())),
            &registry,
            &registry,
        );
    }));

    assert!(unwind.is_err());
    assert_eq!(executor.try_backend().unwrap().ended_pass_count(), 1);
}

#[test]
fn executor_reentrancy_reports_busy_instead_of_blocking() {
    let executor = Arc::new(executor());
    let callback_executor = Arc::clone(&executor);
    let mut graph = RenderGraph::new();
    let pass = graph.add_compute_pass(
        "reentrant",
        |_| ((), ()),
        move |_, _, _, _| {
            assert!(matches!(
                callback_executor.collect_retired(),
                Err(ExecutionError::ExecutorBusy)
            ));
            Ok(())
        },
    );
    graph.mark_side_effect(pass.id, SideEffectReason::Diagnostic("retain".into()));
    let graph = graph.compile(&capabilities()).unwrap().graph;
    let registry = TestRegistry::new(device());

    executor
        .execute(
            &graph,
            graph.instantiate_local(FrameInputs::new(())),
            &registry,
            &registry,
        )
        .unwrap();
}
