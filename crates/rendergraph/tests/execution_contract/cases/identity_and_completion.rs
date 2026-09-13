//! Physical-identity and completion contract cases.

use super::*;

#[test]
fn texture_and_buffer_physical_identities_use_separate_namespaces() {
    let mut graph = RenderGraph::new();
    let imported_texture = graph.import_texture_slot(
        "texture",
        ImportTextureContract {
            descriptor: texture(),
            initial_state: ResourceAccessState::ShaderSampledRead,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let imported_buffer = graph.import_buffer_slot(
        "buffer",
        ImportBufferContract {
            descriptor: buffer(),
            initial_state: ResourceAccessState::ShaderStorageRead,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let pass = graph.add_compute_pass(
        "read-both-kinds",
        |pass| {
            pass.read_texture(
                &imported_texture.version,
                TextureReadUse::Sampled,
                TextureRange::whole(),
            );
            pass.read_buffer(
                &imported_buffer.version,
                BufferReadUse::Storage,
                BufferRange::whole(),
            );
            ((), ())
        },
        |_, _, _, _| Ok(()),
    );
    graph.mark_side_effect(pass.id, SideEffectReason::Diagnostic("retain".into()));
    let graph = graph.compile(&capabilities()).unwrap().graph;
    let texture_id = TextureBindingId::new(31);
    let buffer_id = BufferBindingId::new(32);
    let mut registry = TestRegistry::new(device());
    registry.register_texture(
        texture_id,
        fluxel_rendergraph::test_rhi::TestTexture::new(1),
        texture(),
        TextureUsage::from_kinds([TextureUsageKind::Sampled]),
        ResourceAccessState::ShaderSampledRead,
    );
    registry.register_buffer(
        buffer_id,
        TestBuffer::new(1),
        buffer(),
        BufferUsage::from_kinds([BufferUsageKind::StorageRead]),
        ResourceAccessState::ShaderStorageRead,
    );
    let mut inputs = FrameInputs::new(());
    inputs.bind_texture(imported_texture.slot, texture_id);
    inputs.bind_buffer(imported_buffer.slot, buffer_id);

    executor()
        .execute(
            &graph,
            graph.instantiate_local(inputs),
            &registry,
            &registry,
        )
        .unwrap();
}

#[test]
fn resolver_rejects_a_handle_declared_by_a_different_pass() {
    let mut graph = RenderGraph::new();
    let imported = graph.import_buffer_slot(
        "input",
        ImportBufferContract {
            descriptor: buffer(),
            initial_state: ResourceAccessState::ShaderStorageRead,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let foreign = graph.add_compute_pass(
        "source",
        |pass| {
            let read = pass.read_buffer(
                &imported.version,
                BufferReadUse::Storage,
                BufferRange::whole(),
            );
            (read, ())
        },
        |_, _, _, _| Ok(()),
    );
    let retained = graph.add_compute_pass(
        "consumer",
        |_| ((), foreign.output),
        |_, resolver, foreign, _| {
            resolver.resolve_bindings(
                BindingSetId::new(91),
                &[BindingResource::BufferRead(foreign)],
                &[],
            )?;
            Ok(())
        },
    );
    graph.mark_side_effect(retained.id, SideEffectReason::Diagnostic("retain".into()));
    let compiled = graph.compile(&capabilities()).unwrap().graph;
    let registry = TestRegistry::new(device());
    assert!(matches!(
        executor().execute(
            &compiled,
            compiled.instantiate_local(FrameInputs::new(())),
            &registry,
            &registry,
        ),
        Err(ExecutionError::Recording(error))
            if error.kind == RecordingErrorKind::ForeignOrUndeclaredPassAccess
    ));
}
#[test]
fn two_pending_completions_are_independent_and_local_send_instantiate() {
    let (graph, slot) = imported_compute_graph();
    let registry = registry_with_buffers();
    let executor = executor();
    let mut first_input = FrameInputs::new(1);
    first_input.bind_buffer(slot, BufferBindingId::new(1));
    let mut second_input = FrameInputs::new(1);
    second_input.bind_buffer(slot, BufferBindingId::new(2));
    let mut first = executor
        .execute(
            &graph,
            graph.instantiate_local(first_input),
            &registry,
            &registry,
        )
        .unwrap();
    let mut second = executor
        .execute(
            &graph,
            graph.instantiate_send(second_input),
            &registry,
            &registry,
        )
        .unwrap();
    let completion: TestCompletion = *first.submission.completion();
    assert_eq!(
        first.submission.status().unwrap(),
        CompletionStatus::Pending
    );
    assert_eq!(
        second.submission.status().unwrap(),
        CompletionStatus::Pending
    );
    executor.try_backend().unwrap().complete(completion);
    assert_eq!(
        first.submission.status().unwrap(),
        CompletionStatus::Complete
    );
    assert_eq!(
        second.submission.status().unwrap(),
        CompletionStatus::Pending
    );
}

#[test]
fn wrong_graph_and_surface_binding_are_rejected_before_recording() {
    let (first, _) = imported_compute_graph();
    let (second, slot) = imported_compute_graph();
    let registry = registry_with_buffers();
    let mut inputs = FrameInputs::new(1);
    inputs.bind_buffer(slot, BufferBindingId::new(1));
    assert!(matches!(
        executor().execute(
            &first,
            second.instantiate_local(inputs),
            &registry,
            &registry
        ),
        Err(ExecutionError::WrongCompiledGraph)
    ));

    let mut caps = capabilities();
    caps.queues[0].capabilities.present = true;
    caps.surface = Some(SurfaceCapabilities::new(
        vec![TextureFormat::Rgba8Unorm],
        true,
        false,
    ));
    let mut surface = RenderGraph::new();
    let image = surface.import_surface_texture_slot(
        "surface",
        SurfaceTextureContract {
            descriptor: texture(),
        },
    );
    let pass = surface.add_raster_pass(
        "surface",
        |pass| {
            let out = pass.color_attachment(
                image.version,
                ColorAttachmentDesc {
                    index: 0,
                    range: TextureRange::whole(),
                    operations: AttachmentOps {
                        load: LoadOp::Clear([0.0; 4]),
                        store: StoreOp::Store,
                        write_coverage: WriteCoverage::Full,
                    },
                },
            );
            (out, ())
        },
        |_, _, _, _| Ok(()),
    );
    surface.present(pass.output, PresentContract::new());
    let compiled = surface.compile(&caps).unwrap().graph;
    let executor = FrameExecutor::new(TestRhi::new(caps, device()));
    assert!(matches!(
        executor.execute(
            &compiled,
            compiled.instantiate_local(FrameInputs::new(())),
            &TestRegistry::new(device()),
            &TestRegistry::new(device())
        ),
        Err(ExecutionError::FrameBinding(error))
            if error.kind == FrameBindingErrorKind::MissingSurface
    ));
}

#[test]
fn acquired_surface_image_transitions_and_presents_with_its_root() {
    let mut caps = capabilities();
    caps.queues[0].capabilities.present = true;
    caps.surface = Some(SurfaceCapabilities::new(
        vec![TextureFormat::Rgba8Unorm],
        true,
        false,
    ));
    let mut graph = RenderGraph::<bool>::new();
    let image = graph.import_surface_texture_slot(
        "surface",
        SurfaceTextureContract {
            descriptor: texture(),
        },
    );
    let pass = graph.add_raster_pass(
        "surface",
        |pass| {
            let out = pass.color_attachment(
                image.version,
                ColorAttachmentDesc {
                    index: 0,
                    range: TextureRange::whole(),
                    operations: AttachmentOps {
                        load: LoadOp::Clear([0.0; 4]),
                        store: StoreOp::Store,
                        write_coverage: WriteCoverage::Full,
                    },
                },
            );
            (out, ())
        },
        |_, _, _, abort| {
            if *abort {
                Err(RecordingError {
                    kind: RecordingErrorKind::InvalidCommandArgument,
                    context: DiagnosticContext {
                        passes: Vec::new(),
                        resource: None,
                        texture_slot: None,
                        buffer_slot: None,
                        detail: "test surface callback failure".into(),
                        unsupported: None,
                    },
                })
            } else {
                Ok(())
            }
        },
    );
    graph.present(pass.output, PresentContract::new());
    let compiled = graph.compile(&caps).unwrap().graph;
    let surface_binding = SurfaceBindingId::new(901);
    let mut registry = TestRegistry::new(device());
    let (presented, presented_probe) = TestPresentationToken::fresh(44);
    registry.register_surface(
        surface_binding,
        TestTexture::new(900),
        texture(),
        TextureUsage::from_kinds([TextureUsageKind::ColorAttachment, TextureUsageKind::Present]),
        ResourceAccessState::Present,
        presented,
    );
    let mut inputs = FrameInputs::new(false);
    inputs.bind_surface(image.slot, surface_binding);
    let executor = FrameExecutor::new(TestRhi::new(caps, device()));
    executor
        .execute(
            &compiled,
            compiled.instantiate_local(inputs),
            &registry,
            &registry,
        )
        .unwrap();
    let trace = executor.try_backend().unwrap().trace().to_vec();
    assert!(trace.iter().any(|event| matches!(
        event,
        TestTraceEvent::TransitionTexture {
            texture,
            before: ResourceAccessState::Present,
            after: ResourceAccessState::ColorAttachmentWrite,
            ..
        } if *texture == TestTexture::new(900)
    )));
    assert!(trace.iter().any(|event| matches!(
        event,
        TestTraceEvent::TransitionTexture {
            texture,
            before: ResourceAccessState::ColorAttachmentWrite,
            after: ResourceAccessState::Present,
            ..
        } if *texture == TestTexture::new(900)
    )));
    assert!(trace.iter().any(|event| matches!(
        event,
        TestTraceEvent::Submit { presentations, .. }
            if presentations.len() == 1 && presentations[0].1 == 44
    )));
    assert_eq!(presented_probe.cancellation_count(), 0);

    let mut repeated = FrameInputs::new(false);
    repeated.bind_surface(image.slot, surface_binding);
    assert!(matches!(
        executor.execute(
            &compiled,
            compiled.instantiate_local(repeated),
            &registry,
            &registry,
        ),
        Err(ExecutionError::FrameBinding(error))
            if error.kind == FrameBindingErrorKind::MissingSurface
                && error.surface_binding == Some(surface_binding)
    ));

    let trace_disabled_binding = SurfaceBindingId::new(906);
    let (trace_disabled_token, trace_disabled_probe) = TestPresentationToken::fresh(49);
    registry.register_surface(
        trace_disabled_binding,
        TestTexture::new(906),
        texture(),
        TextureUsage::from_kinds([TextureUsageKind::ColorAttachment, TextureUsageKind::Present]),
        ResourceAccessState::Present,
        trace_disabled_token,
    );
    executor.try_backend().unwrap().set_trace_enabled(false);
    let mut trace_disabled_inputs = FrameInputs::new(false);
    trace_disabled_inputs.bind_surface(image.slot, trace_disabled_binding);
    executor
        .execute(
            &compiled,
            compiled.instantiate_local(trace_disabled_inputs),
            &registry,
            &registry,
        )
        .unwrap();
    assert_eq!(trace_disabled_probe.cancellation_count(), 0);
    executor.try_backend().unwrap().set_trace_enabled(true);

    let validation_binding = SurfaceBindingId::new(902);
    let (validation_token, validation_probe) = TestPresentationToken::fresh(45);
    let mut wrong_descriptor = texture();
    wrong_descriptor.extent.width += 1;
    registry.register_surface(
        validation_binding,
        TestTexture::new(902),
        wrong_descriptor,
        TextureUsage::from_kinds([TextureUsageKind::ColorAttachment, TextureUsageKind::Present]),
        ResourceAccessState::Present,
        validation_token,
    );
    let mut validation_inputs = FrameInputs::new(false);
    validation_inputs.bind_surface(image.slot, validation_binding);
    assert!(matches!(
        executor.execute(
            &compiled,
            compiled.instantiate_local(validation_inputs),
            &registry,
            &registry,
        ),
        Err(ExecutionError::FrameBinding(error))
            if error.kind == FrameBindingErrorKind::DescriptorMismatch
    ));
    assert_eq!(validation_probe.cancellation_count(), 1);

    let callback_binding = SurfaceBindingId::new(903);
    let (callback_token, callback_probe) = TestPresentationToken::fresh(46);
    registry.register_surface(
        callback_binding,
        TestTexture::new(903),
        texture(),
        TextureUsage::from_kinds([TextureUsageKind::ColorAttachment, TextureUsageKind::Present]),
        ResourceAccessState::Present,
        callback_token,
    );
    let mut callback_inputs = FrameInputs::new(true);
    callback_inputs.bind_surface(image.slot, callback_binding);
    assert!(matches!(
        executor.execute(
            &compiled,
            compiled.instantiate_local(callback_inputs),
            &registry,
            &registry,
        ),
        Err(ExecutionError::Recording(_))
    ));
    assert_eq!(callback_probe.cancellation_count(), 1);

    let finish_binding = SurfaceBindingId::new(904);
    let (finish_token, finish_probe) = TestPresentationToken::fresh(47);
    registry.register_surface(
        finish_binding,
        TestTexture::new(904),
        texture(),
        TextureUsage::from_kinds([TextureUsageKind::ColorAttachment, TextureUsageKind::Present]),
        ResourceAccessState::Present,
        finish_token,
    );
    executor
        .try_backend()
        .unwrap()
        .fail_finish_encoder(TestRhiError::new("finish"));
    let mut finish_inputs = FrameInputs::new(false);
    finish_inputs.bind_surface(image.slot, finish_binding);
    assert!(matches!(
        executor.execute(
            &compiled,
            compiled.instantiate_local(finish_inputs),
            &registry,
            &registry,
        ),
        Err(ExecutionError::Backend(_))
    ));
    assert_eq!(finish_probe.cancellation_count(), 1);

    let submit_binding = SurfaceBindingId::new(905);
    let (submit_token, submit_probe) = TestPresentationToken::fresh(48);
    registry.register_surface(
        submit_binding,
        TestTexture::new(905),
        texture(),
        TextureUsage::from_kinds([TextureUsageKind::ColorAttachment, TextureUsageKind::Present]),
        ResourceAccessState::Present,
        submit_token,
    );
    executor
        .try_backend()
        .unwrap()
        .fail_submit(TestRhiError::new("submit"));
    let mut submit_inputs = FrameInputs::new(false);
    submit_inputs.bind_surface(image.slot, submit_binding);
    assert!(matches!(
        executor.execute(
            &compiled,
            compiled.instantiate_local(submit_inputs),
            &registry,
            &registry,
        ),
        Err(ExecutionError::Backend(_))
    ));
    assert_eq!(submit_probe.cancellation_count(), 1);
}

#[test]
fn retained_surface_without_present_root_is_rejected_at_compile_time() {
    let mut caps = capabilities();
    caps.queues[0].capabilities.present = true;
    caps.surface = Some(SurfaceCapabilities::new(
        vec![TextureFormat::Rgba8Unorm],
        true,
        false,
    ));
    let mut graph = RenderGraph::<()>::new();
    let image = graph.import_surface_texture_slot(
        "surface",
        SurfaceTextureContract {
            descriptor: texture(),
        },
    );
    let pass = graph.add_raster_pass(
        "surface-side-effect",
        |pass| {
            pass.color_attachment(
                image.version,
                ColorAttachmentDesc {
                    index: 0,
                    range: TextureRange::whole(),
                    operations: AttachmentOps {
                        load: LoadOp::Clear([0.0; 4]),
                        store: StoreOp::Store,
                        write_coverage: WriteCoverage::Full,
                    },
                },
            );
            ((), ())
        },
        |_, _, _, _| Ok(()),
    );
    graph.mark_side_effect(pass.id, SideEffectReason::Diagnostic("retain".into()));
    assert!(matches!(
        graph.compile(&caps),
        Err(CompileError {
            kind: CompileErrorKind::InvalidExportOrPresent,
            ..
        })
    ));
}

#[test]
fn dead_surface_without_present_is_culled_with_its_pass() {
    let mut caps = capabilities();
    caps.queues[0].capabilities.present = true;
    caps.surface = Some(SurfaceCapabilities::new(
        vec![TextureFormat::Rgba8Unorm],
        true,
        false,
    ));
    let mut graph = RenderGraph::<()>::new();
    let image = graph.import_surface_texture_slot(
        "dead-surface",
        SurfaceTextureContract {
            descriptor: texture(),
        },
    );
    let pass = graph.add_raster_pass(
        "dead-surface-pass",
        |pass| {
            pass.color_attachment(
                image.version,
                ColorAttachmentDesc {
                    index: 0,
                    range: TextureRange::whole(),
                    operations: AttachmentOps {
                        load: LoadOp::Clear([0.0; 4]),
                        store: StoreOp::Store,
                        write_coverage: WriteCoverage::Full,
                    },
                },
            );
            ((), ())
        },
        |_, _, _, _| Ok(()),
    );
    let output = graph.compile(&caps).unwrap();
    assert_eq!(output.report.culled_passes, vec![pass.id]);
}
