//! Completion-gated cross-frame transient reuse contracts.

use super::*;
use fluxel_rendergraph::test_rhi::TestRhiError;

fn transient_graph() -> CompiledGraph {
    let mut graph = RenderGraph::new();
    let transient = graph.create_buffer("transient", buffer(16));
    let write = graph.add_compute_pass(
        "write",
        |pass| {
            let (output, _) = pass.write_buffer(
                transient,
                BufferWriteUse::Storage,
                BufferRange::whole(),
                WriteCoverage::Full,
            );
            (output, ())
        },
        |_, _, _, _| Ok(()),
    );
    let pass = graph.add_compute_pass(
        "read",
        |pass| {
            pass.read_buffer(&write.output, BufferReadUse::Storage, BufferRange::whole());
            ((), ())
        },
        |_, _, _, _| Ok(()),
    );
    graph.mark_side_effect(
        pass.id,
        SideEffectReason::Diagnostic("retain transient".into()),
    );
    graph.compile(&caps()).unwrap().graph
}

fn created_textures(executor: &FrameExecutor<TestRhi>) -> usize {
    executor
        .try_backend()
        .unwrap()
        .trace()
        .iter()
        .filter(|event| matches!(event, TestTraceEvent::CreateTexture { .. }))
        .count()
}

fn created_buffers(executor: &FrameExecutor<TestRhi>) -> usize {
    executor
        .try_backend()
        .unwrap()
        .trace()
        .iter()
        .filter(|event| matches!(event, TestTraceEvent::CreateBuffer { .. }))
        .count()
}

fn execute_transient(
    executor: &FrameExecutor<TestRhi>,
    graph: &CompiledGraph,
) -> ExecutedFrame<TestRhi> {
    let registry = TestRegistry::new(executor.try_backend().unwrap().device_identity());
    executor
        .execute(
            graph,
            graph.instantiate_local(FrameInputs::new(())),
            &registry,
            &registry,
        )
        .unwrap()
}

#[test]
fn completed_transients_reuse_but_pending_and_unknown_slots_do_not() {
    let graph = transient_graph();
    let frame_executor = executor();
    let first = execute_transient(&frame_executor, &graph);
    let first_completion = *first.submission.completion();
    let second = execute_transient(&frame_executor, &graph);
    assert_eq!(
        created_buffers(&frame_executor),
        2,
        "pending work cannot be reused"
    );
    frame_executor
        .try_backend()
        .unwrap()
        .complete(first_completion);
    let _third = execute_transient(&frame_executor, &graph);
    assert_eq!(
        created_buffers(&frame_executor),
        2,
        "a completed slot is reused"
    );
    assert!(
        frame_executor
            .try_backend()
            .unwrap()
            .trace()
            .iter()
            .any(|event| matches!(
                event,
                TestTraceEvent::TransitionBuffer {
                    before: ResourceAccessState::ShaderStorageRead,
                    ..
                }
            )),
        "the reused physical buffer starts from its prior terminal state"
    );

    let unknown_graph = transient_graph();
    let unknown_executor = executor();
    let unknown = execute_transient(&unknown_executor, &unknown_graph);
    let completion = *unknown.submission.completion();
    unknown_executor.try_backend().unwrap().unknown(completion);
    let _next = execute_transient(&unknown_executor, &unknown_graph);
    assert_eq!(
        created_buffers(&unknown_executor),
        2,
        "unknown work remains quarantined"
    );
    drop(second);
}

#[test]
fn failed_device_and_graph_generations_are_isolated() {
    let first_graph = transient_graph();
    let second_graph = transient_graph();
    let frame_executor = executor();
    let first = execute_transient(&frame_executor, &first_graph);
    let completion = *first.submission.completion();
    frame_executor.try_backend().unwrap().fail(completion);
    let _after_failure = execute_transient(&frame_executor, &first_graph);
    assert_eq!(
        created_buffers(&frame_executor),
        2,
        "failed work is discarded, not reused"
    );
    let _other_graph = execute_transient(&frame_executor, &second_graph);
    assert_eq!(
        created_buffers(&frame_executor),
        3,
        "compiled graphs do not share transients"
    );
    frame_executor
        .try_backend()
        .unwrap()
        .set_device_identity(DeviceIdentity::new(89));
    let _other_device = execute_transient(&frame_executor, &first_graph);
    assert_eq!(
        created_buffers(&frame_executor),
        4,
        "device generations do not share transients"
    );
}

#[test]
fn exported_transients_are_not_cached_and_submit_failure_returns_checkout() {
    let mut graph = RenderGraph::new();
    let color = graph.create_texture("export", texture(TextureFormat::Rgba8Unorm));
    let pass = graph.add_raster_pass(
        "clear",
        |pass| {
            let output = pass.color_attachment(
                color,
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
            (output, ())
        },
        |_, _, _, _| Ok(()),
    );
    graph.export_texture(
        pass.output,
        ExportTextureContract {
            final_state: ResourceAccessState::ShaderSampledRead,
        },
    );
    let exported = graph.compile(&caps()).unwrap().graph;
    let frame_executor = executor();
    let first = execute_transient(&frame_executor, &exported);
    frame_executor
        .try_backend()
        .unwrap()
        .complete(*first.submission.completion());
    let _second = execute_transient(&frame_executor, &exported);
    assert_eq!(
        created_textures(&frame_executor),
        2,
        "caller-visible exports never enter the pool"
    );

    let reusable = transient_graph();
    let failed_executor = executor();
    failed_executor
        .try_backend()
        .unwrap()
        .fail_submit(TestRhiError::new("reject submit"));
    let registry = TestRegistry::new(device());
    assert!(
        failed_executor
            .execute(
                &reusable,
                reusable.instantiate_local(FrameInputs::new(())),
                &registry,
                &registry
            )
            .is_err()
    );
    let _retry = execute_transient(&failed_executor, &reusable);
    assert_eq!(
        created_buffers(&failed_executor),
        1,
        "unsubmitted checkout returns to the pool"
    );
}

#[test]
fn provider_imports_never_enter_the_transient_pool() {
    let mut graph = RenderGraph::new();
    let imported = graph.import_texture_slot(
        "persistent provider texture",
        ImportTextureContract {
            descriptor: texture(TextureFormat::Rgba8Unorm),
            initial_state: ResourceAccessState::ShaderSampledRead,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let pass = graph.add_compute_pass(
        "read provider texture",
        |pass| {
            pass.read_texture(
                &imported.version,
                TextureReadUse::Sampled,
                TextureRange::whole(),
            );
            ((), ())
        },
        |_, _, _, _| Ok(()),
    );
    graph.mark_side_effect(
        pass.id,
        SideEffectReason::Diagnostic("retain import".into()),
    );
    let compiled = graph.compile(&caps()).unwrap().graph;
    let frame_executor = executor();
    let mut registry = TestRegistry::new(device());
    registry.register_texture(
        TextureBindingId::new(41),
        TestTexture::new(41),
        texture(TextureFormat::Rgba8Unorm),
        TextureUsage::from_kinds([TextureUsageKind::Sampled]),
        ResourceAccessState::ShaderSampledRead,
    );
    let mut inputs = FrameInputs::new(());
    inputs.bind_texture(imported.slot, TextureBindingId::new(41));
    frame_executor
        .execute(
            &compiled,
            compiled.instantiate_local(inputs),
            &registry,
            &registry,
        )
        .unwrap();
    assert_eq!(
        created_textures(&frame_executor),
        0,
        "provider-owned imports bypass transient allocation"
    );
}

#[test]
fn invalidation_releases_available_slots_and_unknown_retirement_stays_quarantined() {
    let graph = transient_graph();
    let frame_executor = executor();
    let first = execute_transient(&frame_executor, &graph);
    frame_executor
        .try_backend()
        .unwrap()
        .complete(*first.submission.completion());
    let _reused = execute_transient(&frame_executor, &graph);
    assert_eq!(created_buffers(&frame_executor), 1);
    frame_executor.invalidate_graph(&graph).unwrap();
    let pending = execute_transient(&frame_executor, &graph);
    assert_eq!(
        created_buffers(&frame_executor),
        2,
        "invalidation drops available allocations"
    );
    let completion = *pending.submission.completion();
    frame_executor.try_backend().unwrap().unknown(completion);
    drop(pending);
    let _ = frame_executor.collect_retired().unwrap();
    assert_eq!(
        frame_executor.try_backend().unwrap().retired_count(),
        1,
        "unknown retirement retains its leases"
    );
}

#[test]
fn partial_buffer_plans_are_never_cached() {
    let mut graph = RenderGraph::new();
    let transient = graph.create_buffer("partial", buffer(16));
    let pass = graph.add_compute_pass(
        "partial write",
        |pass| {
            let (output, _) = pass.write_buffer(
                transient,
                BufferWriteUse::Storage,
                BufferRange::Bytes { offset: 0, size: 8 },
                WriteCoverage::Full,
            );
            (output, ())
        },
        |_, _, _, _| Ok(()),
    );
    graph.mark_side_effect(
        pass.id,
        SideEffectReason::Diagnostic("retain partial".into()),
    );
    let compiled = graph.compile(&caps()).unwrap().graph;
    let frame_executor = executor();
    let first = execute_transient(&frame_executor, &compiled);
    frame_executor
        .try_backend()
        .unwrap()
        .complete(*first.submission.completion());
    let _second = execute_transient(&frame_executor, &compiled);
    assert_eq!(
        created_buffers(&frame_executor),
        2,
        "partial ranges fail closed to fresh allocations"
    );
}
