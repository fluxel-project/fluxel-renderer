//! Shared fixtures and execution-contract test groups.

//! Black-box execution-contract regression tests.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use fluxel_rendergraph::{
    test_rhi::{
        TestBindingResource, TestBindings, TestBuffer, TestCompletion, TestComputePipeline,
        TestPresentationToken, TestRegistry, TestRhi, TestRhiError, TestTexture, TestTraceEvent,
    },
    *,
};

fn device() -> DeviceIdentity {
    DeviceIdentity::new(77)
}

fn capabilities() -> DeviceCapabilities {
    DeviceCapabilities::builder()
        .queue(QueueDescriptor::new(
            QueueId::new(0),
            QueueCapabilities::new(true, true, true, false),
        ))
        .recording(RecordingCapabilities::new(
            RecordingModel::DeferredCommandBuffers,
            false,
        ))
        .transitions(TransitionCapabilities::GraphManagedExplicit)
        .synchronization(SynchronizationCapabilities::SingleQueueOrdering)
        .limits(DeviceLimits::new(4, 256).with_max_compute_workgroups_per_dimension([65_535; 3]))
        .buffers(BufferCapabilities::new(true, true, true))
        .texture_format(
            TextureFormatCapabilities::builder(TextureFormat::Rgba8Unorm)
                .sampled(true, true)
                .storage(true, true)
                .attachments(true, false, vec![1])
                .copies(true, true)
                .build(),
        )
        .build()
}

fn buffer() -> BufferDesc {
    BufferDesc { size: 64 }
}

fn texture() -> TextureDesc {
    TextureDesc {
        dimension: TextureDimension::D2,
        extent: Extent3d {
            width: 8,
            height: 8,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
    }
}

fn executor() -> FrameExecutor<TestRhi> {
    FrameExecutor::new(TestRhi::new(capabilities(), device()))
}

fn imported_compute_graph() -> (CompiledGraph<u32>, ImportBufferSlot) {
    let mut graph = RenderGraph::new();
    let imported = graph.import_buffer_slot(
        "dynamic",
        ImportBufferContract {
            descriptor: buffer(),
            initial_state: ResourceAccessState::ShaderStorageRead,
            ownership: ExternalOwnership::Caller,
            initial_contents: InitialContents::Defined,
        },
    );
    let pass = graph.add_compute_pass(
        "dynamic",
        |pass| {
            pass.read_buffer(
                &imported.version,
                BufferReadUse::Storage,
                BufferRange::whole(),
            );
            ((), ())
        },
        |commands, _, _, frame| {
            if *frame == 0 {
                return Err(RecordingError {
                    kind: RecordingErrorKind::InvalidCommandArgument,
                    context: DiagnosticContext {
                        passes: Vec::new(),
                        resource: None,
                        texture_slot: None,
                        buffer_slot: None,
                        detail: "test callback failure".into(),
                        unsupported: None,
                    },
                });
            }
            commands.dispatch([*frame, 1, 1])
        },
    );
    graph.mark_side_effect(pass.id, SideEffectReason::Diagnostic("retain".into()));
    (graph.compile(&capabilities()).unwrap().graph, imported.slot)
}

fn registry_with_buffers() -> TestRegistry {
    let mut registry = TestRegistry::new(device());
    registry.register_buffer(
        BufferBindingId::new(1),
        TestBuffer::new(1),
        buffer(),
        BufferUsage::from_kinds([BufferUsageKind::StorageRead]),
        ResourceAccessState::ShaderStorageRead,
    );
    registry.register_buffer(
        BufferBindingId::new(2),
        TestBuffer::new(2),
        buffer(),
        BufferUsage::from_kinds([BufferUsageKind::StorageRead]),
        ResourceAccessState::ShaderStorageRead,
    );
    registry
}

mod identity_and_completion;
mod recording;
