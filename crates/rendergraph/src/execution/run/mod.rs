//! Executes one immutable plan as an ordered, single-lock frame transaction.
//!
//! Identity and unsupported-feature checks precede the backend lock; capability
//! validation, resource resolution, recording, submission, and retirement then share
//! that lock deliberately. Providers and callbacks therefore cannot re-enter the same
//! executor, and no partially resolved frame is submitted after a failed precondition.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use crate::{
    CompiledGraph,
    backend::{ExecutionBackend, ExecutionError, FrameResourceProvider, RenderObjectProvider},
};

use super::{
    FrameExecution,
    recording::{RecordPassRequest, record_pass},
    submission::{FrameSubmission, RetirementInbox, drain_retirement_inbox, try_lock_backend},
};

mod exports;
mod resolution;
mod transients;
mod transitions;

pub use exports::{ExecutedFrame, ExportedBuffer, ExportedTexture, FrameExports};

use exports::build_exports;
use resolution::{live_resources, resolve_resources};
use transitions::emit_transitions;

/// Serial single-queue executor for one backend device.
pub struct FrameExecutor<B: ExecutionBackend> {
    backend: Arc<Mutex<B>>,
    retirement_inbox: RetirementInbox<B>,
    transients: Mutex<transients::TransientPool<B>>,
}

impl<B: ExecutionBackend> FrameExecutor<B> {
    /// Creates an executor owning one backend device instance.
    pub fn new(backend: B) -> Self {
        Self {
            backend: Arc::new(Mutex::new(backend)),
            retirement_inbox: Arc::new(Mutex::new(Vec::new())),
            transients: Mutex::new(transients::TransientPool::default()),
        }
    }

    /// Tries to borrow the backend for diagnostics or explicit test control.
    ///
    /// This returns `None` while a frame is executing, so providers and pass
    /// callbacks cannot accidentally deadlock by re-entering the executor.
    pub fn try_backend(&self) -> Option<std::sync::MutexGuard<'_, B>> {
        try_lock_backend(&self.backend).ok()
    }

    /// Polls the backend retirement queue.
    pub fn collect_retired(&self) -> Result<usize, ExecutionError<B::Error>> {
        // Keep one backend guard through resolution, provider callbacks, native
        // recording, and submit. `try_backend` turns any re-entry into Busy rather
        // than permitting interleaved state or a self-deadlock.
        let mut backend =
            try_lock_backend(&self.backend).map_err(|()| ExecutionError::ExecutorBusy)?;
        drain_retirement_inbox(&mut *backend, &self.retirement_inbox);
        lock_transients(&self.transients).reclaim(&*backend);
        backend.collect_retired().map_err(ExecutionError::Backend)
    }

    /// Invalidates cached transients belonging to one compiled graph.
    ///
    /// Available allocations are released immediately. Allocations referenced
    /// by accepted work stay quarantined until their completion is terminal.
    /// Call this when replacing a compiled graph (for example after extent or
    /// format recompilation) so the executor does not retain an unbounded
    /// history of obsolete graph caches.
    pub fn invalidate_graph<F>(
        &self,
        graph: &CompiledGraph<F>,
    ) -> Result<(), ExecutionError<B::Error>> {
        let _backend =
            try_lock_backend(&self.backend).map_err(|()| ExecutionError::ExecutorBusy)?;
        lock_transients(&self.transients).invalidate_graph(graph.identity);
        Ok(())
    }

    /// Executes one frame instance through the retained single-queue plan.
    pub fn execute<F, M, R, O>(
        &self,
        graph: &CompiledGraph<F>,
        execution: FrameExecution<F, M>,
        resources: &R,
        objects: &O,
    ) -> Result<ExecutedFrame<B>, ExecutionError<B::Error>>
    where
        R: FrameResourceProvider<B>,
        O: RenderObjectProvider<B>,
    {
        if execution.graph_identity != graph.identity {
            return Err(ExecutionError::WrongCompiledGraph);
        }
        let live = live_resources(graph);
        let queue =
            graph
                .execution_plan()
                .queue()
                .ok_or(ExecutionError::UnsupportedExecutionFeature(
                    "an empty graph has no submission queue",
                ))?;
        let mut backend =
            try_lock_backend(&self.backend).map_err(|()| ExecutionError::ExecutorBusy)?;
        if graph.capability_fingerprint != backend.capabilities().fingerprint() {
            return Err(ExecutionError::CapabilityMismatch);
        }
        drain_retirement_inbox(&mut *backend, &self.retirement_inbox);
        backend.collect_retired().map_err(ExecutionError::Backend)?;
        let mut transients = lock_transients(&self.transients);
        transients.reclaim(&*backend);

        let resolved = match resolve_resources(
            graph,
            &execution,
            resources,
            &mut *backend,
            &mut transients,
            &live,
        ) {
            Ok(resolved) => resolved,
            Err(error) => {
                transients.abort_checked_out();
                return Err(error);
            }
        };
        let resolution::ResolvedFrameResources {
            physical,
            resource_leases,
            mut retained,
            presentations,
            transient_reservations,
        } = resolved;
        let result = (|| {
            let exports = build_exports(graph, &physical, &resource_leases);
            let mut encoder = backend
                .begin_encoder(queue)
                .map_err(ExecutionError::Backend)?;
            let passes: HashMap<_, _> = graph.passes.iter().map(|pass| (pass.id, pass)).collect();
            for planned in graph.execution_plan().passes() {
                emit_transitions(&mut *backend, &mut encoder, &physical, &planned.transitions)?;
                record_pass(RecordPassRequest {
                    backend: &mut *backend,
                    encoder: &mut encoder,
                    pass: passes[&planned.pass],
                    planned,
                    physical: &physical,
                    objects,
                    frame: &execution.inputs.frame_data,
                    retained: &mut retained,
                })?;
            }
            emit_transitions(
                &mut *backend,
                &mut encoder,
                &physical,
                graph.execution_plan().final_transitions(),
            )?;
            let command_buffer = backend
                .finish_encoder(encoder)
                .map_err(ExecutionError::Backend)?;
            let completion = backend
                .submit(queue, command_buffer, presentations)
                .map_err(ExecutionError::Backend)?;
            transients.commit(completion.clone(), &transient_reservations);
            Ok(ExecutedFrame {
                exports,
                submission: FrameSubmission::new(
                    Arc::clone(&self.backend),
                    Arc::clone(&self.retirement_inbox),
                    completion,
                    retained,
                ),
            })
        })();
        if result.is_err() {
            transients.abort(&transient_reservations);
        }
        drop(transients);
        drop(backend);
        result
    }
}

fn lock_transients<B: ExecutionBackend>(
    transients: &Mutex<transients::TransientPool<B>>,
) -> std::sync::MutexGuard<'_, transients::TransientPool<B>> {
    transients
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}
