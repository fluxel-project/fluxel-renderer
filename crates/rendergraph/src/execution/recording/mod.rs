//! Records declared passes into a backend encoder without exposing native objects.
//!
//! This module owns pass-local resolver sessions and their retained leases; command
//! validation lives in the pass-kind bridges, while orchestration opens and closes
//! backend passes. A session nonce prevents bindings resolved for one callback from
//! being used by another, and every resolved object remains leased through submission.

mod compute;
mod copy;
mod orchestration;
mod raster;
mod resolver;
mod shared;

use std::{
    cell::RefCell,
    collections::HashMap,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::{
    backend::{BoundBindings, ExecutionBackend},
    internal::PassDecl,
    plan::PlannedPass,
};

pub(crate) use orchestration::record_pass;

pub(crate) enum PhysicalResource<B: ExecutionBackend> {
    Texture {
        physical: B::Texture,
        descriptor: crate::rhi::TextureDesc,
        initial_state: crate::rhi::ResourceAccessState,
    },
    Buffer {
        physical: B::Buffer,
        descriptor: crate::rhi::BufferDesc,
        initial_state: crate::rhi::ResourceAccessState,
    },
}

pub(crate) type PhysicalResources<B> = HashMap<crate::handles::ResourceId, PhysicalResource<B>>;

static NEXT_SESSION: AtomicU64 = AtomicU64::new(1);

pub(super) struct SessionObjects<B: ExecutionBackend> {
    pub(super) session: u64,
    pub(super) bindings: RefCell<Vec<BoundBindings<B::Bindings, B::Lease>>>,
    pub(super) leases: RefCell<Vec<B::Lease>>,
}

impl<B: ExecutionBackend> SessionObjects<B> {
    pub(super) fn new() -> Self {
        Self {
            session: NEXT_SESSION.fetch_add(1, Ordering::Relaxed),
            bindings: RefCell::new(Vec::new()),
            leases: RefCell::new(Vec::new()),
        }
    }

    pub(super) fn finish(self, retained: &mut Vec<B::Lease>) {
        retained.extend(self.leases.into_inner());
        retained.extend(
            self.bindings
                .into_inner()
                .into_iter()
                .map(|binding| binding.lease),
        );
    }
}

pub(crate) struct RecordPassRequest<'a, B: ExecutionBackend, O, F> {
    pub backend: &'a mut B,
    pub encoder: &'a mut B::Encoder,
    pub pass: &'a PassDecl<F>,
    pub planned: &'a PlannedPass,
    pub physical: &'a PhysicalResources<B>,
    pub objects: &'a O,
    pub frame: &'a F,
    pub retained: &'a mut Vec<B::Lease>,
}
