//! Executor-private, completion-gated transient resource reuse.
//!
//! This cache only reuses non-exported graph transients between executions of
//! the same compiled graph on the same device. It never aliases two logical
//! resources within a frame, and it treats every non-complete completion as
//! unavailable.

use std::collections::HashMap;

use crate::{
    backend::{
        BoundBuffer, BoundTexture, CompletionFailure, CompletionStatus, DeviceIdentity,
        ExecutionBackend, PhysicalResourceIdentity,
    },
    handles::ResourceId,
    plan::{BufferUsage, TextureUsage},
    rhi::{BufferDesc, ResourceAccessState, TextureDesc},
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct TransientKey {
    device: DeviceIdentity,
    graph: u64,
    resource: ResourceId,
}

enum CachedResource<B: ExecutionBackend> {
    Texture {
        physical: B::Texture,
        identity: PhysicalResourceIdentity,
        descriptor: TextureDesc,
        usage: TextureUsage,
        state: ResourceAccessState,
        post_submit_state: ResourceAccessState,
        lease: B::Lease,
    },
    Buffer {
        physical: B::Buffer,
        identity: PhysicalResourceIdentity,
        descriptor: BufferDesc,
        usage: BufferUsage,
        state: ResourceAccessState,
        post_submit_state: ResourceAccessState,
        lease: B::Lease,
    },
}

enum SlotState<C> {
    Available,
    CheckedOut,
    InFlight(C),
    RetireOnTerminal(C),
}

struct Slot<B: ExecutionBackend> {
    id: u64,
    resource: CachedResource<B>,
    state: SlotState<B::Completion>,
}

/// A checkout returned by resolution and committed only after submit accepts.
pub(super) struct TransientReservations<B: ExecutionBackend> {
    slots: Vec<(TransientKey, u64)>,
    _backend: std::marker::PhantomData<B>,
}

impl<B: ExecutionBackend> Default for TransientReservations<B> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            _backend: std::marker::PhantomData,
        }
    }
}

/// Per-executor cache. It intentionally has no public API.
pub(super) struct TransientPool<B: ExecutionBackend> {
    slots: HashMap<TransientKey, Vec<Slot<B>>>,
    next_slot: u64,
}

impl<B: ExecutionBackend> Default for TransientPool<B> {
    fn default() -> Self {
        Self {
            slots: HashMap::new(),
            next_slot: 1,
        }
    }
}

impl<B: ExecutionBackend> TransientPool<B> {
    /// Drops entries from old device generations and observes submitted work.
    pub(super) fn reclaim(&mut self, backend: &B) {
        let device = backend.device_identity();
        for (key, entries) in &mut self.slots {
            let old_device = key.device != device;
            if old_device {
                for slot in &mut *entries {
                    slot.state = match std::mem::replace(&mut slot.state, SlotState::Available) {
                        SlotState::InFlight(completion)
                        | SlotState::RetireOnTerminal(completion) => {
                            SlotState::RetireOnTerminal(completion)
                        }
                        SlotState::Available | SlotState::CheckedOut => continue,
                    };
                }
            }
            entries.retain_mut(|slot| match &slot.state {
                SlotState::Available => !old_device,
                // Execution is serial under the backend lock. A checked-out
                // slot observed at the beginning of a later execution can
                // only belong to an earlier recording/submit failure, which
                // never reached the GPU and is therefore safe to return.
                SlotState::CheckedOut => {
                    slot.state = SlotState::Available;
                    !old_device
                }
                SlotState::InFlight(completion) => match backend.completion_status(completion) {
                    CompletionStatus::Complete => {
                        slot.state = SlotState::Available;
                        true
                    }
                    CompletionStatus::Failed(CompletionFailure::DeviceLost) => false,
                    // A failed submission is not reused. Unknown is not a
                    // terminal proof and stays quarantined with Pending.
                    CompletionStatus::Failed(_) => false,
                    CompletionStatus::Pending | CompletionStatus::Unknown => true,
                },
                SlotState::RetireOnTerminal(completion) => matches!(
                    backend.completion_status(completion),
                    CompletionStatus::Pending | CompletionStatus::Unknown
                ),
            });
        }
        self.slots.retain(|_, entries| !entries.is_empty());
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "cache key, allocation contract, and frame reservation are distinct ownership inputs"
    )]
    pub(super) fn checkout_texture(
        &mut self,
        backend: &mut B,
        graph: u64,
        resource: ResourceId,
        descriptor: TextureDesc,
        usage: TextureUsage,
        final_state: ResourceAccessState,
        reservations: &mut TransientReservations<B>,
    ) -> Result<BoundTexture<B::Texture, B::Lease>, B::Error> {
        let key = TransientKey {
            device: backend.device_identity(),
            graph,
            resource,
        };
        if let Some(slot) = self.available_slot(key) {
            let CachedResource::Texture {
                physical,
                identity,
                descriptor,
                usage,
                state,
                lease,
                ..
            } = &slot.resource
            else {
                unreachable!("one logical transient cannot change resource kind")
            };
            slot.state = SlotState::CheckedOut;
            reservations.slots.push((key, slot.id));
            return Ok(BoundTexture {
                device: key.device,
                identity: *identity,
                physical: physical.clone(),
                descriptor: *descriptor,
                usage: *usage,
                initial_state: *state,
                lease: lease.clone(),
            });
        }
        let bound = backend.create_transient_texture(descriptor, usage)?;
        let id = self.insert(
            key,
            CachedResource::Texture {
                physical: bound.physical.clone(),
                identity: bound.identity,
                descriptor: bound.descriptor,
                usage: bound.usage,
                state: bound.initial_state,
                post_submit_state: final_state,
                lease: bound.lease.clone(),
            },
        );
        reservations.slots.push((key, id));
        Ok(bound)
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "cache key, allocation contract, and frame reservation are distinct ownership inputs"
    )]
    pub(super) fn checkout_buffer(
        &mut self,
        backend: &mut B,
        graph: u64,
        resource: ResourceId,
        descriptor: BufferDesc,
        usage: BufferUsage,
        final_state: ResourceAccessState,
        reservations: &mut TransientReservations<B>,
    ) -> Result<BoundBuffer<B::Buffer, B::Lease>, B::Error> {
        let key = TransientKey {
            device: backend.device_identity(),
            graph,
            resource,
        };
        if let Some(slot) = self.available_slot(key) {
            let CachedResource::Buffer {
                physical,
                identity,
                descriptor,
                usage,
                state,
                lease,
                ..
            } = &slot.resource
            else {
                unreachable!("one logical transient cannot change resource kind")
            };
            slot.state = SlotState::CheckedOut;
            reservations.slots.push((key, slot.id));
            return Ok(BoundBuffer {
                device: key.device,
                identity: *identity,
                physical: physical.clone(),
                descriptor: *descriptor,
                usage: *usage,
                initial_state: *state,
                lease: lease.clone(),
            });
        }
        let bound = backend.create_transient_buffer(descriptor, usage)?;
        let id = self.insert(
            key,
            CachedResource::Buffer {
                physical: bound.physical.clone(),
                identity: bound.identity,
                descriptor: bound.descriptor,
                usage: bound.usage,
                state: bound.initial_state,
                post_submit_state: final_state,
                lease: bound.lease.clone(),
            },
        );
        reservations.slots.push((key, id));
        Ok(bound)
    }

    pub(super) fn abort(&mut self, reservations: &TransientReservations<B>) {
        for (key, id) in &reservations.slots {
            let slot = self.find_slot(*key, *id);
            assert!(
                matches!(slot.state, SlotState::CheckedOut),
                "transient reservation was settled twice"
            );
            slot.state = SlotState::Available;
        }
    }

    /// Returns all current-frame checkouts after resolution failed before it
    /// could hand its reservation list back to the executor.
    pub(super) fn abort_checked_out(&mut self) {
        for entries in self.slots.values_mut() {
            for slot in entries {
                if matches!(slot.state, SlotState::CheckedOut) {
                    slot.state = SlotState::Available;
                }
            }
        }
    }

    pub(super) fn commit(
        &mut self,
        completion: B::Completion,
        reservations: &TransientReservations<B>,
    ) {
        for (key, id) in &reservations.slots {
            let slot = self.find_slot(*key, *id);
            assert!(
                matches!(slot.state, SlotState::CheckedOut),
                "transient reservation was settled twice"
            );
            match &mut slot.resource {
                CachedResource::Texture {
                    state,
                    post_submit_state,
                    ..
                }
                | CachedResource::Buffer {
                    state,
                    post_submit_state,
                    ..
                } => *state = *post_submit_state,
            }
            slot.state = SlotState::InFlight(completion.clone());
        }
    }

    pub(super) fn invalidate_graph(&mut self, graph: u64) {
        for (key, entries) in &mut self.slots {
            if key.graph != graph {
                continue;
            }
            entries.retain_mut(|slot| {
                match std::mem::replace(&mut slot.state, SlotState::Available) {
                    SlotState::InFlight(completion) | SlotState::RetireOnTerminal(completion) => {
                        slot.state = SlotState::RetireOnTerminal(completion);
                        true
                    }
                    SlotState::Available | SlotState::CheckedOut => false,
                }
            });
        }
        self.slots.retain(|_, entries| !entries.is_empty());
    }

    fn available_slot(&mut self, key: TransientKey) -> Option<&mut Slot<B>> {
        self.slots
            .get_mut(&key)?
            .iter_mut()
            .find(|slot| matches!(slot.state, SlotState::Available))
    }

    fn insert(&mut self, key: TransientKey, resource: CachedResource<B>) -> u64 {
        let id = self.next_slot;
        self.next_slot = self
            .next_slot
            .checked_add(1)
            .expect("transient cache slot space exhausted");
        self.slots.entry(key).or_default().push(Slot {
            id,
            resource,
            state: SlotState::CheckedOut,
        });
        id
    }

    fn find_slot(&mut self, key: TransientKey, id: u64) -> &mut Slot<B> {
        self.slots
            .get_mut(&key)
            .and_then(|entries| entries.iter_mut().find(|slot| slot.id == id))
            .expect("live transient reservation belongs to this pool")
    }
}
