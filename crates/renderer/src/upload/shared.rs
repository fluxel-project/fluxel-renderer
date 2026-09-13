//! Shared upload-generation synchronization and retained-state contracts.

use std::{
    sync::atomic::AtomicU64,
    sync::{Arc, Mutex},
};

use fluxel_rendergraph::CompletionStatus;

pub(super) static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);

/// Whether completion remains non-terminal and therefore retains every lease.
///
/// `Unknown` means the backend cannot prove completion, not that it proved a
/// failure.  Renderer state machines must retain their pending upload/frame
/// and report retryable progress until a terminal status is observed.
pub(crate) const fn completion_requires_retention(status: CompletionStatus) -> bool {
    matches!(
        status,
        CompletionStatus::Pending | CompletionStatus::Unknown
    )
}

/// Why a ready snapshot cannot start another renderer draw.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SnapshotUseError {
    /// A prior accepted draw did not prove that the fixed outgoing states held.
    Poisoned,
}

/// One private immutable-read lease for a snapshot generation.
///
/// Several immutable readers may coexist. Dropping an accepted operation without
/// proving completion poisons the whole generation; a pre-submit caller explicitly
/// releases its own lease instead.
#[derive(Debug)]
pub(crate) struct SnapshotDrawReservation {
    gate: Arc<SnapshotUseGate>,
    active: bool,
}

impl SnapshotDrawReservation {
    /// Releases a reservation whose graph execution was rejected before submit.
    pub(crate) fn release_before_submit(mut self) {
        self.gate.release();
        self.active = false;
    }

    /// Releases a reservation after a complete submission restored known state.
    pub(crate) fn release_complete(&mut self) {
        if self.active {
            self.gate.release();
            self.active = false;
        }
    }

    /// Makes this generation permanently unusable after an uncertain outcome.
    pub(crate) fn poison(&mut self) {
        if self.active {
            self.gate.poison();
            self.active = false;
        }
    }
}

impl Drop for SnapshotDrawReservation {
    fn drop(&mut self) {
        if self.active {
            self.gate.poison();
        }
    }
}

#[derive(Debug)]
pub(super) struct SnapshotUseGate {
    state: Mutex<SnapshotUseState>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SnapshotUseState {
    Ready,
    Readers(usize),
    Poisoned,
}

impl SnapshotUseGate {
    pub(super) fn new() -> Self {
        Self {
            state: Mutex::new(SnapshotUseState::Ready),
        }
    }

    pub(super) fn reserve(self: &Arc<Self>) -> Result<SnapshotDrawReservation, SnapshotUseError> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match *state {
            SnapshotUseState::Ready => {
                *state = SnapshotUseState::Readers(1);
                Ok(SnapshotDrawReservation {
                    gate: Arc::clone(self),
                    active: true,
                })
            }
            SnapshotUseState::Readers(readers) => {
                *state = SnapshotUseState::Readers(
                    readers
                        .checked_add(1)
                        .expect("snapshot reader count overflow"),
                );
                Ok(SnapshotDrawReservation {
                    gate: Arc::clone(self),
                    active: true,
                })
            }
            SnapshotUseState::Poisoned => Err(SnapshotUseError::Poisoned),
        }
    }

    fn release(&self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match *state {
            SnapshotUseState::Readers(1) => *state = SnapshotUseState::Ready,
            SnapshotUseState::Readers(readers) => *state = SnapshotUseState::Readers(readers - 1),
            // A sibling accepted submission became unknown. Its poison is
            // monotonic, so a later known reader cannot restore this generation.
            SnapshotUseState::Poisoned => {}
            SnapshotUseState::Ready => {
                debug_assert!(false, "released snapshot lease was not active")
            }
        }
    }

    fn poison(&self) {
        *self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = SnapshotUseState::Poisoned;
    }

    #[cfg(test)]
    pub(super) fn active_readers(&self) -> usize {
        match *self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
        {
            SnapshotUseState::Ready | SnapshotUseState::Poisoned => 0,
            SnapshotUseState::Readers(readers) => readers,
        }
    }
}

/// Readiness retained by a closed multi-stream mesh upload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RetainedUploadState {
    Missing,
    Pending,
    Ready,
}
