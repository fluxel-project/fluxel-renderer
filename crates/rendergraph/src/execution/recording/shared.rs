//! Shares fail-closed recording validation and backend-error latching across pass kinds.
//!
//! A callback may observe a portable recording error, but once a backend command has failed its
//! native error is authoritative and must prevent further success from escaping the pass bridge.

use crate::access::BufferRange;

pub(super) fn buffer_range_contains(range: BufferRange, offset: u64, size: u64) -> bool {
    match range {
        BufferRange::Whole => true,
        BufferRange::Bytes {
            offset: declared,
            size: declared_size,
        } => declared.checked_add(declared_size).is_some_and(|end| {
            offset >= declared && offset.checked_add(size).is_some_and(|value| value <= end)
        }),
    }
}
// Shared command bridge and recording diagnostics.

use std::collections::HashMap;

use crate::{
    backend::ExecutionBackend,
    error::{DiagnosticContext, RecordResult, RecordingError, RecordingErrorKind},
    handles::{PassId, ResourceId},
    internal::AccessDecl,
};

use super::{PhysicalResources, SessionObjects};

pub(super) struct CommandBridge<'a, B: ExecutionBackend, O> {
    pub(super) pass: PassId,
    pub(super) device: crate::backend::DeviceIdentity,
    pub(super) backend: &'a mut B,
    pub(super) encoder: &'a mut B::Encoder,
    pub(super) accesses: &'a HashMap<u64, &'a AccessDecl>,
    pub(super) physical: &'a PhysicalResources<B>,
    pub(super) objects: &'a O,
    pub(super) session: &'a SessionObjects<B>,
    pub(super) backend_error: Option<B::Error>,
}

impl<B: ExecutionBackend, O> CommandBridge<'_, B, O> {
    pub(super) fn finish(
        &mut self,
        callback: RecordResult,
    ) -> Result<(), crate::backend::ExecutionError<B::Error>> {
        // A backend command may return a portable recording error to stop the
        // callback, but its native failure is the root cause and must win here.
        if let Some(error) = self.backend_error.take() {
            Err(crate::backend::ExecutionError::Backend(error))
        } else {
            callback.map_err(crate::backend::ExecutionError::Recording)
        }
    }

    pub(super) fn fail_backend(&mut self, error: B::Error) -> RecordingError {
        self.backend_error = Some(error);
        recording_error(
            RecordingErrorKind::InvalidCommandArgument,
            self.pass,
            None,
            "backend rejected a recording command",
        )
    }

    pub(super) fn access(&self, handle: u64) -> RecordResult<&AccessDecl> {
        self.accesses.get(&handle).copied().ok_or_else(|| {
            recording_error(
                RecordingErrorKind::ForeignOrUndeclaredPassAccess,
                self.pass,
                None,
                "command access was not declared by this pass",
            )
        })
    }

    pub(super) fn set_bindings(&mut self, ticket: u64, session: u64) -> RecordResult {
        if session != self.session.session {
            return Err(recording_error(
                RecordingErrorKind::ForeignOrUndeclaredPassAccess,
                self.pass,
                None,
                "resolved bindings belong to another recording session",
            ));
        }
        let bindings = self.session.bindings.borrow();
        let binding = bindings.get(ticket as usize).ok_or_else(|| {
            recording_error(
                RecordingErrorKind::ForeignOrUndeclaredPassAccess,
                self.pass,
                None,
                "resolved binding ticket is invalid",
            )
        })?;
        self.backend
            .set_bindings(self.encoder, &binding.physical)
            .map_err(|error| self.fail_backend(error))
    }
}

pub(super) fn recording_error(
    kind: RecordingErrorKind,
    pass: PassId,
    resource: Option<ResourceId>,
    detail: impl Into<String>,
) -> RecordingError {
    RecordingError {
        kind,
        context: DiagnosticContext {
            passes: vec![pass],
            resource,
            texture_slot: None,
            buffer_slot: None,
            detail: detail.into(),
            unsupported: None,
        },
    }
}
