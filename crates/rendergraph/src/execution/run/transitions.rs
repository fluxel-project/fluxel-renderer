//! Lowers planned state transitions to backend encoder calls.

use crate::{
    backend::{ExecutionBackend, ExecutionError},
    plan::{PlannedResourceRange, PlannedTransition},
};

use super::super::recording::{PhysicalResource, PhysicalResources};

pub(super) fn emit_transitions<B: ExecutionBackend>(
    backend: &mut B,
    encoder: &mut B::Encoder,
    physical: &PhysicalResources<B>,
    transitions: &[PlannedTransition],
) -> Result<(), ExecutionError<B::Error>> {
    for transition in transitions {
        match (&physical[&transition.resource], transition.range) {
            (
                PhysicalResource::Texture {
                    physical,
                    initial_state,
                    ..
                },
                PlannedResourceRange::Texture(range),
            ) => backend
                .transition_texture(
                    encoder,
                    physical,
                    range,
                    if transition.before == crate::rhi::ResourceAccessState::Undefined {
                        *initial_state
                    } else {
                        transition.before
                    },
                    transition.after,
                )
                .map_err(ExecutionError::Backend)?,
            (
                PhysicalResource::Buffer {
                    physical,
                    initial_state,
                    ..
                },
                PlannedResourceRange::Buffer(range),
            ) => backend
                .transition_buffer(
                    encoder,
                    physical,
                    range,
                    if transition.before == crate::rhi::ResourceAccessState::Undefined {
                        *initial_state
                    } else {
                        transition.before
                    },
                    transition.after,
                )
                .map_err(ExecutionError::Backend)?,
            _ => unreachable!("planned transition resource kind matches its range"),
        }
    }
    Ok(())
}
