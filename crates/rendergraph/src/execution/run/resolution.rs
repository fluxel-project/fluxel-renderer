//! Resolves and validates graph resources for one frame.

use std::collections::{HashMap, HashSet};

use crate::{
    CompiledGraph,
    backend::{
        BoundBuffer, BoundTexture, ExecutionBackend, ExecutionError, FrameBindingError,
        FrameBindingErrorKind, FrameResourceProvider, PresentationSubmission,
    },
    handles::ResourceId,
    internal::{ResourceKind, ResourceOrigin, RootDecl},
    plan::{BufferUsage, ResourceUsageSummary, TextureUsage},
    rhi::{BufferDesc, ResourceAccessState, TextureDesc},
};

use super::super::{
    FrameExecution,
    recording::{PhysicalResource, PhysicalResources},
};
use super::transients::{TransientPool, TransientReservations};

pub(super) fn live_resources<F>(graph: &CompiledGraph<F>) -> HashSet<ResourceId> {
    let mut live = HashSet::new();
    for pass in &graph.passes {
        live.extend(pass.accesses.iter().map(|access| access.resource));
    }
    for root in &graph.roots {
        match root {
            RootDecl::Texture(_, resource, _, _)
            | RootDecl::Buffer(_, resource, _, _)
            | RootDecl::Present(_, resource, _, _) => {
                live.insert(*resource);
            }
            RootDecl::SideEffect(..) => {}
        }
    }
    live
}

pub(super) struct ResolvedFrameResources<B: ExecutionBackend> {
    pub(super) physical: PhysicalResources<B>,
    pub(super) resource_leases: HashMap<ResourceId, B::Lease>,
    pub(super) retained: Vec<B::Lease>,
    pub(super) presentations: Vec<PresentationSubmission<B::PresentationToken>>,
    pub(super) transient_reservations: TransientReservations<B>,
}

struct TextureBindingContract {
    resource: ResourceId,
    descriptor: TextureDesc,
    state: ResourceAccessState,
    usage: TextureUsage,
    texture_slot: Option<crate::ImportTextureSlot>,
    surface_binding: Option<crate::SurfaceBindingId>,
}

pub(super) fn resolve_resources<B, R, F, M>(
    graph: &CompiledGraph<F>,
    execution: &FrameExecution<F, M>,
    provider: &R,
    backend: &mut B,
    transient_pool: &mut TransientPool<B>,
    live: &HashSet<ResourceId>,
) -> Result<ResolvedFrameResources<B>, ExecutionError<B::Error>>
where
    B: ExecutionBackend,
    R: FrameResourceProvider<B>,
{
    let texture_bindings: HashMap<_, _> = execution.inputs.textures.iter().copied().collect();
    let buffer_bindings: HashMap<_, _> = execution.inputs.buffers.iter().copied().collect();
    let surface_bindings: HashMap<_, _> = execution.inputs.surfaces.iter().copied().collect();
    prevalidate_import_bindings(
        graph,
        live,
        &texture_bindings,
        &buffer_bindings,
        &surface_bindings,
    )?;
    let exported: HashSet<_> = graph
        .roots
        .iter()
        .filter_map(|root| match root {
            RootDecl::Texture(_, resource, _, _) | RootDecl::Buffer(_, resource, _, _) => {
                Some(*resource)
            }
            RootDecl::Present(..) | RootDecl::SideEffect(..) => None,
        })
        .collect();
    let mut physical = HashMap::new();
    let mut resource_leases = HashMap::new();
    let mut retained = Vec::new();
    let mut surface_tokens = HashMap::new();
    let mut transient_reservations = TransientReservations::default();
    // Texture and buffer identities live in independent backend namespaces.
    let mut texture_identities = HashMap::new();
    let mut buffer_identities = HashMap::new();
    for resource in graph
        .resources
        .iter()
        .filter(|resource| live.contains(&resource.id))
    {
        match (resource.kind, resource.origin) {
            (ResourceKind::Texture(descriptor), ResourceOrigin::Transient) => {
                let usage = match graph.execution_plan().resource_requirement(resource.id) {
                    Some(ResourceUsageSummary::Texture(usage)) => usage,
                    _ => unreachable!("live transient texture has a texture usage requirement"),
                };
                let cache_final = cacheable_final_state(graph, resource.id);
                let final_state = cache_final.unwrap_or(ResourceAccessState::Undefined);
                let bound = if exported.contains(&resource.id) || cache_final.is_none() {
                    backend.create_transient_texture(descriptor, usage)
                } else {
                    transient_pool.checkout_texture(
                        backend,
                        graph.identity,
                        resource.id,
                        descriptor,
                        usage,
                        final_state,
                        &mut transient_reservations,
                    )
                }
                .map_err(ExecutionError::Backend)?;
                validate_bound_texture(
                    backend,
                    TextureBindingContract {
                        resource: resource.id,
                        descriptor,
                        state: bound.initial_state,
                        usage,
                        texture_slot: None,
                        surface_binding: None,
                    },
                    &bound,
                )?;
                reject_alias(&mut texture_identities, bound.identity, resource.id)?;
                resource_leases.insert(resource.id, bound.lease.clone());
                retained.push(bound.lease);
                physical.insert(
                    resource.id,
                    PhysicalResource::Texture {
                        physical: bound.physical,
                        descriptor,
                        initial_state: bound.initial_state,
                    },
                );
            }
            (ResourceKind::Buffer(descriptor), ResourceOrigin::Transient) => {
                let usage = match graph.execution_plan().resource_requirement(resource.id) {
                    Some(ResourceUsageSummary::Buffer(usage)) => usage,
                    _ => unreachable!("live transient buffer has a buffer usage requirement"),
                };
                let cache_final = cacheable_final_state(graph, resource.id);
                let final_state = cache_final.unwrap_or(ResourceAccessState::Undefined);
                let bound = if exported.contains(&resource.id) || cache_final.is_none() {
                    backend.create_transient_buffer(descriptor, usage)
                } else {
                    transient_pool.checkout_buffer(
                        backend,
                        graph.identity,
                        resource.id,
                        descriptor,
                        usage,
                        final_state,
                        &mut transient_reservations,
                    )
                }
                .map_err(ExecutionError::Backend)?;
                validate_bound_buffer(
                    backend,
                    resource.id,
                    descriptor,
                    bound.initial_state,
                    usage,
                    None,
                    &bound,
                )?;
                reject_alias(&mut buffer_identities, bound.identity, resource.id)?;
                resource_leases.insert(resource.id, bound.lease.clone());
                retained.push(bound.lease);
                physical.insert(
                    resource.id,
                    PhysicalResource::Buffer {
                        physical: bound.physical,
                        descriptor,
                        initial_state: bound.initial_state,
                    },
                );
            }
            (ResourceKind::Texture(descriptor), ResourceOrigin::TextureImport(slot, contract)) => {
                if surface_bindings.contains_key(&slot) {
                    return Err(frame_error(
                        FrameBindingErrorKind::ConflictingSlotBinding,
                        Some(slot),
                        None,
                        "ordinary texture import slot also has a surface binding",
                    ));
                }
                let id = texture_bindings.get(&slot).copied().ok_or_else(|| {
                    frame_error(
                        FrameBindingErrorKind::MissingTexture,
                        Some(slot),
                        None,
                        "required texture import was not bound",
                    )
                })?;
                let bound = provider.texture(id).map_err(ExecutionError::FrameBinding)?;
                let usage = match graph.execution_plan().resource_requirement(resource.id) {
                    Some(ResourceUsageSummary::Texture(usage)) => usage,
                    _ => unreachable!("live imported texture has a texture usage requirement"),
                };
                validate_bound_texture(
                    backend,
                    TextureBindingContract {
                        resource: resource.id,
                        descriptor,
                        state: contract.initial_state,
                        usage,
                        texture_slot: Some(slot),
                        surface_binding: None,
                    },
                    &bound,
                )?;
                reject_alias(&mut texture_identities, bound.identity, resource.id)?;
                resource_leases.insert(resource.id, bound.lease.clone());
                retained.push(bound.lease);
                physical.insert(
                    resource.id,
                    PhysicalResource::Texture {
                        physical: bound.physical,
                        descriptor,
                        initial_state: bound.initial_state,
                    },
                );
            }
            (ResourceKind::Buffer(descriptor), ResourceOrigin::BufferImport(slot, contract)) => {
                let id = buffer_bindings.get(&slot).copied().ok_or_else(|| {
                    frame_error(
                        FrameBindingErrorKind::MissingBuffer,
                        None,
                        Some(slot),
                        "required buffer import was not bound",
                    )
                })?;
                let bound = provider.buffer(id).map_err(ExecutionError::FrameBinding)?;
                let usage = match graph.execution_plan().resource_requirement(resource.id) {
                    Some(ResourceUsageSummary::Buffer(usage)) => usage,
                    _ => unreachable!("live imported buffer has a buffer usage requirement"),
                };
                validate_bound_buffer(
                    backend,
                    resource.id,
                    descriptor,
                    contract.initial_state,
                    usage,
                    Some(slot),
                    &bound,
                )?;
                reject_alias(&mut buffer_identities, bound.identity, resource.id)?;
                resource_leases.insert(resource.id, bound.lease.clone());
                retained.push(bound.lease);
                physical.insert(
                    resource.id,
                    PhysicalResource::Buffer {
                        physical: bound.physical,
                        descriptor,
                        initial_state: bound.initial_state,
                    },
                );
            }
            (ResourceKind::Texture(descriptor), ResourceOrigin::Surface(slot, _)) => {
                if texture_bindings.contains_key(&slot) {
                    return Err(frame_error(
                        FrameBindingErrorKind::ConflictingSlotBinding,
                        Some(slot),
                        None,
                        "ordinary texture import slot also has a surface binding",
                    ));
                }
                let binding = surface_bindings.get(&slot).copied().ok_or_else(|| {
                    frame_error(
                        FrameBindingErrorKind::MissingSurface,
                        Some(slot),
                        None,
                        "required acquired surface image was not bound",
                    )
                })?;
                let bound = provider
                    .surface(binding)
                    .map_err(ExecutionError::FrameBinding)?;
                let usage = match graph.execution_plan().resource_requirement(resource.id) {
                    Some(ResourceUsageSummary::Texture(usage)) => usage,
                    _ => unreachable!("live surface texture has a texture usage requirement"),
                };
                validate_bound_texture(
                    backend,
                    TextureBindingContract {
                        resource: resource.id,
                        descriptor,
                        state: ResourceAccessState::Present,
                        usage,
                        texture_slot: Some(slot),
                        surface_binding: Some(binding),
                    },
                    &bound.texture,
                )?;
                reject_alias(&mut texture_identities, bound.texture.identity, resource.id)?;
                resource_leases.insert(resource.id, bound.texture.lease.clone());
                retained.push(bound.texture.lease);
                surface_tokens.insert(resource.id, bound.presentation);
                physical.insert(
                    resource.id,
                    PhysicalResource::Texture {
                        physical: bound.texture.physical,
                        descriptor,
                        initial_state: bound.texture.initial_state,
                    },
                );
            }
            _ => unreachable!("validated resource kind and origin match"),
        }
    }
    let presentations = graph
        .roots
        .iter()
        .filter_map(|root| match root {
            RootDecl::Present(target, resource, _, _) => Some((*target, *resource)),
            _ => None,
        })
        .map(|(target, resource)| PresentationSubmission {
            target,
            token: surface_tokens
                .remove(&resource)
                .expect("validated present root retains one resolved surface image"),
        })
        .collect();
    Ok(ResolvedFrameResources {
        physical,
        resource_leases,
        retained,
        presentations,
        transient_reservations,
    })
}

/// Finds the one state a cached allocation will have after this plan, but only
/// for whole-resource transitions. Partial state tracking would require a
/// subresource state vector, so those plans deliberately allocate afresh.
fn cacheable_final_state<F>(
    graph: &CompiledGraph<F>,
    resource: ResourceId,
) -> Option<ResourceAccessState> {
    use crate::plan::PlannedResourceRange;

    let mut final_state = None;
    for transition in graph
        .execution_plan()
        .passes()
        .iter()
        .flat_map(|pass| pass.transitions.iter())
        .chain(graph.execution_plan().final_transitions())
    {
        if transition.resource != resource {
            continue;
        }
        let whole = match transition.range {
            PlannedResourceRange::Texture(crate::TextureRange::Whole) => true,
            PlannedResourceRange::Buffer(crate::BufferRange::Whole) => true,
            PlannedResourceRange::Buffer(crate::BufferRange::Bytes { offset: 0, size }) => {
                matches!(
                    graph.resources.iter().find(|candidate| candidate.id == resource).map(|candidate| candidate.kind),
                    Some(ResourceKind::Buffer(descriptor)) if size == descriptor.size
                )
            }
            _ => false,
        };
        if !whole {
            return None;
        }
        // The immutable compiler has already proven transition continuity;
        // this helper only decides whether its ranges are cacheable.
        final_state = Some(transition.after);
    }
    final_state
}

/// Checks frame-owned import identities before resolution can allocate or lease
/// anything. Providers may acquire or account for resources, so a missing
/// retained import must fail without observable provider or backend effects.
fn prevalidate_import_bindings<E, F>(
    graph: &CompiledGraph<F>,
    live: &HashSet<ResourceId>,
    texture_bindings: &HashMap<crate::ImportTextureSlot, crate::TextureBindingId>,
    buffer_bindings: &HashMap<crate::ImportBufferSlot, crate::BufferBindingId>,
    surface_bindings: &HashMap<crate::ImportTextureSlot, crate::SurfaceBindingId>,
) -> Result<(), ExecutionError<E>> {
    for resource in graph
        .resources
        .iter()
        .filter(|resource| live.contains(&resource.id))
    {
        match resource.origin {
            ResourceOrigin::TextureImport(slot, _) if surface_bindings.contains_key(&slot) => {
                return Err(frame_error(
                    FrameBindingErrorKind::ConflictingSlotBinding,
                    Some(slot),
                    None,
                    "ordinary texture import slot also has a surface binding",
                ));
            }
            ResourceOrigin::TextureImport(slot, _) if !texture_bindings.contains_key(&slot) => {
                return Err(frame_error(
                    FrameBindingErrorKind::MissingTexture,
                    Some(slot),
                    None,
                    "required texture import was not bound",
                ));
            }
            ResourceOrigin::BufferImport(slot, _) if !buffer_bindings.contains_key(&slot) => {
                return Err(frame_error(
                    FrameBindingErrorKind::MissingBuffer,
                    None,
                    Some(slot),
                    "required buffer import was not bound",
                ));
            }
            ResourceOrigin::Surface(slot, _) if !surface_bindings.contains_key(&slot) => {
                return Err(frame_error(
                    FrameBindingErrorKind::MissingSurface,
                    Some(slot),
                    None,
                    "required acquired surface image was not bound",
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

fn reject_alias<E>(
    identities: &mut HashMap<crate::PhysicalResourceIdentity, ResourceId>,
    physical: crate::PhysicalResourceIdentity,
    logical: ResourceId,
) -> Result<(), ExecutionError<E>> {
    // The plan tracks versions and transitions per logical resource. Because no
    // physical-alias barrier model exists, sharing one generation between two
    // logical IDs would make both state histories unsound and must fail closed.
    if let Some(previous) = identities.insert(physical, logical) {
        if previous != logical {
            return Err(frame_error(
                FrameBindingErrorKind::AliasedPhysicalResource,
                None,
                None,
                "distinct logical resources resolved to one physical generation",
            ));
        }
    }
    Ok(())
}

fn validate_bound_texture<B: ExecutionBackend>(
    backend: &B,
    contract: TextureBindingContract,
    bound: &BoundTexture<B::Texture, B::Lease>,
) -> Result<(), ExecutionError<B::Error>> {
    if bound.device != backend.device_identity() {
        return Err(texture_binding_error(
            FrameBindingErrorKind::DeviceMismatch,
            contract.resource,
            contract.texture_slot,
            contract.surface_binding,
            "texture belongs to another backend device",
        ));
    }
    if bound.descriptor != contract.descriptor {
        return Err(texture_binding_error(
            FrameBindingErrorKind::DescriptorMismatch,
            contract.resource,
            contract.texture_slot,
            contract.surface_binding,
            "texture descriptor differs from the compiled import contract",
        ));
    }
    if bound.initial_state != contract.state {
        return Err(texture_binding_error(
            FrameBindingErrorKind::InitialStateMismatch,
            contract.resource,
            contract.texture_slot,
            contract.surface_binding,
            "texture incoming state differs from the compiled import contract",
        ));
    }
    if !bound.usage.contains_all(contract.usage) {
        return Err(ExecutionError::FrameBinding(FrameBindingError {
            kind: FrameBindingErrorKind::UsageMismatch,
            texture_slot: contract.texture_slot,
            buffer_slot: None,
            resource: Some(contract.resource),
            surface_binding: contract.surface_binding,
            detail: format!(
                "texture allowed operations do not cover the compiled requirement: required {:?}, actual {:?}",
                contract.usage, bound.usage
            ),
        }));
    }
    Ok(())
}

fn texture_binding_error<E>(
    kind: FrameBindingErrorKind,
    resource: ResourceId,
    texture_slot: Option<crate::ImportTextureSlot>,
    surface_binding: Option<crate::SurfaceBindingId>,
    detail: impl Into<String>,
) -> ExecutionError<E> {
    ExecutionError::FrameBinding(FrameBindingError {
        kind,
        texture_slot,
        buffer_slot: None,
        resource: Some(resource),
        surface_binding,
        detail: detail.into(),
    })
}

fn validate_bound_buffer<B: ExecutionBackend>(
    backend: &B,
    resource: ResourceId,
    descriptor: BufferDesc,
    state: ResourceAccessState,
    required_usage: BufferUsage,
    buffer_slot: Option<crate::ImportBufferSlot>,
    bound: &BoundBuffer<B::Buffer, B::Lease>,
) -> Result<(), ExecutionError<B::Error>> {
    if bound.device != backend.device_identity() {
        return Err(frame_error(
            FrameBindingErrorKind::DeviceMismatch,
            None,
            buffer_slot,
            "buffer belongs to another backend device",
        ));
    }
    if bound.descriptor != descriptor {
        return Err(frame_error(
            FrameBindingErrorKind::DescriptorMismatch,
            None,
            buffer_slot,
            "buffer descriptor differs from the compiled import contract",
        ));
    }
    if bound.initial_state != state {
        return Err(frame_error(
            FrameBindingErrorKind::InitialStateMismatch,
            None,
            buffer_slot,
            "buffer incoming state differs from the compiled import contract",
        ));
    }
    if !bound.usage.contains_all(required_usage) {
        return Err(ExecutionError::FrameBinding(FrameBindingError {
            kind: FrameBindingErrorKind::UsageMismatch,
            texture_slot: None,
            buffer_slot,
            resource: Some(resource),
            surface_binding: None,
            detail: format!(
                "buffer allowed operations do not cover the compiled requirement: required {required_usage:?}, actual {:?}",
                bound.usage
            ),
        }));
    }
    Ok(())
}

fn frame_error<E>(
    kind: FrameBindingErrorKind,
    texture_slot: Option<crate::ImportTextureSlot>,
    buffer_slot: Option<crate::ImportBufferSlot>,
    detail: impl Into<String>,
) -> ExecutionError<E> {
    ExecutionError::FrameBinding(FrameBindingError {
        kind,
        texture_slot,
        buffer_slot,
        resource: None,
        surface_binding: None,
        detail: detail.into(),
    })
}
