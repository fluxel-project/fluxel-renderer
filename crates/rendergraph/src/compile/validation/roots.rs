//! Validates imports, exports, presentation, and their boundary states.

use super::super::*;

pub(in crate::compile) fn validate_roots<F>(
    graph: &RenderGraph<F>,
    resources: &HashMap<ResourceId, &ResourceDecl>,
    writers: &HashMap<VersionKey, PassId>,
    retained: &HashSet<PassId>,
    caps: &DeviceCapabilities,
) -> Result<(), CompileError> {
    let mut resource_roots = HashMap::new();
    let mut present_resources = HashSet::new();
    for root in &graph.roots {
        let root_fact = match *root {
            RootDecl::Texture(_, resource, version, contract) => {
                Some((resource, version, contract.final_state))
            }
            RootDecl::Buffer(_, resource, version, contract) => {
                Some((resource, version, contract.final_state))
            }
            RootDecl::Present(_, resource, version, _) => {
                Some((resource, version, ResourceAccessState::Present))
            }
            RootDecl::SideEffect(..) => None,
        };
        if let Some((resource, version, state)) = root_fact {
            if let Some(&(previous_version, previous_state)) = resource_roots.get(&resource) {
                if previous_version != version || previous_state != state {
                    return Err(err(
                        CompileErrorKind::InvalidExportOrPresent,
                        Vec::new(),
                        Some(resource),
                        "one physical resource cannot expose conflicting root versions or final states",
                        None,
                    ));
                }
            } else {
                resource_roots.insert(resource, (version, state));
            }
        }
        match *root {
            RootDecl::Texture(_, r, v, contract) => {
                if !matches!(resources[&r].kind, ResourceKind::Texture(_))
                    || v > 0 && !writers.contains_key(&(r, v))
                    || !texture_boundary_state(
                        match resources[&r].kind {
                            ResourceKind::Texture(desc) => desc.format,
                            _ => unreachable!(),
                        },
                        contract.final_state,
                        false,
                    )
                {
                    return Err(err(
                        CompileErrorKind::InvalidExportOrPresent,
                        Vec::new(),
                        Some(r),
                        "invalid texture export",
                        None,
                    ));
                }
                let ResourceKind::Texture(desc) = resources[&r].kind else {
                    unreachable!()
                };
                if !texture_state_supported(caps, desc.format, contract.final_state) {
                    return Err(unsupported_root(
                        r,
                        "texture export state is unsupported for its format",
                        CapabilityRequirement::TextureState {
                            format: desc.format,
                            sample_count: desc.sample_count,
                            state: contract.final_state,
                        },
                        caps,
                    ));
                }
            }
            RootDecl::Buffer(_, r, v, contract) => {
                if !matches!(resources[&r].kind, ResourceKind::Buffer(_))
                    || v > 0 && !writers.contains_key(&(r, v))
                    || !buffer_boundary_state(contract.final_state, false)
                {
                    return Err(err(
                        CompileErrorKind::InvalidExportOrPresent,
                        Vec::new(),
                        Some(r),
                        "invalid buffer export",
                        None,
                    ));
                }
                if !buffer_state_supported(caps, contract.final_state) {
                    return Err(unsupported_root(
                        r,
                        "buffer export state is unsupported",
                        CapabilityRequirement::BufferState {
                            state: contract.final_state,
                        },
                        caps,
                    ));
                }
            }
            RootDecl::Present(_, r, v, _) => {
                if !present_resources.insert(r) {
                    return Err(err(
                        CompileErrorKind::InvalidExportOrPresent,
                        Vec::new(),
                        Some(r),
                        "one acquired presentation image cannot satisfy multiple present roots",
                        None,
                    ));
                }
                let d = resources[&r];
                let ResourceKind::Texture(t) = d.kind else {
                    return Err(err(
                        CompileErrorKind::InvalidExportOrPresent,
                        Vec::new(),
                        Some(r),
                        "present root is not a texture",
                        None,
                    ));
                };
                if !matches!(d.origin, ResourceOrigin::Surface(_, _))
                    || v > 0 && !writers.contains_key(&(r, v))
                {
                    return Err(err(
                        CompileErrorKind::InvalidExportOrPresent,
                        Vec::new(),
                        Some(r),
                        "present root is not a produced surface texture",
                        None,
                    ));
                }
                let Some(s) = &caps.surface else {
                    return Err(unsupported_root(
                        r,
                        "surface capabilities unavailable",
                        CapabilityRequirement::Surface {
                            operation: SurfaceCapabilityOperation::Availability,
                            format: t.format,
                        },
                        caps,
                    ));
                };
                if !s.formats.contains(&t.format) {
                    return Err(unsupported_root(
                        r,
                        "surface format is unsupported for presentation",
                        CapabilityRequirement::Surface {
                            operation: SurfaceCapabilityOperation::Format,
                            format: t.format,
                        },
                        caps,
                    ));
                }
                if !caps.queues.iter().any(|q| q.capabilities.present) {
                    return Err(unsupported_root(
                        r,
                        "no queue supports presentation",
                        CapabilityRequirement::Queue {
                            pass_kinds: Vec::new(),
                            present: true,
                        },
                        caps,
                    ));
                }
            }
            RootDecl::SideEffect(..) => {}
        }
    }
    let live_surface_resources: HashSet<_> = graph
        .passes
        .iter()
        .filter(|pass| retained.contains(&pass.id))
        .flat_map(|pass| pass.accesses.iter().map(|access| access.resource))
        .filter(|resource| matches!(resources[resource].origin, ResourceOrigin::Surface(_, _)))
        .chain(present_resources.iter().copied())
        .collect();
    for resource in live_surface_resources {
        if !present_resources.contains(&resource) {
            return Err(err(
                CompileErrorKind::InvalidExportOrPresent,
                Vec::new(),
                Some(resource),
                "every acquired surface image must have exactly one present root",
                None,
            ));
        }
    }
    Ok(())
}

pub(in crate::compile) fn texture_boundary_state(
    format: TextureFormat,
    state: ResourceAccessState,
    allow_undefined: bool,
) -> bool {
    if allow_undefined && state == ResourceAccessState::Undefined {
        return true;
    }
    match state {
        ResourceAccessState::ColorAttachmentRead
        | ResourceAccessState::ColorAttachmentWrite
        | ResourceAccessState::ColorAttachmentReadWrite => format != TextureFormat::Depth32Float,
        ResourceAccessState::DepthStencilRead
        | ResourceAccessState::DepthStencilWrite
        | ResourceAccessState::DepthStencilReadWrite => format == TextureFormat::Depth32Float,
        ResourceAccessState::ShaderSampledRead
        | ResourceAccessState::ShaderStorageRead
        | ResourceAccessState::ShaderStorageWrite
        | ResourceAccessState::ShaderStorageReadWrite
        | ResourceAccessState::CopySource
        | ResourceAccessState::CopyDestination => true,
        _ => false,
    }
}

pub(in crate::compile) fn buffer_boundary_state(
    state: ResourceAccessState,
    allow_undefined: bool,
) -> bool {
    (allow_undefined && state == ResourceAccessState::Undefined)
        || matches!(
            state,
            ResourceAccessState::ShaderStorageRead
                | ResourceAccessState::ShaderStorageWrite
                | ResourceAccessState::ShaderStorageReadWrite
                | ResourceAccessState::UniformRead
                | ResourceAccessState::VertexRead
                | ResourceAccessState::IndexRead
                | ResourceAccessState::IndirectRead
                | ResourceAccessState::CopySource
                | ResourceAccessState::CopyDestination
        )
}

pub(in crate::compile) fn texture_state_supported(
    caps: &DeviceCapabilities,
    format: TextureFormat,
    state: ResourceAccessState,
) -> bool {
    let Some(facts) = caps
        .texture_formats
        .iter()
        .find(|facts| facts.format == format)
    else {
        return false;
    };
    match state {
        ResourceAccessState::ColorAttachmentRead
        | ResourceAccessState::ColorAttachmentWrite
        | ResourceAccessState::ColorAttachmentReadWrite => facts.color_attachment,
        ResourceAccessState::DepthStencilRead
        | ResourceAccessState::DepthStencilWrite
        | ResourceAccessState::DepthStencilReadWrite => facts.depth_stencil_attachment,
        ResourceAccessState::ShaderSampledRead => facts.sampled,
        ResourceAccessState::ShaderStorageRead => facts.storage_read,
        ResourceAccessState::ShaderStorageWrite => facts.storage_write,
        ResourceAccessState::ShaderStorageReadWrite => facts.storage_read && facts.storage_write,
        ResourceAccessState::CopySource => facts.copy_source,
        ResourceAccessState::CopyDestination => facts.copy_destination,
        _ => false,
    }
}

pub(in crate::compile) fn buffer_state_supported(
    caps: &DeviceCapabilities,
    state: ResourceAccessState,
) -> bool {
    match state {
        ResourceAccessState::ShaderStorageRead => caps.buffers.storage_read,
        ResourceAccessState::ShaderStorageWrite => caps.buffers.storage_write,
        ResourceAccessState::ShaderStorageReadWrite => {
            caps.buffers.storage_read && caps.buffers.storage_write
        }
        ResourceAccessState::IndirectRead => caps.buffers.indirect_read,
        ResourceAccessState::UniformRead
        | ResourceAccessState::VertexRead
        | ResourceAccessState::IndexRead
        | ResourceAccessState::CopySource
        | ResourceAccessState::CopyDestination => true,
        _ => false,
    }
}
