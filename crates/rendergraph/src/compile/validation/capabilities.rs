//! Rejects retained graph semantics unsupported by the selected device.

use super::super::*;

pub(in crate::compile) fn validate_capabilities<F>(
    graph: &RenderGraph<F>,
    resources: &HashMap<ResourceId, &ResourceDecl>,
    retained: &HashSet<PassId>,
    caps: &DeviceCapabilities,
) -> Result<(), CompileError> {
    let mut queue_ids = HashSet::new();
    for queue in &caps.queues {
        if !queue_ids.insert(queue.id) {
            return Err(err(
                CompileErrorKind::UnsupportedSemanticRequirement,
                Vec::new(),
                None,
                "device capabilities contain duplicate logical queue identifiers",
                Some(Box::new(UnsupportedCapability {
                    requirement: CapabilityRequirement::QueueConfiguration,
                    observed: Box::new(caps.clone()),
                })),
            ));
        }
    }
    for resource in &graph.resources {
        match resource.origin {
            ResourceOrigin::TextureImport(_, contract)
                if contract.initial_state != ResourceAccessState::Undefined
                    && !texture_state_supported(
                        caps,
                        contract.descriptor.format,
                        contract.initial_state,
                    ) =>
            {
                return Err(unsupported_root(
                    resource.id,
                    "texture import state is unsupported for its format",
                    CapabilityRequirement::TextureState {
                        format: contract.descriptor.format,
                        sample_count: contract.descriptor.sample_count,
                        state: contract.initial_state,
                    },
                    caps,
                ));
            }
            ResourceOrigin::BufferImport(_, contract)
                if contract.initial_state != ResourceAccessState::Undefined
                    && !buffer_state_supported(caps, contract.initial_state) =>
            {
                return Err(unsupported_root(
                    resource.id,
                    "buffer import state is unsupported",
                    CapabilityRequirement::BufferState {
                        state: contract.initial_state,
                    },
                    caps,
                ));
            }
            _ => {}
        }
    }
    let needs_present = graph
        .roots
        .iter()
        .any(|root| matches!(root, RootDecl::Present(_, _, _, _)));
    let needs_queue = !retained.is_empty()
        || graph.roots.iter().any(|root| {
            matches!(
                root,
                RootDecl::Texture(_, _, _, _)
                    | RootDecl::Buffer(_, _, _, _)
                    | RootDecl::Present(_, _, _, _)
            )
        });
    let single_queue = caps.queues.iter().any(|queue| {
        (!needs_present || queue.capabilities.present)
            && graph
                .passes
                .iter()
                .filter(|pass| retained.contains(&pass.id))
                .all(|pass| match pass.kind {
                    PassKind::Raster => queue.capabilities.raster,
                    PassKind::Compute => queue.capabilities.compute,
                    PassKind::Copy => queue.capabilities.copy,
                })
    });
    if needs_queue && !single_queue {
        let passes = graph
            .passes
            .iter()
            .filter(|pass| retained.contains(&pass.id))
            .map(|pass| pass.id)
            .collect();
        let mut pass_kinds: Vec<_> = graph
            .passes
            .iter()
            .filter(|pass| retained.contains(&pass.id))
            .map(|pass| pass.kind)
            .collect();
        pass_kinds.sort_by_key(|kind| match kind {
            PassKind::Raster => 0,
            PassKind::Compute => 1,
            PassKind::Copy => 2,
        });
        pass_kinds.dedup();
        return Err(err(
            CompileErrorKind::UnsupportedSemanticRequirement,
            passes,
            None,
            "the current single-queue compiler requires one logical queue that supports every retained pass kind and presentation requirement",
            Some(Box::new(UnsupportedCapability {
                requirement: CapabilityRequirement::Queue {
                    pass_kinds,
                    present: needs_present,
                },
                observed: Box::new(caps.clone()),
            })),
        ));
    }
    for p in graph.passes.iter().filter(|p| retained.contains(&p.id)) {
        let queue = caps.queues.iter().any(|q| match p.kind {
            PassKind::Raster => q.capabilities.raster,
            PassKind::Compute => q.capabilities.compute,
            PassKind::Copy => q.capabilities.copy,
        });
        if !queue {
            return Err(unsupported(
                p.id,
                None,
                "no queue supports this pass kind",
                CapabilityRequirement::Queue {
                    pass_kinds: vec![p.kind],
                    present: false,
                },
                caps,
            ));
        }
        let color_indices = p
            .accesses
            .iter()
            .filter_map(|a| {
                if let AccessSemantic::ColorAttachment { index } = a.semantic {
                    Some(index)
                } else {
                    None
                }
            })
            .collect::<HashSet<_>>();
        let required_color_slots = color_indices.iter().max().map_or(0, |index| index + 1);
        if required_color_slots > caps.limits.max_color_attachments {
            return Err(unsupported(
                p.id,
                None,
                "color attachment count exceeds limit",
                CapabilityRequirement::ColorAttachmentCount {
                    required: required_color_slots,
                    supported: caps.limits.max_color_attachments,
                },
                caps,
            ));
        }
        for a in &p.accesses {
            let r = resources[&a.resource];
            match (r.kind, a.semantic) {
                (ResourceKind::Buffer(_), AccessSemantic::BufferRead(BufferReadUse::Storage))
                    if !caps.buffers.storage_read =>
                {
                    return Err(unsupported(
                        p.id,
                        Some(r.id),
                        "storage buffer reads unsupported",
                        CapabilityRequirement::BufferState {
                            state: ResourceAccessState::ShaderStorageRead,
                        },
                        caps,
                    ));
                }
                (ResourceKind::Buffer(_), AccessSemantic::BufferRead(BufferReadUse::Indirect))
                    if !caps.buffers.indirect_read =>
                {
                    return Err(unsupported(
                        p.id,
                        Some(r.id),
                        "indirect buffer reads unsupported",
                        CapabilityRequirement::BufferState {
                            state: ResourceAccessState::IndirectRead,
                        },
                        caps,
                    ));
                }
                (ResourceKind::Buffer(_), AccessSemantic::BufferWrite(BufferWriteUse::Storage))
                    if !caps.buffers.storage_write =>
                {
                    return Err(unsupported(
                        p.id,
                        Some(r.id),
                        "storage buffer writes unsupported",
                        CapabilityRequirement::BufferState {
                            state: ResourceAccessState::ShaderStorageWrite,
                        },
                        caps,
                    ));
                }
                (
                    ResourceKind::Buffer(_),
                    AccessSemantic::BufferReadWrite(BufferReadWriteUse::Storage),
                ) if !caps.buffers.storage_read || !caps.buffers.storage_write => {
                    return Err(unsupported(
                        p.id,
                        Some(r.id),
                        "storage buffer read-write unsupported",
                        CapabilityRequirement::BufferState {
                            state: ResourceAccessState::ShaderStorageReadWrite,
                        },
                        caps,
                    ));
                }
                (ResourceKind::Texture(d), semantic) => {
                    let Some(f) = caps.texture_formats.iter().find(|f| f.format == d.format) else {
                        return Err(unsupported(
                            p.id,
                            Some(r.id),
                            "missing texture format capabilities",
                            CapabilityRequirement::TextureFormat { format: d.format },
                            caps,
                        ));
                    };
                    let ok = match semantic {
                        AccessSemantic::TextureRead(TextureReadUse::Sampled) => f.sampled,
                        AccessSemantic::TextureRead(TextureReadUse::Storage) => f.storage_read,
                        AccessSemantic::TextureRead(TextureReadUse::CopySource) => f.copy_source,
                        AccessSemantic::TextureWrite(TextureWriteUse::Storage) => f.storage_write,
                        AccessSemantic::TextureReadWrite(TextureReadWriteUse::Storage) => {
                            f.storage_read && f.storage_write
                        }
                        AccessSemantic::TextureWrite(TextureWriteUse::CopyDestination) => {
                            f.copy_destination
                        }
                        AccessSemantic::ColorAttachment { .. } => {
                            f.color_attachment
                                && f.attachment_sample_counts.contains(&d.sample_count)
                        }
                        AccessSemantic::DepthStencilAttachment { .. } => {
                            f.depth_stencil_attachment
                                && f.attachment_sample_counts.contains(&d.sample_count)
                        }
                        _ => true,
                    };
                    if !ok {
                        return Err(unsupported(
                            p.id,
                            Some(r.id),
                            "texture semantic unsupported for format",
                            CapabilityRequirement::TextureState {
                                format: d.format,
                                sample_count: d.sample_count,
                                state: texture_state_for_semantic(semantic),
                            },
                            caps,
                        ));
                    }
                    if matches!(r.origin, ResourceOrigin::Surface(_, _)) {
                        let Some(s) = &caps.surface else {
                            return Err(unsupported(
                                p.id,
                                Some(r.id),
                                "surface capabilities unavailable",
                                CapabilityRequirement::Surface {
                                    operation: SurfaceCapabilityOperation::Availability,
                                    format: d.format,
                                },
                                caps,
                            ));
                        };
                        let ok = match semantic {
                            AccessSemantic::ColorAttachment { .. } => s.color_attachment,
                            AccessSemantic::TextureWrite(TextureWriteUse::CopyDestination) => {
                                s.copy_destination
                            }
                            _ => false,
                        };
                        if !ok {
                            return Err(unsupported(
                                p.id,
                                Some(r.id),
                                "surface destination semantic unsupported",
                                CapabilityRequirement::Surface {
                                    operation: surface_operation(semantic),
                                    format: d.format,
                                },
                                caps,
                            ));
                        }
                    }
                }
                _ => {}
            }
        }
    }
    Ok(())
}

fn texture_state_for_semantic(semantic: AccessSemantic) -> ResourceAccessState {
    match semantic {
        AccessSemantic::TextureRead(TextureReadUse::Sampled) => {
            ResourceAccessState::ShaderSampledRead
        }
        AccessSemantic::TextureRead(TextureReadUse::Storage) => {
            ResourceAccessState::ShaderStorageRead
        }
        AccessSemantic::TextureRead(TextureReadUse::CopySource) => ResourceAccessState::CopySource,
        AccessSemantic::TextureWrite(TextureWriteUse::Storage) => {
            ResourceAccessState::ShaderStorageWrite
        }
        AccessSemantic::TextureReadWrite(TextureReadWriteUse::Storage) => {
            ResourceAccessState::ShaderStorageReadWrite
        }
        AccessSemantic::TextureWrite(TextureWriteUse::CopyDestination) => {
            ResourceAccessState::CopyDestination
        }
        AccessSemantic::ColorAttachment { .. } => ResourceAccessState::ColorAttachmentWrite,
        AccessSemantic::DepthStencilAttachment { .. } => ResourceAccessState::DepthStencilWrite,
        _ => unreachable!("texture capability validation only receives texture semantics"),
    }
}

fn surface_operation(semantic: AccessSemantic) -> SurfaceCapabilityOperation {
    match semantic {
        AccessSemantic::ColorAttachment { .. } => SurfaceCapabilityOperation::ColorAttachment,
        AccessSemantic::TextureWrite(TextureWriteUse::CopyDestination) => {
            SurfaceCapabilityOperation::CopyDestination
        }
        AccessSemantic::TextureRead(TextureReadUse::Sampled) => {
            SurfaceCapabilityOperation::SampledRead
        }
        AccessSemantic::TextureRead(TextureReadUse::Storage) => {
            SurfaceCapabilityOperation::StorageRead
        }
        AccessSemantic::TextureWrite(TextureWriteUse::Storage) => {
            SurfaceCapabilityOperation::StorageWrite
        }
        AccessSemantic::TextureReadWrite(TextureReadWriteUse::Storage) => {
            SurfaceCapabilityOperation::StorageReadWrite
        }
        AccessSemantic::TextureRead(TextureReadUse::CopySource) => {
            SurfaceCapabilityOperation::CopySource
        }
        AccessSemantic::DepthStencilAttachment { .. } => {
            SurfaceCapabilityOperation::DepthStencilAttachment
        }
        _ => unreachable!("surface capability validation only receives texture semantics"),
    }
}
