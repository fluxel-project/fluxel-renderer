//! Native resource construction and deterministic owner destruction.

use super::*;

/// Retains a presentation ticket when accepted work lacks a completion proof.
///
/// This is intentionally separate from ordinary resource quarantine so the
/// surface capacity gate cannot become free merely because completion Drop
/// runs after a failed/unknown presentation.
#[cfg(any(feature = "dx12", feature = "vulkan"))]
pub(crate) fn quarantine_presentation_lease(lease: Option<NativePresentationLease>) {
    if let Some(lease) = lease {
        lease.quarantine_surface();
        std::mem::forget(lease);
    }
}

impl Drop for OwnedBuffer {
    fn drop(&mut self) {
        let native = self.native.take().expect("owned buffer destroyed once");
        #[allow(
            unreachable_patterns,
            reason = "single-feature builds have one backend variant"
        )]
        match (&self.owner.native, native) {
            #[cfg(feature = "dx12")]
            (NativeDevice::Dx12 { device, .. }, NativeBuffer::Dx12(buffer)) => {
                // SAFETY: this buffer was created by this live device. Submission
                // bundles retain a strong resource lease through completion, so
                // the unique final owner cannot run while commands reference it.
                unsafe { device.destroy_buffer(buffer) };
            }
            #[cfg(feature = "vulkan")]
            (NativeDevice::Vulkan { device, .. }, NativeBuffer::Vulkan(buffer)) => {
                // SAFETY: same-device ownership and in-flight leases are retained;
                // the Option enforces one native destruction.
                unsafe { device.destroy_buffer(buffer) };
            }
            _ => unreachable!("resource and device backend always match"),
        }
        #[cfg(test)]
        DESTROYED_BUFFERS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

impl Drop for OwnedTexture {
    fn drop(&mut self) {
        let native = self.native.take().expect("owned texture destroyed once");
        #[allow(
            unreachable_patterns,
            reason = "single-feature builds have one backend variant"
        )]
        match (&self.owner.native, native) {
            #[cfg(feature = "dx12")]
            (NativeDevice::Dx12 { device, .. }, NativeTexture::Dx12(texture)) => {
                // SAFETY: this texture belongs to the retained live device and
                // in-flight command bundles keep its resource lease alive.
                unsafe { device.destroy_texture(texture) };
            }
            #[cfg(feature = "vulkan")]
            (NativeDevice::Vulkan { device, .. }, NativeTexture::Vulkan(texture)) => {
                // SAFETY: same-device ownership, completion retention, and unique
                // final destruction are enforced structurally.
                unsafe { device.destroy_texture(texture) };
            }
            _ => unreachable!("resource and device backend always match"),
        }
        #[cfg(test)]
        DESTROYED_TEXTURES.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

#[cfg(test)]
static DESTROYED_BUFFERS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static DESTROYED_TEXTURES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(test)]
pub(crate) fn destruction_counts() -> (usize, usize) {
    use std::sync::atomic::Ordering;
    (
        DESTROYED_BUFFERS.load(Ordering::SeqCst),
        DESTROYED_TEXTURES.load(Ordering::SeqCst),
    )
}

pub(crate) fn create_buffer(
    owner: &Arc<OpenedDevice>,
    descriptor: BufferDescriptor,
) -> Result<(OwnedBuffer, BufferUsage), ResourceCreateError> {
    let native_usage = lower_buffer_usage(descriptor.usage);
    let hal_desc = wgpu_hal::BufferDescriptor {
        label: Some("fluxel owned buffer"),
        size: descriptor.buffer.size,
        usage: native_usage,
        memory_flags: wgpu_hal::MemoryFlags::empty(),
    };
    let native = match &owner.native {
        #[cfg(feature = "dx12")]
        NativeDevice::Dx12 { device, .. } => {
            // SAFETY: the safe layer validated non-zero size and domain usage;
            // lowering uses only HAL-defined flags and no mapping is requested.
            NativeBuffer::Dx12(
                unsafe { device.create_buffer(&hal_desc) }
                    .map_err(|error| resource_error(Backend::Dx12, error))?,
            )
        }
        #[cfg(feature = "vulkan")]
        NativeDevice::Vulkan { device, .. } => {
            // SAFETY: identical portable validation precedes backend-specific
            // creation and MemoryFlags is the validated DeviceOnly policy.
            NativeBuffer::Vulkan(
                unsafe { device.create_buffer(&hal_desc) }
                    .map_err(|error| resource_error(Backend::Vulkan, error))?,
            )
        }
    };
    Ok((
        OwnedBuffer {
            native: Some(native),
            owner: Arc::clone(owner),
            size: descriptor.buffer.size,
            allowed_usage: buffer_usage_from_native(native_usage),
        },
        buffer_usage_from_native(native_usage),
    ))
}

pub(crate) fn create_texture(
    owner: &Arc<OpenedDevice>,
    descriptor: TextureDescriptor,
) -> Result<(OwnedTexture, TextureUsage), ResourceCreateError> {
    let native_usage = lower_texture_usage(descriptor.usage);
    let texture = descriptor.texture;
    let hal_desc = wgpu_hal::TextureDescriptor {
        label: Some("fluxel owned texture"),
        size: wgt::Extent3d {
            width: texture.extent.width,
            height: texture.extent.height,
            depth_or_array_layers: match texture.dimension {
                TextureDimension::D2 => texture.array_layers,
                TextureDimension::D3 => texture.extent.depth,
                _ => 1,
            },
        },
        mip_level_count: texture.mip_levels,
        sample_count: texture.sample_count,
        dimension: lower_dimension(texture.dimension),
        format: lower_format(texture.format),
        usage: native_usage,
        memory_flags: wgpu_hal::MemoryFlags::empty(),
        view_formats: Vec::new(),
    };
    let native = match &owner.native {
        #[cfg(feature = "dx12")]
        NativeDevice::Dx12 {
            device, adapter, ..
        } => {
            // SAFETY: the format is a valid wgpu format and the retained
            // adapter is the exact parent used to open this device.
            let caps = unsafe { adapter.texture_format_capabilities(hal_desc.format) };
            validate_texture_capabilities(caps, descriptor)?;
            // SAFETY: extent, dimension, mip, sample, format and usage
            // compatibility were validated before this private HAL boundary.
            NativeTexture::Dx12(
                unsafe { device.create_texture(&hal_desc) }
                    .map_err(|error| resource_error(Backend::Dx12, error))?,
            )
        }
        #[cfg(feature = "vulkan")]
        NativeDevice::Vulkan {
            device, adapter, ..
        } => {
            // SAFETY: the retained live adapter accepts this portable format;
            // the query performs no resource operation.
            let caps = unsafe { adapter.texture_format_capabilities(hal_desc.format) };
            validate_texture_capabilities(caps, descriptor)?;
            // SAFETY: all HAL descriptor preconditions consumed by this slice
            // are validated, and the returned texture remains device-owned.
            NativeTexture::Vulkan(
                unsafe { device.create_texture(&hal_desc) }
                    .map_err(|error| resource_error(Backend::Vulkan, error))?,
            )
        }
    };
    let allowed_usage = texture_usage_from_native(native_usage, texture.format);
    Ok((
        OwnedTexture {
            native: Some(native),
            owner: Arc::clone(owner),
            descriptor: texture,
            allowed_usage,
        },
        allowed_usage,
    ))
}

impl Drop for NativeComputePipelineShared {
    fn drop(&mut self) {
        // SAFETY: every object below was created by `owner`; this shared owner
        // keeps that device live. The declaration order is intentionally the
        // reverse native dependency order: pipeline, layout, BGL, shader.
        #[allow(
            unreachable_patterns,
            reason = "single-feature builds have one backend variant"
        )]
        let native = self.native.take().expect("compute pipeline destroyed once");
        match (&self.owner.native, native) {
            #[cfg(feature = "dx12")]
            (
                NativeDevice::Dx12 { device, .. },
                NativeComputePipelineInner::Dx12 {
                    shader,
                    bind_group_layout,
                    pipeline_layout,
                    pipeline,
                },
            ) => unsafe {
                // SAFETY: these same-device objects are uniquely owned here and
                // remain unpublished at Drop; destroy in reverse dependency order.
                device.destroy_compute_pipeline(pipeline);
                device.destroy_pipeline_layout(pipeline_layout);
                device.destroy_bind_group_layout(bind_group_layout);
                device.destroy_shader_module(shader);
            },
            #[cfg(feature = "vulkan")]
            (
                NativeDevice::Vulkan { device, .. },
                NativeComputePipelineInner::Vulkan {
                    shader,
                    bind_group_layout,
                    pipeline_layout,
                    pipeline,
                },
            ) => unsafe {
                // SAFETY: these same-device objects are uniquely owned here and
                // remain unpublished at Drop; destroy in reverse dependency order.
                device.destroy_compute_pipeline(pipeline);
                device.destroy_pipeline_layout(pipeline_layout);
                device.destroy_bind_group_layout(bind_group_layout);
                device.destroy_shader_module(shader);
            },
            _ => unreachable!("pipeline and device backend always match"),
        }
    }
}

impl Drop for NativeRasterPipelineShared {
    fn drop(&mut self) {
        // SAFETY: every object was made by `owner`, which is retained here;
        // pipeline dependencies are destroyed in reverse creation order.
        let native = self.native.take().expect("raster pipeline destroyed once");
        #[allow(unreachable_patterns, reason = "single-backend build")]
        match (&self.owner.native, native) {
            #[cfg(feature = "dx12")]
            (
                NativeDevice::Dx12 { device, .. },
                NativeRasterPipelineInner::Dx12 {
                    vertex_shader,
                    fragment_shader,
                    bind_group_layout,
                    pipeline_layout,
                    pipeline,
                    depth_pipeline,
                },
            ) => unsafe {
                // SAFETY: `owner` created every object, remains live, and no
                // in-flight lease exists when the shared owner reaches Drop.
                device.destroy_render_pipeline(pipeline);
                device.destroy_render_pipeline(depth_pipeline);
                device.destroy_pipeline_layout(pipeline_layout);
                if let Some(layout) = bind_group_layout {
                    device.destroy_bind_group_layout(layout);
                }
                device.destroy_shader_module(fragment_shader);
                device.destroy_shader_module(vertex_shader);
            },
            #[cfg(feature = "vulkan")]
            (
                NativeDevice::Vulkan { device, .. },
                NativeRasterPipelineInner::Vulkan {
                    vertex_shader,
                    fragment_shader,
                    bind_group_layout,
                    pipeline_layout,
                    pipeline,
                    depth_pipeline,
                },
            ) => unsafe {
                // SAFETY: same retained-device, terminal-lifetime, and reverse
                // dependency destruction proof as the DX12 branch.
                device.destroy_render_pipeline(pipeline);
                device.destroy_render_pipeline(depth_pipeline);
                device.destroy_pipeline_layout(pipeline_layout);
                if let Some(layout) = bind_group_layout {
                    device.destroy_bind_group_layout(layout);
                }
                device.destroy_shader_module(fragment_shader);
                device.destroy_shader_module(vertex_shader);
            },
            _ => unreachable!("pipeline and device backend always match"),
        }
    }
}

impl Drop for NativeRasterUniformBindings {
    fn drop(&mut self) {
        let native = self
            .native
            .take()
            .expect("raster uniform bindings destroyed once");
        match (&self.pipeline.0.owner.native, native) {
            #[cfg(feature = "dx12")]
            (NativeDevice::Dx12 { device, .. }, NativeRasterUniformBindingsInner::Dx12(group)) => unsafe {
                // SAFETY: the retained pipeline owns the matching BGL and all
                // in-flight leases retain this group until terminal completion.
                device.destroy_bind_group(group)
            },
            #[cfg(feature = "vulkan")]
            (
                NativeDevice::Vulkan { device, .. },
                NativeRasterUniformBindingsInner::Vulkan(group),
            ) => unsafe {
                // SAFETY: same device/layout/lifetime proof as DX12.
                device.destroy_bind_group(group)
            },
            _ => unreachable!("raster uniform binding and device backend always match"),
        }
    }
}

impl Drop for NativeRasterTextureBindings {
    fn drop(&mut self) {
        let native = self
            .native
            .take()
            .expect("raster texture bindings destroyed once");
        match (&self.pipeline.0.owner.native, native) {
            #[cfg(feature = "dx12")]
            (
                NativeDevice::Dx12 { device, .. },
                NativeRasterTextureBindingsInner::Dx12 {
                    group,
                    view,
                    sampler,
                },
            ) => unsafe {
                // SAFETY: the retained pipeline owns this matching device and
                // layout; ResourceLease keeps the group/view and their bound
                // resources alive until terminal queue completion.
                device.destroy_bind_group(group);
                if let Some(sampler) = sampler {
                    device.destroy_sampler(sampler);
                }
                device.destroy_texture_view(view);
            },
            #[cfg(feature = "vulkan")]
            (
                NativeDevice::Vulkan { device, .. },
                NativeRasterTextureBindingsInner::Vulkan {
                    group,
                    view,
                    sampler,
                },
            ) => unsafe {
                // SAFETY: same device/layout/lifetime proof as DX12.
                device.destroy_bind_group(group);
                if let Some(sampler) = sampler {
                    device.destroy_sampler(sampler);
                }
                device.destroy_texture_view(view);
            },
            _ => unreachable!("textured raster bindings and device backend always match"),
        }
    }
}

impl Drop for NativeComputeBindings {
    fn drop(&mut self) {
        let native = self.native.take().expect("compute bindings destroyed once");
        // SAFETY: `pipeline` keeps the matching device and BGL alive. The safe
        // layer retains the bound buffer lease until submission completion.
        #[allow(
            unreachable_patterns,
            reason = "single-feature builds have one backend variant"
        )]
        match (&self.pipeline.0.owner.native, native) {
            #[cfg(feature = "dx12")]
            (NativeDevice::Dx12 { device, .. }, NativeComputeBindingsInner::Dx12(group)) => unsafe {
                // SAFETY: this same-device group is uniquely owned and its pipeline
                // keeps the layout/device live until this one-time destruction.
                device.destroy_bind_group(group);
            },
            #[cfg(feature = "vulkan")]
            (NativeDevice::Vulkan { device, .. }, NativeComputeBindingsInner::Vulkan(group)) => unsafe {
                // SAFETY: this same-device group is uniquely owned and its pipeline
                // keeps the layout/device live until this one-time destruction.
                device.destroy_bind_group(group);
            },
            #[cfg(feature = "dx12")]
            (
                NativeDevice::Dx12 { device, .. },
                NativeComputeBindingsInner::Dx12Texture { group, view },
            ) => unsafe {
                device.destroy_bind_group(group);
                device.destroy_texture_view(view);
            },
            #[cfg(feature = "vulkan")]
            (
                NativeDevice::Vulkan { device, .. },
                NativeComputeBindingsInner::VulkanTexture { group, view },
            ) => unsafe {
                device.destroy_bind_group(group);
                device.destroy_texture_view(view);
            },
            _ => unreachable!("bindings and pipeline backend always match"),
        }
    }
}

impl Drop for NativeTexturePackBindings {
    fn drop(&mut self) {
        let native = self
            .native
            .take()
            .expect("texture-pack bindings destroyed once");
        // SAFETY: `pipeline` retains the matching device and layout until the
        // bind group is destroyed; resource leases are held by the safe owner.
        #[allow(unreachable_patterns, reason = "single-backend build")]
        match (&self.pipeline.0.owner.native, native) {
            #[cfg(feature = "dx12")]
            (
                NativeDevice::Dx12 { device, .. },
                NativeTexturePackBindingsInner::Dx12 { group, view },
            ) => unsafe {
                // SAFETY: the retained pipeline owns the matching live device;
                // the uniquely consumed group is destroyed before its view.
                device.destroy_bind_group(group);
                device.destroy_texture_view(view)
            },
            #[cfg(feature = "vulkan")]
            (
                NativeDevice::Vulkan { device, .. },
                NativeTexturePackBindingsInner::Vulkan { group, view },
            ) => unsafe {
                // SAFETY: same device ownership, terminal lifetime, and reverse
                // dependency destruction order as DX12.
                device.destroy_bind_group(group);
                device.destroy_texture_view(view)
            },
            _ => unreachable!("bindings and device backend always match"),
        }
    }
}

impl Drop for CopyEncoder {
    fn drop(&mut self) {
        if let Some(mut native) = self.native.take() {
            // SAFETY: a live CopyEncoder is always recording; an unconsumed
            // encoder must discard exactly once and is never submitted.
            unsafe {
                match &mut native {
                    #[cfg(feature = "dx12")]
                    NativeEncoder::Dx12(encoder) => encoder.discard_encoding(),
                    #[cfg(feature = "vulkan")]
                    NativeEncoder::Vulkan(encoder) => encoder.discard_encoding(),
                }
            }
        }
        let mut views = std::mem::take(&mut self.render_views);
        if let Some(view) = self.active_render_view.take() {
            views.push(view);
        }
        destroy_render_views(&self.owner, views);
    }
}

impl Drop for CopyCommandBuffer {
    fn drop(&mut self) {
        if let Some(finished) = self.native.take() {
            reset_finished(finished);
        }
    }
}

pub(crate) fn reset_finished(finished: NativeFinished) {
    match finished {
        #[cfg(feature = "dx12")]
        NativeFinished::Dx12 {
            owner,
            mut encoder,
            command_buffer,
            render_views,
        } => {
            // SAFETY: encoder is closed, the command buffer was never accepted
            // or has completed, and it is the only live buffer from this encoder.
            unsafe { encoder.reset_all(core::iter::once(command_buffer)) };
            destroy_render_views(&owner, render_views);
        }
        #[cfg(feature = "vulkan")]
        NativeFinished::Vulkan {
            owner,
            mut encoder,
            command_buffer,
            render_views,
        } => {
            // SAFETY: same closed-encoder and complete/unsubmitted ownership
            // proof as the DX12 branch.
            unsafe { encoder.reset_all(core::iter::once(command_buffer)) };
            destroy_render_views(&owner, render_views);
        }
    }
}

impl Drop for NativeSubmission {
    fn drop(&mut self) {
        match self {
            #[cfg(feature = "dx12")]
            Self::Dx12 {
                owner,
                encoder,
                command_buffer,
                fence,
                leases,
                staging_buffers,
                render_views,
                presentation_lease,
                presentation_completion_hold,
                failure,
            } => {
                // No fence means a submit failure may already have accepted
                // command lists, while leaving no completion proof. Do not
                // destructure-and-return here: that would drop encoder/views/
                // leases still used by GPU.
                if fence.is_none() {
                    let retained = std::mem::take(leases);
                    let retained_staging = std::mem::take(staging_buffers);
                    let retained_views = std::mem::take(render_views);
                    quarantine_presentation_lease(presentation_lease.take());
                    let retained_presentation_completion_hold = presentation_completion_hold.take();
                    std::mem::forget((
                        owner.clone(),
                        encoder.take(),
                        command_buffer.take(),
                        retained,
                        retained_staging,
                        retained_views,
                        retained_presentation_completion_hold,
                    ));
                    return;
                }
                // A malformed internal bundle is not proof that a prior
                // queue operation was rejected. Retain all pieces (including
                // the post-present gate) rather than allowing field Drop to
                // make surface teardown appear safe.
                if encoder.is_none() || command_buffer.is_none() {
                    let retained = std::mem::take(leases);
                    let retained_staging = std::mem::take(staging_buffers);
                    let retained_views = std::mem::take(render_views);
                    quarantine_presentation_lease(presentation_lease.take());
                    let retained_presentation_completion_hold = presentation_completion_hold.take();
                    std::mem::forget((
                        owner.clone(),
                        encoder.take(),
                        command_buffer.take(),
                        fence.take(),
                        retained,
                        retained_staging,
                        retained_views,
                        retained_presentation_completion_hold,
                    ));
                    return;
                }
                let (
                    NativeDevice::Dx12 { device, .. },
                    Some(mut encoder),
                    Some(command_buffer),
                    Some(fence),
                ) = (
                    &owner.native,
                    encoder.take(),
                    command_buffer.take(),
                    fence.take(),
                )
                else {
                    return;
                };
                // A submit/query failure can mean ExecuteCommandLists was accepted
                // before fence signaling failed. In that state no fence value can
                // prove completion. Retain the complete native/resource bundle for
                // process lifetime rather than resetting storage still in use.
                if failure.is_some() {
                    let retained = std::mem::take(leases);
                    let retained_staging = std::mem::take(staging_buffers);
                    let retained_views = std::mem::take(render_views);
                    quarantine_presentation_lease(presentation_lease.take());
                    let retained_presentation_completion_hold = presentation_completion_hold.take();
                    std::mem::forget((
                        owner.clone(),
                        encoder,
                        command_buffer,
                        fence,
                        retained,
                        retained_staging,
                        retained_views,
                        retained_presentation_completion_hold,
                    ));
                } else if unsafe { device.wait(&fence, 1, None) }.is_ok() {
                    // SAFETY: the sole command buffer completed at fence value 1.
                    unsafe {
                        encoder.reset_all(core::iter::once(command_buffer));
                        device.destroy_fence(fence);
                    }
                    destroy_render_views(owner, std::mem::take(render_views));
                    // Views are now destroyed and fence completion proved, so
                    // a subsequent resize/unconfigure may retire this surface.
                    drop(presentation_lease.take());
                    drop(presentation_completion_hold.take());
                } else {
                    let retained = std::mem::take(leases);
                    let retained_staging = std::mem::take(staging_buffers);
                    let retained_views = std::mem::take(render_views);
                    quarantine_presentation_lease(presentation_lease.take());
                    let retained_presentation_completion_hold = presentation_completion_hold.take();
                    std::mem::forget((
                        owner.clone(),
                        encoder,
                        command_buffer,
                        fence,
                        retained,
                        retained_staging,
                        retained_views,
                        retained_presentation_completion_hold,
                    ));
                }
            }
            #[cfg(feature = "vulkan")]
            Self::Vulkan {
                owner,
                encoder,
                command_buffer,
                fence,
                leases,
                staging_buffers,
                render_views,
                presentation_lease,
                failure,
            } => {
                let (
                    NativeDevice::Vulkan { device, .. },
                    Some(mut encoder),
                    Some(command_buffer),
                    Some(fence),
                ) = (
                    &owner.native,
                    encoder.take(),
                    command_buffer.take(),
                    fence.take(),
                )
                else {
                    return;
                };
                let (fence_ref, target) = match &fence {
                    NativeVulkanFence::Owned(fence) => (fence, 1),
                    NativeVulkanFence::Presentation { sync, value } => match sync.fence() {
                        Ok(fence) => (fence, *value),
                        Err(_) => {
                            let retained = std::mem::take(leases);
                            let retained_staging = std::mem::take(staging_buffers);
                            let retained_views = std::mem::take(render_views);
                            quarantine_presentation_lease(presentation_lease.take());
                            std::mem::forget((
                                owner.clone(),
                                encoder,
                                command_buffer,
                                fence,
                                retained,
                                retained_staging,
                                retained_views,
                            ));
                            return;
                        }
                    },
                };
                if failure.is_some() {
                    let retained = std::mem::take(leases);
                    let retained_staging = std::mem::take(staging_buffers);
                    let retained_views = std::mem::take(render_views);
                    quarantine_presentation_lease(presentation_lease.take());
                    std::mem::forget((
                        owner.clone(),
                        encoder,
                        command_buffer,
                        fence,
                        retained,
                        retained_staging,
                        retained_views,
                    ));
                } else if unsafe { device.wait(fence_ref, target, None) }.is_ok() {
                    // SAFETY: the sole command buffer completed at fence value 1.
                    unsafe {
                        encoder.reset_all(core::iter::once(command_buffer));
                        if let NativeVulkanFence::Owned(fence) = fence {
                            device.destroy_fence(fence);
                        }
                    }
                    destroy_render_views(owner, std::mem::take(render_views));
                    drop(presentation_lease.take());
                } else {
                    let retained = std::mem::take(leases);
                    let retained_staging = std::mem::take(staging_buffers);
                    let retained_views = std::mem::take(render_views);
                    quarantine_presentation_lease(presentation_lease.take());
                    std::mem::forget((
                        owner.clone(),
                        encoder,
                        command_buffer,
                        fence,
                        retained,
                        retained_staging,
                        retained_views,
                    ));
                }
            }
            Self::TerminalFailure(_) => {}
        }
    }
}
