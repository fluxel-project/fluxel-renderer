//! Native raster-pass command recording.
use super::*;

/// Returns the least non-empty byte range accepted for one closed vertex slot.
///
/// This is a native-boundary range sanity check, not the whole-stream
/// contract: typed bindings below still require the exact stream size. In
/// particular, indexed triangles may legally reuse vertices, so slot one of
/// the vertex-color recipe must accept two RGBA8 vertices (eight bytes).
pub(crate) const fn raster_vertex_minimum_size(kernel: crate::RasterKernel, slot: u32) -> u64 {
    match (kernel, slot) {
        (
            crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUv
            | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
            | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb,
            1,
        ) => 8,
        (crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialVertexColor, 1) => 4,
        _ => 12,
    }
}

/// Begins the fixed color pass and its optional Depth32Float attachment.
///
/// Both attachment views are retained until the closed command buffer reaches
/// a terminal outcome. The safe caller has already constrained these to whole,
/// single-sample D2 views with matching extents.
#[allow(
    clippy::too_many_arguments,
    reason = "the private native boundary receives separately validated color and optional depth attachment facts"
)]
pub(crate) fn begin_raster(
    encoder: &mut CopyEncoder,
    texture: &OwnedTexture,
    desc: TextureDesc,
    clear: Option<[f32; 4]>,
    load: bool,
    store: bool,
    depth: Option<(&OwnedTexture, TextureDesc, Option<f32>, bool)>,
    label: &str,
) -> Result<(), String> {
    if encoder.active_render_view.is_some() {
        return Err("a raster pass is already active".into());
    }
    // A native binding is valid only after this pass selects its pipeline.
    // Clear any terminal value left by a prior pass before creating the view.
    encoder.active_raster_pipeline = None;
    if !Arc::ptr_eq(&encoder.owner, &texture.owner) {
        return Err("raster attachment belongs to another native device".into());
    }
    if let Some((depth_texture, _, _, _)) = depth {
        if !Arc::ptr_eq(&encoder.owner, &depth_texture.owner) {
            return Err("depth attachment belongs to another native device".into());
        }
    }
    let expected_state = if load {
        ResourceAccessState::ColorAttachmentReadWrite
    } else {
        ResourceAccessState::ColorAttachmentWrite
    };
    let key = TextureStateKey {
        allocation: core::ptr::from_ref(texture).addr(),
        mip_level: 0,
        array_layer: 0,
        aspect: TextureAspect::Color,
    };
    if encoder.texture_states.get(&key).copied() != Some(expected_state) {
        return Err(format!(
            "raster attachment requires encoder state {expected_state:?}"
        ));
    }
    if let Some((depth_texture, _, _, depth_load)) = depth {
        let expected_state = if depth_load {
            ResourceAccessState::DepthStencilReadWrite
        } else {
            ResourceAccessState::DepthStencilWrite
        };
        let key = TextureStateKey {
            allocation: core::ptr::from_ref(depth_texture).addr(),
            mip_level: 0,
            array_layer: 0,
            aspect: TextureAspect::Depth,
        };
        if encoder.texture_states.get(&key).copied() != Some(expected_state) {
            return Err(format!(
                "depth attachment requires encoder state {expected_state:?}"
            ));
        }
    }
    let ops = match clear {
        Some(_) => wgpu_hal::AttachmentOps::LOAD_CLEAR,
        None if load => wgpu_hal::AttachmentOps::LOAD,
        None => wgpu_hal::AttachmentOps::LOAD_DONT_CARE,
    } | if store {
        wgpu_hal::AttachmentOps::STORE
    } else {
        wgpu_hal::AttachmentOps::STORE_DISCARD
    };
    let clear_value = clear.unwrap_or([0.0; 4]);
    let view_desc = wgpu_hal::TextureViewDescriptor {
        label: Some("fluxel fixed raster color view"),
        format: wgt::TextureFormat::Rgba8Unorm,
        dimension: wgt::TextureViewDimension::D2,
        usage: wgt::TextureUses::COLOR_TARGET,
        range: wgt::ImageSubresourceRange {
            aspect: wgt::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        },
    };
    let depth_view_desc = wgpu_hal::TextureViewDescriptor {
        label: Some("fluxel fixed raster depth view"),
        format: wgt::TextureFormat::Depth32Float,
        dimension: wgt::TextureViewDimension::D2,
        usage: wgt::TextureUses::DEPTH_STENCIL_WRITE,
        range: wgt::ImageSubresourceRange {
            aspect: wgt::TextureAspect::DepthOnly,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        },
    };
    let extent = wgt::Extent3d {
        width: desc.extent.width,
        height: desc.extent.height,
        depth_or_array_layers: 1,
    };
    match (
        encoder.native.as_mut().expect("live encoder"),
        &encoder.owner.native,
        texture.native.as_ref(),
    ) {
        #[cfg(feature = "dx12")]
        (
            NativeEncoder::Dx12(command),
            NativeDevice::Dx12 { device, .. },
            Some(NativeTexture::Dx12(texture)),
        ) => {
            let view = unsafe {
                // SAFETY: the safe layer fixed the full single-sample Rgba8
                // descriptor/usage and the encoder tracker proves color state.
                device.create_texture_view(texture, &view_desc)
            }
            .map_err(|e| format!("DX12 raster view creation failed: {e}"))?;
            let depth_view = match depth {
                Some((depth_texture, _, depth_clear, depth_load)) => {
                    let Some(NativeTexture::Dx12(depth_texture)) = depth_texture.native.as_ref()
                    else {
                        unsafe { device.destroy_texture_view(view) };
                        return Err("depth attachment belongs to another native backend".into());
                    };
                    // SAFETY: the validated depth texture and retained device
                    // have the same native lineage, matching the color view.
                    let depth_view = match unsafe {
                        device.create_texture_view(depth_texture, &depth_view_desc)
                    } {
                        Ok(view) => view,
                        Err(error) => {
                            // SAFETY: depth view creation failed, so the color
                            // view is unreferenced and must be consumed here.
                            unsafe { device.destroy_texture_view(view) };
                            return Err(format!("DX12 depth view creation failed: {error}"));
                        }
                    };
                    let depth_ops = match depth_clear {
                        Some(_) => wgpu_hal::AttachmentOps::LOAD_CLEAR,
                        None if depth_load => wgpu_hal::AttachmentOps::LOAD,
                        None => wgpu_hal::AttachmentOps::LOAD_DONT_CARE,
                    } | wgpu_hal::AttachmentOps::STORE;
                    Some((depth_view, depth_ops, depth_clear.unwrap_or(0.0)))
                }
                None => None,
            };
            let colors = [Some(wgpu_hal::ColorAttachment {
                target: wgpu_hal::Attachment {
                    view: &view,
                    usage: wgt::TextureUses::COLOR_TARGET,
                },
                depth_slice: None,
                resolve_target: None,
                ops,
                clear_value: wgt::Color {
                    r: f64::from(clear_value[0]),
                    g: f64::from(clear_value[1]),
                    b: f64::from(clear_value[2]),
                    a: f64::from(clear_value[3]),
                },
            })];
            if let Err(e) = unsafe {
                // SAFETY: encoder is recording outside another pass; attachment
                // view/state/extent/ops are validated and remain retained.
                command.begin_render_pass(&wgpu_hal::RenderPassDescriptor {
                    label: Some(label),
                    extent,
                    sample_count: 1,
                    color_attachments: &colors,
                    depth_stencil_attachment: depth_view.as_ref().map(|(view, ops, clear)| {
                        wgpu_hal::DepthStencilAttachment {
                            target: wgpu_hal::Attachment {
                                view,
                                usage: wgt::TextureUses::DEPTH_STENCIL_WRITE,
                            },
                            depth_ops: *ops,
                            // Vulkan's HAL render-pass lowering requires a
                            // complete load/store pair even when the selected
                            // view has no stencil aspect.
                            stencil_ops: wgpu_hal::AttachmentOps::LOAD_DONT_CARE
                                | wgpu_hal::AttachmentOps::STORE_DISCARD,
                            clear_value: (*clear, 0),
                        }
                    }),
                    multiview_mask: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                })
            } {
                unsafe {
                    // SAFETY: begin failed and did not retain the uniquely owned
                    // same-device view, so it may be destroyed immediately.
                    device.destroy_texture_view(view);
                    if let Some((depth, _, _)) = depth_view {
                        device.destroy_texture_view(depth);
                    }
                };
                return Err(format!("DX12 begin raster pass failed: {e}"));
            }
            encoder.active_render_view = Some(NativeRenderView::Dx12 {
                color: view,
                depth: depth_view.map(|(view, _, _)| view),
            });
        }
        #[cfg(feature = "vulkan")]
        (
            NativeEncoder::Vulkan(command),
            NativeDevice::Vulkan { device, .. },
            Some(NativeTexture::Vulkan(texture)),
        ) => {
            let view = unsafe {
                // SAFETY: same validated full Rgba8 attachment and tracked
                // color state/lifetime proof as DX12.
                device.create_texture_view(texture, &view_desc)
            }
            .map_err(|e| format!("Vulkan raster view creation failed: {e}"))?;
            let depth_view = match depth {
                Some((depth_texture, _, depth_clear, depth_load)) => {
                    let Some(NativeTexture::Vulkan(depth_texture)) = depth_texture.native.as_ref()
                    else {
                        unsafe { device.destroy_texture_view(view) };
                        return Err("depth attachment belongs to another native backend".into());
                    };
                    // SAFETY: the validated depth texture and retained device
                    // have the same native lineage, matching the color view.
                    let depth_view = match unsafe {
                        device.create_texture_view(depth_texture, &depth_view_desc)
                    } {
                        Ok(view) => view,
                        Err(error) => {
                            // SAFETY: depth view creation failed, so the color
                            // view is unreferenced and must be consumed here.
                            unsafe { device.destroy_texture_view(view) };
                            return Err(format!("Vulkan depth view creation failed: {error}"));
                        }
                    };
                    let depth_ops = match depth_clear {
                        Some(_) => wgpu_hal::AttachmentOps::LOAD_CLEAR,
                        None if depth_load => wgpu_hal::AttachmentOps::LOAD,
                        None => wgpu_hal::AttachmentOps::LOAD_DONT_CARE,
                    } | wgpu_hal::AttachmentOps::STORE;
                    Some((depth_view, depth_ops, depth_clear.unwrap_or(0.0)))
                }
                None => None,
            };
            let colors = [Some(wgpu_hal::ColorAttachment {
                target: wgpu_hal::Attachment {
                    view: &view,
                    usage: wgt::TextureUses::COLOR_TARGET,
                },
                depth_slice: None,
                resolve_target: None,
                ops,
                clear_value: wgt::Color {
                    r: f64::from(clear_value[0]),
                    g: f64::from(clear_value[1]),
                    b: f64::from(clear_value[2]),
                    a: f64::from(clear_value[3]),
                },
            })];
            if let Err(e) = unsafe {
                // SAFETY: the recording encoder has no active pass and all
                // attachment descriptor/state/lifetime requirements hold.
                command.begin_render_pass(&wgpu_hal::RenderPassDescriptor {
                    label: Some(label),
                    extent,
                    sample_count: 1,
                    color_attachments: &colors,
                    depth_stencil_attachment: depth_view.as_ref().map(|(view, ops, clear)| {
                        wgpu_hal::DepthStencilAttachment {
                            target: wgpu_hal::Attachment {
                                view,
                                usage: wgt::TextureUses::DEPTH_STENCIL_WRITE,
                            },
                            depth_ops: *ops,
                            stencil_ops: wgpu_hal::AttachmentOps::LOAD_DONT_CARE
                                | wgpu_hal::AttachmentOps::STORE_DISCARD,
                            clear_value: (*clear, 0),
                        }
                    }),
                    multiview_mask: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                })
            } {
                unsafe {
                    // SAFETY: failed begin left this same-device view
                    // unreferenced; it is consumed exactly once here.
                    device.destroy_texture_view(view);
                    if let Some((depth, _, _)) = depth_view {
                        device.destroy_texture_view(depth);
                    }
                };
                return Err(format!("Vulkan begin raster pass failed: {e}"));
            }
            encoder.active_render_view = Some(NativeRenderView::Vulkan {
                color: view,
                depth: depth_view.map(|(view, _, _)| view),
            });
        }
        _ => return Err("raster attachment belongs to another native backend".into()),
    }
    Ok(())
}

pub(crate) fn end_raster(encoder: &mut CopyEncoder) -> Result<(), String> {
    let view = encoder
        .active_render_view
        .take()
        .ok_or_else(|| "no active raster pass".to_owned())?;
    match (
        encoder.native.as_mut().expect("live encoder"),
        &encoder.owner.native,
        view,
    ) {
        #[cfg(feature = "dx12")]
        (
            NativeEncoder::Dx12(command),
            NativeDevice::Dx12 { .. },
            view @ NativeRenderView::Dx12 { .. },
        ) => unsafe {
            // SAFETY: the safe layer established one active raster pass; its
            // view is moved into command-buffer retention after this call.
            command.end_render_pass();
            encoder.render_views.push(view);
        },
        #[cfg(feature = "vulkan")]
        (
            NativeEncoder::Vulkan(command),
            NativeDevice::Vulkan { .. },
            view @ NativeRenderView::Vulkan { .. },
        ) => unsafe {
            // SAFETY: same active-pass and retained-view proof as DX12.
            command.end_render_pass();
            encoder.render_views.push(view);
        },
        _ => return Err("raster pass belongs to another native backend".into()),
    }
    // The next pass must explicitly select a pipeline before any binding.
    encoder.active_raster_pipeline = None;
    Ok(())
}

pub(crate) fn set_raster_pipeline(
    encoder: &mut CopyEncoder,
    pipeline: &NativeRasterPipeline,
) -> Result<(), String> {
    if !Arc::ptr_eq(&encoder.owner, &pipeline.0.owner) {
        return Err("raster pipeline belongs to another native device".into());
    }
    if encoder.active_render_view.is_none() {
        return Err("raster pipeline requires an active raster pass".into());
    }
    let has_depth = match encoder.active_render_view.as_ref() {
        #[cfg(feature = "dx12")]
        Some(NativeRenderView::Dx12 { depth, .. }) => depth.is_some(),
        #[cfg(feature = "vulkan")]
        Some(NativeRenderView::Vulkan { depth, .. }) => depth.is_some(),
        None => unreachable!("active raster view was checked above"),
    };
    match (
        encoder.native.as_mut().expect("live encoder"),
        pipeline.0.native.as_ref(),
    ) {
        #[cfg(feature = "dx12")]
        (
            NativeEncoder::Dx12(encoder),
            Some(NativeRasterPipelineInner::Dx12 {
                pipeline,
                depth_pipeline,
                ..
            }),
        ) => unsafe {
            // SAFETY: safe checks prove same device and an active compatible
            // raster pass; the pipeline lease survives command completion.
            encoder.set_render_pipeline(if has_depth { depth_pipeline } else { pipeline })
        },
        #[cfg(feature = "vulkan")]
        (
            NativeEncoder::Vulkan(encoder),
            Some(NativeRasterPipelineInner::Vulkan {
                pipeline,
                depth_pipeline,
                ..
            }),
        ) => unsafe {
            // SAFETY: same device, active-pass, compatibility, and lifetime
            // proof as the DX12 branch.
            encoder.set_render_pipeline(if has_depth { depth_pipeline } else { pipeline })
        },
        _ => return Err("raster pipeline belongs to another native backend".into()),
    };
    encoder.active_raster_pipeline = Some(Arc::as_ptr(&pipeline.0) as usize);
    Ok(())
}

#[allow(
    clippy::too_many_arguments,
    reason = "the private native boundary deliberately receives every independently validated closed-ABI fact"
)]
pub(crate) fn set_vertex_buffer(
    encoder: &mut CopyEncoder,
    buffer: &OwnedBuffer,
    offset: u64,
    size: u64,
    kernel: crate::RasterKernel,
    slot: u32,
    actual_identity: fluxel_rendergraph::PhysicalResourceIdentity,
    uv_bindings: Option<&NativeRasterTextureBindings>,
    normal_bindings: Option<&NativeRasterUniformBindings>,
    vertex_color_bindings: Option<&NativeRasterUniformBindings>,
) -> Result<(), String> {
    if matches!(
        kernel,
        crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUv
            | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
            | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
    ) {
        if encoder.active_render_view.is_none() {
            return Err("explicit-UV vertex binding requires an active raster pass".into());
        }
        let bindings = uv_bindings
            .ok_or_else(|| "explicit-UV vertex binding requires UV bindings".to_owned())?;
        if !Arc::ptr_eq(&encoder.owner, &bindings.pipeline.0.owner)
            || encoder.active_raster_pipeline != Some(Arc::as_ptr(&bindings.pipeline.0) as usize)
        {
            return Err("explicit-UV vertex binding requires its active pipeline".into());
        }
        if slot > 1 {
            return Err("explicit-UV vertex slot is out of range".into());
        }
        if !buffer.allowed_usage.contains(BufferUsageKind::Vertex) {
            return Err("explicit-UV vertex buffer requires vertex usage".into());
        }
        let expected = bindings
            .expected_uv_vertex_streams
            .ok_or_else(|| "explicit-UV native binding lacks stream identities".to_owned())?;
        let (expected_identity, expected_size) = if slot == 0 {
            (expected.0, expected.1)
        } else {
            (expected.2, expected.3)
        };
        if actual_identity != expected_identity {
            return Err("explicit-UV vertex role identity mismatch".into());
        }
        if offset != 0 || size != expected_size || buffer.size != expected_size {
            return Err("explicit-UV vertex stream range mismatch".into());
        }
    }
    if kernel == crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialNormalLambert {
        if encoder.active_render_view.is_none() {
            return Err("normal-Lambert vertex binding requires an active raster pass".into());
        }
        let bindings = normal_bindings
            .ok_or_else(|| "normal-Lambert vertex binding requires normal bindings".to_owned())?;
        if !Arc::ptr_eq(&encoder.owner, &bindings.pipeline.0.owner)
            || encoder.active_raster_pipeline != Some(Arc::as_ptr(&bindings.pipeline.0) as usize)
            || slot > 1
            || !buffer.allowed_usage.contains(BufferUsageKind::Vertex)
        {
            return Err(
                "normal-Lambert vertex binding has an invalid pipeline, slot, or usage".into(),
            );
        }
        let expected = bindings
            .expected_normal_vertex_streams
            .ok_or_else(|| "normal-Lambert native binding lacks stream identities".to_owned())?;
        let (expected_identity, expected_size) = if slot == 0 {
            (expected.0, expected.1)
        } else {
            (expected.2, expected.3)
        };
        if actual_identity != expected_identity
            || offset != 0
            || size != expected_size
            || buffer.size != expected_size
        {
            return Err("normal-Lambert vertex role or range mismatch".into());
        }
    }
    if kernel == crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialVertexColor {
        if encoder.active_render_view.is_none() {
            return Err("vertex-color vertex binding requires an active raster pass".into());
        }
        let bindings = vertex_color_bindings
            .ok_or_else(|| "vertex-color vertex binding requires bindings".to_owned())?;
        if !Arc::ptr_eq(&encoder.owner, &bindings.pipeline.0.owner)
            || encoder.active_raster_pipeline != Some(Arc::as_ptr(&bindings.pipeline.0) as usize)
            || slot > 1
            || !buffer.allowed_usage.contains(BufferUsageKind::Vertex)
        {
            return Err("vertex-color vertex binding has invalid pipeline, slot, or usage".into());
        }
        let expected = bindings
            .expected_vertex_color_streams
            .ok_or_else(|| "vertex-color native binding lacks stream identities".to_owned())?;
        let (expected_identity, expected_size) = if slot == 0 {
            (expected.0, expected.1)
        } else {
            (expected.2, expected.3)
        };
        if actual_identity != expected_identity
            || offset != 0
            || size != expected_size
            || buffer.size != expected_size
        {
            return Err("vertex-color vertex role or range mismatch".into());
        }
    }
    if !Arc::ptr_eq(&encoder.owner, &buffer.owner)
        || !offset.is_multiple_of(4)
        || (matches!(
            kernel,
            crate::RasterKernel::IndexedPositionFloat32x3
                | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterial
                | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTexture
                | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialNormalLambert
                | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialVertexColor
        ) && offset != 0)
        || (matches!(
            kernel,
            crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUv
                | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
                | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
                | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialNormalLambert
                | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialVertexColor
        ) && (slot > 1 || offset != 0 || size < raster_vertex_minimum_size(kernel, slot)))
        || offset.checked_add(size).is_none_or(|end| end > buffer.size)
        || size < raster_vertex_minimum_size(kernel, slot)
    {
        return Err("invalid raster vertex buffer range".into());
    }
    let size = NonZeroU64::new(size).expect("checked non-zero vertex range");
    match (
        encoder.native.as_mut().expect("live encoder"),
        buffer.native.as_ref(),
    ) {
        #[cfg(feature = "dx12")]
        (NativeEncoder::Dx12(encoder), Some(NativeBuffer::Dx12(buffer))) => unsafe {
            // SAFETY: safe and native checks prove same device, active raster
            // pass, aligned in-bounds non-empty range, and retained buffer life.
            encoder.set_vertex_buffer(
                slot,
                wgpu_hal::BufferBinding::new_unchecked(buffer, offset, size),
            )
        },
        #[cfg(feature = "vulkan")]
        (NativeEncoder::Vulkan(encoder), Some(NativeBuffer::Vulkan(buffer))) => unsafe {
            // SAFETY: same pass/device/range/alignment/lifetime proof as DX12.
            encoder.set_vertex_buffer(
                slot,
                wgpu_hal::BufferBinding::new_unchecked(buffer, offset, size),
            )
        },
        _ => return Err("vertex buffer belongs to another native backend".into()),
    };
    Ok(())
}

pub(crate) fn set_index_buffer(
    encoder: &mut CopyEncoder,
    buffer: &OwnedBuffer,
    offset: u64,
    size: u64,
    format: IndexFormat,
) -> Result<(), String> {
    let alignment = match format {
        IndexFormat::Uint16 => 2,
        IndexFormat::Uint32 => 4,
    };
    if !Arc::ptr_eq(&encoder.owner, &buffer.owner)
        || !offset.is_multiple_of(alignment)
        || (format == IndexFormat::Uint32 && offset != 0)
        || !size.is_multiple_of(alignment)
        || offset.checked_add(size).is_none_or(|end| end > buffer.size)
        || size == 0
    {
        return Err("invalid raster index buffer range".into());
    }
    let size = NonZeroU64::new(size).expect("checked non-zero index range");
    match (
        encoder.native.as_mut().expect("live encoder"),
        buffer.native.as_ref(),
    ) {
        #[cfg(feature = "dx12")]
        (NativeEncoder::Dx12(encoder), Some(NativeBuffer::Dx12(buffer))) => unsafe {
            // SAFETY: safe and native checks prove active pass, recipe-specific alignment,
            // in-bounds non-empty range, same device, and retained lifetime.
            encoder.set_index_buffer(
                wgpu_hal::BufferBinding::new_unchecked(buffer, offset, size),
                match format {
                    IndexFormat::Uint16 => wgt::IndexFormat::Uint16,
                    IndexFormat::Uint32 => wgt::IndexFormat::Uint32,
                },
            )
        },
        #[cfg(feature = "vulkan")]
        (NativeEncoder::Vulkan(encoder), Some(NativeBuffer::Vulkan(buffer))) => unsafe {
            // SAFETY: same active-pass, recipe-specific range, device, and lifetime proof.
            encoder.set_index_buffer(
                wgpu_hal::BufferBinding::new_unchecked(buffer, offset, size),
                match format {
                    IndexFormat::Uint16 => wgt::IndexFormat::Uint16,
                    IndexFormat::Uint32 => wgt::IndexFormat::Uint32,
                },
            )
        },
        _ => return Err("index buffer belongs to another native backend".into()),
    };
    Ok(())
}

pub(crate) fn set_viewport(
    encoder: &mut CopyEncoder,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    min_depth: f32,
    max_depth: f32,
) -> Result<(), String> {
    if ![x, y, width, height, min_depth, max_depth]
        .iter()
        .all(|v| v.is_finite())
        || width <= 0.0
        || height <= 0.0
        || !(0.0..=1.0).contains(&min_depth)
        || !(0.0..=1.0).contains(&max_depth)
        || min_depth > max_depth
    {
        return Err("invalid raster viewport".into());
    }
    let rect = wgpu_hal::Rect {
        x,
        y,
        w: width,
        h: height,
    };
    match encoder.native.as_mut().expect("live encoder") {
        #[cfg(feature = "dx12")]
        NativeEncoder::Dx12(e) => unsafe {
            // SAFETY: finite positive extent and ordered [0,1] depth range were
            // checked twice; safe layer proves an active raster pass.
            e.set_viewport(&rect, min_depth..max_depth)
        },
        #[cfg(feature = "vulkan")]
        NativeEncoder::Vulkan(e) => unsafe {
            // SAFETY: same active-pass and validated numeric range as DX12.
            e.set_viewport(&rect, min_depth..max_depth)
        },
    };
    Ok(())
}

pub(crate) fn set_scissor(
    encoder: &mut CopyEncoder,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
) -> Result<(), String> {
    if width == 0 || height == 0 {
        return Err("invalid raster scissor".into());
    }
    let rect = wgpu_hal::Rect {
        x,
        y,
        w: width,
        h: height,
    };
    match encoder.native.as_mut().expect("live encoder") {
        #[cfg(feature = "dx12")]
        NativeEncoder::Dx12(e) => unsafe {
            // SAFETY: safe layer bounds this non-empty rectangle to the active
            // attachment and proves an active raster pass.
            e.set_scissor_rect(&rect)
        },
        #[cfg(feature = "vulkan")]
        NativeEncoder::Vulkan(e) => unsafe {
            // SAFETY: same active-pass and validated rectangle proof as DX12.
            e.set_scissor_rect(&rect)
        },
    };
    Ok(())
}

pub(crate) fn draw(
    encoder: &mut CopyEncoder,
    first_vertex: u32,
    vertex_count: u32,
    first_instance: u32,
    instance_count: u32,
) -> Result<(), String> {
    if vertex_count == 0 || instance_count == 0 {
        return Err("empty raster draw".into());
    }
    match encoder.native.as_mut().expect("live encoder") {
        #[cfg(feature = "dx12")]
        NativeEncoder::Dx12(e) => unsafe {
            // SAFETY: safe fixed recipe proves active compatible pipeline and
            // non-empty exact vertex/instance ranges; leases retain all objects.
            e.draw(first_vertex, vertex_count, first_instance, instance_count)
        },
        #[cfg(feature = "vulkan")]
        NativeEncoder::Vulkan(e) => unsafe {
            // SAFETY: same fixed draw/pass/pipeline/lifetime proof as DX12.
            e.draw(first_vertex, vertex_count, first_instance, instance_count)
        },
    };
    Ok(())
}

pub(crate) fn draw_indexed(
    encoder: &mut CopyEncoder,
    first_index: u32,
    index_count: u32,
    base_vertex: i32,
    first_instance: u32,
    instance_count: u32,
) -> Result<(), String> {
    if index_count == 0 || instance_count == 0 {
        return Err("empty indexed raster draw".into());
    }
    match encoder.native.as_mut().expect("live encoder") {
        #[cfg(feature = "dx12")]
        NativeEncoder::Dx12(e) => unsafe {
            // SAFETY: safe fixed recipe proves active compatible pipeline,
            // bound vertex/index ranges, and non-empty validated draw ranges.
            e.draw_indexed(
                first_index,
                index_count,
                base_vertex,
                first_instance,
                instance_count,
            )
        },
        #[cfg(feature = "vulkan")]
        NativeEncoder::Vulkan(e) => unsafe {
            // SAFETY: same active indexed recipe, range, and retained-lifetime
            // proof as the DX12 branch.
            e.draw_indexed(
                first_index,
                index_count,
                base_vertex,
                first_instance,
                instance_count,
            )
        },
    };
    Ok(())
}
