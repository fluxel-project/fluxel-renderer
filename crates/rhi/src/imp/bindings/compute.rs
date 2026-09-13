//! Native compute binding construction.

use super::*;

pub(crate) fn create_compute_bindings(
    owner: &Arc<OpenedDevice>,
    pipeline: &NativeComputePipeline,
    buffer: &OwnedBuffer,
    offset: u64,
    size: u64,
) -> Result<NativeComputeBindings, String> {
    if !Arc::ptr_eq(owner, &pipeline.0.owner) {
        return Err("compute pipeline belongs to another native device".into());
    }
    validate_native_compute_binding_range(
        offset,
        size,
        buffer.size,
        owner.capabilities.min_storage_buffer_offset_alignment,
        owner.capabilities.max_storage_buffer_binding_size,
    )?;
    let size = NonZeroU64::new(size).expect("checked non-zero storage binding size");
    let entry = [wgpu_hal::BindGroupEntry {
        binding: 0,
        resource_index: 0,
        count: 1,
    }];
    let buffer = buffer
        .native
        .as_ref()
        .ok_or_else(|| "compute bindings referenced a destroyed buffer".to_owned())?;
    let native = match (&owner.native, pipeline.0.native.as_ref(), buffer) {
        #[cfg(feature = "dx12")]
        (
            NativeDevice::Dx12 { device, .. },
            Some(NativeComputePipelineInner::Dx12 {
                bind_group_layout, ..
            }),
            NativeBuffer::Dx12(buffer),
        ) => {
            let binding = wgpu_hal::BufferBinding::new_unchecked(buffer, offset, size);
            unsafe {
                // SAFETY: caller's safe boundary proved buffer/device identity,
                // range containment and storage alignment; this fixed descriptor
                // has exactly the one matching RW storage entry.
                device.create_bind_group(&wgpu_hal::BindGroupDescriptor {
                    label: Some("fluxel fixed compute RW storage bindings"),
                    layout: bind_group_layout,
                    buffers: &[binding],
                    samplers: &[],
                    textures: &[],
                    entries: &entry,
                    acceleration_structures: &[],
                    external_textures: &[],
                })
            }
            .map(NativeComputeBindingsInner::Dx12)
            .map_err(|error| format!("DX12 compute bind-group creation failed: {error}"))?
        }
        #[cfg(feature = "vulkan")]
        (
            NativeDevice::Vulkan { device, .. },
            Some(NativeComputePipelineInner::Vulkan {
                bind_group_layout, ..
            }),
            NativeBuffer::Vulkan(buffer),
        ) => {
            let binding = wgpu_hal::BufferBinding::new_unchecked(buffer, offset, size);
            unsafe {
                // SAFETY: caller's safe boundary proved buffer/device identity,
                // range containment and storage alignment; this fixed descriptor
                // has exactly the one matching RW storage entry.
                device.create_bind_group(&wgpu_hal::BindGroupDescriptor {
                    label: Some("fluxel fixed compute RW storage bindings"),
                    layout: bind_group_layout,
                    buffers: &[binding],
                    samplers: &[],
                    textures: &[],
                    entries: &entry,
                    acceleration_structures: &[],
                    external_textures: &[],
                })
            }
            .map(NativeComputeBindingsInner::Vulkan)
            .map_err(|error| format!("Vulkan compute bind-group creation failed: {error}"))?
        }
        _ => return Err("compute bindings reference a foreign device or backend".into()),
    };
    Ok(NativeComputeBindings {
        native: Some(native),
        pipeline: pipeline.clone(),
    })
}

/// Creates X01's sampled-texture plus RW-storage-buffer binding.  This check
/// intentionally remains at the unsafe boundary as well as the safe layer:
/// constructing unchecked HAL views or bindings from a foreign resource would
/// violate HAL's device/lifetime contract.
pub(crate) fn create_texture_pack_bindings(
    owner: &Arc<OpenedDevice>,
    pipeline: &NativeComputePipeline,
    texture: &OwnedTexture,
    buffer: &OwnedBuffer,
    offset: u64,
    size: u64,
) -> Result<NativeTexturePackBindings, String> {
    if !Arc::ptr_eq(owner, &pipeline.0.owner)
        || !Arc::ptr_eq(owner, &texture.owner)
        || !Arc::ptr_eq(owner, &buffer.owner)
    {
        return Err("texture-pack objects belong to another native device".into());
    }
    validate_native_compute_binding_range(
        offset,
        size,
        buffer.size,
        owner.capabilities.min_storage_buffer_offset_alignment,
        owner.capabilities.max_storage_buffer_binding_size,
    )?;
    let view_desc = wgpu_hal::TextureViewDescriptor {
        label: Some("fluxel X01 sampled Rgba8Unorm view"),
        format: wgt::TextureFormat::Rgba8Unorm,
        dimension: wgt::TextureViewDimension::D2,
        usage: wgt::TextureUses::RESOURCE,
        range: wgt::ImageSubresourceRange {
            aspect: wgt::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        },
    };
    let entries = [
        wgpu_hal::BindGroupEntry {
            binding: 0,
            resource_index: 0,
            count: 1,
        },
        wgpu_hal::BindGroupEntry {
            binding: 1,
            resource_index: 0,
            count: 1,
        },
    ];
    let size = NonZeroU64::new(size).expect("validated non-zero storage size");
    match (
        &owner.native,
        pipeline.0.native.as_ref(),
        texture.native.as_ref(),
        buffer.native.as_ref(),
    ) {
        #[cfg(feature = "dx12")]
        (
            NativeDevice::Dx12 { device, .. },
            Some(NativeComputePipelineInner::Dx12 {
                bind_group_layout, ..
            }),
            Some(NativeTexture::Dx12(texture)),
            Some(NativeBuffer::Dx12(buffer)),
        ) => {
            let view = unsafe {
                // SAFETY: safe and native checks prove same-device ownership,
                // complete Rgba8 range, SAMPLE usage, and retained lifetime.
                device.create_texture_view(texture, &view_desc)
            }
            .map_err(|e| format!("DX12 texture-pack view creation failed: {e}"))?;
            let group = unsafe {
                // SAFETY: BGL/resources are live and same-device; the unchecked
                // buffer range was revalidated for bounds and alignment above.
                device.create_bind_group(&wgpu_hal::BindGroupDescriptor {
                    label: Some("fluxel X01 texture-pack bindings"),
                    layout: bind_group_layout,
                    buffers: &[wgpu_hal::BufferBinding::new_unchecked(buffer, offset, size)],
                    samplers: &[],
                    textures: &[wgpu_hal::TextureBinding {
                        view: &view,
                        usage: wgt::TextureUses::RESOURCE,
                    }],
                    entries: &entries,
                    acceleration_structures: &[],
                    external_textures: &[],
                })
            };
            match group {
                Ok(group) => Ok(NativeTexturePackBindings {
                    native: Some(NativeTexturePackBindingsInner::Dx12 { group, view }),
                    pipeline: pipeline.clone(),
                }),
                Err(e) => {
                    unsafe {
                        // SAFETY: group creation failed, so this same-device
                        // uniquely owned view has no remaining native users.
                        device.destroy_texture_view(view)
                    };
                    Err(format!("DX12 texture-pack bind-group creation failed: {e}"))
                }
            }
        }
        #[cfg(feature = "vulkan")]
        (
            NativeDevice::Vulkan { device, .. },
            Some(NativeComputePipelineInner::Vulkan {
                bind_group_layout, ..
            }),
            Some(NativeTexture::Vulkan(texture)),
            Some(NativeBuffer::Vulkan(buffer)),
        ) => {
            let view = unsafe {
                // SAFETY: same validated descriptor, device identity, usage,
                // and retained texture lifetime as the DX12 branch.
                device.create_texture_view(texture, &view_desc)
            }
            .map_err(|e| format!("Vulkan texture-pack view creation failed: {e}"))?;
            let group = unsafe {
                // SAFETY: fixed entries match the live same-device BGL; buffer
                // bounds/alignment and texture view range were revalidated.
                device.create_bind_group(&wgpu_hal::BindGroupDescriptor {
                    label: Some("fluxel X01 texture-pack bindings"),
                    layout: bind_group_layout,
                    buffers: &[wgpu_hal::BufferBinding::new_unchecked(buffer, offset, size)],
                    samplers: &[],
                    textures: &[wgpu_hal::TextureBinding {
                        view: &view,
                        usage: wgt::TextureUses::RESOURCE,
                    }],
                    entries: &entries,
                    acceleration_structures: &[],
                    external_textures: &[],
                })
            };
            match group {
                Ok(group) => Ok(NativeTexturePackBindings {
                    native: Some(NativeTexturePackBindingsInner::Vulkan { group, view }),
                    pipeline: pipeline.clone(),
                }),
                Err(e) => {
                    unsafe {
                        // SAFETY: no group was created; consume this unreferenced
                        // same-device view exactly once.
                        device.destroy_texture_view(view)
                    };
                    Err(format!(
                        "Vulkan texture-pack bind-group creation failed: {e}"
                    ))
                }
            }
        }
        _ => Err("texture-pack objects belong to another native backend".into()),
    }
}

pub(crate) fn create_texture_store_bindings(
    owner: &Arc<OpenedDevice>,
    pipeline: &NativeComputePipeline,
    texture: &OwnedTexture,
) -> Result<NativeComputeBindings, String> {
    create_storage_texture_bindings(
        owner,
        pipeline,
        texture,
        None,
        0,
        0,
        wgt::TextureUses::STORAGE_WRITE_ONLY,
    )
}

pub(crate) fn create_texture_load_bindings(
    owner: &Arc<OpenedDevice>,
    pipeline: &NativeComputePipeline,
    texture: &OwnedTexture,
    buffer: &OwnedBuffer,
    offset: u64,
    size: u64,
) -> Result<NativeComputeBindings, String> {
    validate_native_compute_binding_range(
        offset,
        size,
        buffer.size,
        owner.capabilities.min_storage_buffer_offset_alignment,
        owner.capabilities.max_storage_buffer_binding_size,
    )?;
    create_storage_texture_bindings(
        owner,
        pipeline,
        texture,
        Some(buffer),
        offset,
        size,
        wgt::TextureUses::STORAGE_READ_ONLY,
    )
}

fn create_storage_texture_bindings(
    owner: &Arc<OpenedDevice>,
    pipeline: &NativeComputePipeline,
    texture: &OwnedTexture,
    buffer: Option<&OwnedBuffer>,
    offset: u64,
    size: u64,
    usage: wgt::TextureUses,
) -> Result<NativeComputeBindings, String> {
    if !Arc::ptr_eq(owner, &pipeline.0.owner) || !Arc::ptr_eq(owner, &texture.owner) {
        return Err("storage texture bindings belong to another native device".into());
    }
    let desc = wgpu_hal::TextureViewDescriptor {
        label: Some("fluxel fixed RGBA8 storage view"),
        format: wgt::TextureFormat::Rgba8Unorm,
        dimension: wgt::TextureViewDimension::D2,
        usage,
        range: wgt::ImageSubresourceRange {
            aspect: wgt::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        },
    };
    let entries_one = [wgpu_hal::BindGroupEntry {
        binding: 0,
        resource_index: 0,
        count: 1,
    }];
    let entries_two = [
        entries_one[0].clone(),
        wgpu_hal::BindGroupEntry {
            binding: 1,
            resource_index: 0,
            count: 1,
        },
    ];
    macro_rules! make {
        ($device:expr, $pipe:expr, $texture:expr, $buf:expr, $variant:ident) => {{
            let view = unsafe {
                // SAFETY: the safe constructor and the ownership check above
                // prove same-device texture/pipeline ownership; its closed
                // descriptor is the complete single-sample RGBA8 image and
                // `usage` is selected only by the fixed read or write recipe.
                $device.create_texture_view($texture, &desc)
            }
            .map_err(|e| format!("storage texture view creation failed: {e}"))?;
            let group = match $buf {
                Some(buf) => {
                    let bindings = [wgpu_hal::BufferBinding::new_unchecked(
                        buf,
                        offset,
                        NonZeroU64::new(size).expect("validated storage range"),
                    )];
                    unsafe {
                        // SAFETY: the closed two-entry descriptor exactly
                        // matches TextureLoadRgba8's BGL. The buffer range was
                        // revalidated before this helper; view/pipeline/device
                        // ownership is retained by the returned binding object.
                        $device.create_bind_group(&wgpu_hal::BindGroupDescriptor {
                            label: Some("fluxel fixed RGBA8 storage bindings"),
                            layout: $pipe,
                            buffers: &bindings,
                            samplers: &[],
                            textures: &[wgpu_hal::TextureBinding { view: &view, usage }],
                            entries: &entries_two,
                            acceleration_structures: &[],
                            external_textures: &[],
                        })
                    }
                }
                None => unsafe {
                    // SAFETY: the closed one-entry descriptor exactly matches
                    // TextureStoreRgba8's BGL, with a same-device live view.
                    $device.create_bind_group(&wgpu_hal::BindGroupDescriptor {
                        label: Some("fluxel fixed RGBA8 storage bindings"),
                        layout: $pipe,
                        buffers: &[],
                        samplers: &[],
                        textures: &[wgpu_hal::TextureBinding { view: &view, usage }],
                        entries: &entries_one,
                        acceleration_structures: &[],
                        external_textures: &[],
                    })
                },
            };
            match group {
                Ok(group) => Ok(NativeComputeBindings {
                    native: Some(NativeComputeBindingsInner::$variant { group, view }),
                    pipeline: pipeline.clone(),
                }),
                Err(e) => {
                    unsafe {
                        // SAFETY: bind-group creation failed, so no group can
                        // reference this uniquely owned same-device view.
                        $device.destroy_texture_view(view)
                    };
                    Err(format!("storage texture bind-group creation failed: {e}"))
                }
            }
        }};
    }
    match (
        &owner.native,
        pipeline.0.native.as_ref(),
        texture.native.as_ref(),
        buffer.and_then(|b| b.native.as_ref()),
    ) {
        #[cfg(feature = "dx12")]
        (
            NativeDevice::Dx12 { device, .. },
            Some(NativeComputePipelineInner::Dx12 {
                bind_group_layout, ..
            }),
            Some(NativeTexture::Dx12(texture)),
            b,
        ) => {
            let b = match b {
                Some(NativeBuffer::Dx12(v)) => Some(v),
                None => None,
                _ => return Err("storage buffer backend mismatch".into()),
            };
            make!(device, bind_group_layout, texture, b, Dx12Texture)
        }
        #[cfg(feature = "vulkan")]
        (
            NativeDevice::Vulkan { device, .. },
            Some(NativeComputePipelineInner::Vulkan {
                bind_group_layout, ..
            }),
            Some(NativeTexture::Vulkan(texture)),
            b,
        ) => {
            let b = match b {
                Some(NativeBuffer::Vulkan(v)) => Some(v),
                None => None,
                _ => return Err("storage buffer backend mismatch".into()),
            };
            make!(device, bind_group_layout, texture, b, VulkanTexture)
        }
        _ => Err("storage texture bindings belong to another native backend".into()),
    }
}
