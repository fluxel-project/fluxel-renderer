//! Native compute-pipeline construction.

use super::*;

/// Compiles a previously selected fixed WGSL artifact into a single-RW-storage
/// pipeline. The public layer never forwards caller-controlled source here.
pub(crate) fn create_compute_pipeline(
    owner: &Arc<OpenedDevice>,
    source: &str,
    entry: &str,
) -> Result<NativeComputePipeline, ComputePipelineCreateError> {
    let module = naga::front::wgsl::Frontend::new()
        .parse(source)
        .map_err(|error| {
            ComputePipelineCreateError::ShaderValidation(format!("WGSL parse failed: {error}"))
        })?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .map_err(|error| {
        ComputePipelineCreateError::ShaderValidation(format!("WGSL validation failed: {error}"))
    })?;
    let shader = wgpu_hal::NagaShader {
        module: Cow::Owned(module),
        info,
        debug_source: None,
    };
    let texture_pack = source.contains("texture_2d<");
    let storage_write = source.contains("texture_storage_2d<rgba8unorm, write>");
    let storage_read = source.contains("texture_storage_2d<rgba8unorm, read>");
    let layout_entries = if storage_write {
        vec![wgt::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgt::ShaderStages::COMPUTE,
            ty: wgt::BindingType::StorageTexture {
                access: wgt::StorageTextureAccess::WriteOnly,
                format: wgt::TextureFormat::Rgba8Unorm,
                view_dimension: wgt::TextureViewDimension::D2,
            },
            count: None,
        }]
    } else if storage_read {
        vec![
            wgt::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgt::ShaderStages::COMPUTE,
                ty: wgt::BindingType::StorageTexture {
                    access: wgt::StorageTextureAccess::ReadOnly,
                    format: wgt::TextureFormat::Rgba8Unorm,
                    view_dimension: wgt::TextureViewDimension::D2,
                },
                count: None,
            },
            wgt::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgt::ShaderStages::COMPUTE,
                ty: wgt::BindingType::Buffer {
                    ty: wgt::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ]
    } else if texture_pack {
        vec![
            wgt::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgt::ShaderStages::COMPUTE,
                ty: wgt::BindingType::Texture {
                    sample_type: wgt::TextureSampleType::Float { filterable: false },
                    view_dimension: wgt::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgt::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgt::ShaderStages::COMPUTE,
                ty: wgt::BindingType::Buffer {
                    ty: wgt::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ]
    } else {
        vec![wgt::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgt::ShaderStages::COMPUTE,
            ty: wgt::BindingType::Buffer {
                ty: wgt::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }]
    };

    // Each arm creates objects in dependency order and eagerly tears down any
    // prefix on failure. This prevents a failed setup from retaining native
    // state or leaving a half-formed public object.
    let native = match &owner.native {
        #[cfg(feature = "dx12")]
        NativeDevice::Dx12 { device, .. } => {
            let shader_module = unsafe {
                // SAFETY: Naga parsed and validated this controlled source;
                // the descriptor enables runtime safety checks.
                device.create_shader_module(
                    &wgpu_hal::ShaderModuleDescriptor {
                        label: Some("fluxel fixed compute shader"),
                        runtime_checks: wgt::ShaderRuntimeChecks::checked(),
                    },
                    wgpu_hal::ShaderInput::Naga(shader),
                )
            }
            .map_err(|error| {
                ComputePipelineCreateError::ShaderCompilation(format!(
                    "DX12 shader creation failed: {error}"
                ))
            })?;
            let bind_group_layout = unsafe {
                // SAFETY: one statically-valid storage binding, sorted at binding zero.
                device.create_bind_group_layout(&wgpu_hal::BindGroupLayoutDescriptor {
                    label: Some("fluxel fixed compute RW storage layout"),
                    flags: wgpu_hal::BindGroupLayoutFlags::empty(),
                    entries: &layout_entries,
                })
            };
            let bind_group_layout = match bind_group_layout {
                Ok(value) => value,
                Err(error) => {
                    // SAFETY: creation never published this unique live module;
                    // no dependent layout or pipeline exists on this failure path.
                    unsafe { device.destroy_shader_module(shader_module) };
                    return Err(ComputePipelineCreateError::NativeObjectCreation(format!(
                        "DX12 bind-group-layout creation failed: {error}"
                    )));
                }
            };
            let layouts = [Some(&bind_group_layout)];
            let pipeline_layout = unsafe {
                // SAFETY: the BGL is live and belongs to this device.
                device.create_pipeline_layout(&wgpu_hal::PipelineLayoutDescriptor {
                    label: Some("fluxel fixed compute pipeline layout"),
                    flags: wgpu_hal::PipelineLayoutFlags::empty(),
                    bind_group_layouts: &layouts,
                    immediate_size: 0,
                })
            };
            let pipeline_layout = match pipeline_layout {
                Ok(value) => value,
                Err(error) => {
                    unsafe {
                        // SAFETY: both unique objects belong to this device and
                        // were never published; destroy dependent BGL before shader.
                        device.destroy_bind_group_layout(bind_group_layout);
                        device.destroy_shader_module(shader_module);
                    }
                    return Err(ComputePipelineCreateError::NativeObjectCreation(format!(
                        "DX12 pipeline-layout creation failed: {error}"
                    )));
                }
            };
            let constants = naga::back::PipelineConstants::default();
            let pipeline = unsafe {
                // SAFETY: module, entry, and layout are live and all belong to this device.
                device.create_compute_pipeline(&wgpu_hal::ComputePipelineDescriptor {
                    label: Some("fluxel fixed compute pipeline"),
                    layout: &pipeline_layout,
                    stage: wgpu_hal::ProgrammableStage {
                        module: &shader_module,
                        entry_point: entry,
                        constants: &constants,
                        zero_initialize_workgroup_memory: true,
                    },
                    cache: None,
                })
            };
            let pipeline = match pipeline {
                Ok(value) => value,
                Err(error) => {
                    unsafe {
                        // SAFETY: the failed pipeline was not created. These unique
                        // unpublished dependencies are destroyed in reverse order.
                        device.destroy_pipeline_layout(pipeline_layout);
                        device.destroy_bind_group_layout(bind_group_layout);
                        device.destroy_shader_module(shader_module);
                    }
                    return Err(ComputePipelineCreateError::ShaderCompilation(format!(
                        "DX12 compute-pipeline creation failed: {error}"
                    )));
                }
            };
            NativeComputePipelineInner::Dx12 {
                shader: shader_module,
                bind_group_layout,
                pipeline_layout,
                pipeline,
            }
        }
        #[cfg(feature = "vulkan")]
        NativeDevice::Vulkan { device, .. } => {
            let shader_module = unsafe {
                // SAFETY: Naga parsed and validated this controlled source;
                // the descriptor enables runtime safety checks.
                device.create_shader_module(
                    &wgpu_hal::ShaderModuleDescriptor {
                        label: Some("fluxel fixed compute shader"),
                        runtime_checks: wgt::ShaderRuntimeChecks::checked(),
                    },
                    wgpu_hal::ShaderInput::Naga(shader),
                )
            }
            .map_err(|error| {
                ComputePipelineCreateError::ShaderCompilation(format!(
                    "Vulkan shader creation failed: {error}"
                ))
            })?;
            let bind_group_layout = unsafe {
                // SAFETY: one statically-valid storage binding, sorted at binding zero.
                device.create_bind_group_layout(&wgpu_hal::BindGroupLayoutDescriptor {
                    label: Some("fluxel fixed compute RW storage layout"),
                    flags: wgpu_hal::BindGroupLayoutFlags::empty(),
                    entries: &layout_entries,
                })
            };
            let bind_group_layout = match bind_group_layout {
                Ok(value) => value,
                Err(error) => {
                    // SAFETY: creation never published this unique live module;
                    // no dependent layout or pipeline exists on this failure path.
                    unsafe { device.destroy_shader_module(shader_module) };
                    return Err(ComputePipelineCreateError::NativeObjectCreation(format!(
                        "Vulkan bind-group-layout creation failed: {error}"
                    )));
                }
            };
            let layouts = [Some(&bind_group_layout)];
            let pipeline_layout = unsafe {
                // SAFETY: the BGL is live and belongs to this device.
                device.create_pipeline_layout(&wgpu_hal::PipelineLayoutDescriptor {
                    label: Some("fluxel fixed compute pipeline layout"),
                    flags: wgpu_hal::PipelineLayoutFlags::empty(),
                    bind_group_layouts: &layouts,
                    immediate_size: 0,
                })
            };
            let pipeline_layout = match pipeline_layout {
                Ok(value) => value,
                Err(error) => {
                    unsafe {
                        // SAFETY: both unique objects belong to this device and
                        // were never published; destroy dependent BGL before shader.
                        device.destroy_bind_group_layout(bind_group_layout);
                        device.destroy_shader_module(shader_module);
                    }
                    return Err(ComputePipelineCreateError::NativeObjectCreation(format!(
                        "Vulkan pipeline-layout creation failed: {error}"
                    )));
                }
            };
            let constants = naga::back::PipelineConstants::default();
            let pipeline = unsafe {
                // SAFETY: module, entry, and layout are live and all belong to this device.
                device.create_compute_pipeline(&wgpu_hal::ComputePipelineDescriptor {
                    label: Some("fluxel fixed compute pipeline"),
                    layout: &pipeline_layout,
                    stage: wgpu_hal::ProgrammableStage {
                        module: &shader_module,
                        entry_point: entry,
                        constants: &constants,
                        zero_initialize_workgroup_memory: true,
                    },
                    cache: None,
                })
            };
            let pipeline = match pipeline {
                Ok(value) => value,
                Err(error) => {
                    unsafe {
                        // SAFETY: the failed pipeline was not created. These unique
                        // unpublished dependencies are destroyed in reverse order.
                        device.destroy_pipeline_layout(pipeline_layout);
                        device.destroy_bind_group_layout(bind_group_layout);
                        device.destroy_shader_module(shader_module);
                    }
                    return Err(ComputePipelineCreateError::ShaderCompilation(format!(
                        "Vulkan compute-pipeline creation failed: {error}"
                    )));
                }
            };
            NativeComputePipelineInner::Vulkan {
                shader: shader_module,
                bind_group_layout,
                pipeline_layout,
                pipeline,
            }
        }
    };

    Ok(NativeComputePipeline(Arc::new(
        NativeComputePipelineShared {
            native: Some(native),
            owner: Arc::clone(owner),
        },
    )))
}
