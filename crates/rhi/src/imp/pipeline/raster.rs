//! Native raster-pipeline construction.
use super::*;
/// Creates one closed graphics recipe. Source is parsed and validated before
/// any unsafe HAL call; callers cannot pass a raw module or layout.
pub(crate) fn create_raster_pipeline(
    owner: &Arc<OpenedDevice>,
    source: &str,
    vertex_entry: &str,
    fragment_entry: &str,
    kernel: crate::RasterKernel,
) -> Result<NativeRasterPipeline, RasterPipelineCreateError> {
    let module = naga::front::wgsl::Frontend::new()
        .parse(source)
        .map_err(|e| {
            RasterPipelineCreateError::ShaderValidation(format!("WGSL parse failed: {e}"))
        })?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .map_err(|e| {
        RasterPipelineCreateError::ShaderValidation(format!("WGSL validation failed: {e}"))
    })?;
    let position_color_attributes = [
        wgt::VertexAttribute {
            format: wgt::VertexFormat::Float32x2,
            offset: 0,
            shader_location: 0,
        },
        wgt::VertexAttribute {
            format: wgt::VertexFormat::Unorm8x4,
            offset: 8,
            shader_location: 1,
        },
    ];
    let position_f32x3_attributes = [wgt::VertexAttribute {
        format: wgt::VertexFormat::Float32x3,
        offset: 0,
        shader_location: 0,
    }];
    let texture_coordinate_f32x2_attributes = [wgt::VertexAttribute {
        format: wgt::VertexFormat::Float32x2,
        offset: 0,
        shader_location: 1,
    }];
    let normal_f32x3_attributes = [wgt::VertexAttribute {
        format: wgt::VertexFormat::Float32x3,
        offset: 0,
        shader_location: 1,
    }];
    let color_unorm8x4_attributes = [wgt::VertexAttribute {
        format: wgt::VertexFormat::Unorm8x4,
        offset: 0,
        shader_location: 1,
    }];
    let buffers = match kernel {
        crate::RasterKernel::Triangle => vec![],
        crate::RasterKernel::IndexedPositionColor => vec![Some(wgpu_hal::VertexBufferLayout {
            array_stride: 12,
            step_mode: wgt::VertexStepMode::Vertex,
            attributes: &position_color_attributes,
        })],
        crate::RasterKernel::IndexedPositionFloat32x3 => vec![Some(wgpu_hal::VertexBufferLayout {
            array_stride: 12,
            step_mode: wgt::VertexStepMode::Vertex,
            attributes: &position_f32x3_attributes,
        })],
        crate::RasterKernel::IndexedPositionFloat32x3CameraMaterial => {
            vec![Some(wgpu_hal::VertexBufferLayout {
                array_stride: 12,
                step_mode: wgt::VertexStepMode::Vertex,
                attributes: &position_f32x3_attributes,
            })]
        }
        crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTexture => {
            vec![Some(wgpu_hal::VertexBufferLayout {
                array_stride: 12,
                step_mode: wgt::VertexStepMode::Vertex,
                attributes: &position_f32x3_attributes,
            })]
        }
        crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUv
        | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
        | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb => {
            vec![
                Some(wgpu_hal::VertexBufferLayout {
                    array_stride: 12,
                    step_mode: wgt::VertexStepMode::Vertex,
                    attributes: &position_f32x3_attributes,
                }),
                Some(wgpu_hal::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgt::VertexStepMode::Vertex,
                    attributes: &texture_coordinate_f32x2_attributes,
                }),
            ]
        }
        crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialNormalLambert => {
            vec![
                Some(wgpu_hal::VertexBufferLayout {
                    array_stride: 12,
                    step_mode: wgt::VertexStepMode::Vertex,
                    attributes: &position_f32x3_attributes,
                }),
                Some(wgpu_hal::VertexBufferLayout {
                    array_stride: 12,
                    step_mode: wgt::VertexStepMode::Vertex,
                    attributes: &normal_f32x3_attributes,
                }),
            ]
        }
        crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialVertexColor => {
            vec![
                Some(wgpu_hal::VertexBufferLayout {
                    array_stride: 12,
                    step_mode: wgt::VertexStepMode::Vertex,
                    attributes: &position_f32x3_attributes,
                }),
                Some(wgpu_hal::VertexBufferLayout {
                    array_stride: 4,
                    step_mode: wgt::VertexStepMode::Vertex,
                    attributes: &color_unorm8x4_attributes,
                }),
            ]
        }
    };
    let color_targets = [Some(wgt::ColorTargetState::from(
        wgt::TextureFormat::Rgba8Unorm,
    ))];
    let frame_uniform_entries = [wgt::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgt::ShaderStages::VERTEX_FRAGMENT,
        ty: wgt::BindingType::Buffer {
            ty: wgt::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: Some(NonZeroU64::new(80).expect("fixed non-zero uniform size")),
        },
        count: None,
    }];
    let textured_frame_entries = [
        frame_uniform_entries[0],
        wgt::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgt::ShaderStages::FRAGMENT,
            ty: wgt::BindingType::Texture {
                sample_type: wgt::TextureSampleType::Float { filterable: false },
                view_dimension: wgt::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
    ];
    let linear_clamp_textured_frame_entries = [
        frame_uniform_entries[0],
        wgt::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgt::ShaderStages::FRAGMENT,
            ty: wgt::BindingType::Texture {
                sample_type: wgt::TextureSampleType::Float { filterable: true },
                view_dimension: wgt::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgt::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgt::ShaderStages::FRAGMENT,
            ty: wgt::BindingType::Sampler(wgt::SamplerBindingType::Filtering),
            count: None,
        },
    ];
    let raster_binding_entries: &[wgt::BindGroupLayoutEntry] = if matches!(
        kernel,
        crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTexture
            | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUv
    ) {
        &textured_frame_entries
    } else if matches!(
        kernel,
        crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
            | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
    ) {
        let filterable = match kernel {
            crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp => {
                owner.capabilities.rgba8_unorm_filterable
            }
            crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb => {
                owner.capabilities.rgba8_unorm_srgb_filterable
            }
            _ => unreachable!("linear clamp kernel already matched"),
        };
        if !filterable {
            return Err(RasterPipelineCreateError::NativeObjectCreation(
                "linear sampling is not supported for this texture format on the selected adapter"
                    .into(),
            ));
        }
        &linear_clamp_textured_frame_entries
    } else {
        &frame_uniform_entries
    };
    let constants = naga::back::PipelineConstants::default();
    // Keep the public API closed: every fixed recipe has two private native
    // realizations. The original one preserves color-only passes; this sibling
    // realization is selected only while a Depth32Float attachment is active.
    let depth_stencil = wgt::DepthStencilState {
        format: wgt::TextureFormat::Depth32Float,
        depth_write_enabled: Some(true),
        depth_compare: Some(wgt::CompareFunction::LessEqual),
        stencil: wgt::StencilState::default(),
        bias: wgt::DepthBiasState::default(),
    };
    let native = match &owner.native {
        #[cfg(feature = "dx12")]
        NativeDevice::Dx12 { device, .. } => {
            let make_shader = |label| unsafe {
                // SAFETY: WGSL was parsed and validated into owned Naga IR;
                // runtime checks are enabled and `device` stays retained.
                device.create_shader_module(
                    &wgpu_hal::ShaderModuleDescriptor {
                        label: Some(label),
                        runtime_checks: wgt::ShaderRuntimeChecks::checked(),
                    },
                    wgpu_hal::ShaderInput::Naga(wgpu_hal::NagaShader {
                        module: Cow::Owned(module.clone()),
                        info: info.clone(),
                        debug_source: None,
                    }),
                )
            };
            let vertex_shader = make_shader("fluxel fixed raster vertex shader").map_err(|e| {
                RasterPipelineCreateError::ShaderCompilation(format!(
                    "DX12 vertex shader creation failed: {e}"
                ))
            })?;
            let fragment_shader = match make_shader("fluxel fixed raster fragment shader") {
                Ok(v) => v,
                Err(e) => {
                    unsafe {
                        // SAFETY: this device uniquely owns the live module;
                        // no pipeline was created after fragment creation failed.
                        device.destroy_shader_module(vertex_shader)
                    };
                    return Err(RasterPipelineCreateError::ShaderCompilation(format!(
                        "DX12 fragment shader creation failed: {e}"
                    )));
                }
            };
            let bind_group_layout = if matches!(
                kernel,
                crate::RasterKernel::IndexedPositionFloat32x3CameraMaterial
                    | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTexture
                    | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUv
                    | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
                    | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
                    | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialNormalLambert
                    | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialVertexColor
            ) {
                match unsafe {
                    // SAFETY: the static one-entry descriptor is valid and
                    // its entries live through this creation call.
                    device.create_bind_group_layout(&wgpu_hal::BindGroupLayoutDescriptor {
                        label: Some("fluxel fixed camera/material frame layout"),
                        flags: wgpu_hal::BindGroupLayoutFlags::empty(),
                        entries: raster_binding_entries,
                    })
                } {
                    Ok(value) => Some(value),
                    Err(e) => {
                        unsafe {
                            // SAFETY: BGL creation failed before any dependent
                            // layout/pipeline existed; these same-device shader
                            // modules are unique, unpublished, and destroyed once.
                            device.destroy_shader_module(fragment_shader);
                            device.destroy_shader_module(vertex_shader)
                        };
                        return Err(RasterPipelineCreateError::NativeObjectCreation(format!(
                            "DX12 raster bind-group-layout creation failed: {e}"
                        )));
                    }
                }
            } else {
                None
            };
            let layout_result = if let Some(ref bgl) = bind_group_layout {
                let layouts = [Some(bgl)];
                unsafe {
                    // SAFETY: this retained same-device BGL is live for the call.
                    device.create_pipeline_layout(&wgpu_hal::PipelineLayoutDescriptor {
                        label: Some("fluxel fixed raster pipeline layout"),
                        flags: wgpu_hal::PipelineLayoutFlags::empty(),
                        bind_group_layouts: &layouts,
                        immediate_size: 0,
                    })
                }
            } else {
                let layouts: [Option<&wgpu_hal::dx12::BindGroupLayout>; 0] = [];
                unsafe {
                    // SAFETY: the zero-bind-group descriptor is internally valid
                    // and all referenced storage lives through this call.
                    device.create_pipeline_layout(&wgpu_hal::PipelineLayoutDescriptor {
                        label: Some("fluxel fixed raster pipeline layout"),
                        flags: wgpu_hal::PipelineLayoutFlags::empty(),
                        bind_group_layouts: &layouts,
                        immediate_size: 0,
                    })
                }
            };
            let layout = match layout_result {
                Ok(v) => v,
                Err(e) => {
                    unsafe {
                        // SAFETY: layout creation failed, so both same-device
                        // modules are unreferenced and uniquely consumed here.
                        device.destroy_shader_module(fragment_shader);
                        device.destroy_shader_module(vertex_shader);
                        if let Some(bgl) = bind_group_layout {
                            device.destroy_bind_group_layout(bgl);
                        }
                    };
                    return Err(RasterPipelineCreateError::NativeObjectCreation(format!(
                        "DX12 raster pipeline-layout creation failed: {e}"
                    )));
                }
            };
            let desc = wgpu_hal::RenderPipelineDescriptor {
                label: Some("fluxel fixed Rgba8Unorm raster pipeline"),
                layout: &layout,
                vertex_processor: wgpu_hal::VertexProcessor::Standard {
                    vertex_buffers: &buffers,
                    vertex_stage: wgpu_hal::ProgrammableStage {
                        module: &vertex_shader,
                        entry_point: vertex_entry,
                        constants: &constants,
                        zero_initialize_workgroup_memory: true,
                    },
                },
                primitive: wgt::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgt::MultisampleState::default(),
                fragment_stage: Some(wgpu_hal::ProgrammableStage {
                    module: &fragment_shader,
                    entry_point: fragment_entry,
                    constants: &constants,
                    zero_initialize_workgroup_memory: true,
                }),
                color_targets: &color_targets,
                multiview_mask: None,
                cache: None,
            };
            let pipeline = match unsafe {
                // SAFETY: descriptor modules/layout/entries are validated,
                // live, same-device objects and outlive this creation call.
                device.create_render_pipeline(&desc)
            } {
                Ok(v) => v,
                Err(e) => {
                    unsafe {
                        // SAFETY: pipeline creation failed; the same-device
                        // dependencies are uniquely consumed in reverse order.
                        device.destroy_pipeline_layout(layout);
                        if let Some(bgl) = bind_group_layout {
                            device.destroy_bind_group_layout(bgl);
                        }
                        device.destroy_shader_module(fragment_shader);
                        device.destroy_shader_module(vertex_shader)
                    };
                    return Err(RasterPipelineCreateError::ShaderCompilation(format!(
                        "DX12 raster pipeline creation failed: {e}"
                    )));
                }
            };
            let mut depth_desc = desc;
            depth_desc.depth_stencil = Some(depth_stencil.clone());
            let depth_pipeline = match unsafe {
                // SAFETY: this differs from the color-only fixed descriptor
                // solely by the validated Depth32Float attachment contract.
                device.create_render_pipeline(&depth_desc)
            } {
                Ok(value) => value,
                Err(e) => {
                    unsafe {
                        device.destroy_render_pipeline(pipeline);
                        device.destroy_pipeline_layout(layout);
                        if let Some(bgl) = bind_group_layout {
                            device.destroy_bind_group_layout(bgl);
                        }
                        device.destroy_shader_module(fragment_shader);
                        device.destroy_shader_module(vertex_shader);
                    }
                    return Err(RasterPipelineCreateError::ShaderCompilation(format!(
                        "DX12 depth raster pipeline creation failed: {e}"
                    )));
                }
            };
            NativeRasterPipelineInner::Dx12 {
                vertex_shader,
                fragment_shader,
                bind_group_layout,
                pipeline_layout: layout,
                pipeline,
                depth_pipeline,
            }
        }
        #[cfg(feature = "vulkan")]
        NativeDevice::Vulkan { device, .. } => {
            let make_shader = |label| unsafe {
                // SAFETY: same validated owned Naga IR and retained-device
                // proof as the DX12 shader creation branch.
                device.create_shader_module(
                    &wgpu_hal::ShaderModuleDescriptor {
                        label: Some(label),
                        runtime_checks: wgt::ShaderRuntimeChecks::checked(),
                    },
                    wgpu_hal::ShaderInput::Naga(wgpu_hal::NagaShader {
                        module: Cow::Owned(module.clone()),
                        info: info.clone(),
                        debug_source: None,
                    }),
                )
            };
            let vertex_shader = make_shader("fluxel fixed raster vertex shader").map_err(|e| {
                RasterPipelineCreateError::ShaderCompilation(format!(
                    "Vulkan vertex shader creation failed: {e}"
                ))
            })?;
            let fragment_shader = match make_shader("fluxel fixed raster fragment shader") {
                Ok(v) => v,
                Err(e) => {
                    unsafe {
                        // SAFETY: fragment creation failed before any pipeline;
                        // this same-device vertex module is uniquely consumed.
                        device.destroy_shader_module(vertex_shader)
                    };
                    return Err(RasterPipelineCreateError::ShaderCompilation(format!(
                        "Vulkan fragment shader creation failed: {e}"
                    )));
                }
            };
            let bind_group_layout = if matches!(
                kernel,
                crate::RasterKernel::IndexedPositionFloat32x3CameraMaterial
                    | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTexture
                    | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUv
                    | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
                    | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
                    | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialNormalLambert
                    | crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialVertexColor
            ) {
                match unsafe {
                    // SAFETY: static binding-zero uniform layout is valid and borrowed only for this call.
                    device.create_bind_group_layout(&wgpu_hal::BindGroupLayoutDescriptor {
                        label: Some("fluxel fixed camera/material frame layout"),
                        flags: wgpu_hal::BindGroupLayoutFlags::empty(),
                        entries: raster_binding_entries,
                    })
                } {
                    Ok(value) => Some(value),
                    Err(e) => {
                        unsafe {
                            // SAFETY: BGL creation failed before any dependent
                            // layout/pipeline existed; these same-device shader
                            // modules are unique, unpublished, and destroyed once.
                            device.destroy_shader_module(fragment_shader);
                            device.destroy_shader_module(vertex_shader)
                        };
                        return Err(RasterPipelineCreateError::NativeObjectCreation(format!(
                            "Vulkan raster bind-group-layout creation failed: {e}"
                        )));
                    }
                }
            } else {
                None
            };
            let layout_result = if let Some(ref bgl) = bind_group_layout {
                let layouts = [Some(bgl)];
                unsafe {
                    // SAFETY: the retained same-device BGL outlives layout creation.
                    device.create_pipeline_layout(&wgpu_hal::PipelineLayoutDescriptor {
                        label: Some("fluxel fixed raster pipeline layout"),
                        flags: wgpu_hal::PipelineLayoutFlags::empty(),
                        bind_group_layouts: &layouts,
                        immediate_size: 0,
                    })
                }
            } else {
                let layouts: [Option<&wgpu_hal::vulkan::BindGroupLayout>; 0] = [];
                unsafe {
                    // SAFETY: the zero-group descriptor is valid and its borrowed
                    // storage remains live for the complete call.
                    device.create_pipeline_layout(&wgpu_hal::PipelineLayoutDescriptor {
                        label: Some("fluxel fixed raster pipeline layout"),
                        flags: wgpu_hal::PipelineLayoutFlags::empty(),
                        bind_group_layouts: &layouts,
                        immediate_size: 0,
                    })
                }
            };
            let layout = match layout_result {
                Ok(v) => v,
                Err(e) => {
                    unsafe {
                        // SAFETY: failed layout creation left both same-device
                        // modules unreferenced; consume them exactly once.
                        device.destroy_shader_module(fragment_shader);
                        device.destroy_shader_module(vertex_shader);
                        if let Some(bgl) = bind_group_layout {
                            device.destroy_bind_group_layout(bgl);
                        }
                    };
                    return Err(RasterPipelineCreateError::NativeObjectCreation(format!(
                        "Vulkan raster pipeline-layout creation failed: {e}"
                    )));
                }
            };
            let desc = wgpu_hal::RenderPipelineDescriptor {
                label: Some("fluxel fixed Rgba8Unorm raster pipeline"),
                layout: &layout,
                vertex_processor: wgpu_hal::VertexProcessor::Standard {
                    vertex_buffers: &buffers,
                    vertex_stage: wgpu_hal::ProgrammableStage {
                        module: &vertex_shader,
                        entry_point: vertex_entry,
                        constants: &constants,
                        zero_initialize_workgroup_memory: true,
                    },
                },
                primitive: wgt::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgt::MultisampleState::default(),
                fragment_stage: Some(wgpu_hal::ProgrammableStage {
                    module: &fragment_shader,
                    entry_point: fragment_entry,
                    constants: &constants,
                    zero_initialize_workgroup_memory: true,
                }),
                color_targets: &color_targets,
                multiview_mask: None,
                cache: None,
            };
            let pipeline = match unsafe {
                // SAFETY: validated same-device modules and layout are live;
                // vertex/color descriptors satisfy the fixed pipeline recipe.
                device.create_render_pipeline(&desc)
            } {
                Ok(v) => v,
                Err(e) => {
                    unsafe {
                        // SAFETY: no pipeline exists; destroy its uniquely owned
                        // same-device dependencies in reverse creation order.
                        device.destroy_pipeline_layout(layout);
                        if let Some(bgl) = bind_group_layout {
                            device.destroy_bind_group_layout(bgl);
                        }
                        device.destroy_shader_module(fragment_shader);
                        device.destroy_shader_module(vertex_shader)
                    };
                    return Err(RasterPipelineCreateError::ShaderCompilation(format!(
                        "Vulkan raster pipeline creation failed: {e}"
                    )));
                }
            };
            let mut depth_desc = desc;
            depth_desc.depth_stencil = Some(depth_stencil.clone());
            let depth_pipeline = match unsafe {
                // SAFETY: this is the fixed Depth32Float sibling of the
                // already validated color-only recipe descriptor.
                device.create_render_pipeline(&depth_desc)
            } {
                Ok(value) => value,
                Err(e) => {
                    unsafe {
                        device.destroy_render_pipeline(pipeline);
                        device.destroy_pipeline_layout(layout);
                        if let Some(bgl) = bind_group_layout {
                            device.destroy_bind_group_layout(bgl);
                        }
                        device.destroy_shader_module(fragment_shader);
                        device.destroy_shader_module(vertex_shader);
                    }
                    return Err(RasterPipelineCreateError::ShaderCompilation(format!(
                        "Vulkan depth raster pipeline creation failed: {e}"
                    )));
                }
            };
            NativeRasterPipelineInner::Vulkan {
                vertex_shader,
                fragment_shader,
                bind_group_layout,
                pipeline_layout: layout,
                pipeline,
                depth_pipeline,
            }
        }
    };
    Ok(NativeRasterPipeline(Arc::new(NativeRasterPipelineShared {
        native: Some(native),
        owner: Arc::clone(owner),
        kernel,
    })))
}
