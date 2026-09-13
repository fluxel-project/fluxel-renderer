//! Raster state-machine rejection witness.

use super::native_negative_support::{raster_negative_color, raster_negative_ops};
use super::native_raster_recipes::run_raster_vertex_recipe_negative_paths;
use crate::*;
use fluxel_rendergraph::*;
pub(super) fn run_raster_negative_paths(backend_kind: crate::Backend) {
    let device = Device::open(
        backend_kind,
        crate::DeviceOptions {
            validation: crate::Validation::Required,
            ..crate::DeviceOptions::default()
        },
    )
    .unwrap();
    crate::imp::clear_validation_diagnostics(&device.inner);
    let target_descriptor = TextureDesc {
        dimension: fluxel_rendergraph::TextureDimension::D2,
        extent: Extent3d {
            width: 8,
            height: 8,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
    };
    let target = device
        .create_texture(TextureDescriptor {
            texture: target_descriptor,
            usage: TextureUsage::from_kinds([TextureUsageKind::ColorAttachment]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let overlap_descriptor = TextureDesc {
        mip_levels: 2,
        ..target_descriptor
    };
    let overlap_texture = device
        .create_texture(TextureDescriptor {
            texture: overlap_descriptor,
            usage: TextureUsage::from_kinds([TextureUsageKind::ColorAttachment]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let pipeline = device
        .create_raster_pipeline(crate::RasterKernel::IndexedPositionColor)
        .unwrap();
    let triangle = device
        .create_raster_pipeline(crate::RasterKernel::Triangle)
        .unwrap();
    let camera_pipeline = device
        .create_raster_pipeline(crate::RasterKernel::IndexedPositionFloat32x3CameraMaterial)
        .unwrap();
    let other_camera_pipeline = device
        .create_raster_pipeline(crate::RasterKernel::IndexedPositionFloat32x3CameraMaterial)
        .unwrap();
    let textured_pipeline = device
        .create_raster_pipeline(crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTexture)
        .unwrap();
    let uniform = device
        .create_buffer(BufferDescriptor {
            buffer: BufferDesc { size: 80 },
            usage: BufferUsage::from_kinds([BufferUsageKind::Uniform]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let camera_bindings = RasterBindings::RasterUniform(
        device
            .create_raster_uniform_bindings(&camera_pipeline, &uniform)
            .unwrap(),
    );
    let other_camera_bindings = RasterBindings::RasterUniform(
        device
            .create_raster_uniform_bindings(&other_camera_pipeline, &uniform)
            .unwrap(),
    );
    let sampled_texture_descriptor = TextureDesc {
        dimension: fluxel_rendergraph::TextureDimension::D2,
        extent: Extent3d {
            width: 2,
            height: 2,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
    };
    let sampled_texture = device
        .create_texture(TextureDescriptor {
            texture: sampled_texture_descriptor,
            usage: TextureUsage::from_kinds([TextureUsageKind::Sampled]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let textured_bindings = RasterBindings::RasterTexture(
        device
            .create_raster_texture_bindings(&textured_pipeline, &uniform, &sampled_texture)
            .unwrap(),
    );
    let compute_pipeline = device
        .create_compute_pipeline(crate::ComputeKernel::WrappingAdd)
        .unwrap();
    let compute_buffer = device
        .create_buffer(BufferDescriptor {
            buffer: BufferDesc { size: 64 },
            usage: BufferUsage::from_kinds([
                BufferUsageKind::StorageRead,
                BufferUsageKind::StorageWrite,
            ]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let compute_bindings = RasterBindings::Compute(
        device
            .create_compute_bindings(&compute_pipeline, &compute_buffer, 0, 64)
            .unwrap(),
    );
    let foreign_device = Device::open(backend_kind, crate::DeviceOptions::default()).unwrap();
    let foreign_pipeline = foreign_device
        .create_raster_pipeline(crate::RasterKernel::Triangle)
        .unwrap();
    let foreign_textured_pipeline = foreign_device
        .create_raster_pipeline(crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTexture)
        .unwrap();
    let foreign_target = foreign_device
        .create_texture(TextureDescriptor {
            texture: target_descriptor,
            usage: TextureUsage::from_kinds([TextureUsageKind::ColorAttachment]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let foreign_sampled_texture = foreign_device
        .create_texture(TextureDescriptor {
            texture: sampled_texture_descriptor,
            usage: TextureUsage::from_kinds([TextureUsageKind::Sampled]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let mut provider = RasterObjectProvider::new(&device);
    assert_eq!(
        provider.register_raster_pipeline(RasterPipelineId::new(999), foreign_pipeline),
        Err(NativeExecutionError::ForeignResource)
    );
    let camera_pipeline_id = RasterPipelineId::new(997);
    let camera_binding_id = BindingSetId::new(998);
    provider
        .register_raster_pipeline(camera_pipeline_id, camera_pipeline.clone())
        .unwrap();
    provider
        .register_raster_uniform_bindings(camera_binding_id, camera_pipeline_id)
        .unwrap();
    let textured_pipeline_id = RasterPipelineId::new(995);
    let textured_binding_id = BindingSetId::new(996);
    provider
        .register_raster_pipeline(textured_pipeline_id, textured_pipeline.clone())
        .unwrap();
    provider
        .register_raster_textured_bindings(textured_binding_id, textured_pipeline_id)
        .unwrap();
    let uniform_resource = [ResolvedBindingResource::Buffer {
        physical: &uniform,
        range: BufferRange::Whole,
        semantic: fluxel_rendergraph::BindingResourceSemantic::BufferRead(
            fluxel_rendergraph::BufferReadUse::Uniform,
        ),
    }];
    assert!(
        provider
            .bindings(camera_binding_id, &uniform_resource, &[0])
            .is_err()
    );
    let ranged_uniform = [ResolvedBindingResource::Buffer {
        physical: &uniform,
        range: BufferRange::Bytes {
            offset: 0,
            size: 80,
        },
        semantic: fluxel_rendergraph::BindingResourceSemantic::BufferRead(
            fluxel_rendergraph::BufferReadUse::Uniform,
        ),
    }];
    assert!(
        provider
            .bindings(camera_binding_id, &ranged_uniform, &[])
            .is_err()
    );
    let wrong_uniform_semantic = [ResolvedBindingResource::Buffer {
        physical: &uniform,
        range: BufferRange::Whole,
        semantic: fluxel_rendergraph::BindingResourceSemantic::BufferRead(
            fluxel_rendergraph::BufferReadUse::Storage,
        ),
    }];
    assert!(
        provider
            .bindings(camera_binding_id, &wrong_uniform_semantic, &[])
            .is_err()
    );
    assert!(
        provider
            .bindings(camera_binding_id, &uniform_resource, &[])
            .is_ok()
    );
    let textured_resources = [
        ResolvedBindingResource::Buffer {
            physical: &uniform,
            range: BufferRange::Whole,
            semantic: fluxel_rendergraph::BindingResourceSemantic::BufferRead(
                fluxel_rendergraph::BufferReadUse::Uniform,
            ),
        },
        ResolvedBindingResource::Texture {
            physical: &sampled_texture,
            range: TextureRange::Whole,
            semantic: fluxel_rendergraph::BindingResourceSemantic::TextureRead(
                fluxel_rendergraph::TextureReadUse::Sampled,
            ),
        },
    ];
    // The graph-facing provider validates the exact two-resource recipe and
    // refuses dynamic offsets before constructing a native bind group.
    assert!(
        provider
            .bindings(textured_binding_id, &textured_resources, &[0])
            .is_err()
    );
    assert!(
        provider
            .bindings(textured_binding_id, &textured_resources[..1], &[])
            .is_err()
    );
    assert!(
        provider
            .bindings(textured_binding_id, &textured_resources, &[])
            .is_ok()
    );

    // The public safe boundary must reject every descriptor fact the
    // closed texture_2d<f32> ABI cannot represent.  These objects are
    // intentionally created on the real backend; no malformed binding is
    // ever lowered or submitted.
    // D3 is not a supported RHI texture resource at all, so this must be
    // rejected at creation rather than pretending it reached the binding
    // boundary.
    assert!(
        device
            .create_texture(TextureDescriptor {
                texture: TextureDesc {
                    dimension: fluxel_rendergraph::TextureDimension::D3,
                    extent: Extent3d {
                        width: 2,
                        height: 2,
                        depth: 2,
                    },
                    ..sampled_texture_descriptor
                },
                usage: TextureUsage::from_kinds([TextureUsageKind::Sampled]),
                memory: MemoryPolicy::DeviceOnly,
            })
            .is_err()
    );
    for invalid_texture in [
        TextureDesc {
            format: TextureFormat::Depth32Float,
            ..sampled_texture_descriptor
        },
        TextureDesc {
            mip_levels: 2,
            ..sampled_texture_descriptor
        },
        TextureDesc {
            array_layers: 2,
            ..sampled_texture_descriptor
        },
    ] {
        let texture = device
            .create_texture(TextureDescriptor {
                texture: invalid_texture,
                usage: TextureUsage::from_kinds([TextureUsageKind::Sampled]),
                memory: MemoryPolicy::DeviceOnly,
            })
            .unwrap();
        assert!(matches!(
            device.create_raster_texture_bindings(&textured_pipeline, &uniform, &texture),
            Err(crate::RasterCreateError::BindingRecipeMismatch)
        ));
    }
    // Invalid sample counts are rejected by the resource boundary.  Do
    // not manufacture a supported multisample resource here: its failure
    // to match this single-sample ABI is already checked by the binding
    // contract, and this negative fixture must not risk device loss.
    assert!(
        device
            .create_texture(TextureDescriptor {
                texture: TextureDesc {
                    sample_count: 0,
                    ..sampled_texture_descriptor
                },
                usage: TextureUsage::from_kinds([TextureUsageKind::Sampled]),
                memory: MemoryPolicy::DeviceOnly,
            })
            .is_err()
    );
    // Reuse the already-real attachment texture to prove a resource that
    // lacks Sampled usage is rejected before binding/submission.
    assert!(matches!(
        device.create_raster_texture_bindings(&textured_pipeline, &uniform, &target),
        Err(crate::RasterCreateError::BindingRecipeMismatch)
    ));
    assert!(matches!(
        device.create_raster_texture_bindings(&camera_pipeline, &uniform, &sampled_texture),
        Err(crate::RasterCreateError::BindingRecipeMismatch)
    ));
    assert!(matches!(
        device.create_raster_texture_bindings(
            &foreign_textured_pipeline,
            &uniform,
            &sampled_texture
        ),
        Err(crate::RasterCreateError::ForeignDevice)
    ));
    assert!(matches!(
        device.create_raster_texture_bindings(
            &textured_pipeline,
            &uniform,
            &foreign_sampled_texture
        ),
        Err(crate::RasterCreateError::ForeignDevice)
    ));

    let color = raster_negative_color(&target);
    let descriptor = RasterPassDescriptor {
        label: "negative-raster",
        colors: std::slice::from_ref(&color),
        depth_stencil: None,
    };
    let mismatch = NativeExecutionError::RasterStateMismatch;
    let mut backend = RasterBackend::new(device.clone());
    let depth_descriptor = TextureDesc {
        dimension: fluxel_rendergraph::TextureDimension::D2,
        extent: Extent3d {
            width: 8,
            height: 8,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        sample_count: 1,
        format: TextureFormat::Depth32Float,
    };
    let depth_target = device
        .create_texture(TextureDescriptor {
            texture: depth_descriptor,
            usage: TextureUsage::from_kinds([TextureUsageKind::DepthStencilAttachment]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let depth_ops = AttachmentOps {
        load: LoadOp::Clear(0.5),
        store: StoreOp::Store,
        write_coverage: WriteCoverage::Full,
    };

    // All malformed pass descriptors fail before opening a native render pass.
    let foreign_color = raster_negative_color(&foreign_target);
    let foreign_descriptor = RasterPassDescriptor {
        label: "foreign",
        colors: std::slice::from_ref(&foreign_color),
        depth_stencil: None,
    };
    let mut encoder = backend.begin_encoder(QueueId::new(0)).unwrap();
    assert!(matches!(
        backend.begin_raster(&mut encoder, &descriptor),
        Err(NativeExecutionError::Recording(_))
    ));
    for invalid_range in [
        TextureRange::Subresources {
            base_mip_level: 0,
            mip_level_count: 0,
            base_array_layer: 0,
            array_layer_count: 1,
            aspect: fluxel_rendergraph::TextureAspect::Color,
        },
        TextureRange::Subresources {
            base_mip_level: 1,
            mip_level_count: 1,
            base_array_layer: 0,
            array_layer_count: 1,
            aspect: fluxel_rendergraph::TextureAspect::Color,
        },
    ] {
        assert!(matches!(
            backend.transition_texture(
                &mut encoder,
                &target,
                invalid_range,
                ResourceAccessState::Undefined,
                ResourceAccessState::ColorAttachmentWrite,
            ),
            Err(NativeExecutionError::Recording(_))
        ));
    }
    backend
        .transition_texture(
            &mut encoder,
            &target,
            TextureRange::Subresources {
                base_mip_level: 0,
                mip_level_count: 1,
                base_array_layer: 0,
                array_layer_count: 1,
                aspect: fluxel_rendergraph::TextureAspect::Color,
            },
            ResourceAccessState::Undefined,
            ResourceAccessState::ColorAttachmentWrite,
        )
        .unwrap();
    assert!(matches!(
        backend.transition_texture(
            &mut encoder,
            &target,
            TextureRange::Whole,
            ResourceAccessState::Undefined,
            ResourceAccessState::CopySource,
        ),
        Err(NativeExecutionError::Recording(_))
    ));
    let mut overlap_encoder = backend.begin_encoder(QueueId::new(0)).unwrap();
    backend
        .transition_texture(
            &mut overlap_encoder,
            &overlap_texture,
            TextureRange::Whole,
            ResourceAccessState::Undefined,
            ResourceAccessState::ColorAttachmentWrite,
        )
        .unwrap();
    assert!(matches!(
        backend.transition_texture(
            &mut overlap_encoder,
            &overlap_texture,
            TextureRange::Subresources {
                base_mip_level: 1,
                mip_level_count: 1,
                base_array_layer: 0,
                array_layer_count: 1,
                aspect: fluxel_rendergraph::TextureAspect::Color,
            },
            ResourceAccessState::Undefined,
            ResourceAccessState::CopySource,
        ),
        Err(NativeExecutionError::Recording(_))
    ));
    drop(overlap_encoder);
    let mut overlap_encoder = backend.begin_encoder(QueueId::new(0)).unwrap();
    backend
        .transition_texture(
            &mut overlap_encoder,
            &overlap_texture,
            TextureRange::Subresources {
                base_mip_level: 1,
                mip_level_count: 1,
                base_array_layer: 0,
                array_layer_count: 1,
                aspect: fluxel_rendergraph::TextureAspect::Color,
            },
            ResourceAccessState::Undefined,
            ResourceAccessState::ColorAttachmentWrite,
        )
        .unwrap();
    assert!(matches!(
        backend.transition_texture(
            &mut overlap_encoder,
            &overlap_texture,
            TextureRange::Whole,
            ResourceAccessState::Undefined,
            ResourceAccessState::CopySource,
        ),
        Err(NativeExecutionError::Recording(_))
    ));
    drop(overlap_encoder);
    assert_eq!(
        backend.set_compute_pipeline(&mut encoder, &compute_pipeline),
        Err(NativeExecutionError::ComputeBindingMismatch)
    );
    assert_eq!(
        backend.set_bindings(&mut encoder, &compute_bindings),
        Err(NativeExecutionError::ComputeBindingMismatch)
    );
    assert_eq!(
        backend.dispatch(&mut encoder, [1, 1, 1]),
        Err(NativeExecutionError::ComputeBindingMismatch)
    );
    assert_eq!(
        backend.begin_raster(&mut encoder, &foreign_descriptor),
        Err(NativeExecutionError::ForeignResource)
    );
    let ranged_color = RasterColorAttachment {
        index: 0,
        texture: &target,
        range: TextureRange::Subresources {
            base_mip_level: 0,
            mip_level_count: 1,
            base_array_layer: 0,
            array_layer_count: 1,
            aspect: fluxel_rendergraph::TextureAspect::Color,
        },
        operations: raster_negative_ops(),
    };
    assert_eq!(
        backend.begin_raster(
            &mut encoder,
            &RasterPassDescriptor {
                label: "range",
                colors: std::slice::from_ref(&ranged_color),
                depth_stencil: None,
            }
        ),
        Err(mismatch.clone())
    );
    // Every depth descriptor fact is rejected by the safe adapter before a
    // native pass opens. These deliberately use an encoder without depth
    // transitions: a valid descriptor would subsequently require the tracked
    // DepthStencilWrite/ReadWrite state, while each case below must fail first.
    let depth_without_usage = device
        .create_texture(TextureDescriptor {
            texture: depth_descriptor,
            usage: TextureUsage::from_kinds([TextureUsageKind::CopySource]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let depth_wrong_extent = device
        .create_texture(TextureDescriptor {
            texture: TextureDesc {
                extent: Extent3d {
                    width: 4,
                    height: 8,
                    depth: 1,
                },
                ..depth_descriptor
            },
            usage: TextureUsage::from_kinds([TextureUsageKind::DepthStencilAttachment]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    for attachment in [
        RasterDepthStencilAttachment {
            texture: &depth_without_usage,
            range: TextureRange::Whole,
            depth: Some(depth_ops),
            stencil: None,
        },
        RasterDepthStencilAttachment {
            texture: &depth_target,
            range: TextureRange::Subresources {
                base_mip_level: 0,
                mip_level_count: 1,
                base_array_layer: 0,
                array_layer_count: 1,
                aspect: fluxel_rendergraph::TextureAspect::Depth,
            },
            depth: Some(depth_ops),
            stencil: None,
        },
        RasterDepthStencilAttachment {
            texture: &target,
            range: TextureRange::Whole,
            depth: Some(depth_ops),
            stencil: None,
        },
        RasterDepthStencilAttachment {
            texture: &depth_wrong_extent,
            range: TextureRange::Whole,
            depth: Some(depth_ops),
            stencil: None,
        },
        RasterDepthStencilAttachment {
            texture: &depth_target,
            range: TextureRange::Whole,
            depth: None,
            stencil: Some(AttachmentOps {
                load: LoadOp::Clear(0),
                store: StoreOp::Store,
                write_coverage: WriteCoverage::Full,
            }),
        },
        RasterDepthStencilAttachment {
            texture: &depth_target,
            range: TextureRange::Whole,
            depth: None,
            stencil: None,
        },
        RasterDepthStencilAttachment {
            texture: &depth_target,
            range: TextureRange::Whole,
            depth: Some(AttachmentOps {
                load: LoadOp::DontCare,
                store: StoreOp::Store,
                write_coverage: WriteCoverage::Unknown,
            }),
            stencil: None,
        },
        RasterDepthStencilAttachment {
            texture: &depth_target,
            range: TextureRange::Whole,
            depth: Some(AttachmentOps {
                load: LoadOp::Clear(f32::NAN),
                store: StoreOp::Store,
                write_coverage: WriteCoverage::Full,
            }),
            stencil: None,
        },
        RasterDepthStencilAttachment {
            texture: &depth_target,
            range: TextureRange::Whole,
            depth: Some(AttachmentOps {
                load: LoadOp::Clear(1.01),
                store: StoreOp::Store,
                write_coverage: WriteCoverage::Full,
            }),
            stencil: None,
        },
    ] {
        assert_eq!(
            backend.begin_raster(
                &mut encoder,
                &RasterPassDescriptor {
                    label: "invalid-depth",
                    colors: std::slice::from_ref(&color),
                    depth_stencil: Some(attachment),
                }
            ),
            Err(mismatch.clone())
        );
    }
    let depth = RasterDepthStencilAttachment {
        texture: &target,
        range: TextureRange::Whole,
        depth: None,
        stencil: None,
    };
    assert_eq!(
        backend.begin_raster(
            &mut encoder,
            &RasterPassDescriptor {
                label: "depth",
                colors: std::slice::from_ref(&color),
                depth_stencil: Some(depth),
            }
        ),
        Err(mismatch.clone())
    );
    assert_eq!(
        backend.begin_raster(
            &mut encoder,
            &RasterPassDescriptor {
                label: "mrt",
                colors: &[
                    raster_negative_color(&target),
                    raster_negative_color(&target),
                ],
                depth_stencil: None,
            }
        ),
        Err(mismatch.clone())
    );
    drop(encoder);

    // Native proof for the minimal depth path: a fixed-recipe draw uses the
    // private Depth32Float pipeline sibling, while both attachments remain
    // retained through submission without caller-owned texture handles.
    let retained_color = device
        .create_texture(TextureDescriptor {
            texture: target_descriptor,
            usage: TextureUsage::from_kinds([TextureUsageKind::ColorAttachment]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let retained_depth = device
        .create_texture(TextureDescriptor {
            texture: depth_descriptor,
            usage: TextureUsage::from_kinds([TextureUsageKind::DepthStencilAttachment]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let depth_clear = RasterDepthStencilAttachment {
        texture: &retained_depth,
        range: TextureRange::Whole,
        depth: Some(depth_ops),
        stencil: None,
    };
    let color_clear = RasterColorAttachment {
        index: 0,
        texture: &retained_color,
        range: TextureRange::Whole,
        operations: AttachmentOps {
            load: LoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
            store: StoreOp::Store,
            write_coverage: WriteCoverage::Full,
        },
    };
    let mut encoder = backend.begin_encoder(QueueId::new(0)).unwrap();
    backend
        .transition_texture(
            &mut encoder,
            &retained_color,
            TextureRange::Whole,
            ResourceAccessState::Undefined,
            ResourceAccessState::ColorAttachmentWrite,
        )
        .unwrap();
    backend
        .transition_texture(
            &mut encoder,
            &retained_depth,
            TextureRange::Whole,
            ResourceAccessState::Undefined,
            ResourceAccessState::DepthStencilWrite,
        )
        .unwrap();
    backend
        .begin_raster(
            &mut encoder,
            &RasterPassDescriptor {
                label: "depth-clear-store",
                colors: std::slice::from_ref(&color_clear),
                depth_stencil: Some(depth_clear),
            },
        )
        .unwrap();
    backend
        .set_raster_pipeline(&mut encoder, &triangle)
        .unwrap();
    backend.draw(&mut encoder, 0..3, 0..1).unwrap();
    backend.end_raster(&mut encoder).unwrap();
    let command = backend.finish_encoder(encoder).unwrap();
    drop(retained_color);
    drop(retained_depth);
    let completion = backend
        .submit(QueueId::new(0), command, Vec::new())
        .unwrap();
    backend
        .wait(&completion, std::time::Duration::from_secs(10))
        .unwrap();
    assert_eq!(
        backend.completion_status(&completion),
        CompletionStatus::Complete
    );
    let diagnostics = crate::imp::validation_diagnostics(&device.inner);
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");

    // A separate two-submission witness proves Load selects the read/write
    // depth state rather than treating it like a clear. The first submission
    // establishes stored contents; the second transitions from that exact
    // outgoing state before recording Load plus a real fixed-recipe draw.
    let initial_depth = RasterDepthStencilAttachment {
        texture: &depth_target,
        range: TextureRange::Whole,
        depth: Some(depth_ops),
        stencil: None,
    };
    let initial_color = RasterColorAttachment {
        index: 0,
        texture: &target,
        range: TextureRange::Whole,
        operations: AttachmentOps {
            load: LoadOp::Clear([0.0, 0.0, 0.0, 1.0]),
            store: StoreOp::Store,
            write_coverage: WriteCoverage::Full,
        },
    };
    let mut encoder = backend.begin_encoder(QueueId::new(0)).unwrap();
    backend
        .transition_texture(
            &mut encoder,
            &target,
            TextureRange::Whole,
            ResourceAccessState::Undefined,
            ResourceAccessState::ColorAttachmentWrite,
        )
        .unwrap();
    backend
        .transition_texture(
            &mut encoder,
            &depth_target,
            TextureRange::Whole,
            ResourceAccessState::Undefined,
            ResourceAccessState::DepthStencilWrite,
        )
        .unwrap();
    backend
        .begin_raster(
            &mut encoder,
            &RasterPassDescriptor {
                label: "depth-initialize-for-load",
                colors: std::slice::from_ref(&initial_color),
                depth_stencil: Some(initial_depth),
            },
        )
        .unwrap();
    backend
        .set_raster_pipeline(&mut encoder, &triangle)
        .unwrap();
    backend.draw(&mut encoder, 0..3, 0..1).unwrap();
    backend.end_raster(&mut encoder).unwrap();
    let command = backend.finish_encoder(encoder).unwrap();
    let completion = backend
        .submit(QueueId::new(0), command, Vec::new())
        .unwrap();
    backend
        .wait(&completion, std::time::Duration::from_secs(10))
        .unwrap();

    let loaded_depth = RasterDepthStencilAttachment {
        texture: &depth_target,
        range: TextureRange::Whole,
        depth: Some(AttachmentOps {
            load: LoadOp::Load,
            store: StoreOp::Store,
            write_coverage: WriteCoverage::Unknown,
        }),
        stencil: None,
    };
    let loaded_color = RasterColorAttachment {
        index: 0,
        texture: &target,
        range: TextureRange::Whole,
        operations: raster_negative_ops(),
    };
    let mut encoder = backend.begin_encoder(QueueId::new(0)).unwrap();
    backend
        .transition_texture(
            &mut encoder,
            &target,
            TextureRange::Whole,
            ResourceAccessState::ColorAttachmentWrite,
            ResourceAccessState::ColorAttachmentReadWrite,
        )
        .unwrap();
    backend
        .transition_texture(
            &mut encoder,
            &depth_target,
            TextureRange::Whole,
            ResourceAccessState::DepthStencilWrite,
            ResourceAccessState::DepthStencilReadWrite,
        )
        .unwrap();
    backend
        .begin_raster(
            &mut encoder,
            &RasterPassDescriptor {
                label: "depth-load-store",
                colors: std::slice::from_ref(&loaded_color),
                depth_stencil: Some(loaded_depth),
            },
        )
        .unwrap();
    backend
        .set_raster_pipeline(&mut encoder, &triangle)
        .unwrap();
    backend.draw(&mut encoder, 0..3, 0..1).unwrap();
    backend.end_raster(&mut encoder).unwrap();
    let command = backend.finish_encoder(encoder).unwrap();
    let completion = backend
        .submit(QueueId::new(0), command, Vec::new())
        .unwrap();
    backend
        .wait(&completion, std::time::Duration::from_secs(10))
        .unwrap();
    assert_eq!(
        backend.completion_status(&completion),
        CompletionStatus::Complete
    );
    let diagnostics = crate::imp::validation_diagnostics(&device.inner);
    assert!(diagnostics.is_empty(), "{diagnostics:#?}");

    let no_attachment_usage = device
        .create_texture(TextureDescriptor {
            texture: target_descriptor,
            usage: TextureUsage::from_kinds([TextureUsageKind::CopyDestination]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let usage_color = raster_negative_color(&no_attachment_usage);
    let mut encoder = backend.begin_encoder(QueueId::new(0)).unwrap();
    assert_eq!(
        backend.begin_raster(
            &mut encoder,
            &RasterPassDescriptor {
                label: "usage",
                colors: std::slice::from_ref(&usage_color),
                depth_stencil: None,
            }
        ),
        Err(mismatch.clone())
    );
    drop(encoder);

    // Nested/mismatched pass state and finish-with-open-pass are rejected.
    let mut encoder = backend.begin_encoder(QueueId::new(0)).unwrap();
    backend
        .transition_texture(
            &mut encoder,
            &target,
            TextureRange::Whole,
            ResourceAccessState::Undefined,
            ResourceAccessState::ColorAttachmentReadWrite,
        )
        .unwrap();
    backend.begin_raster(&mut encoder, &descriptor).unwrap();
    assert_eq!(
        backend.transition_texture(
            &mut encoder,
            &target,
            TextureRange::Whole,
            ResourceAccessState::ColorAttachmentWrite,
            ResourceAccessState::CopySource,
        ),
        Err(NativeExecutionError::CopyStateMismatch)
    );
    assert_eq!(
        backend.begin_copy(&mut encoder, "nested-copy"),
        Err(NativeExecutionError::CopyStateMismatch)
    );
    assert_eq!(
        backend.copy_buffer(
            &mut encoder,
            &compute_buffer,
            &compute_buffer,
            BufferCopyRegion {
                source_offset: 0,
                destination_offset: 0,
                size: 4,
            },
        ),
        Err(NativeExecutionError::CopyStateMismatch)
    );
    assert_eq!(
        backend.begin_raster(&mut encoder, &descriptor),
        Err(mismatch.clone())
    );
    assert_eq!(
        backend.begin_compute(&mut encoder, "nested"),
        Err(mismatch.clone())
    );
    assert_eq!(
        backend.end_compute(&mut encoder),
        Err(NativeExecutionError::ComputeBindingMismatch)
    );
    assert!(matches!(
        backend.finish_encoder(encoder),
        Err(NativeExecutionError::RasterStateMismatch)
    ));
    let mut encoder = backend.begin_encoder(QueueId::new(0)).unwrap();
    backend.begin_compute(&mut encoder, "compute").unwrap();
    assert_eq!(backend.end_raster(&mut encoder), Err(mismatch.clone()));
    assert_eq!(
        backend.begin_raster(&mut encoder, &descriptor),
        Err(mismatch.clone())
    );
    assert!(matches!(
        backend.finish_encoder(encoder),
        Err(NativeExecutionError::RasterStateMismatch)
    ));

    // Draw validation happens before native draw calls and never submits this encoder.
    let mut encoder = backend.begin_encoder(QueueId::new(0)).unwrap();
    backend
        .transition_texture(
            &mut encoder,
            &target,
            TextureRange::Whole,
            ResourceAccessState::Undefined,
            ResourceAccessState::ColorAttachmentReadWrite,
        )
        .unwrap();
    backend.begin_raster(&mut encoder, &descriptor).unwrap();
    assert_eq!(
        backend.draw(&mut encoder, 0..3, 0..1),
        Err(mismatch.clone())
    );
    assert_eq!(
        backend.set_viewport(
            &mut encoder,
            Viewport {
                x: 7.0,
                y: 0.0,
                width: 2.0,
                height: 1.0,
                min_depth: 0.0,
                max_depth: 1.0
            }
        ),
        Err(mismatch.clone())
    );
    assert_eq!(
        backend.set_scissor(
            &mut encoder,
            ScissorRect {
                x: 7,
                y: 0,
                width: 2,
                height: 1
            }
        ),
        Err(mismatch.clone())
    );
    backend
        .set_raster_pipeline(&mut encoder, &pipeline)
        .unwrap();
    assert_eq!(
        backend.draw_indexed(&mut encoder, 0..3, 0, 0..1),
        Err(mismatch.clone())
    );
    let wrong_usage = device
        .create_buffer(BufferDescriptor {
            buffer: BufferDesc { size: 64 },
            usage: BufferUsage::from_kinds([BufferUsageKind::StorageRead]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    assert_eq!(
        backend.set_vertex_buffer(&mut encoder, 0, &wrong_usage, 0),
        Err(mismatch.clone())
    );
    assert_eq!(
        backend.set_index_buffer(&mut encoder, &wrong_usage, 0, IndexFormat::Uint16),
        Err(mismatch.clone())
    );
    let vertex = device
        .create_buffer(BufferDescriptor {
            buffer: BufferDesc { size: 64 },
            usage: BufferUsage::from_kinds([BufferUsageKind::Vertex]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    let index = device
        .create_buffer(BufferDescriptor {
            buffer: BufferDesc { size: 6 },
            usage: BufferUsage::from_kinds([BufferUsageKind::Index]),
            memory: MemoryPolicy::DeviceOnly,
        })
        .unwrap();
    backend
        .set_vertex_buffer(&mut encoder, 0, &vertex, 0)
        .unwrap();
    assert_eq!(
        backend.set_index_buffer(&mut encoder, &index, 0, IndexFormat::Uint32),
        Err(mismatch.clone())
    );
    backend
        .set_index_buffer(&mut encoder, &index, 0, IndexFormat::Uint16)
        .unwrap();
    assert_eq!(
        backend.draw_indexed(&mut encoder, 0..4, 0, 0..1),
        Err(mismatch)
    );
    backend.end_raster(&mut encoder).unwrap();
    drop(encoder);

    run_raster_vertex_recipe_negative_paths(
        &device,
        &mut backend,
        &descriptor,
        &target,
        &camera_pipeline,
        &camera_bindings,
        &textured_pipeline,
        &textured_bindings,
        &uniform,
        &other_camera_bindings,
        &sampled_texture,
        &foreign_device,
    );
}
