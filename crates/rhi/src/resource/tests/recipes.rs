//! Resource recipes contract tests.

use super::*;

pub(super) fn portable_compute_identity_shares_module_but_distinguishes_entries() {
    let add = ComputeKernel::WrappingAdd.portable_identity();
    let multiply = ComputeKernel::WrappingMultiply.portable_identity();
    assert_eq!(add.module_source_hash, multiply.module_source_hash);
    assert_ne!(add.entry_point, multiply.entry_point);
    assert_eq!(add.workgroup_size, [64, 1, 1]);
    assert_eq!(add.binding_recipe_version, 1);
}

pub(super) fn texture_pack_is_a_distinct_closed_compute_recipe() {
    let pack = ComputeKernel::TexturePackRgba8.portable_identity();
    let add = ComputeKernel::WrappingAdd.portable_identity();
    assert_ne!(pack.module_source_hash, add.module_source_hash);
    assert_eq!(pack.entry_point, "pack_rgba8");
    assert_eq!(pack.workgroup_size, [8, 8, 1]);
    assert_eq!(pack.binding_recipe_version, 2);
}

pub(super) fn storage_texture_recipes_are_distinct_and_closed() {
    let store = ComputeKernel::TextureStoreRgba8.portable_identity();
    let load = ComputeKernel::TextureLoadRgba8.portable_identity();
    assert_eq!(store.entry_point, "store_rgba8");
    assert_eq!(load.entry_point, "load_rgba8");
    assert_eq!(store.workgroup_size, [8, 8, 1]);
    assert_eq!(load.workgroup_size, [8, 8, 1]);
    assert_eq!(store.binding_recipe_version, 3);
    assert_eq!(load.binding_recipe_version, 4);
    assert_ne!(store.module_source_hash, load.module_source_hash);
}

pub(super) fn raster_identities_are_portable_but_recipes_remain_distinct() {
    let triangle = RasterKernel::Triangle.portable_identity();
    let indexed = RasterKernel::IndexedPositionColor.portable_identity();
    let indexed_f32x3 = RasterKernel::IndexedPositionFloat32x3.portable_identity();
    assert_eq!(triangle.target_format, TextureFormat::Rgba8Unorm);
    assert_eq!(triangle.vertex_stride, 0);
    assert_eq!(triangle.vertex_layout, RasterVertexLayout::None);
    assert_eq!(triangle.index_format, None);
    assert_eq!(indexed.vertex_stride, 12);
    assert_eq!(
        indexed.vertex_layout,
        RasterVertexLayout::PositionFloat32x2ColorUnorm8x4
    );
    assert_ne!(triangle.module_source_hash, indexed.module_source_hash);
    assert_eq!(triangle.fragment_entry_point, indexed.fragment_entry_point);
    assert_eq!(indexed_f32x3.vertex_stride, 12);
    assert_eq!(indexed_f32x3.index_format, Some(IndexFormat::Uint32));
    assert_eq!(
        indexed_f32x3.vertex_layout,
        RasterVertexLayout::PositionFloat32x3
    );
    assert_ne!(indexed.module_source_hash, indexed_f32x3.module_source_hash);
    // Pre-0.2.2 artifacts retain the no-binding identity facts exactly.
    assert_eq!(triangle.binding_count, 0);
    assert_eq!(triangle.uniform_binding_size, 0);
    assert_eq!(triangle.binding_recipe_version, 0);
    assert_eq!(indexed_f32x3.binding_count, 0);
    assert_eq!(
        format!("{indexed_f32x3:?}"),
        "RasterArtifactIdentity { module_source_hash: 16318329126220124131, vertex_entry_point: \"position_f32x3_vertex\", fragment_entry_point: \"color_fragment\", target_format: Rgba8Unorm, vertex_layout: PositionFloat32x3, vertex_stride: 12, index_format: Some(Uint32), recipe_version: 1 }"
    );
    let camera = RasterKernel::IndexedPositionFloat32x3CameraMaterial.portable_identity();
    assert_eq!(camera.binding_count, 1);
    assert_eq!(camera.uniform_binding_size, 80);
    assert_eq!(camera.binding_recipe_version, 1);
    assert_eq!(
        format!("{camera:?}"),
        "RasterArtifactIdentity { module_source_hash: 8563197035213614468, vertex_entry_point: \"camera_material_vertex\", fragment_entry_point: \"color_fragment\", target_format: Rgba8Unorm, vertex_layout: PositionFloat32x3, vertex_stride: 12, index_format: Some(Uint32), binding_count: 1, uniform_binding_size: 80, binding_recipe_version: 1, recipe_version: 1 }"
    );
    assert_ne!(camera.module_source_hash, indexed_f32x3.module_source_hash);
}

pub(super) fn camera_material_uniform_contract_is_closed_and_fail_closed() {
    let uniform =
        BufferUsage::from_kinds([BufferUsageKind::Uniform, BufferUsageKind::CopyDestination]);
    assert_eq!(
        validate_raster_uniform_contract(
            RasterKernel::IndexedPositionFloat32x3CameraMaterial,
            80,
            uniform,
        ),
        Ok(())
    );
    assert_eq!(
        validate_raster_uniform_contract(RasterKernel::IndexedPositionFloat32x3, 80, uniform),
        Err(RasterCreateError::BindingRecipeMismatch)
    );
    assert_eq!(
        validate_raster_uniform_contract(
            RasterKernel::IndexedPositionFloat32x3CameraMaterial,
            64,
            uniform,
        ),
        Err(RasterCreateError::InvalidBindingRange)
    );
    assert_eq!(
        validate_raster_uniform_contract(
            RasterKernel::IndexedPositionFloat32x3CameraMaterial,
            80,
            BufferUsage::from_kinds([BufferUsageKind::CopyDestination]),
        ),
        Err(RasterCreateError::UniformUsageRequired)
    );
    assert_eq!(
        validate_raster_uniform_contract(
            RasterKernel::IndexedPositionFloat32x3CameraMaterialNormalLambert,
            80,
            uniform,
        ),
        Ok(())
    );
}

pub(super) fn normal_lambert_identity_is_closed_and_additive() {
    let legacy = format!(
        "{:?}",
        RasterKernel::IndexedPositionFloat32x3.portable_identity()
    );
    assert!(!legacy.contains("lighting_recipe_version"));
    let identity =
        RasterKernel::IndexedPositionFloat32x3CameraMaterialNormalLambert.portable_identity();
    assert_eq!(
        identity.vertex_layout,
        RasterVertexLayout::PositionFloat32x3AndNormalFloat32x3
    );
    assert_eq!(identity.binding_count, 1);
    assert_eq!(identity.uniform_binding_size, 80);
    assert_eq!(identity.lighting_recipe_version, 1);
    assert_eq!(
        identity.normal_interpolation,
        Some(RasterNormalInterpolation::PerspectiveCenter)
    );
    assert_eq!(
        identity.normal_normalization,
        Some(RasterNormalNormalization::ExactPositiveUnitAfterInterpolationZeroLambert)
    );
    assert_eq!(identity.lighting_space, Some(RasterLightingSpace::Object));
    assert_eq!(
        identity.fixed_light_direction,
        Some(RasterFixedLightDirection::PositiveZ)
    );
    assert_eq!(
        identity.lighting_model,
        Some(RasterLightingModel::LambertRgbAlphaPassthrough)
    );
    assert_eq!(identity.position_vertex_slot, Some(0));
    assert_eq!(identity.position_shader_location, Some(0));
    assert_eq!(identity.normal_vertex_slot, Some(1));
    assert_eq!(identity.normal_vertex_stride, Some(12));
    assert_eq!(identity.normal_shader_location, Some(1));
    assert!(format!("{identity:?}").contains("lighting_recipe_version"));
    let source = RasterKernel::IndexedPositionFloat32x3CameraMaterialNormalLambert.wgsl_source();
    assert!(source.contains("len2 > 0.0"));
    assert!(source.contains("inverseSqrt(len2)"));
    assert!(source.contains("frame.base_color.a"));
}

pub(super) fn vertex_color_identity_is_closed_and_additive() {
    let legacy = format!(
        "{:?}",
        RasterKernel::IndexedPositionFloat32x3CameraMaterial.portable_identity()
    );
    assert!(!legacy.contains("vertex_color_recipe_version"));
    let identity =
        RasterKernel::IndexedPositionFloat32x3CameraMaterialVertexColor.portable_identity();
    assert_eq!(
        identity.vertex_layout,
        RasterVertexLayout::PositionFloat32x3AndColorUnorm8x4
    );
    assert_eq!(identity.index_format, Some(IndexFormat::Uint32));
    assert_eq!(identity.binding_count, 1);
    assert_eq!(identity.uniform_binding_size, 80);
    assert_eq!(identity.vertex_color_recipe_version, 1);
    assert_eq!(identity.vertex_color_position_slot, Some(0));
    assert_eq!(identity.vertex_color_position_shader_location, Some(0));
    assert_eq!(identity.vertex_color_slot, Some(1));
    assert_eq!(identity.vertex_color_stride, Some(4));
    assert_eq!(identity.vertex_color_shader_location, Some(1));
    let source = RasterKernel::IndexedPositionFloat32x3CameraMaterialVertexColor.wgsl_source();
    assert!(source.contains("@interpolate(perspective, center) color"));
    assert!(source.contains("input.color * frame.base_color"));
    let uniform = BufferUsage::from_kinds([BufferUsageKind::Uniform]);
    assert_eq!(
        validate_raster_uniform_contract(
            RasterKernel::IndexedPositionFloat32x3CameraMaterialVertexColor,
            80,
            uniform
        ),
        Ok(())
    );
}

pub(super) fn texture_pack_accepts_only_full_single_sampled_rgba8_images() {
    let image = TextureDesc {
        dimension: TextureDimension::D2,
        extent: Extent3d {
            width: 7,
            height: 3,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
    };
    let sampled = TextureUsage::empty().with(TextureUsageKind::Sampled);
    assert_eq!(validate_texture_pack_texture_desc(image, sampled), Ok(()));
    assert_eq!(texture_pack_required_size(image), Ok(84));
    assert_eq!(
        validate_texture_pack_texture_desc(
            TextureDesc {
                mip_levels: 2,
                ..image
            },
            sampled
        ),
        Err(ComputeCreateError::BindingRecipeMismatch)
    );
    assert_eq!(
        validate_texture_pack_texture_desc(image, TextureUsage::empty()),
        Err(ComputeCreateError::BindingRecipeMismatch)
    );
}

pub(super) fn immutable_texture_upload_is_closed_and_requires_exact_tight_rgba8() {
    let image = TextureDesc {
        dimension: TextureDimension::D2,
        extent: Extent3d {
            width: 3,
            height: 2,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
    };
    let descriptor = TextureDescriptor {
        texture: image,
        usage: TextureUsage::from_kinds([
            TextureUsageKind::CopyDestination,
            TextureUsageKind::Sampled,
        ]),
        memory: MemoryPolicy::DeviceOnly,
    };
    assert_eq!(
        validate_immutable_texture_upload_descriptor(descriptor, &[0; 24]),
        Ok(())
    );
    assert_eq!(
        validate_immutable_texture_upload_descriptor(descriptor, &[0; 23]),
        Err(TextureUploadError::InvalidRequest(
            InvalidTextureUploadReason::DataLengthMismatch
        ))
    );
    assert_eq!(
        validate_immutable_texture_upload_descriptor(
            TextureDescriptor {
                usage: TextureUsage::empty().with(TextureUsageKind::Sampled),
                ..descriptor
            },
            &[0; 24]
        ),
        Err(TextureUploadError::InvalidRequest(
            InvalidTextureUploadReason::CopyDestinationUsageRequired
        ))
    );
    assert_eq!(
        validate_immutable_texture_upload_descriptor(
            TextureDescriptor {
                usage: descriptor.usage.with(TextureUsageKind::StorageWrite),
                ..descriptor
            },
            &[0; 24]
        ),
        Err(TextureUploadError::InvalidRequest(
            InvalidTextureUploadReason::UnexpectedUsage
        ))
    );
    assert_eq!(
        validate_immutable_texture_upload_descriptor(
            TextureDescriptor {
                usage: descriptor.usage.with(TextureUsageKind::ColorAttachment),
                ..descriptor
            },
            &[0; 24]
        ),
        Err(TextureUploadError::InvalidRequest(
            InvalidTextureUploadReason::UnexpectedUsage
        ))
    );
    // Readback is a test-only widening, not a production upload feature.
    assert_eq!(
        validate_immutable_texture_upload_descriptor(
            TextureDescriptor {
                usage: descriptor.usage.with(TextureUsageKind::CopySource),
                ..descriptor
            },
            &[0; 24]
        ),
        Ok(())
    );
    assert_eq!(
        validate_immutable_texture_upload_descriptor(
            TextureDescriptor {
                texture: TextureDesc {
                    mip_levels: 2,
                    ..image
                },
                ..descriptor
            },
            &[0; 24]
        ),
        Err(TextureUploadError::InvalidRequest(
            InvalidTextureUploadReason::MipLevelsUnsupported
        ))
    );
}

pub(super) fn textured_raster_identity_is_explicit_without_mutating_legacy_debug() {
    let legacy = format!(
        "{:?}",
        RasterKernel::IndexedPositionFloat32x3.portable_identity()
    );
    assert!(!legacy.contains("texture_binding"));
    let identity = RasterKernel::IndexedPositionFloat32x3CameraMaterialTexture.portable_identity();
    assert_eq!(identity.binding_count, 2);
    assert_eq!(identity.uniform_binding_size, 80);
    assert_eq!(identity.uniform_binding, Some(0));
    assert_eq!(
        identity.uniform_visibility,
        Some(RasterBindingVisibility::VertexFragment)
    );
    assert_eq!(identity.texture_binding, Some(1));
    assert_eq!(identity.texture_dimension, Some(TextureDimension::D2));
    assert_eq!(identity.texture_format, Some(TextureFormat::Rgba8Unorm));
    assert_eq!(
        identity.texture_visibility,
        Some(RasterBindingVisibility::Fragment)
    );
    assert_eq!(
        identity.texture_sample_type,
        Some(RasterTextureSampleType::Float)
    );
    assert_eq!(identity.texture_mip_level, Some(0));
    assert_eq!(identity.texture_mapping_recipe_version, 1);
    assert_eq!(
        format!("{identity:?}"),
        "RasterArtifactIdentity { module_source_hash: 8313758755579089945, vertex_entry_point: \"camera_material_texture_vertex\", fragment_entry_point: \"texture_color_fragment\", target_format: Rgba8Unorm, vertex_layout: PositionFloat32x3, vertex_stride: 12, index_format: Some(Uint32), binding_count: 2, uniform_binding_size: 80, binding_recipe_version: 1, texture_binding_recipe_version: 1, texture_binding: Some(1), texture_dimension: Some(D2), texture_format: Some(Rgba8Unorm), uniform_binding: Some(0), uniform_visibility: Some(VertexFragment), texture_visibility: Some(Fragment), texture_sample_type: Some(Float), texture_mip_level: Some(0), texture_mapping_recipe_version: 1, recipe_version: 1 }"
    );
    let uv = RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUv.portable_identity();
    assert_eq!(uv.texture_mapping_recipe_version, 2);
    assert_eq!(
        uv.vertex_layout,
        RasterVertexLayout::PositionFloat32x3AndTextureCoordinateFloat32x2
    );
    assert_eq!(
        uv.texture_coordinate_vertex_layout,
        Some(RasterVertexLayout::PositionFloat32x3AndTextureCoordinateFloat32x2)
    );
    assert_eq!(uv.texture_coordinate_vertex_stride, Some(8));
    assert_eq!(uv.texture_coordinate_shader_location, Some(1));
    assert_eq!(
        format!("{uv:?}"),
        "RasterArtifactIdentity { module_source_hash: 13201511842499328436, vertex_entry_point: \"camera_material_texture_uv_vertex\", fragment_entry_point: \"texture_color_fragment\", target_format: Rgba8Unorm, vertex_layout: PositionFloat32x3AndTextureCoordinateFloat32x2, vertex_stride: 12, index_format: Some(Uint32), binding_count: 2, uniform_binding_size: 80, binding_recipe_version: 1, texture_binding_recipe_version: 1, texture_binding: Some(1), texture_dimension: Some(D2), texture_format: Some(Rgba8Unorm), uniform_binding: Some(0), uniform_visibility: Some(VertexFragment), texture_visibility: Some(Fragment), texture_sample_type: Some(Float), texture_mip_level: Some(0), texture_mapping_recipe_version: 2, texture_coordinate_vertex_layout: Some(PositionFloat32x3AndTextureCoordinateFloat32x2), texture_coordinate_vertex_stride: Some(8), texture_coordinate_shader_location: Some(1), recipe_version: 1 }"
    );
    // Conditional UV fields must not perturb the exact legacy artifact
    // representation consumed by the earlier conformance evidence.
    assert!(!format!("{identity:?}").contains("texture_coordinate_vertex"));
}

pub(super) fn linear_clamp_identity_is_additive_and_records_the_closed_sampler() {
    let old_uv = RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUv.portable_identity();
    assert!(!format!("{old_uv:?}").contains("sampler_binding"));

    let identity = RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
        .portable_identity();
    assert_eq!(identity.binding_count, 3);
    assert_eq!(identity.texture_binding, Some(1));
    assert_eq!(identity.sampler_binding, Some(2));
    assert_eq!(identity.sampler_binding_recipe_version, 1);
    assert_eq!(
        identity.sampler_binding_type,
        Some(RasterSamplerBindingType::Filtering)
    );
    assert_eq!(
        identity.sampler_min_filter,
        Some(RasterSamplerFilter::Linear)
    );
    assert_eq!(
        identity.sampler_mag_filter,
        Some(RasterSamplerFilter::Linear)
    );
    assert_eq!(
        identity.sampler_mipmap_filter,
        Some(RasterSamplerFilter::Nearest)
    );
    assert_eq!(
        identity.sampler_address_mode_u,
        Some(RasterSamplerAddressMode::ClampToEdge)
    );
    assert_eq!(
        identity.sampler_address_mode_v,
        Some(RasterSamplerAddressMode::ClampToEdge)
    );
    assert_eq!(
        identity.sampler_address_mode_w,
        Some(RasterSamplerAddressMode::ClampToEdge)
    );
    assert_eq!(identity.sampler_explicit_mip_level, Some(0));
    assert!(
        RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
            .wgsl_source()
            .contains("textureSampleLevel(tex, linear_clamp_sampler, input.uv, 0.0)")
    );
    assert!(
        !RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
            .wgsl_source()
            .contains("textureSample(tex,")
    );
}

pub(super) fn srgb_linear_clamp_identity_is_distinct_and_preserves_encoded_upload_contract() {
    let srgb = RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
        .portable_identity();
    let unorm = RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
        .portable_identity();
    assert_eq!(srgb.texture_format, Some(TextureFormat::Rgba8UnormSrgb));
    assert_eq!(srgb.binding_count, 3);
    assert_eq!(srgb.texture_mapping_recipe_version, 2);
    assert_eq!(
        srgb.texture_coordinate_vertex_layout,
        Some(RasterVertexLayout::PositionFloat32x3AndTextureCoordinateFloat32x2)
    );
    assert_eq!(srgb.texture_coordinate_vertex_stride, Some(8));
    assert_eq!(srgb.texture_coordinate_shader_location, Some(1));
    assert_eq!(srgb.sampler_binding, Some(2));
    assert_eq!(srgb.sampler_binding_recipe_version, 1);
    assert_eq!(
        srgb.sampler_binding_type,
        Some(RasterSamplerBindingType::Filtering)
    );
    assert_eq!(srgb.sampler_min_filter, Some(RasterSamplerFilter::Linear));
    assert_eq!(srgb.sampler_mag_filter, Some(RasterSamplerFilter::Linear));
    assert_eq!(
        srgb.sampler_mipmap_filter,
        Some(RasterSamplerFilter::Nearest)
    );
    assert_eq!(
        srgb.sampler_address_mode_u,
        Some(RasterSamplerAddressMode::ClampToEdge)
    );
    assert_eq!(
        srgb.sampler_address_mode_v,
        Some(RasterSamplerAddressMode::ClampToEdge)
    );
    assert_eq!(
        srgb.sampler_address_mode_w,
        Some(RasterSamplerAddressMode::ClampToEdge)
    );
    assert_eq!(srgb.sampler_explicit_mip_level, Some(0));
    assert_ne!(srgb.module_source_hash, unorm.module_source_hash);
    assert!(
        RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
            .wgsl_source()
            .contains("textureSampleLevel(tex, linear_clamp_sampler, input.uv, 0.0)")
    );

    let descriptor = TextureDescriptor {
        texture: TextureDesc {
            dimension: TextureDimension::D2,
            extent: Extent3d {
                width: 1,
                height: 1,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            sample_count: 1,
            format: TextureFormat::Rgba8UnormSrgb,
        },
        usage: TextureUsage::from_kinds([
            TextureUsageKind::CopyDestination,
            TextureUsageKind::Sampled,
        ]),
        memory: MemoryPolicy::DeviceOnly,
    };
    assert_eq!(
        validate_immutable_texture_upload_descriptor(descriptor, &[7, 8, 9, 10]),
        Ok(())
    );
    assert_eq!(
        validate_raster_srgb_texture_desc(descriptor.texture, descriptor.usage),
        Ok(())
    );
}

pub(super) fn not_filterable_error_is_structured() {
    assert_eq!(
        RasterCreateError::TextureFormatNotFilterable {
            format: TextureFormat::Rgba8Unorm,
        }
        .to_string(),
        "raster texture format is not filterable: Rgba8Unorm"
    );
}
