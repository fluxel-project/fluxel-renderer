//! Portable execution contract tests.

use crate::ComputeCreateError;
use crate::execution::helpers::{require_device, validate_buffer_copy, validate_texture_copy};
use crate::execution::raster::operations::{
    raster_recipe_allows_first_index, raster_recipe_allows_index_offset,
    raster_recipe_allows_vertex_offset,
};
use crate::execution::raster_capabilities_from_limit_and_filterability;
use crate::execution::*;
use fluxel_rendergraph::{Extent3d, TextureDimension};

#[test]
fn binding_creation_errors_preserve_contract_vs_backend_failure() {
    assert_eq!(
        compute_binding_error_kind(&ComputeCreateError::InvalidBindingRange),
        fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe
    );
    assert_eq!(
        compute_binding_error_kind(&ComputeCreateError::NativeFailure("descriptor".into())),
        fluxel_rendergraph::RecordingErrorKind::BackendObjectCreation
    );
}

#[test]
fn copy_profile_exposes_only_one_copy_queue() {
    let capabilities = CopyBackend::portable_capabilities();
    assert_eq!(capabilities.queues.len(), 1);
    let queue = &capabilities.queues[0];
    assert!(queue.capabilities.copy);
    assert!(!queue.capabilities.compute);
    assert!(!queue.capabilities.raster);
}

#[test]
fn compute_profile_adds_compute_without_mutating_copy_profile() {
    let copy = CopyBackend::portable_capabilities();
    let compute = ComputeBackend::portable_capabilities();
    assert!(!copy.queues[0].capabilities.compute);
    assert!(compute.queues[0].capabilities.compute);
    assert!(compute.queues[0].capabilities.copy);
    assert_eq!(
        compute.limits.max_compute_workgroups_per_dimension,
        [65_535; 3]
    );
}

#[test]
fn native_profiles_support_completion_gated_cross_frame_transient_pooling() {
    for capabilities in [
        CopyBackend::portable_capabilities(),
        ComputeBackend::portable_capabilities(),
        RasterBackend::portable_capabilities(),
    ] {
        let transients = capabilities.transient_resources;
        assert!(
            transients.cross_frame_object_pooling,
            "native DeviceOnly allocations keep their lease through completion"
        );
        assert!(
            !transients.in_frame_object_reuse,
            "fixed native profiles do not record alias/reuse barriers"
        );
        assert!(
            !transients.aliased_memory,
            "fixed native profiles do not expose heap aliasing"
        );
    }
}

#[test]
fn raster_profile_is_single_queue_raster_compute_copy_rgba8() {
    let capabilities = RasterBackend::portable_capabilities();
    assert_eq!(capabilities.queues.len(), 1);
    assert!(capabilities.queues[0].capabilities.raster);
    assert!(capabilities.queues[0].capabilities.compute);
    assert!(capabilities.queues[0].capabilities.copy);
    let rgba8 = capabilities
        .texture_formats
        .iter()
        .find(|facts| facts.format == TextureFormat::Rgba8Unorm)
        .unwrap();
    assert!(rgba8.color_attachment && rgba8.sampled && rgba8.copy_source);
    assert!(!rgba8.storage_read && !rgba8.storage_write);
    assert_eq!(rgba8.attachment_sample_counts, vec![1]);
    let depth = capabilities
        .texture_formats
        .iter()
        .find(|facts| facts.format == TextureFormat::Depth32Float)
        .expect("Depth32Float fixed raster format");
    assert!(depth.depth_stencil_attachment);
    assert_eq!(depth.attachment_sample_counts, vec![1]);
    assert!(!depth.sampled && !depth.storage_read && !depth.storage_write);
    assert!(!depth.copy_source && !depth.copy_destination);
}

#[test]
fn raster_profile_propagates_actual_rgba8_filterability() {
    let filterable = raster_capabilities_from_limit_and_filterability([1, 1, 1], true, true);
    let not_filterable = raster_capabilities_from_limit_and_filterability([1, 1, 1], false, false);
    let filterable_fact = filterable
        .texture_formats
        .iter()
        .find(|facts| facts.format == TextureFormat::Rgba8Unorm)
        .expect("RGBA8 fixed raster format");
    let not_filterable_fact = not_filterable
        .texture_formats
        .iter()
        .find(|facts| facts.format == TextureFormat::Rgba8Unorm)
        .expect("RGBA8 fixed raster format");
    assert!(filterable_fact.filterable);
    assert!(!not_filterable_fact.filterable);
    // Existing abstract fixtures remain able to compile the old profile.
    assert!(
        RasterBackend::portable_capabilities()
            .texture_formats
            .iter()
            .find(|facts| facts.format == TextureFormat::Rgba8Unorm)
            .expect("RGBA8 fixed raster format")
            .filterable
    );
}

#[test]
fn raster_profile_keeps_unorm_and_srgb_filterability_independent() {
    let capabilities = raster_capabilities_from_limit_and_filterability([1, 1, 1], true, false);
    let unorm = capabilities
        .texture_formats
        .iter()
        .find(|facts| facts.format == TextureFormat::Rgba8Unorm)
        .expect("UNORM fixed raster format");
    let srgb = capabilities
        .texture_formats
        .iter()
        .find(|facts| facts.format == TextureFormat::Rgba8UnormSrgb)
        .expect("sRGB fixed raster format");
    assert!(unorm.filterable);
    assert!(!srgb.filterable);
    assert!(!srgb.color_attachment);
}

#[test]
fn both_explicit_uv_kernels_share_the_epoch_family() {
    assert!(RasterBackend::is_uv_kernel(
        crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUv
    ));
    assert!(RasterBackend::is_uv_kernel(
        crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
    ));
    assert!(RasterBackend::is_uv_kernel(
        crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
    ));
    assert!(!RasterBackend::is_uv_kernel(crate::RasterKernel::Triangle));
}

#[test]
fn f32x3_u32_recipe_rejects_nonzero_bases_before_native_recording() {
    use crate::RasterKernel::{IndexedPositionColor, IndexedPositionFloat32x3};

    // These predicates are evaluated by RasterBackend before it calls the
    // private HAL boundary, so rejected values cannot form a submission.
    assert!(raster_recipe_allows_vertex_offset(
        IndexedPositionFloat32x3,
        0
    ));
    assert!(!raster_recipe_allows_vertex_offset(
        IndexedPositionFloat32x3,
        4
    ));
    assert!(raster_recipe_allows_index_offset(
        IndexedPositionFloat32x3,
        0
    ));
    assert!(!raster_recipe_allows_index_offset(
        IndexedPositionFloat32x3,
        4
    ));
    assert!(raster_recipe_allows_first_index(
        Some(IndexedPositionFloat32x3),
        0
    ));
    assert!(!raster_recipe_allows_first_index(
        Some(IndexedPositionFloat32x3),
        1
    ));

    // R02 is deliberately unchanged: its fixed Uint16 recipe continues
    // to permit aligned buffer offsets and an indexed subrange.
    assert!(raster_recipe_allows_vertex_offset(IndexedPositionColor, 4));
    assert!(raster_recipe_allows_index_offset(IndexedPositionColor, 2));
    assert!(raster_recipe_allows_first_index(
        Some(IndexedPositionColor),
        1
    ));
}

#[test]
fn vertex_color_recipe_rejects_nonzero_bases_before_native_recording() {
    use crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialVertexColor;
    assert!(raster_recipe_allows_vertex_offset(
        IndexedPositionFloat32x3CameraMaterialVertexColor,
        0
    ));
    assert!(!raster_recipe_allows_vertex_offset(
        IndexedPositionFloat32x3CameraMaterialVertexColor,
        4
    ));
    assert!(raster_recipe_allows_index_offset(
        IndexedPositionFloat32x3CameraMaterialVertexColor,
        0
    ));
    assert!(!raster_recipe_allows_index_offset(
        IndexedPositionFloat32x3CameraMaterialVertexColor,
        4
    ));
    assert!(raster_recipe_allows_first_index(
        Some(IndexedPositionFloat32x3CameraMaterialVertexColor),
        0
    ));
    assert!(!raster_recipe_allows_first_index(
        Some(IndexedPositionFloat32x3CameraMaterialVertexColor),
        1
    ));
}

#[test]
fn vertex_color_slot_one_accepts_two_rgba8_vertices_at_native_range_gate() {
    use crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialVertexColor;
    // Three indices may repeat either vertex; the closed geometry contract
    // therefore permits a two-vertex, eight-byte color stream.
    assert_eq!(
        crate::imp::raster_vertex_minimum_size(
            IndexedPositionFloat32x3CameraMaterialVertexColor,
            1
        ),
        4
    );
    assert!(
        8 >= crate::imp::raster_vertex_minimum_size(
            IndexedPositionFloat32x3CameraMaterialVertexColor,
            1
        )
    );
    assert_eq!(
        crate::imp::raster_vertex_minimum_size(
            crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUv,
            1
        ),
        8
    );
}

#[test]
fn fixed_compute_artifacts_have_stable_entries_workgroups_and_source_hash() {
    use crate::ComputeKernel;
    for (kernel, entry) in [
        (ComputeKernel::WrappingAdd, "wrapping_add"),
        (ComputeKernel::WrappingMultiply, "wrapping_multiply"),
        (ComputeKernel::TextureStoreRgba8, "store_rgba8"),
        (ComputeKernel::TextureLoadRgba8, "load_rgba8"),
    ] {
        assert_eq!(kernel.entry_point(), entry);
        assert!(matches!(kernel.workgroup_size(), [64, 1, 1] | [8, 8, 1]));
        assert_ne!(kernel.source_hash(), 0);
        assert_eq!(
            kernel.source_hash(),
            crate::resource::fnv1a64(kernel.wgsl_source().as_bytes())
        );
    }
}

#[test]
fn dispatch_dimensions_fail_closed_before_native_recording() {
    let maximum = [4, 8, 16];
    assert!(valid_compute_dispatch([4, 8, 16], maximum));
    assert!(!valid_compute_dispatch([0, 1, 1], maximum));
    assert!(!valid_compute_dispatch([5, 1, 1], maximum));
    assert!(!valid_compute_dispatch([1, 9, 1], maximum));
    assert!(!valid_compute_dispatch([1, 1, 17], maximum));
}

#[test]
fn uv_vertex_failures_remain_structured_and_slot_specific() {
    assert_eq!(
        NativeExecutionError::RasterVertexRoleMismatch { slot: 0 },
        NativeExecutionError::RasterVertexRoleMismatch { slot: 0 }
    );
    assert_ne!(
        NativeExecutionError::RasterVertexRoleMismatch { slot: 0 },
        NativeExecutionError::RasterVertexRoleMismatch { slot: 1 }
    );
    assert_ne!(
        NativeExecutionError::RasterVertexSlotMissing { slot: 0 },
        NativeExecutionError::RasterEpochMismatch
    );
}

#[test]
fn encoder_and_command_buffer_identity_checks_fail_closed() {
    let first = fluxel_rendergraph::DeviceIdentity::new(1);
    let second = fluxel_rendergraph::DeviceIdentity::new(2);
    assert_eq!(
        require_device(first, second, NativeExecutionError::ForeignEncoder),
        Err(NativeExecutionError::ForeignEncoder)
    );
    assert_eq!(
        require_device(first, second, NativeExecutionError::ForeignCommandBuffer),
        Err(NativeExecutionError::ForeignCommandBuffer)
    );
    assert_eq!(
        require_device(first, first, NativeExecutionError::ForeignEncoder),
        Ok(())
    );
}

#[test]
fn native_buffer_validation_rejects_zero_misaligned_overflow_and_out_of_bounds() {
    let descriptor = BufferDesc { size: 32 };
    for region in [
        BufferCopyRegion {
            source_offset: 0,
            destination_offset: 0,
            size: 0,
        },
        BufferCopyRegion {
            source_offset: 2,
            destination_offset: 0,
            size: 4,
        },
        BufferCopyRegion {
            source_offset: 0,
            destination_offset: 2,
            size: 4,
        },
        BufferCopyRegion {
            source_offset: 0,
            destination_offset: 0,
            size: 6,
        },
        BufferCopyRegion {
            source_offset: u64::MAX,
            destination_offset: 0,
            size: 2,
        },
        BufferCopyRegion {
            source_offset: 16,
            destination_offset: 17,
            size: 16,
        },
    ] {
        assert!(matches!(
            validate_buffer_copy(descriptor, descriptor, region),
            Err(NativeExecutionError::InvalidTransfer(_))
        ));
    }
}

#[test]
fn native_texture_validation_rejects_format_and_extent_mismatches() {
    let descriptor = TextureDesc {
        dimension: TextureDimension::D2,
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
    let region = TextureCopyRegion {
        source_origin: [0, 0, 0],
        destination_origin: [4, 0, 0],
        extent: [5, 1, 1],
        source_mip_level: 0,
        destination_mip_level: 0,
    };
    assert!(matches!(
        validate_texture_copy(descriptor, descriptor, region),
        Err(NativeExecutionError::InvalidTransfer(_))
    ));
    assert!(matches!(
        validate_texture_copy(
            descriptor,
            TextureDesc {
                format: TextureFormat::Bgra8Unorm,
                ..descriptor
            },
            TextureCopyRegion {
                destination_origin: [0, 0, 0],
                extent: [1, 1, 1],
                ..region
            }
        ),
        Err(NativeExecutionError::InvalidTransfer(_))
    ));
}
