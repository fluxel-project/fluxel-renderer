//! Resource tests grouped by upload, validation, and closed-recipe contracts.

use super::validation::*;
use super::*;

#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
use crate::resource::artifact::map_compute_pipeline_create_error;
use fluxel_rendergraph::{Extent3d, TextureDimension};

mod recipes;
mod uploads;
mod validation;

#[test]
fn immutable_upload_requires_exact_device_only_copy_destination_bytes() {
    uploads::immutable_upload_requires_exact_device_only_copy_destination_bytes();
}

#[cfg(all(windows, feature = "dx12"))]
#[test]
#[ignore = "requires a Windows DX12 device with required validation"]
fn immutable_upload_dx12_failure_contract() {
    uploads::immutable_upload_dx12_failure_contract();
}

#[cfg(all(windows, feature = "vulkan"))]
#[test]
#[ignore = "requires a Windows Vulkan device with required validation"]
fn immutable_upload_vulkan_failure_contract() {
    uploads::immutable_upload_vulkan_failure_contract();
}

#[cfg(all(windows, feature = "dx12"))]
#[test]
#[ignore = "requires a Windows DX12 device with required validation"]
fn immutable_texture_upload_dx12_failure_contract() {
    uploads::immutable_texture_upload_dx12_failure_contract();
}

#[cfg(all(windows, feature = "vulkan"))]
#[test]
#[ignore = "requires a Windows Vulkan device with required validation"]
fn immutable_texture_upload_vulkan_failure_contract() {
    uploads::immutable_texture_upload_vulkan_failure_contract();
}

#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
#[test]
fn compute_pipeline_creation_stages_map_to_public_error_variants() {
    validation::compute_pipeline_creation_stages_map_to_public_error_variants();
}

#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
#[test]
fn raster_pipeline_creation_stages_map_to_public_error_variants() {
    validation::raster_pipeline_creation_stages_map_to_public_error_variants();
}

#[test]
fn rejects_zero_buffer_and_empty_usage() {
    validation::rejects_zero_buffer_and_empty_usage();
}

#[test]
fn rejects_surface_and_incompatible_texture_contracts() {
    validation::rejects_surface_and_incompatible_texture_contracts();
}

#[test]
fn fixed_workgroup_requires_each_native_dimension_and_total_invocations() {
    validation::fixed_workgroup_requires_each_native_dimension_and_total_invocations();
}

#[test]
fn storage_bindings_require_native_alignment_and_limited_in_bounds_ranges() {
    validation::storage_bindings_require_native_alignment_and_limited_in_bounds_ranges();
}

#[test]
fn portable_compute_identity_shares_module_but_distinguishes_entries() {
    recipes::portable_compute_identity_shares_module_but_distinguishes_entries();
}

#[test]
fn texture_pack_is_a_distinct_closed_compute_recipe() {
    recipes::texture_pack_is_a_distinct_closed_compute_recipe();
}

#[test]
fn storage_texture_recipes_are_distinct_and_closed() {
    recipes::storage_texture_recipes_are_distinct_and_closed();
}

#[test]
fn raster_identities_are_portable_but_recipes_remain_distinct() {
    recipes::raster_identities_are_portable_but_recipes_remain_distinct();
}

#[test]
fn camera_material_uniform_contract_is_closed_and_fail_closed() {
    recipes::camera_material_uniform_contract_is_closed_and_fail_closed();
}

#[test]
fn normal_lambert_identity_is_closed_and_additive() {
    recipes::normal_lambert_identity_is_closed_and_additive();
}

#[test]
fn vertex_color_identity_is_closed_and_additive() {
    recipes::vertex_color_identity_is_closed_and_additive();
}

#[test]
fn texture_pack_accepts_only_full_single_sampled_rgba8_images() {
    recipes::texture_pack_accepts_only_full_single_sampled_rgba8_images();
}

#[test]
fn immutable_texture_upload_is_closed_and_requires_exact_tight_rgba8() {
    recipes::immutable_texture_upload_is_closed_and_requires_exact_tight_rgba8();
}

#[test]
fn textured_raster_identity_is_explicit_without_mutating_legacy_debug() {
    recipes::textured_raster_identity_is_explicit_without_mutating_legacy_debug();
}

#[test]
fn linear_clamp_identity_is_additive_and_records_the_closed_sampler() {
    recipes::linear_clamp_identity_is_additive_and_records_the_closed_sampler();
}

#[test]
fn srgb_linear_clamp_identity_is_distinct_and_preserves_encoded_upload_contract() {
    recipes::srgb_linear_clamp_identity_is_distinct_and_preserves_encoded_upload_contract();
}

#[test]
fn not_filterable_error_is_structured() {
    recipes::not_filterable_error_is_structured();
}
