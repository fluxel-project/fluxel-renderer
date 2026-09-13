//! Upload contract tests grouped by their closed resource domain.

use super::indexed::SnapshotPublication;
use super::normal::*;
use super::shared::{RetainedUploadState, SnapshotUseGate, completion_requires_retention};
use super::textured::*;
use super::*;
use std::sync::Arc;

use fluxel_rendergraph::CompletionFailure;
use fluxel_rhi::BufferUploadError;
#[cfg(windows)]
use fluxel_rhi::Device;

use crate::Geometry;

mod texture;
mod vertex_color;

#[test]
fn rgba8_image_requires_nonzero_extent_and_exact_tight_length() {
    texture::rgba8_image_requires_nonzero_extent_and_exact_tight_length_contract();
}

#[test]
fn rgba8_image_checked_length_handles_largest_constructible_dimensions() {
    texture::rgba8_image_checked_length_handles_largest_constructible_dimensions_contract();
}

#[test]
fn base_color_texture_snapshot_debug_is_exact_safe_metadata_only() {
    texture::base_color_texture_snapshot_debug_is_exact_safe_metadata_only_contract();
}

#[test]
fn indexed_mesh_snapshot_debug_is_exact_safe_metadata_only() {
    texture::indexed_mesh_snapshot_debug_is_exact_safe_metadata_only_contract();
}

#[test]
fn mesh_snapshot_retains_original_cpu_metadata() {
    texture::mesh_snapshot_retains_original_cpu_metadata_contract();
}

mod contract;
mod gate;
mod geometry;
mod publication;

#[test]
fn normal_geometry_canonicalizes_with_fixed_f64_recipe_and_positive_zero_bytes() {
    geometry::normal_geometry_canonicalizes_with_fixed_f64_recipe_and_positive_zero_bytes();
}

#[test]
fn normal_geometry_rejects_closed_stream_and_normalization_contract_violations() {
    geometry::normal_geometry_rejects_closed_stream_and_normalization_contract_violations();
}

#[test]
fn normal_payload_uses_only_canonical_metadata_bytes() {
    geometry::normal_payload_uses_only_canonical_metadata_bytes();
}

#[test]
fn normal_snapshot_debug_and_publication_are_opaque_and_atomic() {
    geometry::normal_snapshot_debug_and_publication_are_opaque_and_atomic();
}

#[test]
fn normal_upload_failure_mapping_preserves_stream_role_and_acceptance_boundary() {
    geometry::normal_upload_failure_mapping_preserves_stream_role_and_acceptance_boundary();
}

#[cfg(windows)]
#[test]
fn normal_fault_commit_helper_accepts_only_full_hex_sha_values() {
    geometry::normal_fault_commit_helper_contract();
}

#[test]
fn srgba8_image_keeps_encoded_bytes_and_rejects_invalid_shape() {
    geometry::srgba8_image_keeps_encoded_bytes_and_rejects_invalid_shape();
}

#[test]
fn srgba8_image_debug_and_error_are_explicit_about_encoded_domain() {
    geometry::srgba8_image_debug_and_error_are_explicit_about_encoded_domain();
}

#[test]
fn textured_geometry_rejects_closed_stream_contract_violations() {
    geometry::textured_geometry_rejects_closed_stream_contract_violations();
}

#[test]
fn textured_payload_has_exact_three_little_endian_streams() {
    geometry::textured_payload_has_exact_three_little_endian_streams();
}

#[test]
fn textured_snapshot_debug_is_opaque_metadata_only() {
    geometry::textured_snapshot_debug_is_opaque_metadata_only();
}

#[test]
fn textured_three_stream_publication_never_exposes_partial_or_failed_output() {
    geometry::textured_three_stream_publication_never_exposes_partial_or_failed_output();
}

#[test]
fn textured_upload_failure_mapping_preserves_stream_role_and_acceptance_boundary() {
    geometry::textured_upload_failure_mapping_preserves_stream_and_acceptance_boundary();
}

#[test]
fn vertex_color_geometry_material_and_payload_preserve_closed_contract() {
    vertex_color::geometry_material_and_payload_preserve_closed_contract();
}

#[test]
fn vertex_color_publication_and_failure_mapping_preserve_three_stream_contract() {
    vertex_color::publication_and_failure_mapping_preserve_three_stream_contract();
}

#[cfg(windows)]
#[test]
#[ignore = "requires a Windows DX12 device with required validation and test-support fault injection"]
fn u09_vertex_color_three_upload_fault_contract_dx12() {
    vertex_color::u09_vertex_color_three_upload_fault_contract_dx12();
}

#[cfg(windows)]
#[test]
#[ignore = "requires a Windows Vulkan device with required validation and test-support fault injection"]
fn u09_vertex_color_three_upload_fault_contract_vulkan() {
    vertex_color::u09_vertex_color_three_upload_fault_contract_vulkan();
}

#[test]
fn serializes_positions_as_exact_little_endian_bits_and_indices_in_order() {
    contract::serializes_positions_as_exact_little_endian_bits_and_indices_in_order();
}

#[test]
fn payload_rejects_empty_or_non_indexed_geometry_before_upload() {
    contract::payload_rejects_empty_or_non_indexed_geometry_before_upload();
}

#[test]
fn publication_policy_never_exposes_a_partially_ready_or_failed_pair() {
    publication::publication_policy_never_exposes_a_partially_ready_or_failed_pair();
}

#[test]
fn publication_policy_represents_partial_acceptance_without_ready_output() {
    publication::publication_policy_represents_partial_acceptance_without_ready_output();
}

#[test]
fn generation_use_gate_allows_concurrent_immutable_readers_and_poison_is_monotonic() {
    gate::generation_use_gate_allows_concurrent_immutable_readers_and_poison_is_monotonic();
}

#[test]
fn unknown_completion_keeps_the_reservation_until_a_terminal_observation() {
    gate::unknown_completion_keeps_the_reservation_until_a_terminal_observation();
}

#[cfg(windows)]
#[test]
#[ignore = "requires a Windows DX12 device with required validation"]
fn u05_textured_three_upload_fault_contract_dx12() {
    gate::u05_textured_three_upload_fault_contract_dx12();
}

#[cfg(windows)]
#[test]
#[ignore = "requires a Windows Vulkan device with required validation"]
fn u05_textured_three_upload_fault_contract_vulkan() {
    gate::u05_textured_three_upload_fault_contract_vulkan();
}

#[cfg(windows)]
#[test]
#[ignore = "requires a Windows DX12 device with required validation"]
fn u08_normal_three_upload_fault_contract_dx12() {
    gate::u08_normal_three_upload_fault_contract_dx12();
}

#[cfg(windows)]
#[test]
#[ignore = "requires a Windows Vulkan device with required validation"]
fn u08_normal_three_upload_fault_contract_vulkan() {
    gate::u08_normal_three_upload_fault_contract_vulkan();
}

#[cfg(windows)]
#[test]
#[ignore = "requires a Windows DX12 device with required validation"]
fn u01_indexed_mesh_upload_dx12() {
    gate::u01_indexed_mesh_upload_dx12();
}

#[cfg(windows)]
#[test]
#[ignore = "requires a Windows Vulkan device with required validation"]
fn u01_indexed_mesh_upload_vulkan() {
    gate::u01_indexed_mesh_upload_vulkan();
}
