//! Portable validation helpers for owned-resource creation.
use super::*;
pub(super) fn validate_raster_uniform_contract(
    kernel: RasterKernel,
    size: u64,
    usage: BufferUsage,
) -> Result<(), RasterCreateError> {
    if !matches!(
        kernel,
        RasterKernel::IndexedPositionFloat32x3CameraMaterial
            | RasterKernel::IndexedPositionFloat32x3CameraMaterialNormalLambert
            | RasterKernel::IndexedPositionFloat32x3CameraMaterialVertexColor
    ) {
        return Err(RasterCreateError::BindingRecipeMismatch);
    }
    if size != 80 {
        return Err(RasterCreateError::InvalidBindingRange);
    }
    if !usage.contains(BufferUsageKind::Uniform) {
        return Err(RasterCreateError::UniformUsageRequired);
    }
    Ok(())
}

pub(super) fn validate_immutable_upload_descriptor(
    descriptor: BufferDescriptor,
    bytes: &[u8],
) -> Result<(), BufferUploadError> {
    if bytes.is_empty() {
        return Err(BufferUploadError::InvalidRequest(
            InvalidBufferUploadReason::EmptyData,
        ));
    }
    if !bytes.len().is_multiple_of(4) {
        return Err(BufferUploadError::InvalidRequest(
            InvalidBufferUploadReason::DataLengthNotCopyAligned,
        ));
    }
    if descriptor.buffer.size != bytes.len() as u64 {
        return Err(BufferUploadError::InvalidRequest(
            InvalidBufferUploadReason::DescriptorSizeMismatch,
        ));
    }
    if descriptor.memory != MemoryPolicy::DeviceOnly {
        return Err(BufferUploadError::InvalidRequest(
            InvalidBufferUploadReason::MemoryPolicyUnsupported,
        ));
    }
    if !descriptor.usage.contains(BufferUsageKind::CopyDestination) {
        return Err(BufferUploadError::InvalidRequest(
            InvalidBufferUploadReason::CopyDestinationUsageRequired,
        ));
    }
    Ok(())
}

pub(super) fn validate_immutable_texture_upload_descriptor(
    descriptor: TextureDescriptor,
    bytes: &[u8],
) -> Result<(), TextureUploadError> {
    let image = descriptor.texture;
    let invalid = |reason| Err(TextureUploadError::InvalidRequest(reason));
    if image.dimension != TextureDimension::D2 {
        return invalid(InvalidTextureUploadReason::DimensionUnsupported);
    }
    if !matches!(
        image.format,
        TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb
    ) {
        return invalid(InvalidTextureUploadReason::FormatUnsupported);
    }
    if image.mip_levels != 1 {
        return invalid(InvalidTextureUploadReason::MipLevelsUnsupported);
    }
    if image.array_layers != 1 {
        return invalid(InvalidTextureUploadReason::ArrayLayersUnsupported);
    }
    if image.sample_count != 1 {
        return invalid(InvalidTextureUploadReason::SampleCountUnsupported);
    }
    if descriptor.memory != MemoryPolicy::DeviceOnly {
        return invalid(InvalidTextureUploadReason::MemoryPolicyUnsupported);
    }
    if !descriptor.usage.contains(TextureUsageKind::CopyDestination) {
        return invalid(InvalidTextureUploadReason::CopyDestinationUsageRequired);
    }
    if !descriptor.usage.contains(TextureUsageKind::Sampled) {
        return invalid(InvalidTextureUploadReason::SampledUsageRequired);
    }
    let production_usage =
        TextureUsage::from_kinds([TextureUsageKind::CopyDestination, TextureUsageKind::Sampled]);
    // The production object deliberately has no readback or general-purpose
    // usage widening.  Test-support may add exactly CopySource so its private
    // oracle can observe the upload; it may never grant storage or attachment
    // access through this creation path.
    #[cfg(any(test, feature = "test-support"))]
    let usage_is_closed = descriptor.usage == production_usage
        || descriptor.usage == production_usage.with(TextureUsageKind::CopySource);
    #[cfg(not(any(test, feature = "test-support")))]
    let usage_is_closed = descriptor.usage == production_usage;
    if !usage_is_closed {
        return invalid(InvalidTextureUploadReason::UnexpectedUsage);
    }
    let expected = u64::from(image.extent.width)
        .checked_mul(u64::from(image.extent.height))
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(TextureUploadError::InvalidRequest(
            InvalidTextureUploadReason::DataLengthOverflow,
        ))?;
    if u64::try_from(bytes.len()).ok() != Some(expected) {
        return invalid(InvalidTextureUploadReason::DataLengthMismatch);
    }
    Ok(())
}

/// Checks the fixed shader declaration against raw native workgroup facts.
///
/// This is intentionally kept at the RHI boundary: the declaration is a
/// property of the fixed shader artifact, not of RenderGraph dispatch syntax.
pub(super) fn validate_compute_workgroup_limits(
    workgroup_size: [u32; 3],
    maximum_size: [u32; 3],
    maximum_invocations: u32,
) -> Result<(), ComputeCreateError> {
    if workgroup_size
        .into_iter()
        .zip(maximum_size)
        .any(|(requested, maximum)| requested == 0 || requested > maximum)
    {
        return Err(ComputeCreateError::UnsupportedComputeLimits);
    }
    let invocations = workgroup_size
        .into_iter()
        .try_fold(1_u32, |total, component| total.checked_mul(component));
    if invocations.is_none_or(|total| total > maximum_invocations) {
        return Err(ComputeCreateError::UnsupportedComputeLimits);
    }
    Ok(())
}

/// Checks a fixed RW-storage binding against portable and native device facts.
///
/// The fixed WGSL artifacts require four-byte elements, while the native
/// offset must additionally meet the device's storage-buffer alignment.
pub(super) fn validate_compute_binding_range(
    offset: u64,
    size: u64,
    buffer_size: u64,
    minimum_storage_offset_alignment: u32,
    maximum_storage_binding_size: u64,
) -> Result<(), ComputeCreateError> {
    let required_offset_alignment = u64::from(minimum_storage_offset_alignment.max(4));
    let end = offset
        .checked_add(size)
        .ok_or(ComputeCreateError::InvalidBindingRange)?;
    if size == 0
        || !offset.is_multiple_of(required_offset_alignment)
        || !size.is_multiple_of(4)
        || size > maximum_storage_binding_size
        || end > buffer_size
    {
        return Err(ComputeCreateError::InvalidBindingRange);
    }
    Ok(())
}

/// Validates the closed X01 sampled-texture side of the binding recipe.
pub(super) fn validate_texture_pack_texture(texture: &Texture) -> Result<(), ComputeCreateError> {
    validate_texture_pack_texture_desc(texture.descriptor().texture, texture.allowed_usage())
}

/// Validates the exact whole-image storage texture shape used by the two
/// fixed storage recipes. No view ranges, formats, or access modes escape
/// this closed boundary.
pub(super) fn validate_storage_rgba8_texture(
    texture: &Texture,
    required_usage: TextureUsageKind,
) -> Result<(), ComputeCreateError> {
    let image = texture.descriptor().texture;
    if image.dimension != TextureDimension::D2
        || image.format != TextureFormat::Rgba8Unorm
        || image.extent.depth != 1
        || image.mip_levels != 1
        || image.array_layers != 1
        || image.sample_count != 1
        || !texture.allowed_usage().contains(required_usage)
    {
        return Err(ComputeCreateError::BindingRecipeMismatch);
    }
    Ok(())
}

/// Validates the descriptor facts the X01 texture binding cannot generalize.
pub(super) fn validate_texture_pack_texture_desc(
    image: TextureDesc,
    allowed_usage: TextureUsage,
) -> Result<(), ComputeCreateError> {
    if image.dimension != TextureDimension::D2
        || image.format != TextureFormat::Rgba8Unorm
        || image.extent.depth != 1
        || image.mip_levels != 1
        || image.array_layers != 1
        || image.sample_count != 1
        || !allowed_usage.contains(TextureUsageKind::Sampled)
    {
        return Err(ComputeCreateError::BindingRecipeMismatch);
    }
    Ok(())
}

/// Validates the sRGB sampled-texture facts the fixed base-color recipe
/// cannot generalize.  Keeping it distinct from the UNORM compute helper
/// prevents accidentally normalizing an sRGB resource into an UNORM one.
pub(super) fn validate_raster_srgb_texture_desc(
    image: TextureDesc,
    allowed_usage: TextureUsage,
) -> Result<(), RasterCreateError> {
    if image.dimension != TextureDimension::D2
        || image.format != TextureFormat::Rgba8UnormSrgb
        || image.extent.depth != 1
        || image.mip_levels != 1
        || image.array_layers != 1
        || image.sample_count != 1
        || !allowed_usage.contains(TextureUsageKind::Sampled)
    {
        return Err(RasterCreateError::BindingRecipeMismatch);
    }
    Ok(())
}

/// Returns the exact byte extent of X01's one-`u32`-per-pixel output.
pub(super) fn texture_pack_required_size(image: TextureDesc) -> Result<u64, ComputeCreateError> {
    u64::from(image.extent.width)
        .checked_mul(u64::from(image.extent.height))
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or(ComputeCreateError::InvalidBindingRange)
}

pub(super) fn invalid(
    resource: ResourceKind,
    reason: InvalidResourceReason,
) -> ResourceCreateError {
    ResourceCreateError::InvalidDescriptor { resource, reason }
}

pub(super) fn validate_buffer(desc: BufferDescriptor) -> Result<(), ResourceCreateError> {
    if desc.buffer.size == 0 {
        return Err(invalid(
            ResourceKind::Buffer,
            InvalidResourceReason::ZeroSize,
        ));
    }
    if !buffer_has_usage(desc.usage) {
        return Err(invalid(
            ResourceKind::Buffer,
            InvalidResourceReason::EmptyUsage,
        ));
    }
    Ok(())
}

pub(super) fn buffer_has_usage(usage: BufferUsage) -> bool {
    [
        BufferUsageKind::Uniform,
        BufferUsageKind::StorageRead,
        BufferUsageKind::StorageWrite,
        BufferUsageKind::Vertex,
        BufferUsageKind::Index,
        BufferUsageKind::Indirect,
        BufferUsageKind::CopySource,
        BufferUsageKind::CopyDestination,
    ]
    .into_iter()
    .any(|kind| usage.contains(kind))
}

pub(super) fn texture_has_usage(usage: TextureUsage) -> bool {
    [
        TextureUsageKind::Sampled,
        TextureUsageKind::StorageRead,
        TextureUsageKind::StorageWrite,
        TextureUsageKind::ColorAttachment,
        TextureUsageKind::DepthStencilAttachment,
        TextureUsageKind::CopySource,
        TextureUsageKind::CopyDestination,
        TextureUsageKind::Present,
    ]
    .into_iter()
    .any(|kind| usage.contains(kind))
}

pub(super) fn validate_texture(desc: TextureDescriptor) -> Result<(), ResourceCreateError> {
    let d = desc.texture;
    if !texture_has_usage(desc.usage) {
        return Err(invalid(
            ResourceKind::Texture,
            InvalidResourceReason::EmptyUsage,
        ));
    }
    if desc.usage.contains(TextureUsageKind::Present) {
        return Err(invalid(
            ResourceKind::Texture,
            InvalidResourceReason::PresentRequiresSurface,
        ));
    }
    if d.extent.width == 0 || d.extent.height == 0 || d.extent.depth == 0 {
        return Err(invalid(
            ResourceKind::Texture,
            InvalidResourceReason::ZeroExtent,
        ));
    }
    if d.array_layers == 0 {
        return Err(invalid(
            ResourceKind::Texture,
            InvalidResourceReason::InvalidArrayLayers,
        ));
    }
    if d.dimension != TextureDimension::D2 {
        return Err(invalid(
            ResourceKind::Texture,
            InvalidResourceReason::UnsupportedDimension,
        ));
    }
    let dimensions_valid = match d.dimension {
        TextureDimension::D1 => false,
        TextureDimension::D2 => d.extent.depth == 1,
        TextureDimension::D3 => false,
        _ => false,
    };
    if !dimensions_valid {
        return Err(invalid(
            ResourceKind::Texture,
            InvalidResourceReason::InvalidDimension,
        ));
    }
    let max_dimension = d.extent.width.max(d.extent.height).max(d.extent.depth);
    let max_mips = u32::BITS - max_dimension.leading_zeros();
    if d.mip_levels == 0 || d.mip_levels > max_mips {
        return Err(invalid(
            ResourceKind::Texture,
            InvalidResourceReason::InvalidMipLevels,
        ));
    }
    if !matches!(d.sample_count, 1 | 2 | 4 | 8 | 16)
        || (d.sample_count > 1
            && (d.dimension != TextureDimension::D2 || d.mip_levels != 1 || d.array_layers != 1))
    {
        return Err(invalid(
            ResourceKind::Texture,
            InvalidResourceReason::InvalidSampleCount,
        ));
    }
    let depth = d.format == TextureFormat::Depth32Float;
    let color_attachment = desc.usage.contains(TextureUsageKind::ColorAttachment);
    let depth_attachment = desc
        .usage
        .contains(TextureUsageKind::DepthStencilAttachment);
    let storage = desc.usage.contains(TextureUsageKind::StorageRead)
        || desc.usage.contains(TextureUsageKind::StorageWrite);
    let copy = desc.usage.contains(TextureUsageKind::CopySource)
        || desc.usage.contains(TextureUsageKind::CopyDestination);
    if (depth && (color_attachment || storage))
        || (!depth && depth_attachment)
        || (d.sample_count > 1 && (storage || copy))
    {
        return Err(invalid(
            ResourceKind::Texture,
            InvalidResourceReason::IncompatibleUsage,
        ));
    }
    Ok(())
}
