//! Copy-command descriptor checks.

use crate::rhi::{TextureDesc, TextureDimension, TextureFormat};

pub(super) fn supported_texture_copy_descriptor(descriptor: TextureDesc) -> bool {
    descriptor.dimension == TextureDimension::D2
        && descriptor.array_layers == 1
        && descriptor.sample_count == 1
        && descriptor.format != TextureFormat::Depth32Float
}

use crate::{
    access::{
        BufferCopyRegion, BufferRange, BufferReadUse, BufferWriteUse, TextureAspect,
        TextureCopyRegion, TextureRange, TextureReadUse, TextureWriteUse,
    },
    backend::{ExecutionBackend, RenderObjectProvider},
    error::{RecordResult, RecordingErrorKind},
    handles::{BufferRead, BufferWrite, TextureRead, TextureWrite},
    internal::{AccessDecl, AccessSemantic, DeclRange},
    pass::CopyCommandSink,
};

use super::{
    PhysicalResource, PhysicalResources,
    shared::{CommandBridge, recording_error},
};

// Matches wgpu's COPY_BUFFER_ALIGNMENT without making portable RenderGraph
// depend on the native HAL type crate.
const COPY_BUFFER_ALIGNMENT: u64 = 4;

pub(super) struct CopyBridge<'a, B: ExecutionBackend, O> {
    common: CommandBridge<'a, B, O>,
}

impl<'a, B: ExecutionBackend, O> CopyBridge<'a, B, O> {
    pub(super) fn new(common: CommandBridge<'a, B, O>) -> Self {
        Self { common }
    }
    pub(super) fn finish(
        &mut self,
        callback: RecordResult,
    ) -> Result<(), crate::backend::ExecutionError<B::Error>> {
        self.common.finish(callback)
    }
}

impl<B, O> CopyCommandSink for CopyBridge<'_, B, O>
where
    B: ExecutionBackend,
    O: RenderObjectProvider<B>,
{
    fn copy_texture(
        &mut self,
        source: &TextureRead,
        destination: &TextureWrite,
        region: TextureCopyRegion,
    ) -> RecordResult {
        let source_access = self.common.access(source.0)?;
        let destination_access = self.common.access(destination.0)?;
        if !matches!(
            source_access.semantic,
            AccessSemantic::TextureRead(TextureReadUse::CopySource)
        ) || !matches!(
            destination_access.semantic,
            AccessSemantic::TextureWrite(TextureWriteUse::CopyDestination)
        ) {
            return Err(recording_error(
                RecordingErrorKind::DeclaredUseMismatch,
                self.common.pass,
                None,
                "texture copy handles require CopySource and CopyDestination declarations",
            ));
        }
        let (source_texture, source_desc) = texture_physical(self.common.physical, source_access);
        let (destination_texture, destination_desc) =
            texture_physical(self.common.physical, destination_access);
        validate_texture_copy(
            self.common.pass,
            source_access,
            destination_access,
            source_desc,
            destination_desc,
            region,
        )?;
        self.common
            .backend
            .copy_texture(
                self.common.encoder,
                source_texture,
                destination_texture,
                region,
            )
            .map_err(|error| self.common.fail_backend(error))
    }
    fn copy_buffer(
        &mut self,
        source: &BufferRead,
        destination: &BufferWrite,
        region: BufferCopyRegion,
    ) -> RecordResult {
        let source_access = self.common.access(source.0)?;
        let destination_access = self.common.access(destination.0)?;
        if !matches!(
            source_access.semantic,
            AccessSemantic::BufferRead(BufferReadUse::CopySource)
        ) || !matches!(
            destination_access.semantic,
            AccessSemantic::BufferWrite(BufferWriteUse::CopyDestination)
        ) {
            return Err(recording_error(
                RecordingErrorKind::DeclaredUseMismatch,
                self.common.pass,
                None,
                "buffer copy handles require CopySource and CopyDestination declarations",
            ));
        }
        let source_desc = buffer_descriptor(self.common.physical, source_access);
        let destination_desc = buffer_descriptor(self.common.physical, destination_access);
        validate_buffer_copy(
            self.common.pass,
            source_access,
            destination_access,
            source_desc,
            destination_desc,
            region,
        )?;
        let (source_buffer, _) = buffer_physical(self.common.physical, source_access);
        let (destination_buffer, _) = buffer_physical(self.common.physical, destination_access);
        self.common
            .backend
            .copy_buffer(
                self.common.encoder,
                source_buffer,
                destination_buffer,
                region,
            )
            .map_err(|error| self.common.fail_backend(error))
    }
}

fn buffer_physical<'a, B: ExecutionBackend>(
    physical: &'a PhysicalResources<B>,
    access: &AccessDecl,
) -> (&'a B::Buffer, u64) {
    match &physical[&access.resource] {
        PhysicalResource::Buffer { physical, .. } => {
            let offset = match access.range {
                DeclRange::Buffer(BufferRange::Whole) => 0,
                DeclRange::Buffer(BufferRange::Bytes { offset, .. }) => offset,
                _ => unreachable!(),
            };
            (physical, offset)
        }
        PhysicalResource::Texture { .. } => unreachable!(),
    }
}
fn buffer_descriptor<B: ExecutionBackend>(
    physical: &PhysicalResources<B>,
    access: &AccessDecl,
) -> crate::rhi::BufferDesc {
    match &physical[&access.resource] {
        PhysicalResource::Buffer { descriptor, .. } => *descriptor,
        PhysicalResource::Texture { .. } => unreachable!(),
    }
}
fn texture_physical<'a, B: ExecutionBackend>(
    physical: &'a PhysicalResources<B>,
    access: &AccessDecl,
) -> (&'a B::Texture, TextureDesc) {
    match &physical[&access.resource] {
        PhysicalResource::Texture {
            physical,
            descriptor,
            ..
        } => (physical, *descriptor),
        PhysicalResource::Buffer { .. } => unreachable!(),
    }
}
fn validate_buffer_copy(
    pass: crate::handles::PassId,
    source: &AccessDecl,
    destination: &AccessDecl,
    source_desc: crate::rhi::BufferDesc,
    destination_desc: crate::rhi::BufferDesc,
    region: BufferCopyRegion,
) -> RecordResult {
    if region.size == 0
        || !region.source_offset.is_multiple_of(COPY_BUFFER_ALIGNMENT)
        || !region
            .destination_offset
            .is_multiple_of(COPY_BUFFER_ALIGNMENT)
        || !region.size.is_multiple_of(COPY_BUFFER_ALIGNMENT)
        || !buffer_range_contains(source.range, region.source_offset, region.size)
        || !buffer_range_contains(destination.range, region.destination_offset, region.size)
        || !buffer_descriptor_contains(source_desc, region.source_offset, region.size)
        || !buffer_descriptor_contains(destination_desc, region.destination_offset, region.size)
    {
        return Err(recording_error(
            RecordingErrorKind::InvalidCommandArgument,
            pass,
            None,
            "buffer copy requires non-zero 4-byte-aligned offsets and size within its physical and declared access ranges",
        ));
    }
    Ok(())
}
fn buffer_descriptor_contains(d: crate::rhi::BufferDesc, offset: u64, size: u64) -> bool {
    offset.checked_add(size).is_some_and(|end| end <= d.size)
}
fn buffer_range_contains(range: DeclRange, offset: u64, size: u64) -> bool {
    let Some(_) = offset.checked_add(size) else {
        return false;
    };
    match range {
        DeclRange::Buffer(BufferRange::Whole) => true,
        DeclRange::Buffer(range) => super::shared::buffer_range_contains(range, offset, size),
        _ => false,
    }
}
fn validate_texture_copy(
    pass: crate::handles::PassId,
    source: &AccessDecl,
    destination: &AccessDecl,
    source_desc: TextureDesc,
    destination_desc: TextureDesc,
    region: TextureCopyRegion,
) -> RecordResult {
    if !supported_texture_copy_descriptor(source_desc)
        || !supported_texture_copy_descriptor(destination_desc)
        || source_desc.format != destination_desc.format
        || region.extent.contains(&0)
        || region.source_origin[2] != 0
        || region.destination_origin[2] != 0
        || region.extent[2] != 1
        || region.source_mip_level >= source_desc.mip_levels
        || region.destination_mip_level >= destination_desc.mip_levels
        || !texture_region_in_bounds(
            source_desc,
            region.source_mip_level,
            region.source_origin,
            region.extent,
        )
        || !texture_region_in_bounds(
            destination_desc,
            region.destination_mip_level,
            region.destination_origin,
            region.extent,
        )
        || !texture_range_contains(source.range, region.source_mip_level)
        || !texture_range_contains(destination.range, region.destination_mip_level)
    {
        return Err(recording_error(
            RecordingErrorKind::InvalidCommandArgument,
            pass,
            None,
            "texture copy requires matching supported D2 color formats and must fit the declared access range",
        ));
    }
    Ok(())
}
fn texture_region_in_bounds(d: TextureDesc, mip: u32, origin: [u32; 3], extent: [u32; 3]) -> bool {
    let width = (d.extent.width >> mip).max(1);
    let height = (d.extent.height >> mip).max(1);
    origin[0]
        .checked_add(extent[0])
        .is_some_and(|end| end <= width)
        && origin[1]
            .checked_add(extent[1])
            .is_some_and(|end| end <= height)
}
fn texture_range_contains(range: DeclRange, mip: u32) -> bool {
    match range {
        DeclRange::Texture(TextureRange::Whole) => true,
        DeclRange::Texture(TextureRange::Subresources {
            base_mip_level,
            mip_level_count,
            base_array_layer,
            array_layer_count,
            aspect,
        }) => {
            base_array_layer == 0
                && array_layer_count >= 1
                && matches!(aspect, TextureAspect::All | TextureAspect::Color)
                && mip >= base_mip_level
                && mip < base_mip_level + mip_level_count
        }
        _ => false,
    }
}
