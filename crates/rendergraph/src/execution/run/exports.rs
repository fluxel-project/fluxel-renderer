//! Models and lowering for resources exported from an executed frame.

use std::collections::HashMap;

use crate::{
    CompiledGraph,
    backend::ExecutionBackend,
    handles::{ExportBufferSlot, ExportTextureSlot, ResourceId},
    internal::RootDecl,
    rhi::{BufferDesc, TextureDesc},
};

use super::super::{
    recording::{PhysicalResource, PhysicalResources},
    submission::FrameSubmission,
};

/// A physical exported texture and the lease retaining it for the caller.
pub struct ExportedTexture<B: ExecutionBackend> {
    /// Backend-native physical texture object.
    pub physical: B::Texture,
    /// Descriptor of the exported texture.
    pub descriptor: TextureDesc,
    /// State established by the graph's final transition for this export.
    pub outgoing_state: crate::rhi::ResourceAccessState,
    /// Caller-owned strong lease; completion alone does not invalidate it.
    pub lease: B::Lease,
}

/// A physical exported buffer and the lease retaining it for the caller.
pub struct ExportedBuffer<B: ExecutionBackend> {
    /// Backend-native physical buffer object.
    pub physical: B::Buffer,
    /// Descriptor of the exported buffer.
    pub descriptor: BufferDesc,
    /// State established by the graph's final transition for this export.
    pub outgoing_state: crate::rhi::ResourceAccessState,
    /// Caller-owned strong lease; completion alone does not invalidate it.
    pub lease: B::Lease,
}

/// Physical resource results exported by one graph execution.
pub struct FrameExports<B: ExecutionBackend> {
    textures: Vec<(ExportTextureSlot, ExportedTexture<B>)>,
    buffers: Vec<(ExportBufferSlot, ExportedBuffer<B>)>,
}

impl<B: ExecutionBackend> FrameExports<B> {
    /// Returns the texture exported through `slot`, when present.
    pub fn texture(&self, slot: ExportTextureSlot) -> Option<&ExportedTexture<B>> {
        self.textures
            .iter()
            .find(|(candidate, _)| *candidate == slot)
            .map(|(_, value)| value)
    }

    /// Returns the buffer exported through `slot`, when present.
    pub fn buffer(&self, slot: ExportBufferSlot) -> Option<&ExportedBuffer<B>> {
        self.buffers
            .iter()
            .find(|(candidate, _)| *candidate == slot)
            .map(|(_, value)| value)
    }

    /// Iterates all exported textures.
    pub fn textures(&self) -> impl Iterator<Item = (ExportTextureSlot, &ExportedTexture<B>)> {
        self.textures.iter().map(|(slot, value)| (*slot, value))
    }

    /// Iterates all exported buffers.
    pub fn buffers(&self) -> impl Iterator<Item = (ExportBufferSlot, &ExportedBuffer<B>)> {
        self.buffers.iter().map(|(slot, value)| (*slot, value))
    }
}

/// Successful execution result containing exports and in-flight submission.
pub struct ExecutedFrame<B: ExecutionBackend> {
    /// Physical resources handed to the caller.
    pub exports: FrameExports<B>,
    /// GPU completion and executor-held leases for the submitted frame.
    pub submission: FrameSubmission<B>,
}

pub(super) fn build_exports<B: ExecutionBackend, F>(
    graph: &CompiledGraph<F>,
    physical: &PhysicalResources<B>,
    leases: &HashMap<ResourceId, B::Lease>,
) -> FrameExports<B> {
    let mut textures = Vec::new();
    let mut buffers = Vec::new();
    for root in &graph.roots {
        match *root {
            RootDecl::Texture(slot, resource, _, contract) => {
                let PhysicalResource::Texture {
                    physical,
                    descriptor,
                    ..
                } = &physical[&resource]
                else {
                    unreachable!()
                };
                textures.push((
                    slot,
                    ExportedTexture {
                        physical: physical.clone(),
                        descriptor: *descriptor,
                        outgoing_state: contract.final_state,
                        lease: leases[&resource].clone(),
                    },
                ));
            }
            RootDecl::Buffer(slot, resource, _, contract) => {
                let PhysicalResource::Buffer {
                    physical,
                    descriptor,
                    ..
                } = &physical[&resource]
                else {
                    unreachable!()
                };
                buffers.push((
                    slot,
                    ExportedBuffer {
                        physical: physical.clone(),
                        descriptor: *descriptor,
                        outgoing_state: contract.final_state,
                        lease: leases[&resource].clone(),
                    },
                ));
            }
            RootDecl::Present(_, _, _, _) | RootDecl::SideEffect(..) => {}
        }
    }
    FrameExports { textures, buffers }
}
