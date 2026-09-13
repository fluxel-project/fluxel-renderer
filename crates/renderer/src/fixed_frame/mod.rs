//! Coordinates the deliberately closed headless indexed-frame rendering slice.
//!
//! This boundary owns fixed-frame graph declarations, closed raster recipes,
//! draw-start validation, and the two-phase submission lifecycle. It does not
//! expose native resources, configurable pipelines, or general scheduling:
//! those remain RHI and RenderGraph responsibilities. Recipes select the
//! graph and binding contract; the renderer validates and starts work; the
//! submission module preserves the accepted-versus-pre-accept lifecycle and
//! releases or poisons snapshot reservations accordingly.

use core::fmt;
use std::sync::Arc;

use fluxel_rendergraph::{
    AttachmentOps, BindingResource, BindingSetId, BoundBuffer, BufferBindingId, BufferRange,
    ColorAttachmentDesc, CompletionFailure, CompletionStatus, DeviceCapabilities, DeviceIdentity,
    ExecutionError, ExportBufferContract, ExportTextureContract, Extent3d, ExternalOwnership,
    FrameBindingError, FrameBindingErrorKind, FrameInputs, FrameResourceProvider,
    ImportBufferContract, ImportTextureContract, IndexFormat, InitialContents, LoadOp,
    PresentContract, RasterPipelineId, RenderGraph, ResourceAccessState, StoreOp,
    SurfaceTextureContract, TextureBindingId, TextureDesc, TextureDimension, TextureFormat,
    TextureRange, TextureReadUse, Viewport, WriteCoverage,
};
use fluxel_rhi::adapter::fixed_artifacts::{RasterBackend, RasterKernel, RasterObjectProvider};
use fluxel_rhi::{
    Buffer, BufferDescriptor, BufferUploadError, Device, MemoryPolicy, PendingBufferUpload,
    ResourceLease, Texture, UploadedBuffer,
};
#[cfg(windows)]
use fluxel_rhi::{
    TextureLease,
    presentation::{AcquiredSurfaceFrame, PresentationToken, Surface},
};

use crate::upload::{SnapshotDrawReservation, SnapshotUseError, completion_requires_retention};
use crate::{
    BaseColorTextureSnapshot, IndexedMeshSnapshot, NormalIndexedMeshSnapshot,
    SrgbBaseColorTextureSnapshot, SrgbTexturedBasicMaterial, TexturedBasicMaterial,
    TexturedIndexedMeshSnapshot, VertexColorIndexedMeshSnapshot, VertexColorMaterial,
    frame_uniform::{FRAME_UNIFORM_BYTES, FrameUniform},
};

#[cfg(all(test, windows))]
fn native_fixture_guard() -> std::sync::MutexGuard<'static, ()> {
    crate::native_fixture_guard()
}

#[cfg(all(test, windows))]
fn actual_raster_capabilities(device: &Device) -> DeviceCapabilities {
    let backend = RasterBackend::new(device.clone());
    fluxel_rendergraph::ExecutionBackend::capabilities(&backend).clone()
}

/// Fixed-frame graph declarations and resource provider bindings.
mod graph;
/// Shared closed graph declaration used by native and browser execution.
pub(crate) mod graph_shared;
/// Fixed identifiers shared by closed recipes and graph declarations.
mod ids;
/// Owned multi-draw packets and their non-blocking submission lifecycle.
mod packet;
/// Caller-owned snapshot bindings for graph imports.
mod provider;
/// Closed fixed raster-contract mappings.
mod recipe;
/// Fixed-frame renderer construction and draw-start validation.
pub(crate) mod renderer;
/// Fixed-frame two-phase submission completion lifecycle.
mod submission;
#[cfg(test)]
/// CPU and legacy fixed-frame conformance fixtures.
mod tests;
/// Renderer-owned visible fixed-frame presentation lifecycle.
#[cfg(windows)]
mod visible;

pub use packet::{
    RenderPacket, RenderPacketBuildError, RenderPacketDrawBuildError, RenderPacketFailure,
    RenderPacketReservationError, RenderPacketStartError, RenderPacketStatus,
    RenderPacketSubmission,
};
pub use renderer::{
    DrawStartError, FixedFrameExecutionError, FixedFrameFailure, FixedFrameRasterObservationError,
    FixedFrameRenderer, FixedFrameUniformObservationError,
};
pub use submission::{FixedFrameStatus, FixedFrameSubmission, FrameImage};
#[cfg(windows)]
pub use visible::{VisibleFrameStartError, VisibleFrameStatus, VisibleFrameSubmission};

#[cfg(windows)]
use graph::build_presentable_camera_graph;
use graph::{
    CameraGraph, build_camera_graph, build_normal_lambert_camera_graph,
    build_textured_camera_graph, build_uv_textured_camera_graph, build_vertex_color_camera_graph,
};
#[cfg(all(test, windows))]
use graph::{FixedGraph, build_graph};
use ids::*;
use provider::CameraResources;
use recipe::*;
#[cfg(all(test, windows))]
use renderer::UvStartRequest;
use renderer::{FrameTexture, FrameTextureSnapshot};
#[cfg(all(test, windows))]
use submission::SnapshotResources;
use submission::{CameraPhase, FrameMeshSnapshot, missing_binding};
