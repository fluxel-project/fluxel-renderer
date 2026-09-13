//! A typed render graph for dependency-driven GPU work planning.
//!
//! Pass setup declares versioned texture and buffer access. Compilation derives
//! dependencies, culling, capability validation, and future lowering inputs from
//! those declarations. Pass execution can use only the typed access handles its
//! own setup callback produced.
//!
//! The current release has a reusable single-queue execution plan, frame
//! resource and render-object providers, validated command recording,
//! completion retirement, imported presentable-image execution, a deterministic
//! CPU-only `TestRhi`, and typed resource-usage requirements. Native GPU
//! objects remain a deliberate backend boundary. Start with the numbered examples and the
//! repository's `documents/design-rendergraph.md`.
//!
//! The public API is intentionally available from this crate root; implementation
//! modules are private so their layout is not a compatibility contract.
//!
//! ```compile_fail
//! use fluxel_rendergraph::access::TextureRange;
//! ```
//!
//! ```
//! use fluxel_rendergraph::TextureRange;
//! let _ = TextureRange::whole();
//! ```

#![deny(missing_docs)]

mod access;
mod backend;
mod compile;
mod error;
mod execution;
mod graph;
mod handles;
mod internal;
mod pass;
mod plan;
mod recipe;
mod resource;
mod rhi;
pub mod test_rhi;

pub use access::{
    AccessMode, BufferCopyRegion, BufferRange, BufferReadUse, BufferReadWriteUse, BufferWriteUse,
    TextureAspect, TextureCopyRegion, TextureRange, TextureReadUse, TextureReadWriteUse,
    TextureWriteUse, WriteCoverage,
};
pub use backend::{
    BindingResourceSemantic, BoundBindings, BoundBuffer, BoundComputePipeline, BoundRasterPipeline,
    BoundSurfaceTexture, BoundTexture, CompletionFailure, CompletionStatus, DeviceIdentity,
    ExecutionBackend, ExecutionError, FrameBindingError, FrameBindingErrorKind,
    FrameResourceProvider, PhysicalResourceIdentity, PresentationSubmission, RasterColorAttachment,
    RasterDepthStencilAttachment, RasterPassDescriptor, RenderObjectProvider,
    ResolvedBindingResource, SurfaceBindingResult,
};
pub use compile::{
    CapabilityFallback, CompileOutput, CompileReport, CompileResult, CompiledGraph, ExplicitOrder,
    PassDependency, RetainedSideEffect,
};
pub use error::{
    CapabilityRequirement, CompileError, CompileErrorKind, DiagnosticContext, RecordResult,
    RecordingError, RecordingErrorKind, SurfaceCapabilityOperation, UnsupportedCapability,
};
pub use execution::{
    ExecutedFrame, ExportedBuffer, ExportedTexture, FrameExecution, FrameExecutor, FrameExports,
    FrameInputs, FrameSubmission, Local, SendMode,
};
pub use graph::{DeclaredPass, ExplicitOrderReason, RenderGraph, SideEffectReason};
pub use handles::{
    BindingSetId, BufferBindingId, BufferRead, BufferReadWrite, BufferVersion, BufferWrite,
    ComputePipelineId, ExportBufferSlot, ExportTextureSlot, ImportBufferSlot, ImportTextureSlot,
    PassId, PresentTarget, RasterPipelineId, ResourceId, SurfaceBindingId, TextureBindingId,
    TextureRead, TextureReadWrite, TextureVersion, TextureWrite,
};
pub use pass::{
    AttachmentOps, BindingResource, ColorAttachmentDesc, ComputeCommands, ComputePassBuilder,
    CopyCommands, CopyPassBuilder, DepthStencilAttachmentDesc, LoadOp, PassKind,
    PassResourceResolver, RasterCommands, RasterPassBuilder, ResolvedBindings, ScissorRect,
    StoreOp, Viewport,
};
pub use plan::{
    BufferUsage, BufferUsageKind, ExecutionPlan, PlannedColorAttachment,
    PlannedDepthStencilAttachment, PlannedPass, PlannedResourceRange, PlannedTransition,
    RasterPassPlan, ResourceRequirement, ResourceUsageSummary, TextureUsage, TextureUsageKind,
};
pub use recipe::{
    ComputeExecute, ComputeSetup, CopyExecute, CopySetup, PassData, RasterExecute, RasterSetup,
};
pub use resource::{
    ExportBufferContract, ExportTextureContract, ImportBufferContract, ImportTextureContract,
    ImportedBuffer, ImportedTexture, InitialContents, PresentContract, SurfaceTextureContract,
};
pub use rhi::{
    BufferCapabilities, BufferDesc, DeviceCapabilities, DeviceCapabilitiesBuilder, DeviceLimits,
    Extent3d, ExternalOwnership, IndexFormat, QueueCapabilities, QueueDescriptor, QueueId,
    RecordingCapabilities, RecordingModel, ResourceAccessState, SurfaceCapabilities,
    SynchronizationCapabilities, TextureDesc, TextureDimension, TextureFormat,
    TextureFormatCapabilities, TextureFormatCapabilitiesBuilder, TimestampCapabilities,
    TransientResourceCapabilities, TransitionCapabilities,
};
