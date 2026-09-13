//! Structured graph compilation errors.

use std::{error::Error, fmt};

use crate::{
    handles::{ImportBufferSlot, ImportTextureSlot, PassId, ResourceId},
    pass::PassKind,
    rhi::{DeviceCapabilities, ResourceAccessState, TextureFormat},
};

/// Machine-readable categories of graph compilation failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CompileErrorKind {
    /// A resource version is stale or belongs to another graph.
    StaleOrForeignVersion,
    /// A transient resource is read before its first write.
    ReadBeforeInitialization,
    /// More than one writer consumes the same whole-resource version.
    InvalidResourceBranch,
    /// Declared accesses conflict or require unsupported pass-internal ordering.
    ConflictingAccess,
    /// A texture or buffer range is outside its descriptor.
    InvalidSubresourceRange,
    /// Resource and explicit-order edges contain a cycle.
    DependencyCycle,
    /// An imported resource does not have a complete contract.
    MissingImportContract,
    /// The device cannot preserve a requested semantic operation.
    UnsupportedSemanticRequirement,
    /// An export or presentation root is invalid.
    InvalidExportOrPresent,
}

/// Machine-readable categories of command-recording failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RecordingErrorKind {
    /// An access handle was not declared by the pass being recorded.
    ForeignOrUndeclaredPassAccess,
    /// A binding recipe requires a use incompatible with the declared access.
    DeclaredUseMismatch,
    /// A required physical resource was not supplied for this frame.
    MissingFrameBinding,
    /// A dynamic offset, index format, or command range is invalid.
    InvalidCommandArgument,
    /// Renderer/RHI binding metadata is incompatible with the selected pipeline.
    IncompatibleBindingRecipe,
    /// Creation of a backend-owned pipeline or binding object failed after validation.
    BackendObjectCreation,
}

/// A semantic capability requirement that the observed device could not satisfy.
///
/// This describes graph semantics only; it does not expose or imply a general
/// shader or pipeline abstraction.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CapabilityRequirement {
    /// One logical queue must support all requested pass kinds and, optionally,
    /// presentation.
    Queue {
        /// Sorted, deduplicated pass kinds that one queue must support.
        pass_kinds: Vec<PassKind>,
        /// Whether the queue must also support presentation.
        present: bool,
    },
    /// The device capability description is not a legal queue configuration.
    QueueConfiguration,
    /// The requested number of raster color attachments exceeds the device limit.
    ColorAttachmentCount {
        /// Number of attachment slots required by the pass.
        required: u32,
        /// Maximum number of color attachments advertised by the device.
        supported: u32,
    },
    /// A buffer state required by an import, export, or pass access is unavailable.
    BufferState {
        /// The required resource state.
        state: ResourceAccessState,
    },
    /// A texture format has no capability entry on the device.
    TextureFormat {
        /// The required texture format.
        format: TextureFormat,
    },
    /// A texture state is unavailable for its format and sample count.
    TextureState {
        /// The required texture format.
        format: TextureFormat,
        /// The required sample count.
        sample_count: u32,
        /// The required resource state.
        state: ResourceAccessState,
    },
    /// A surface operation is unavailable.
    Surface {
        /// The required surface operation.
        operation: SurfaceCapabilityOperation,
        /// The surface texture format involved in the requirement.
        format: TextureFormat,
    },
}

/// A surface-specific semantic operation requested by a render graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SurfaceCapabilityOperation {
    /// The device did not provide any surface capability facts.
    Availability,
    /// The surface format is unavailable for the requested operation.
    Format,
    /// Use the acquired image as a raster color attachment.
    ColorAttachment,
    /// Copy into the acquired image.
    CopyDestination,
    /// Sample from the acquired image.
    SampledRead,
    /// Read the acquired image through storage access.
    StorageRead,
    /// Write the acquired image through storage access.
    StorageWrite,
    /// Read and write the acquired image through storage access.
    StorageReadWrite,
    /// Copy from the acquired image.
    CopySource,
    /// Use the acquired image as a depth-stencil attachment.
    DepthStencilAttachment,
}

/// Structured evidence for a rejected semantic capability requirement.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct UnsupportedCapability {
    /// The graph semantic requirement that the device cannot satisfy.
    pub requirement: CapabilityRequirement,
    /// The complete capability snapshot observed by the compiler.
    pub observed: Box<DeviceCapabilities>,
}

/// Optional identities and details attached to a compilation error.
#[derive(Clone, Debug)]
pub struct DiagnosticContext {
    /// Passes directly involved in the failure.
    pub passes: Vec<PassId>,
    /// Logical resource involved in the failure.
    pub resource: Option<ResourceId>,
    /// Imported texture slot involved in the failure.
    pub texture_slot: Option<ImportTextureSlot>,
    /// Imported buffer slot involved in the failure.
    pub buffer_slot: Option<ImportBufferSlot>,
    /// Human-readable detail that is not intended for programmatic matching.
    pub detail: String,
    /// Structured evidence when a semantic requirement was rejected.
    pub unsupported: Option<Box<UnsupportedCapability>>,
}

/// A structured failure returned by render graph compilation.
#[derive(Clone, Debug)]
pub struct CompileError {
    /// Stable category for programmatic handling.
    pub kind: CompileErrorKind,
    /// Resource, pass, and capability evidence for diagnostics.
    pub context: DiagnosticContext,
}

impl fmt::Display for CompileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "render graph compilation failed ({:?}): {}",
            self.kind, self.context.detail
        )
    }
}

impl Error for CompileError {}

/// A structured failure produced while resolving pass resources or recording commands.
#[derive(Clone, Debug)]
pub struct RecordingError {
    /// Stable category for programmatic handling.
    pub kind: RecordingErrorKind,
    /// Pass/resource/binding evidence suitable for diagnostics.
    pub context: DiagnosticContext,
}

impl fmt::Display for RecordingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "render graph recording failed ({:?}): {}",
            self.kind, self.context.detail
        )
    }
}

impl Error for RecordingError {}

/// Result returned by pass recording callbacks and validating command methods.
pub type RecordResult<T = ()> = Result<T, RecordingError>;
