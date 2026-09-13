//! Native GPU ownership plus fixed-artifact Copy, Compute, and Raster execution for Fluxel.
//!
//! This crate exposes no `wgpu-hal` types. It opens one headless DX12 or Vulkan
//! device, owns buffers and 2D textures, and executes deliberately fixed Copy,
//! Compute, and Raster subsets of a portable RenderGraph plan through barriers,
//! one submission, completion, and lease-backed retirement. On Windows with
//! either native graphics feature, the narrow presentation façade owns a surface/swapchain
//! while retaining its window through acquired-image retirement. Narrow immutable buffer and
//! whole RGBA8 texture uploads retain staging and destination storage through
//! completion.
//!
//! Fixed raster recipes are intentionally closed adapter contracts:
//!
//! ```
//! use fluxel_rhi::adapter::fixed_artifacts::RasterKernel;
//!
//! let _identity = RasterKernel::Triangle.portable_identity();
//! ```
//!
//! They are not stable root-facade types:
//!
//! ```compile_fail
//! use fluxel_rhi::RasterKernel;
//! ```

#![deny(missing_docs)]

use core::fmt;
#[allow(unused_imports, reason = "crate-private implementation prelude")]
pub(crate) use std::collections::HashMap;
use std::sync::Arc;

mod execution;
/// Explicitly unstable APIs for closed vertical slices.
///
/// These artifacts are intentionally isolated from the RHI root because they
/// encode the renderer's current fixed recipes rather than a general pipeline
/// contract. They may change or be removed in a future minor release.
mod experimental;

/// Closed adapter-facing contracts used by Fluxel's retained rendering slice.
///
/// These APIs connect sibling workspace crates without exposing native GPU
/// objects. They are a deliberately bounded pre-1.0 contract: minor releases
/// may evolve it with their sibling consumers, but it is not a temporary
/// documentation-hidden proof seam. It is intentionally narrower than the
/// resource contract planned for the next stage and must not be expanded into
/// a general pipeline facade.
pub mod adapter {
    /// Closed renderer-shaped raster recipes and execution objects.
    pub use crate::experimental::fixed_artifacts;

    /// Closed browser WebGL2 execution for the retained scene.
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    pub use crate::experimental::webgl2;

    /// Closed browser WebGPU execution for the retained scene.
    #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
    pub use crate::experimental::webgpu;
}
/// Rendering-owned Windows presentation boundary for the current vertical slice.
#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
pub mod presentation;
mod resource;

pub use execution::{
    ComputeBackend, ComputeObjectProvider, CopyBackend, CopyCommandBuffer, CopyEncoder,
    NativeCompletion, NativeExecutionError, UnsupportedBindings, UnsupportedComputePipeline,
    UnsupportedRasterPipeline, WaitError,
};
#[cfg(feature = "test-support")]
pub use execution::{RasterTextureReadback, readback_exported_raster_texture_for_test};
/// Opaque identity of one physical RenderGraph resource generation.
pub use fluxel_rendergraph::PhysicalResourceIdentity;
#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
pub use presentation::{
    AcquiredSurfaceFrame, PresentationToken, Surface, SurfaceError, SurfaceExtent,
    SurfaceGeneration, SurfaceStatus,
};
/// Uninhabited presentation token used by non-presentable builds to keep portable
/// execution profiles structurally total while rejecting presentation.
#[cfg(not(all(windows, any(feature = "dx12", feature = "vulkan"))))]
pub struct PresentationToken {
    _private: (),
}
pub use resource::{
    Buffer, BufferDescriptor, BufferLease, BufferUploadError, BufferUploadStage,
    ComputeArtifactIdentity, ComputeBindings, ComputeBindingsLease, ComputeCreateError,
    ComputeKernel, ComputePipeline, ComputePipelineLease, IncompleteBufferUpload,
    IncompleteTextureUpload, InvalidBufferUploadReason, InvalidResourceReason,
    InvalidTextureUploadReason, MemoryPolicy, PendingBufferUpload, PendingTextureUpload,
    ResourceCreateError, ResourceKind, ResourceLease, Texture, TextureDescriptor, TextureLease,
    TexturePackBindings, TexturePackBindingsLease, TextureUploadError, TextureUploadStage,
    UploadedBuffer, UploadedTexture,
};

// Private implementation modules use the former short paths. These aliases
// have crate visibility only, so they do not preserve the removed root API.
#[allow(unused_imports, reason = "crate-private implementation prelude")]
pub(crate) use execution::*;
#[allow(unused_imports, reason = "crate-private implementation prelude")]
pub(crate) use resource::*;

/// A native graphics API supported by Fluxel's Windows RHI.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum Backend {
    /// Microsoft Direct3D 12.
    Dx12,
    /// Khronos Vulkan.
    Vulkan,
}

/// The required validation behavior when opening a device.
///
/// `Required` is fail-closed. Because the HAL has no portable proof
/// of validation, private backend-specific probes verify the facility before a
/// device is returned.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum Validation {
    /// Do not request native validation.
    #[default]
    Disabled,
    /// Require a verifiably enabled native validation facility. Vulkan also
    /// requires the validation-features extension used for synchronization
    /// validation.
    Required,
}

/// Choices that affect opening a headless device.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct DeviceOptions {
    /// Backend-native index of the adapter to open; default is zero.
    pub adapter_index: usize,
    /// Native validation policy.
    pub validation: Validation,
}

/// The broad kind of physical device reported by the native driver.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum DeviceKind {
    /// The driver did not classify the device more precisely.
    Other,
    /// An integrated GPU.
    Integrated,
    /// A discrete GPU.
    Discrete,
    /// A virtual GPU.
    Virtual,
    /// A CPU or software implementation.
    Cpu,
}

/// Hardware identity reported by the selected native backend.
///
/// These are driver facts, not a conformance profile. Fluxel performs any
/// profile lowering separately and never masks this value.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub struct HardwareInfo {
    /// The selected backend.
    pub backend: Backend,
    /// Driver-reported adapter name.
    pub name: String,
    /// Backend-specific vendor identifier.
    pub vendor_id: u32,
    /// Backend-specific device identifier.
    pub device_id: u32,
    /// Broad device classification.
    pub kind: DeviceKind,
    /// Backend-specific PCI bus identifier, if reported.
    pub pci_bus_id: String,
    /// Driver name reported by the backend.
    pub driver: String,
    /// Additional driver information reported by the backend.
    pub driver_info: String,
}

/// Raw capability facts from the selected native adapter.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub struct HardwareCapabilities {
    /// Whether `Rgba8Unorm` supports filtering linear texture samples.
    ///
    /// This is an unmodified adapter fact. The fixed linear-clamp raster
    /// artifact rejects creation when it is false rather than assuming that
    /// a sampled `Rgba8Unorm` texture is filterable.
    pub rgba8_unorm_filterable: bool,
    /// Whether `Rgba8UnormSrgb` supports filtering linear texture samples.
    ///
    /// This is a separate, unmodified adapter fact. In particular, Fluxel
    /// never infers sRGB filterability from the corresponding UNORM fact.
    pub rgba8_unorm_srgb_filterable: bool,
    /// Whether the selected adapter exposes `Rgba8Unorm` storage-texture reads.
    pub rgba8_unorm_storage_read: bool,
    /// Whether the selected adapter exposes `Rgba8Unorm` storage-texture writes.
    pub rgba8_unorm_storage_write: bool,
    /// Whether adapter-specific texture format features were actually enabled.
    pub rgba8_unorm_storage_read_enabled: bool,
    /// Maximum two-dimensional texture extent reported by the backend.
    pub max_texture_dimension_2d: u32,
    /// Maximum bind groups reported by the backend.
    pub max_bind_groups: u32,
    /// Minimum uniform-buffer dynamic-offset alignment.
    pub min_uniform_buffer_offset_alignment: u32,
    /// Minimum storage-buffer dynamic-offset alignment.
    pub min_storage_buffer_offset_alignment: u32,
    /// Maximum byte size of a single storage-buffer binding.
    pub max_storage_buffer_binding_size: u64,
    /// Maximum workgroup count for each dimension of a compute dispatch.
    ///
    /// A zero component means compute dispatches are unavailable and is
    /// intentionally fail-closed by [`ComputeBackend`].
    pub max_compute_workgroups_per_dimension: [u32; 3],
    /// Maximum local workgroup size for each dimension of a compute shader.
    ///
    /// A zero component means the fixed compute artifacts are unavailable and
    /// is rejected before the private native shader boundary is entered.
    pub max_compute_workgroup_size: [u32; 3],
    /// Maximum total invocations in one compute shader workgroup.
    ///
    /// Zero is treated as unavailable and rejected before native shader
    /// creation.
    pub max_compute_invocations_per_workgroup: u32,
}

/// A headless native device that owns resources and serial graphics-queue execution.
pub struct Device {
    #[allow(
        dead_code,
        reason = "native ownership and the queue are retained for safe shutdown and later slices"
    )]
    inner: Arc<imp::OpenedDevice>,
    identity: fluxel_rendergraph::DeviceIdentity,
    hardware: HardwareInfo,
    capabilities: HardwareCapabilities,
}

impl Clone for Device {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            identity: self.identity,
            hardware: self.hardware.clone(),
            capabilities: self.capabilities,
        }
    }
}

impl Device {
    /// Opens one explicitly selected backend and adapter for headless use.
    ///
    /// This method enters the private HAL boundary, creates no surface, and
    /// exposes no native handles. Copy submission and fixed-artifact compute
    /// execution are available through [`CopyBackend`] and [`ComputeBackend`].
    pub fn open(backend: Backend, options: DeviceOptions) -> Result<Self, OpenError> {
        #[cfg(all(windows, feature = "test-support"))]
        imp::initialize_validation_capture();
        let opened = imp::open(backend, options)?;
        Ok(Self {
            hardware: opened.hardware.clone(),
            capabilities: opened.capabilities,
            inner: Arc::new(opened),
            identity: fluxel_rendergraph::DeviceIdentity::new(next_identity()),
        })
    }

    /// Returns unmodified identity facts for the selected adapter.
    pub fn hardware(&self) -> &HardwareInfo {
        &self.hardware
    }

    /// Returns unmodified capability facts for the selected adapter.
    pub fn capabilities(&self) -> HardwareCapabilities {
        self.capabilities
    }

    /// Returns the identity used to reject resources from another device.
    pub fn identity(&self) -> fluxel_rendergraph::DeviceIdentity {
        self.identity
    }
}

fn next_identity() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    // This counter supplies process-local uniqueness only. Relaxed ordering is
    // sufficient because identity does not publish native objects or synchronize
    // their lifetime; ownership and queue locks provide those guarantees.
    NEXT.fetch_add(1, Ordering::Relaxed)
}

impl fmt::Debug for Device {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Device")
            .field("hardware", &self.hardware)
            .field("capabilities", &self.capabilities)
            .finish_non_exhaustive()
    }
}

/// Why opening a requested native device was not possible.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum OpenError {
    /// This crate currently implements native opening only on Windows.
    PlatformUnsupported {
        /// The requested backend.
        backend: Backend,
    },
    /// The selected backend was not compiled into this build.
    BackendDisabled {
        /// The requested backend.
        backend: Backend,
    },
    /// The requested adapter index was not returned by the native driver.
    AdapterUnavailable {
        /// The requested backend.
        backend: Backend,
        /// The requested index.
        adapter_index: usize,
        /// Number of adapters found.
        available_adapters: usize,
    },
    /// The adapter cannot meet the baseline limits used to open a device.
    RequiredLimitsUnavailable {
        /// The requested backend.
        backend: Backend,
    },
    /// The requested validation mode cannot be positively verified.
    ValidationUnavailable {
        /// The requested backend.
        backend: Backend,
    },
    /// The native loader, driver, or device open operation failed.
    NativeUnavailable {
        /// The requested backend.
        backend: Backend,
        /// A diagnostic captured at the FFI boundary.
        reason: String,
    },
}

impl fmt::Display for OpenError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PlatformUnsupported { backend } => {
                write!(formatter, "{backend:?} is only supported on Windows")
            }
            Self::BackendDisabled { backend } => write!(formatter, "{backend:?} is disabled"),
            Self::AdapterUnavailable {
                backend,
                adapter_index,
                available_adapters,
            } => write!(
                formatter,
                "{backend:?} adapter {adapter_index} is unavailable ({available_adapters} adapters found)"
            ),
            Self::RequiredLimitsUnavailable { backend } => {
                write!(
                    formatter,
                    "{backend:?} cannot meet the baseline device limits"
                )
            }
            Self::ValidationUnavailable { backend } => write!(
                formatter,
                "{backend:?} validation cannot be positively verified during bootstrap"
            ),
            Self::NativeUnavailable { backend, reason } => {
                write!(formatter, "{backend:?} is unavailable: {reason}")
            }
        }
    }
}

impl std::error::Error for OpenError {}

#[cfg(test)]
mod tests;

#[cfg(feature = "test-support")]
#[doc(hidden)]
pub mod test_support;

mod imp;
