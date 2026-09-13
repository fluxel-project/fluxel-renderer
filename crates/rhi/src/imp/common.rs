//! Shared native ownership types and queue synchronization state.

use super::*;

pub(crate) struct OpenedDevice {
    // Drop order: queue, device, adapter, then instance. This retains DX12
    // adapter-owned driver state until all dependent values are gone.
    #[allow(dead_code)]
    pub(crate) native: NativeDevice,
    // All operations on the one native queue are externally synchronized by
    // this lock. In particular, wgpu-hal requires `submit`, `wait_for_idle`,
    // and future presentation operations on a queue to be mutually exclusive.
    // The lock belongs to `OpenedDevice`, so cloned `Device`s and separately
    // constructed execution backends cannot accidentally use the queue
    // concurrently.
    pub(crate) queue_operations: Mutex<()>,
    pub(crate) hardware: HardwareInfo,
    pub(crate) capabilities: HardwareCapabilities,
}

pub(crate) fn lock_queue_operations(queue_operations: &Mutex<()>) -> MutexGuard<'_, ()> {
    // A panic while using the queue does not make later mutual exclusion less
    // necessary. Recover the guard rather than allowing a poisoned mutex to
    // turn into unsynchronized access (or an avoidable safe-API failure).
    queue_operations
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[allow(
    dead_code,
    clippy::large_enum_variant,
    reason = "native objects are retained for safe shutdown and remain inline because device creation is not a hot path"
)]
pub(crate) enum NativeDevice {
    #[cfg(feature = "dx12")]
    Dx12 {
        queue: wgpu_hal::dx12::Queue,
        device: wgpu_hal::dx12::Device,
        adapter: wgpu_hal::dx12::Adapter,
        instance: wgpu_hal::dx12::Instance,
    },
    #[cfg(feature = "vulkan")]
    Vulkan {
        queue: wgpu_hal::vulkan::Queue,
        device: wgpu_hal::vulkan::Device,
        adapter: wgpu_hal::vulkan::Adapter,
        instance: wgpu_hal::vulkan::Instance,
    },
}

pub(crate) enum NativeBuffer {
    #[cfg(feature = "dx12")]
    Dx12(wgpu_hal::dx12::Buffer),
    #[cfg(feature = "vulkan")]
    Vulkan(wgpu_hal::vulkan::Buffer),
}

pub(crate) enum NativeTexture {
    #[cfg(feature = "dx12")]
    Dx12(wgpu_hal::dx12::Texture),
    #[cfg(feature = "vulkan")]
    Vulkan(wgpu_hal::vulkan::Texture),
}

/// A native surface is deliberately kept separate from owned textures: an
/// acquired swapchain image is returned to the surface by present/discard,
/// never destroyed through `Device::destroy_texture`.
#[allow(
    clippy::large_enum_variant,
    reason = "the private surface owns a complete native swapchain state"
)]
pub(crate) enum NativeSurface {
    #[cfg(feature = "dx12")]
    Dx12(wgpu_hal::dx12::Surface),
    #[cfg(feature = "vulkan")]
    Vulkan(wgpu_hal::vulkan::Surface),
}

pub(crate) struct OwnedBuffer {
    pub(crate) native: Option<NativeBuffer>,
    pub(crate) owner: Arc<OpenedDevice>,
    pub(crate) size: u64,
    pub(crate) allowed_usage: BufferUsage,
}

pub(crate) struct OwnedTexture {
    pub(crate) native: Option<NativeTexture>,
    pub(crate) owner: Arc<OpenedDevice>,
    pub(crate) descriptor: TextureDesc,
    pub(crate) allowed_usage: TextureUsage,
}

pub(crate) struct CopyEncoder {
    pub(crate) native: Option<NativeEncoder>,
    pub(crate) owner: Arc<OpenedDevice>,
    // RenderGraph states buffer ranges independently, whereas this minimal
    // HAL path lowers a buffer barrier over the complete allocation. Keep the
    // actual whole-buffer state for this encoder so a later range-local
    // transition that reaches the already-current state becomes a same-state
    // memory dependency rather than an invalid stale DX12 transition.
    pub(crate) buffer_states: HashMap<usize, ResourceAccessState>,
    // This slice lowers texture barriers over one native allocation. Mirror
    // the buffer tracker so stale `before` states fail before HAL and raster
    // attachment state can be checked at the native boundary.
    pub(crate) texture_states: HashMap<TextureStateKey, ResourceAccessState>,
    // A render attachment view must outlive its active pass. It is created at
    // begin and destroyed only after end_render_pass.
    pub(crate) active_render_view: Option<NativeRenderView>,
    pub(crate) active_raster_pipeline: Option<usize>,
    // Ended passes are still referenced by the closed command buffer. Keep
    // their views until the buffer is discarded or terminal completion.
    pub(crate) render_views: Vec<NativeRenderView>,
}

#[allow(
    clippy::large_enum_variant,
    reason = "native encoders are setup objects retained once per in-flight submission, not a dense collection"
)]
pub(crate) enum NativeEncoder {
    #[cfg(feature = "dx12")]
    Dx12(wgpu_hal::dx12::CommandEncoder),
    #[cfg(feature = "vulkan")]
    Vulkan(wgpu_hal::vulkan::CommandEncoder),
}

pub(crate) enum NativeRenderView {
    #[cfg(feature = "dx12")]
    Dx12 {
        color: wgpu_hal::dx12::TextureView,
        depth: Option<wgpu_hal::dx12::TextureView>,
    },
    #[cfg(feature = "vulkan")]
    Vulkan {
        color: wgpu_hal::vulkan::TextureView,
        depth: Option<wgpu_hal::vulkan::TextureView>,
    },
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct TextureStateKey {
    pub(crate) allocation: usize,
    pub(crate) mip_level: u32,
    pub(crate) array_layer: u32,
    pub(crate) aspect: TextureAspect,
}

pub(crate) fn destroy_render_views(owner: &OpenedDevice, views: Vec<NativeRenderView>) {
    // SAFETY: callers invoke this only after discarding an unsubmitted encoder
    // or after the command buffer's terminal completion/reset. Each temporary
    // view was created by this retained device and is consumed exactly once.
    for view in views {
        match (&owner.native, view) {
            #[cfg(feature = "dx12")]
            (NativeDevice::Dx12 { device, .. }, NativeRenderView::Dx12 { color, depth }) => unsafe {
                // SAFETY: the retained device created this uniquely consumed
                // view, and its encoder was discarded or terminally reset.
                device.destroy_texture_view(color);
                if let Some(depth) = depth {
                    device.destroy_texture_view(depth);
                }
            },
            #[cfg(feature = "vulkan")]
            (NativeDevice::Vulkan { device, .. }, NativeRenderView::Vulkan { color, depth }) => unsafe {
                // SAFETY: same unique ownership and terminal-use proof as DX12.
                device.destroy_texture_view(color);
                if let Some(depth) = depth {
                    device.destroy_texture_view(depth);
                }
            },
            _ => unreachable!("render view and device backend always match"),
        }
    }
}

/// One fixed, device-affine compute pipeline. This remains private because the
/// 0.1.3 slice deliberately does not expose arbitrary shader compilation.
#[derive(Clone)]
pub(crate) struct NativeComputePipeline(pub(crate) Arc<NativeComputePipelineShared>);

pub(crate) struct NativeComputePipelineShared {
    pub(crate) native: Option<NativeComputePipelineInner>,
    pub(crate) owner: Arc<OpenedDevice>,
}

#[allow(
    clippy::large_enum_variant,
    reason = "a pipeline owns its backend objects and is an infrequent setup value"
)]
pub(crate) enum NativeComputePipelineInner {
    #[cfg(feature = "dx12")]
    Dx12 {
        shader: wgpu_hal::dx12::ShaderModule,
        bind_group_layout: wgpu_hal::dx12::BindGroupLayout,
        pipeline_layout: wgpu_hal::dx12::PipelineLayout,
        pipeline: wgpu_hal::dx12::ComputePipeline,
    },
    #[cfg(feature = "vulkan")]
    Vulkan {
        shader: wgpu_hal::vulkan::ShaderModule,
        bind_group_layout: wgpu_hal::vulkan::BindGroupLayout,
        pipeline_layout: wgpu_hal::vulkan::PipelineLayout,
        pipeline: wgpu_hal::vulkan::ComputePipeline,
    },
}

/// One fixed RW-storage-buffer bind group. `pipeline` keeps its layout alive
/// until this group has been destroyed.
pub(crate) struct NativeComputeBindings {
    pub(crate) native: Option<NativeComputeBindingsInner>,
    pub(crate) pipeline: NativeComputePipeline,
}

/// Private holder for the closed X01 bind group.  The concrete group is added
/// below with backend-specific ownership so it cannot outlive its pipeline.
pub(crate) struct NativeTexturePackBindings {
    pub(crate) native: Option<NativeTexturePackBindingsInner>,
    pub(crate) pipeline: NativeComputePipeline,
}

pub(crate) enum NativeTexturePackBindingsInner {
    #[cfg(feature = "dx12")]
    Dx12 {
        group: wgpu_hal::dx12::BindGroup,
        view: wgpu_hal::dx12::TextureView,
    },
    #[cfg(feature = "vulkan")]
    Vulkan {
        group: wgpu_hal::vulkan::BindGroup,
        view: wgpu_hal::vulkan::TextureView,
    },
}

/// Pipeline setup stages exposed only to the safe RHI mapping layer. Keeping
/// this private prevents backend or compiler types from entering the public
/// contract while preserving a stable failure category for callers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ComputePipelineCreateError {
    ShaderValidation(String),
    ShaderCompilation(String),
    NativeObjectCreation(String),
}

/// Fixed Rgba8Unorm graphics pipeline.  It deliberately has no public raw
/// handle: all command recording goes through the checked RHI layer.
#[derive(Clone)]
pub(crate) struct NativeRasterPipeline(pub(crate) Arc<NativeRasterPipelineShared>);

pub(crate) struct NativeRasterPipelineShared {
    pub(crate) native: Option<NativeRasterPipelineInner>,
    pub(crate) owner: Arc<OpenedDevice>,
    // The safe layer selects the closed recipe, but this private copy lets
    // every native binding boundary reject a mismatched opaque pipeline too.
    pub(crate) kernel: crate::RasterKernel,
}

#[allow(
    clippy::large_enum_variant,
    reason = "one infrequent pipeline owns its backend objects"
)]
pub(crate) enum NativeRasterPipelineInner {
    #[cfg(feature = "dx12")]
    Dx12 {
        vertex_shader: wgpu_hal::dx12::ShaderModule,
        fragment_shader: wgpu_hal::dx12::ShaderModule,
        bind_group_layout: Option<wgpu_hal::dx12::BindGroupLayout>,
        pipeline_layout: wgpu_hal::dx12::PipelineLayout,
        pipeline: wgpu_hal::dx12::RenderPipeline,
        depth_pipeline: wgpu_hal::dx12::RenderPipeline,
    },
    #[cfg(feature = "vulkan")]
    Vulkan {
        vertex_shader: wgpu_hal::vulkan::ShaderModule,
        fragment_shader: wgpu_hal::vulkan::ShaderModule,
        bind_group_layout: Option<wgpu_hal::vulkan::BindGroupLayout>,
        pipeline_layout: wgpu_hal::vulkan::PipelineLayout,
        pipeline: wgpu_hal::vulkan::RenderPipeline,
        depth_pipeline: wgpu_hal::vulkan::RenderPipeline,
    },
}

/// One closed group-0/binding-0 frame-uniform bind group. Its pipeline keeps
/// the matching layout alive until the group is destroyed.
pub(crate) struct NativeRasterUniformBindings {
    pub(crate) native: Option<NativeRasterUniformBindingsInner>,
    pub(crate) pipeline: NativeRasterPipeline,
    // Only the normal-Lambert closed artifact populates this. Rechecking the
    // opaque generations and exact whole-stream ranges at the unsafe vertex
    // edge prevents a safe caller from exchanging position and normal roles.
    pub(crate) expected_normal_vertex_streams: Option<(
        fluxel_rendergraph::PhysicalResourceIdentity,
        u64,
        fluxel_rendergraph::PhysicalResourceIdentity,
        u64,
    )>,
    /// Vertex-color bindings retain the two stream identities and exact ranges
    /// for a final check at the unsafe vertex command boundary.
    pub(crate) expected_vertex_color_streams: Option<(
        fluxel_rendergraph::PhysicalResourceIdentity,
        u64,
        fluxel_rendergraph::PhysicalResourceIdentity,
        u64,
    )>,
}

/// Private holder for the closed textured raster bind group and texture view.
pub(crate) struct NativeRasterTextureBindings {
    pub(crate) native: Option<NativeRasterTextureBindingsInner>,
    pub(crate) pipeline: NativeRasterPipeline,
    // Only the explicit-UV closed artifact populates this. The opaque
    // generation identities and exact whole-stream ranges are rechecked at
    // the unsafe vertex binding edge.
    pub(crate) expected_uv_vertex_streams: Option<(
        fluxel_rendergraph::PhysicalResourceIdentity,
        u64,
        fluxel_rendergraph::PhysicalResourceIdentity,
        u64,
    )>,
}

pub(crate) enum NativeRasterTextureBindingsInner {
    #[cfg(feature = "dx12")]
    Dx12 {
        group: wgpu_hal::dx12::BindGroup,
        view: wgpu_hal::dx12::TextureView,
        // The sampler is private to the closed linear-clamp recipe. It must
        // outlive its bind group, and is destroyed after that group below.
        sampler: Option<wgpu_hal::dx12::Sampler>,
    },
    #[cfg(feature = "vulkan")]
    Vulkan {
        group: wgpu_hal::vulkan::BindGroup,
        view: wgpu_hal::vulkan::TextureView,
        sampler: Option<wgpu_hal::vulkan::Sampler>,
    },
}

#[allow(
    dead_code,
    reason = "constructed only by the camera/material native binding path"
)]
pub(crate) enum NativeRasterUniformBindingsInner {
    #[cfg(feature = "dx12")]
    Dx12(wgpu_hal::dx12::BindGroup),
    #[cfg(feature = "vulkan")]
    Vulkan(wgpu_hal::vulkan::BindGroup),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum RasterPipelineCreateError {
    ShaderValidation(String),
    ShaderCompilation(String),
    NativeObjectCreation(String),
}

pub(crate) enum NativeComputeBindingsInner {
    #[cfg(feature = "dx12")]
    Dx12(wgpu_hal::dx12::BindGroup),
    #[cfg(feature = "vulkan")]
    Vulkan(wgpu_hal::vulkan::BindGroup),
    /// Closed storage-texture groups retain their sole view until group teardown.
    #[cfg(feature = "dx12")]
    Dx12Texture {
        group: wgpu_hal::dx12::BindGroup,
        view: wgpu_hal::dx12::TextureView,
    },
    #[cfg(feature = "vulkan")]
    VulkanTexture {
        group: wgpu_hal::vulkan::BindGroup,
        view: wgpu_hal::vulkan::TextureView,
    },
}

pub(crate) struct CopyCommandBuffer {
    pub(crate) native: Option<NativeFinished>,
}

#[allow(
    clippy::large_enum_variant,
    reason = "the finished command buffer uniquely owns its native encoder until submission"
)]
pub(crate) enum NativeFinished {
    #[cfg(feature = "dx12")]
    Dx12 {
        owner: Arc<OpenedDevice>,
        encoder: wgpu_hal::dx12::CommandEncoder,
        command_buffer: wgpu_hal::dx12::CommandBuffer,
        render_views: Vec<NativeRenderView>,
    },
    #[cfg(feature = "vulkan")]
    Vulkan {
        owner: Arc<OpenedDevice>,
        encoder: wgpu_hal::vulkan::CommandEncoder,
        command_buffer: wgpu_hal::vulkan::CommandBuffer,
        render_views: Vec<NativeRenderView>,
    },
}

#[derive(Clone)]
pub(crate) struct NativeCompletion(pub(crate) Arc<std::sync::Mutex<NativeSubmission>>);

/// Private accounting for every acquired or accepted-present surface image.
///
/// A ticket is created only by the surface façade and is moved, never copied,
/// through its native token and (after accepted present) completion bundle.
/// The count is deliberately not a public frame/image index: it only answers
/// whether teardown may touch the swapchain at all.
pub(crate) struct NativePresentationTickets {
    live: std::sync::atomic::AtomicUsize,
    acquired: std::sync::atomic::AtomicBool,
    quarantined: std::sync::atomic::AtomicBool,
    capacity: usize,
}

impl NativePresentationTickets {
    pub(crate) fn new(capacity: usize) -> std::sync::Arc<Self> {
        assert!(capacity > 0, "presentation ticket capacity must be nonzero");
        std::sync::Arc::new(Self {
            live: std::sync::atomic::AtomicUsize::new(0),
            acquired: std::sync::atomic::AtomicBool::new(false),
            quarantined: std::sync::atomic::AtomicBool::new(false),
            capacity,
        })
    }

    /// Reserves both the single HAL acquisition and one in-flight retirement
    /// slot. The acquisition lease is dropped at present/discard; the frame
    /// lease survives in the completion bundle until retirement.
    pub(crate) fn try_acquire(
        self: &std::sync::Arc<Self>,
    ) -> Option<(NativeAcquireLease, NativePresentationLease)> {
        if self.quarantined.load(std::sync::atomic::Ordering::Acquire)
            || self
                .acquired
                .compare_exchange(
                    false,
                    true,
                    std::sync::atomic::Ordering::AcqRel,
                    std::sync::atomic::Ordering::Acquire,
                )
                .is_err()
        {
            return None;
        }
        let frame = self
            .live
            .fetch_update(
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
                |live| (live < self.capacity).then_some(live + 1),
            )
            .ok()
            .map(|_| NativePresentationLease {
                tickets: std::sync::Arc::clone(self),
            });
        match frame {
            Some(frame) => Some((
                NativeAcquireLease {
                    tickets: std::sync::Arc::clone(self),
                },
                frame,
            )),
            None => {
                self.acquired
                    .store(false, std::sync::atomic::Ordering::Release);
                None
            }
        }
    }

    pub(crate) fn any_live(&self) -> bool {
        self.live.load(std::sync::atomic::Ordering::Acquire) != 0
    }

    pub(crate) fn poisoned(&self) -> bool {
        self.quarantined.load(std::sync::atomic::Ordering::Acquire)
    }

    pub(crate) fn quarantine(&self) {
        self.quarantined
            .store(true, std::sync::atomic::Ordering::Release);
    }

    #[cfg(test)]
    pub(crate) fn live_count(&self) -> usize {
        self.live.load(std::sync::atomic::Ordering::Acquire)
    }

    #[cfg(test)]
    pub(crate) const fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Proves that exactly one HAL surface texture is currently acquired.
pub(crate) struct NativeAcquireLease {
    tickets: std::sync::Arc<NativePresentationTickets>,
}

impl Drop for NativeAcquireLease {
    fn drop(&mut self) {
        self.tickets
            .acquired
            .store(false, std::sync::atomic::Ordering::Release);
    }
}

/// The post-present part of an acquired-frame gate. It deliberately contains
/// no window or surface ownership, so ordinary completions remain Send/Sync.
/// It is released only after the submission has destroyed derived views.
pub(crate) struct NativePresentationLease {
    tickets: std::sync::Arc<NativePresentationTickets>,
}

/// One Vulkan surface's timeline fence. HAL records the fence/value pair in
/// swapchain metadata at submit, so every later acquire for that surface must
/// use this same fence and monotonically increasing value.
#[cfg(feature = "vulkan")]
pub(crate) struct VulkanPresentationSync {
    owner: Arc<OpenedDevice>,
    fence: Option<wgpu_hal::vulkan::Fence>,
    next_signal_value: std::sync::atomic::AtomicU64,
}

#[cfg(feature = "vulkan")]
impl VulkanPresentationSync {
    pub(crate) fn new(owner: Arc<OpenedDevice>) -> Result<Arc<Self>, String> {
        let NativeDevice::Vulkan { device, .. } = &owner.native else {
            return Err("Vulkan presentation sync received a non-Vulkan device".into());
        };
        let fence = unsafe { device.create_fence() }.map_err(|error| error.to_string())?;
        Ok(Arc::new(Self {
            owner,
            fence: Some(fence),
            next_signal_value: std::sync::atomic::AtomicU64::new(1),
        }))
    }

    pub(crate) fn fence(&self) -> Result<&wgpu_hal::vulkan::Fence, String> {
        self.fence
            .as_ref()
            .ok_or_else(|| "Vulkan presentation fence was already destroyed".into())
    }

    pub(crate) fn reserve_signal_value(&self) -> Result<u64, String> {
        self.next_signal_value
            .fetch_update(
                std::sync::atomic::Ordering::AcqRel,
                std::sync::atomic::Ordering::Acquire,
                |value| value.checked_add(1),
            )
            .map_err(|_| "Vulkan presentation fence value exhausted".to_owned())
    }
}

#[cfg(feature = "vulkan")]
impl Drop for VulkanPresentationSync {
    fn drop(&mut self) {
        if let (NativeDevice::Vulkan { device, .. }, Some(fence)) =
            (&self.owner.native, self.fence.take())
        {
            // No token/completion can retain this final Arc. Ordinary teardown
            // waits for idle; accepted-unknown retains the Arc indefinitely.
            unsafe { device.destroy_fence(fence) };
        }
    }
}

#[cfg(feature = "vulkan")]
pub(crate) enum NativeVulkanFence {
    Owned(wgpu_hal::vulkan::Fence),
    Presentation {
        sync: Arc<VulkanPresentationSync>,
        value: u64,
    },
}

impl Drop for NativePresentationLease {
    fn drop(&mut self) {
        let prior = self
            .tickets
            .live
            .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
        debug_assert!(prior > 0, "presentation ticket accounting underflow");
    }
}

impl NativePresentationLease {
    pub(crate) fn quarantine_surface(&self) {
        self.tickets.quarantine();
    }
}

#[allow(
    clippy::large_enum_variant,
    reason = "one native submission bundle must retain the concrete encoder and command buffer together"
)]
pub(crate) enum NativeSubmission {
    #[cfg(feature = "dx12")]
    Dx12 {
        owner: Arc<OpenedDevice>,
        encoder: Option<wgpu_hal::dx12::CommandEncoder>,
        command_buffer: Option<wgpu_hal::dx12::CommandBuffer>,
        fence: Option<wgpu_hal::dx12::Fence>,
        leases: Vec<ResourceLease>,
        staging_buffers: Vec<OwnedBuffer>,
        render_views: Vec<NativeRenderView>,
        presentation_lease: Option<NativePresentationLease>,
        /// Test-only completion hold consumed only after a successful DX12
        /// present. It is absent for copy/upload and failed present paths.
        presentation_completion_hold: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
        failure: Option<CompletionFailure>,
    },
    #[cfg(feature = "vulkan")]
    Vulkan {
        owner: Arc<OpenedDevice>,
        encoder: Option<wgpu_hal::vulkan::CommandEncoder>,
        command_buffer: Option<wgpu_hal::vulkan::CommandBuffer>,
        fence: Option<NativeVulkanFence>,
        leases: Vec<ResourceLease>,
        staging_buffers: Vec<OwnedBuffer>,
        render_views: Vec<NativeRenderView>,
        presentation_lease: Option<NativePresentationLease>,
        failure: Option<CompletionFailure>,
    },
    TerminalFailure(CompletionFailure),
}
