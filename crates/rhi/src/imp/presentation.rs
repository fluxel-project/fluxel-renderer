//! Private surface creation, acquisition, and presentation lowering.
//!
//! Surface objects never cross this module boundary.  A graph sees only an
//! owned texture wrapper and an opaque one-shot token; the token retains the
//! actual acquired swapchain image until it is consumed by `present` or
//! discarded on drop by the safe façade.

use super::*;

pub(crate) const MAXIMUM_FRAME_LATENCY: u32 = 2;
pub(crate) const PRESENTABLE_IMAGE_COUNT: usize = MAXIMUM_FRAME_LATENCY as usize + 1;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use wgpu_hal::Surface as _;

pub(crate) struct NativePresentationToken {
    pub(crate) owner: Arc<OpenedDevice>,
    pub(crate) surface: Arc<Mutex<NativeSurface>>,
    pub(crate) acquired: Option<NativeAcquiredSurfaceTexture>,
    pub(crate) fence: Option<NativeSurfaceFence>,
    /// Enforces HAL's one-acquired-texture rule until present/discard.
    pub(crate) acquire_lease: Option<NativeAcquireLease>,
    /// Exists from acquire until discard or retirement of an accepted bundle.
    /// It cannot be forged by safe callers and release is completion-driven.
    pub(crate) presentation_ticket: Option<NativePresentationLease>,
    /// Type-erased `Arc<W>` from the public façade. It is intentionally held
    /// by the token because accepted-unknown quarantine can outlive Surface.
    pub(crate) _window: Arc<dyn std::any::Any>,
}

impl Drop for NativePresentationToken {
    fn drop(&mut self) {
        if let Some(acquired) = self.acquired.take() {
            #[cfg(feature = "vulkan")]
            if matches!(acquired, NativeAcquiredSurfaceTexture::Vulkan(_)) {
                // wgpu-hal 30's Vulkan discard is a documented no-op. Reusing
                // this surface would reuse the acquired image's semaphores
                // without a present, violating the HAL acquire contract. Keep
                // every owner and both gates alive for process lifetime and
                // make all later façade operations refuse this surface.
                self.presentation_ticket
                    .as_ref()
                    .expect("acquired token owns a retirement gate")
                    .quarantine_surface();
                let retained = (
                    Arc::clone(&self.owner),
                    Arc::clone(&self.surface),
                    acquired,
                    self.fence.take(),
                    self.acquire_lease.take(),
                    self.presentation_ticket.take(),
                    Arc::clone(&self._window),
                );
                std::mem::forget(retained);
                return;
            }
            let mut surface = self
                .surface
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            match (&mut *surface, acquired) {
                #[cfg(feature = "dx12")]
                (NativeSurface::Dx12(surface), NativeAcquiredSurfaceTexture::Dx12(texture)) => unsafe {
                    // SAFETY: an unconsumed token owns the sole acquired image.
                    surface.discard_texture(texture);
                },
                #[cfg(feature = "vulkan")]
                (NativeSurface::Vulkan(surface), NativeAcquiredSurfaceTexture::Vulkan(texture)) => unsafe {
                    // SAFETY: an unconsumed token owns the sole acquired image.
                    surface.discard_texture(texture);
                },
                _ => unreachable!("native surface/token backend mismatch"),
            }
        }
        #[cfg(feature = "dx12")]
        if let Some(NativeSurfaceFence::Dx12(fence)) = self.fence.take() {
            if let NativeDevice::Dx12 { device, .. } = &self.owner.native {
                // No submit can still reference this fence when its token drops.
                unsafe { device.destroy_fence(fence) };
            }
        }
        #[cfg(feature = "vulkan")]
        // Vulkan surface acquisitions borrow their surface-scoped timeline
        // fence. Dropping this token releases only the Arc; the final sync
        // owner destroys the fence after known teardown or quarantines it.
        drop(self.acquire_lease.take());
        drop(self.presentation_ticket.take());
    }
}

pub(crate) enum NativeAcquiredSurfaceTexture {
    #[cfg(feature = "dx12")]
    Dx12(wgpu_hal::dx12::Texture),
    #[cfg(feature = "vulkan")]
    Vulkan(wgpu_hal::vulkan::SurfaceTexture),
}

pub(crate) enum NativeSurfaceFence {
    #[cfg(feature = "dx12")]
    Dx12(wgpu_hal::dx12::Fence),
    #[cfg(feature = "vulkan")]
    VulkanPresentation {
        sync: Arc<VulkanPresentationSync>,
        value: u64,
    },
}

#[cfg(feature = "vulkan")]
pub(crate) fn create_vulkan_presentation_sync(
    owner: &Arc<OpenedDevice>,
    surface: &Arc<Mutex<NativeSurface>>,
) -> Result<Option<Arc<VulkanPresentationSync>>, String> {
    let native = surface.lock().map_err(|_| "surface lock poisoned")?;
    match (&owner.native, &*native) {
        (NativeDevice::Vulkan { .. }, NativeSurface::Vulkan(_)) => {
            VulkanPresentationSync::new(Arc::clone(owner)).map(Some)
        }
        _ => Ok(None),
    }
}

pub(crate) fn open_surface(
    backend: Backend,
    options: DeviceOptions,
    display: RawDisplayHandle,
    window: RawWindowHandle,
) -> Result<(OpenedDevice, NativeSurface), OpenError> {
    match backend {
        Backend::Dx12 => open_dx12_surface(options, display, window),
        Backend::Vulkan => open_vulkan_surface(options, display, window),
    }
}

pub(crate) fn open_dx12_surface(
    options: DeviceOptions,
    display: RawDisplayHandle,
    window: RawWindowHandle,
) -> Result<(OpenedDevice, NativeSurface), OpenError> {
    #[cfg(feature = "dx12")]
    {
        let descriptor = instance_descriptor(options.validation);
        // SAFETY: raw handles are borrowed from the caller's live window and
        // the returned safe surface lifetime is tied to that window by the
        // public façade.
        let instance = unsafe { wgpu_hal::dx12::Instance::init(&descriptor) }
            .map_err(|e| native_error(Backend::Dx12, e))?;
        // SAFETY: the instance and native window handles remain live while
        // surface creation performs no GPU work.
        let surface = unsafe { instance.create_surface(display, window) }
            .map_err(|e| native_error(Backend::Dx12, e))?;
        // SAFETY: the surface belongs to this live instance and constrains
        // enumeration to adapters which can present to this window.
        let adapters = unsafe { instance.enumerate_adapters(Some(&surface)) };
        let available_adapters = adapters.len();
        let exposed = adapters.into_iter().nth(options.adapter_index).ok_or(
            OpenError::AdapterUnavailable {
                backend: Backend::Dx12,
                adapter_index: options.adapter_index,
                available_adapters,
            },
        )?;
        // SAFETY: `surface` was created by this still-live `instance` from the
        // caller's live window handles. No configuration, acquisition, or
        // destruction can race this read-only compatibility query.
        if unsafe { exposed.adapter.surface_capabilities(&surface) }.is_none() {
            return Err(OpenError::NativeUnavailable {
                backend: Backend::Dx12,
                reason: "selected adapter cannot present to the supplied window".into(),
            });
        }
        let hardware = hardware(Backend::Dx12, &exposed.info);
        let rgba8_unorm_filterable = rgba8_unorm_filterable(&exposed.adapter);
        let rgba8_unorm_srgb_filterable = rgba8_unorm_srgb_filterable(&exposed.adapter);
        let rgba8_unorm_storage_read = rgba8_unorm_storage_read(&exposed.adapter);
        let rgba8_unorm_storage_write = rgba8_unorm_storage_write(&exposed.adapter);
        // Match headless DX12: adapter-reported typed RGBA8 load has no
        // end-to-end conformance proof, so it remains unavailable.
        let enabled_features = wgt::Features::empty();
        let capabilities = capabilities(
            enabled_features,
            &exposed.capabilities,
            rgba8_unorm_filterable,
            rgba8_unorm_srgb_filterable,
            rgba8_unorm_storage_read,
            rgba8_unorm_storage_write,
        );
        let requested_limits = required_limits(Backend::Dx12, &exposed.capabilities)?;
        let adapter = exposed.adapter;
        // SAFETY: requested limits were checked against this exact compatible
        // adapter; features remain intentionally empty for this fixed slice.
        let wgpu_hal::OpenDevice { device, queue } = unsafe {
            adapter.open(
                enabled_features,
                &requested_limits,
                &wgt::MemoryHints::default(),
            )
        }
        .map_err(|e| native_error(Backend::Dx12, e))?;
        if options.validation == Validation::Required && !dx12_validation_is_enabled(&device) {
            return Err(OpenError::ValidationUnavailable {
                backend: Backend::Dx12,
            });
        }
        Ok((
            OpenedDevice {
                native: NativeDevice::Dx12 {
                    queue,
                    device,
                    adapter,
                    instance,
                },
                queue_operations: Mutex::new(()),
                hardware,
                capabilities,
            },
            NativeSurface::Dx12(surface),
        ))
    }
    #[cfg(not(feature = "dx12"))]
    {
        let _ = (options, display, window);
        Err(OpenError::BackendDisabled {
            backend: Backend::Dx12,
        })
    }
}

pub(crate) fn open_vulkan_surface(
    options: DeviceOptions,
    display: RawDisplayHandle,
    window: RawWindowHandle,
) -> Result<(OpenedDevice, NativeSurface), OpenError> {
    #[cfg(feature = "vulkan")]
    {
        if options.validation == Validation::Required && !vulkan_validation_is_available() {
            return Err(OpenError::ValidationUnavailable {
                backend: Backend::Vulkan,
            });
        }
        let descriptor = instance_descriptor(options.validation);
        let instance = unsafe { wgpu_hal::vulkan::Instance::init(&descriptor) }
            .map_err(|e| native_error(Backend::Vulkan, e))?;
        let surface = unsafe { instance.create_surface(display, window) }
            .map_err(|e| native_error(Backend::Vulkan, e))?;
        let adapters = unsafe { instance.enumerate_adapters(Some(&surface)) };
        let available_adapters = adapters.len();
        let exposed = adapters.into_iter().nth(options.adapter_index).ok_or(
            OpenError::AdapterUnavailable {
                backend: Backend::Vulkan,
                adapter_index: options.adapter_index,
                available_adapters,
            },
        )?;
        if unsafe { exposed.adapter.surface_capabilities(&surface) }.is_none() {
            return Err(OpenError::NativeUnavailable {
                backend: Backend::Vulkan,
                reason: "selected adapter cannot present to the supplied window".into(),
            });
        }
        let hardware = hardware(Backend::Vulkan, &exposed.info);
        let rgba8_unorm_filterable = rgba8_unorm_filterable(&exposed.adapter);
        let rgba8_unorm_srgb_filterable = rgba8_unorm_srgb_filterable(&exposed.adapter);
        let rgba8_unorm_storage_read = rgba8_unorm_storage_read(&exposed.adapter);
        let rgba8_unorm_storage_write = rgba8_unorm_storage_write(&exposed.adapter);
        let enabled_features = storage_texture_features(exposed.features, rgba8_unorm_storage_read);
        let capabilities = capabilities(
            enabled_features,
            &exposed.capabilities,
            rgba8_unorm_filterable,
            rgba8_unorm_srgb_filterable,
            rgba8_unorm_storage_read,
            rgba8_unorm_storage_write,
        );
        let requested_limits = required_limits(Backend::Vulkan, &exposed.capabilities)?;
        let adapter = exposed.adapter;
        let wgpu_hal::OpenDevice { device, queue } = unsafe {
            adapter.open(
                enabled_features,
                &requested_limits,
                &wgt::MemoryHints::default(),
            )
        }
        .map_err(|e| native_error(Backend::Vulkan, e))?;
        Ok((
            OpenedDevice {
                native: NativeDevice::Vulkan {
                    queue,
                    device,
                    adapter,
                    instance,
                },
                queue_operations: Mutex::new(()),
                hardware,
                capabilities,
            },
            NativeSurface::Vulkan(surface),
        ))
    }
    #[cfg(not(feature = "vulkan"))]
    {
        let _ = (options, display, window);
        Err(OpenError::BackendDisabled {
            backend: Backend::Vulkan,
        })
    }
}

pub(crate) fn configure_surface(
    owner: &Arc<OpenedDevice>,
    surface: &Mutex<NativeSurface>,
    width: u32,
    height: u32,
) -> Result<(u32, u32), String> {
    let _guard = lock_queue_operations(&owner.queue_operations);
    let mut surface = surface.lock().map_err(|_| "surface lock poisoned")?;
    match (&owner.native, &mut *surface) {
        #[cfg(feature = "dx12")]
        (
            NativeDevice::Dx12 {
                device, adapter, ..
            },
            NativeSurface::Dx12(surface),
        ) => {
            let capabilities = unsafe { adapter.surface_capabilities(surface) }
                .ok_or_else(|| "DX12 adapter no longer supports this surface".to_owned())?;
            let extent = fixed_surface_extent(&capabilities, width, height, "DX12")?;
            let config = wgpu_hal::SurfaceConfiguration {
                maximum_frame_latency: MAXIMUM_FRAME_LATENCY,
                present_mode: wgt::PresentMode::Fifo,
                composite_alpha_mode: wgt::CompositeAlphaMode::Opaque,
                format: wgt::TextureFormat::Rgba8Unorm,
                color_space: wgt::SurfaceColorSpace::Srgb,
                extent,
                usage: wgt::TextureUses::COLOR_TARGET,
                view_formats: Vec::new(),
            };
            // SAFETY: the safe surface façade serializes configure with acquire
            // and requires prior frame shutdown before reconfiguration.
            unsafe { surface.configure(device, &config) }
                .map_err(|e| e.to_string())
                .map(|()| (extent.width, extent.height))
        }
        #[cfg(feature = "vulkan")]
        (
            NativeDevice::Vulkan {
                device, adapter, ..
            },
            NativeSurface::Vulkan(surface),
        ) => {
            let capabilities = unsafe { adapter.surface_capabilities(surface) }
                .ok_or_else(|| "Vulkan adapter no longer supports this surface".to_owned())?;
            let extent = fixed_surface_extent(&capabilities, width, height, "Vulkan")?;
            let config = wgpu_hal::SurfaceConfiguration {
                maximum_frame_latency: MAXIMUM_FRAME_LATENCY,
                present_mode: wgt::PresentMode::Fifo,
                composite_alpha_mode: wgt::CompositeAlphaMode::Opaque,
                format: wgt::TextureFormat::Rgba8Unorm,
                color_space: wgt::SurfaceColorSpace::Srgb,
                extent: wgt::Extent3d {
                    width: extent.width,
                    height: extent.height,
                    depth_or_array_layers: 1,
                },
                usage: wgt::TextureUses::COLOR_TARGET,
                view_formats: Vec::new(),
            };
            unsafe { surface.configure(device, &config) }
                .map_err(|e| e.to_string())
                .map(|()| (extent.width, extent.height))
        }
        _ => Err("surface and device backend mismatch".into()),
    }
}

pub(super) fn fixed_surface_extent(
    capabilities: &wgpu_hal::SurfaceCapabilities,
    width: u32,
    height: u32,
    backend: &str,
) -> Result<wgt::Extent3d, String> {
    let extent = capabilities.current_extent.unwrap_or(wgt::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    });
    if extent.width == 0 || extent.height == 0 {
        return Err(format!("{backend} surface reported a zero drawable extent"));
    }
    let rgba8 = capabilities
        .formats
        .iter()
        .find(|format| format.format == wgt::TextureFormat::Rgba8Unorm)
        .ok_or_else(|| format!("{backend} surface does not support Rgba8Unorm presentation"))?;
    if !rgba8.color_spaces.contains(wgt::SurfaceColorSpaces::SRGB)
        || !capabilities.present_modes.contains(&wgt::PresentMode::Fifo)
        || !capabilities
            .composite_alpha_modes
            .contains(&wgt::CompositeAlphaMode::Opaque)
        || !capabilities.usage.contains(wgt::TextureUses::COLOR_TARGET)
        || !capabilities
            .maximum_frame_latency
            .contains(&MAXIMUM_FRAME_LATENCY)
    {
        return Err(format!(
            "{backend} surface does not support the fixed presentation contract"
        ));
    }
    Ok(extent)
}

pub(crate) struct NativeSurfaceAcquireRequest {
    pub(crate) window: Arc<dyn std::any::Any>,
    pub(crate) acquire_lease: NativeAcquireLease,
    pub(crate) presentation_ticket: NativePresentationLease,
    #[cfg(feature = "vulkan")]
    pub(crate) vulkan_presentation_sync: Option<Arc<VulkanPresentationSync>>,
    pub(crate) descriptor: TextureDesc,
    pub(crate) allowed_usage: TextureUsage,
}

pub(crate) fn acquire_surface(
    owner: &Arc<OpenedDevice>,
    surface: Arc<Mutex<NativeSurface>>,
    request: NativeSurfaceAcquireRequest,
) -> Result<(OwnedTexture, NativePresentationToken), String> {
    let NativeSurfaceAcquireRequest {
        window,
        acquire_lease,
        presentation_ticket,
        #[cfg(feature = "vulkan")]
        vulkan_presentation_sync,
        descriptor,
        allowed_usage,
    } = request;
    let _guard = lock_queue_operations(&owner.queue_operations);
    let mut guard = surface.lock().map_err(|_| "surface lock poisoned")?;
    match (&owner.native, &mut *guard) {
        #[cfg(feature = "dx12")]
        (NativeDevice::Dx12 { device, .. }, NativeSurface::Dx12(native_surface)) => {
            // SAFETY: this fence is uniquely associated with this one acquired
            // image and is passed to both acquire and the sole submit that may
            // use it.
            let fence = unsafe { device.create_fence() }.map_err(|e| e.to_string())?;
            // SAFETY: the surface is configured and queue serialization keeps
            // acquire ordered with configuration/teardown. Each call owns a
            // distinct private ticket until discard or completion retirement.
            let acquired = match unsafe { native_surface.acquire_texture(None, &fence) } {
                Ok(value) => value.texture,
                Err(error) => {
                    // SAFETY: acquisition failed before any queue submission
                    // can reference this uniquely-created fence; it remains
                    // owned solely by this error path and this device is live.
                    unsafe { device.destroy_fence(fence) };
                    return Err(error.to_string());
                }
            };
            // The graph texture is a distinct HAL wrapper around a cloned COM
            // resource. The token retains the original acquired texture, which
            // is the only value later passed to queue submit/present.
            // SAFETY: the cloned COM resource keeps the underlying DX12 image
            // alive; its format, dimension, extent, mip/layer counts exactly
            // match the fixed configured surface descriptor. The token keeps
            // the original acquired texture, surface, and device owner alive
            // until submit/present, discard, or accepted-unknown quarantine.
            let wrapper = unsafe {
                wgpu_hal::dx12::Device::texture_from_raw(
                    acquired.raw_resource().clone(),
                    wgt::TextureFormat::Rgba8Unorm,
                    wgt::TextureDimension::D2,
                    wgt::Extent3d {
                        width: descriptor.extent.width,
                        height: descriptor.extent.height,
                        depth_or_array_layers: 1,
                    },
                    1,
                    1,
                )
            };
            Ok((
                OwnedTexture {
                    native: Some(NativeTexture::Dx12(wrapper)),
                    owner: Arc::clone(owner),
                    descriptor,
                    allowed_usage,
                },
                NativePresentationToken {
                    owner: Arc::clone(owner),
                    surface: Arc::clone(&surface),
                    acquired: Some(NativeAcquiredSurfaceTexture::Dx12(acquired)),
                    fence: Some(NativeSurfaceFence::Dx12(fence)),
                    acquire_lease: Some(acquire_lease),
                    presentation_ticket: Some(presentation_ticket),
                    _window: window,
                },
            ))
        }
        #[cfg(feature = "vulkan")]
        (NativeDevice::Vulkan { device, .. }, NativeSurface::Vulkan(native_surface)) => {
            let sync = vulkan_presentation_sync.ok_or_else(|| {
                "Vulkan surface is missing its presentation synchronization".to_owned()
            })?;
            let value = sync.reserve_signal_value()?;
            let fence = sync.fence()?;
            let acquired = match unsafe { native_surface.acquire_texture(None, fence) } {
                Ok(value) => value.texture,
                Err(error) => return Err(error.to_string()),
            };
            let raw = unsafe {
                <wgpu_hal::vulkan::SurfaceTexture as std::borrow::Borrow<
                    wgpu_hal::vulkan::Texture,
                >>::borrow(&acquired)
                .raw_handle()
            };
            let hal_descriptor = wgpu_hal::TextureDescriptor {
                label: None,
                size: wgt::Extent3d {
                    width: descriptor.extent.width,
                    height: descriptor.extent.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgt::TextureDimension::D2,
                format: wgt::TextureFormat::Rgba8Unorm,
                usage: wgt::TextureUses::COLOR_TARGET,
                memory_flags: wgpu_hal::MemoryFlags::empty(),
                view_formats: Vec::new(),
            };
            let wrapper = unsafe {
                device.texture_from_raw(
                    raw,
                    &hal_descriptor,
                    // Swapchain images are owned and destroyed by the surface.
                    // A present callback marks this HAL wrapper as borrowed so
                    // `destroy_texture` releases only wrapper bookkeeping and
                    // never calls `vkDestroyImage` on the swapchain image.
                    Some(Box::new(|| {})),
                    wgpu_hal::vulkan::TextureMemory::External,
                )
            };
            Ok((
                OwnedTexture {
                    native: Some(NativeTexture::Vulkan(wrapper)),
                    owner: Arc::clone(owner),
                    descriptor,
                    allowed_usage,
                },
                NativePresentationToken {
                    owner: Arc::clone(owner),
                    surface: Arc::clone(&surface),
                    acquired: Some(NativeAcquiredSurfaceTexture::Vulkan(acquired)),
                    fence: Some(NativeSurfaceFence::VulkanPresentation { sync, value }),
                    acquire_lease: Some(acquire_lease),
                    presentation_ticket: Some(presentation_ticket),
                    _window: window,
                },
            ))
        }
        _ => Err("surface and device backend mismatch".into()),
    }
}

pub(crate) fn discard_presentation(token: NativePresentationToken) {
    drop(token);
}

impl NativePresentationToken {
    fn quarantine_surface(&self) {
        self.presentation_ticket
            .as_ref()
            .expect("an acquired surface token owns its live-frame gate")
            .quarantine_surface();
    }

    /// Moves this ticket into the accepted command bundle after present
    /// has consumed the native image. Window/surface ownership stays here and
    /// is released immediately; only derived render-view retirement remains.
    fn take_completion_lease(&mut self) -> Option<NativePresentationLease> {
        drop(self.acquire_lease.take());
        self.presentation_ticket.take()
    }
}

/// Submits a command buffer and consumes exactly one acquired image.
///
/// After a successful queue submission every subsequent failure is returned as
/// completion state, never as a rejection, so graph leases remain quarantined.
pub(crate) fn submit_presented(
    buffer: CopyCommandBuffer,
    leases: Vec<ResourceLease>,
    token: NativePresentationToken,
) -> Result<NativeCompletion, String> {
    match buffer.native.as_ref() {
        #[cfg(feature = "dx12")]
        Some(NativeFinished::Dx12 { .. }) => submit_dx12_presented(buffer, leases, token),
        #[cfg(feature = "vulkan")]
        Some(NativeFinished::Vulkan { .. }) => submit_vulkan_presented(buffer, leases, token),
        None => Err("presentation received an empty command buffer".into()),
    }
}

#[cfg(feature = "dx12")]
fn submit_dx12_presented(
    mut buffer: CopyCommandBuffer,
    leases: Vec<ResourceLease>,
    mut token: NativePresentationToken,
) -> Result<NativeCompletion, String> {
    let finished = buffer.native.take().expect("finished command buffer");
    let NativeFinished::Dx12 {
        owner,
        mut encoder,
        command_buffer,
        render_views,
    } = finished
    else {
        reset_finished(finished);
        return Err("DX12 presentation received a non-DX12 command buffer".into());
    };
    let Some(NativeAcquiredSurfaceTexture::Dx12(surface_texture)) = token.acquired.take() else {
        reset_finished(NativeFinished::Dx12 {
            owner,
            encoder,
            command_buffer,
            render_views,
        });
        return Err("presentation token was already consumed".into());
    };
    let Some(NativeSurfaceFence::Dx12(fence)) = token.fence.take() else {
        reset_finished(NativeFinished::Dx12 {
            owner,
            encoder,
            command_buffer,
            render_views,
        });
        return Err("presentation token has no DX12 fence".into());
    };
    let queue_owner = Arc::clone(&owner);
    let _queue_guard = lock_queue_operations(&queue_owner.queue_operations);
    let NativeDevice::Dx12 { device, queue, .. } = &owner.native else {
        unreachable!()
    };
    // SAFETY: command buffer, acquired texture and unique fence all belong to
    // this queue. The queue guard serializes submit/present with teardown.
    match unsafe { queue.submit(&[&command_buffer], &[&surface_texture], (&fence, 1)) } {
        Ok(()) => {
            let present = {
                let mut surface = token
                    .surface
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let NativeSurface::Dx12(surface) = &mut *surface else {
                    unreachable!()
                };
                // SAFETY: queue accepted the command buffer and this is the
                // unique acquired texture for the configured surface.
                unsafe { queue.present(surface, surface_texture) }.err()
            };
            let presentation_lease = token
                .take_completion_lease()
                .expect("an acquired surface token owns its live-frame gate");
            // A completion hold is deliberately consumed only after both
            // submit and present succeeded. Copy/upload submissions, rejected
            // work, and present failures therefore cannot consume it.
            let presentation_completion_hold = present
                .is_none()
                .then(take_next_dx12_presentation_completion_hold)
                .flatten();
            let injected_accepted_unknown =
                present.is_none() && take_dx12_presentation_accepted_unknown();
            // `Queue::present` takes the acquired texture by value; successful
            // submit has therefore consumed the only surface-owned image.
            // The token now only carries the thread-affine window lease and
            // acquire gate, both of which may be released independently of
            // the ordinary, Send-capable command completion bundle.
            drop(token);
            Ok(NativeCompletion(Arc::new(std::sync::Mutex::new(
                NativeSubmission::Dx12 {
                    owner,
                    encoder: Some(encoder),
                    command_buffer: Some(command_buffer),
                    fence: Some(fence),
                    leases,
                    staging_buffers: Vec::new(),
                    render_views,
                    presentation_lease: Some(presentation_lease),
                    presentation_completion_hold,
                    failure: if injected_accepted_unknown {
                        Some(CompletionFailure::DeviceLost)
                    } else {
                        present.map(|_| CompletionFailure::ExecutionFailed)
                    },
                },
            ))))
        }
        Err(error) => {
            // A failed submit may still have executed command lists. If idle
            // cannot prove otherwise, retain the image token and every lease.
            if unsafe { queue.wait_for_idle() }.is_ok() {
                unsafe {
                    encoder.reset_all(core::iter::once(command_buffer));
                    device.destroy_fence(fence);
                }
                destroy_render_views(&owner, render_views);
                token.acquired = Some(NativeAcquiredSurfaceTexture::Dx12(surface_texture));
                token.fence = None;
                discard_presentation(token);
                Err(error.to_string())
            } else {
                token.acquired = Some(NativeAcquiredSurfaceTexture::Dx12(surface_texture));
                token.fence = Some(NativeSurfaceFence::Dx12(fence));
                // Neither queue acceptance nor completion can be disproven.
                // Quarantine the complete native bundle, including the
                // thread-affine window/surface token, for process lifetime.
                // Returning a terminal failure lets graph retirement progress
                // without pretending this unobservable work became safe.
                token.quarantine_surface();
                std::mem::forget((owner, encoder, command_buffer, leases, render_views, token));
                Ok(NativeCompletion(Arc::new(std::sync::Mutex::new(
                    NativeSubmission::TerminalFailure(CompletionFailure::DeviceLost),
                ))))
            }
        }
    }
}

#[cfg(feature = "vulkan")]
fn submit_vulkan_presented(
    mut buffer: CopyCommandBuffer,
    leases: Vec<ResourceLease>,
    mut token: NativePresentationToken,
) -> Result<NativeCompletion, String> {
    let finished = buffer.native.take().expect("finished command buffer");
    let NativeFinished::Vulkan {
        owner,
        mut encoder,
        command_buffer,
        render_views,
    } = finished
    else {
        reset_finished(finished);
        return Err("Vulkan presentation received a non-Vulkan command buffer".into());
    };
    let Some(NativeAcquiredSurfaceTexture::Vulkan(surface_texture)) = token.acquired.take() else {
        reset_finished(NativeFinished::Vulkan {
            owner,
            encoder,
            command_buffer,
            render_views,
        });
        return Err("presentation token was already consumed".into());
    };
    let Some(NativeSurfaceFence::VulkanPresentation { sync, value }) = token.fence.take() else {
        reset_finished(NativeFinished::Vulkan {
            owner,
            encoder,
            command_buffer,
            render_views,
        });
        return Err("presentation token has no Vulkan fence".into());
    };
    let queue_owner = Arc::clone(&owner);
    let _queue_guard = lock_queue_operations(&queue_owner.queue_operations);
    let NativeDevice::Vulkan { queue, .. } = &owner.native else {
        unreachable!()
    };
    // SAFETY: the acquired surface texture is supplied to submit so HAL wires
    // its acquire/present semaphores around this exact command buffer.
    match unsafe {
        queue.submit(
            &[&command_buffer],
            &[&surface_texture],
            (sync.fence()?, value),
        )
    } {
        Ok(()) => {
            let present = {
                let mut surface = token
                    .surface
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let NativeSurface::Vulkan(surface) = &mut *surface else {
                    unreachable!()
                };
                // SAFETY: queue submission consumed the acquire semaphore and this is the unique image.
                unsafe { queue.present(surface, surface_texture) }.err()
            };
            let presentation_lease = token
                .take_completion_lease()
                .expect("an acquired surface token owns its live-frame gate");
            drop(token);
            Ok(NativeCompletion(Arc::new(Mutex::new(
                NativeSubmission::Vulkan {
                    owner,
                    encoder: Some(encoder),
                    command_buffer: Some(command_buffer),
                    fence: Some(NativeVulkanFence::Presentation { sync, value }),
                    leases,
                    staging_buffers: Vec::new(),
                    render_views,
                    presentation_lease: Some(presentation_lease),
                    failure: present.map(|_| CompletionFailure::ExecutionFailed),
                },
            ))))
        }
        Err(error) => {
            if unsafe { queue.wait_for_idle() }.is_ok() {
                unsafe {
                    encoder.reset_all(core::iter::once(command_buffer));
                }
                destroy_render_views(&owner, render_views);
                token.acquired = Some(NativeAcquiredSurfaceTexture::Vulkan(surface_texture));
                token.fence = None;
                discard_presentation(token);
                Err(error.to_string())
            } else {
                token.acquired = Some(NativeAcquiredSurfaceTexture::Vulkan(surface_texture));
                token.fence = Some(NativeSurfaceFence::VulkanPresentation { sync, value });
                token.quarantine_surface();
                std::mem::forget((owner, encoder, command_buffer, leases, render_views, token));
                Ok(NativeCompletion(Arc::new(Mutex::new(
                    NativeSubmission::TerminalFailure(CompletionFailure::DeviceLost),
                ))))
            }
        }
    }
}

pub(crate) fn unconfigure_surface(
    owner: &Arc<OpenedDevice>,
    surface: &Mutex<NativeSurface>,
) -> Result<(), String> {
    let _guard = lock_queue_operations(&owner.queue_operations);
    let mut surface = surface.lock().map_err(|_| "surface lock poisoned")?;
    match (&owner.native, &mut *surface) {
        #[cfg(feature = "dx12")]
        (NativeDevice::Dx12 { device, queue, .. }, NativeSurface::Dx12(surface)) => {
            // SAFETY: shutdown has stopped acquisition and no frame token can
            // remain. Queue idle establishes the HAL unconfigure precondition.
            unsafe { queue.wait_for_idle() }.map_err(|e| e.to_string())?;
            unsafe { surface.unconfigure(device) };
            Ok(())
        }
        #[cfg(feature = "vulkan")]
        (NativeDevice::Vulkan { device, queue, .. }, NativeSurface::Vulkan(surface)) => {
            unsafe { queue.wait_for_idle() }.map_err(|e| e.to_string())?;
            unsafe { surface.unconfigure(device) };
            Ok(())
        }
        _ => Err("surface and device backend mismatch".into()),
    }
}
