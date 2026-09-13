//! Native adapter discovery, device bootstrap, and hardware-capability reporting.
//!
//! This boundary owns instance and device creation. Portable resource-state and
//! descriptor lowering remain in `lowering`; no opened HAL value escapes `imp`.

use super::*;

pub(crate) fn open(backend: Backend, options: DeviceOptions) -> Result<OpenedDevice, OpenError> {
    match backend {
        Backend::Dx12 => open_dx12(options),
        Backend::Vulkan => open_vulkan(options),
    }
}

#[cfg(feature = "dx12")]
fn open_dx12(options: DeviceOptions) -> Result<OpenedDevice, OpenError> {
    let descriptor = instance_descriptor(options.validation);
    // SAFETY: the descriptor owns no display handle, outlives initialization,
    // and requests only HAL-defined instance flags.
    let instance = unsafe { wgpu_hal::dx12::Instance::init(&descriptor) }
        .map_err(|e| native_error(Backend::Dx12, e))?;
    // SAFETY: the live instance owns the factory used for enumeration; no
    // surface constraint is supplied for this headless device.
    let adapters = unsafe { instance.enumerate_adapters(None) };
    let available_adapters = adapters.len();
    let exposed =
        adapters
            .into_iter()
            .nth(options.adapter_index)
            .ok_or(OpenError::AdapterUnavailable {
                backend: Backend::Dx12,
                adapter_index: options.adapter_index,
                available_adapters,
            })?;
    let hardware = hardware(Backend::Dx12, &exposed.info);
    let rgba8_unorm_filterable = rgba8_unorm_filterable(&exposed.adapter);
    let rgba8_unorm_srgb_filterable = rgba8_unorm_srgb_filterable(&exposed.adapter);
    let rgba8_unorm_storage_read = rgba8_unorm_storage_read(&exposed.adapter);
    let rgba8_unorm_storage_write = rgba8_unorm_storage_write(&exposed.adapter);
    // The adapter reports the format feature, but S01's real D3D12 witness
    // observes zeroes from the typed RGBA8 UAV load. Do not request or expose
    // a feature the fixed RHI recipe cannot prove end-to-end.
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
    // SAFETY: features are empty and default limits are validated by the
    // exposed adapter before HAL creates a device and its matching queue.
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
    Ok(OpenedDevice {
        native: NativeDevice::Dx12 {
            queue,
            device,
            adapter,
            instance,
        },
        queue_operations: Mutex::new(()),
        hardware,
        capabilities,
    })
}
#[cfg(not(feature = "dx12"))]
fn open_dx12(_: DeviceOptions) -> Result<OpenedDevice, OpenError> {
    Err(OpenError::BackendDisabled {
        backend: Backend::Dx12,
    })
}

#[cfg(feature = "vulkan")]
fn open_vulkan(options: DeviceOptions) -> Result<OpenedDevice, OpenError> {
    if options.validation == Validation::Required && !vulkan_validation_is_available() {
        return Err(OpenError::ValidationUnavailable {
            backend: Backend::Vulkan,
        });
    }
    let descriptor = instance_descriptor(options.validation);
    // SAFETY: the descriptor owns no display handle, outlives initialization,
    // and requests only HAL-defined instance flags.
    let instance = unsafe { wgpu_hal::vulkan::Instance::init(&descriptor) }
        .map_err(|e| native_error(Backend::Vulkan, e))?;
    // SAFETY: the live instance owns all Vulkan entry points used during
    // enumeration; no surface constraint is supplied for headless execution.
    let adapters = unsafe { instance.enumerate_adapters(None) };
    let available_adapters = adapters.len();
    let exposed =
        adapters
            .into_iter()
            .nth(options.adapter_index)
            .ok_or(OpenError::AdapterUnavailable {
                backend: Backend::Vulkan,
                adapter_index: options.adapter_index,
                available_adapters,
            })?;
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
    // SAFETY: features are empty and default limits are validated by the
    // exposed adapter before HAL creates a device and its matching queue.
    let wgpu_hal::OpenDevice { device, queue } = unsafe {
        adapter.open(
            enabled_features,
            &requested_limits,
            &wgt::MemoryHints::default(),
        )
    }
    .map_err(|e| native_error(Backend::Vulkan, e))?;
    Ok(OpenedDevice {
        native: NativeDevice::Vulkan {
            queue,
            device,
            adapter,
            instance,
        },
        queue_operations: Mutex::new(()),
        hardware,
        capabilities,
    })
}
#[cfg(not(feature = "vulkan"))]
fn open_vulkan(_: DeviceOptions) -> Result<OpenedDevice, OpenError> {
    Err(OpenError::BackendDisabled {
        backend: Backend::Vulkan,
    })
}

pub(crate) fn instance_descriptor(validation: Validation) -> wgpu_hal::InstanceDescriptor<'static> {
    wgpu_hal::InstanceDescriptor {
        name: "fluxel-rhi",
        flags: match validation {
            Validation::Disabled => wgt::InstanceFlags::empty(),
            Validation::Required => wgt::InstanceFlags::VALIDATION,
        },
        memory_budget_thresholds: wgt::MemoryBudgetThresholds::default(),
        backend_options: wgt::BackendOptions::default(),
        telemetry: None,
        display: None,
    }
}

#[cfg(feature = "dx12")]
pub(crate) fn dx12_validation_is_enabled(device: &wgpu_hal::dx12::Device) -> bool {
    use windows::{Win32::Graphics::Direct3D12::ID3D12InfoQueue, core::Interface as _};

    device.raw_device().cast::<ID3D12InfoQueue>().is_ok()
}

#[cfg(feature = "vulkan")]
pub(crate) fn vulkan_validation_is_available() -> bool {
    let validation_layer = c"VK_LAYER_KHRONOS_validation";
    let validation_features = c"VK_EXT_validation_features";
    // SAFETY: loading the process Vulkan loader performs no device operation;
    // failure is converted into an unavailable validation facility.
    let Ok(entry) = (unsafe { ash::Entry::load() }) else {
        return false;
    };
    // SAFETY: `entry` owns valid loader function pointers for the duration of
    // this enumeration call.
    let Ok(layers) = (unsafe { entry.enumerate_instance_layer_properties() }) else {
        return false;
    };
    let has_layer = layers.iter().any(|layer| {
        layer
            .layer_name_as_c_str()
            .is_ok_and(|name| name == validation_layer)
    });
    if !has_layer {
        return false;
    }
    // SAFETY: `entry` owns valid loader function pointers and the layer name
    // remains alive for the duration of the enumeration call.
    let Ok(extensions) =
        (unsafe { entry.enumerate_instance_extension_properties(Some(validation_layer)) })
    else {
        return false;
    };
    extensions.iter().any(|extension| {
        extension
            .extension_name_as_c_str()
            .is_ok_and(|name| name == validation_features)
    })
}
pub(crate) fn native_error(backend: Backend, error: impl core::fmt::Display) -> OpenError {
    OpenError::NativeUnavailable {
        backend,
        reason: error.to_string(),
    }
}
pub(crate) fn required_limits(
    backend: Backend,
    capabilities: &wgpu_hal::Capabilities,
) -> Result<wgt::Limits, OpenError> {
    let requested = wgt::Limits::default();
    if requested.check_limits(&capabilities.limits) {
        Ok(requested)
    } else {
        Err(OpenError::RequiredLimitsUnavailable { backend })
    }
}
pub(crate) fn hardware(backend: Backend, info: &wgt::AdapterInfo) -> HardwareInfo {
    HardwareInfo {
        backend,
        name: info.name.clone(),
        vendor_id: info.vendor,
        device_id: info.device,
        kind: match info.device_type {
            wgt::DeviceType::Other => DeviceKind::Other,
            wgt::DeviceType::IntegratedGpu => DeviceKind::Integrated,
            wgt::DeviceType::DiscreteGpu => DeviceKind::Discrete,
            wgt::DeviceType::VirtualGpu => DeviceKind::Virtual,
            wgt::DeviceType::Cpu => DeviceKind::Cpu,
        },
        pci_bus_id: info.device_pci_bus_id.clone(),
        driver: info.driver.clone(),
        driver_info: info.driver_info.clone(),
    }
}
pub(crate) fn rgba8_unorm_filterable<A: wgpu_hal::Adapter>(adapter: &A) -> bool {
    // SAFETY: the selected adapter remains retained through device creation;
    // this read-only format query has no resource or queue side effects.
    unsafe { adapter.texture_format_capabilities(wgt::TextureFormat::Rgba8Unorm) }
        .contains(wgpu_hal::TextureFormatCapabilities::SAMPLED_LINEAR)
}

pub(crate) fn rgba8_unorm_srgb_filterable<A: wgpu_hal::Adapter>(adapter: &A) -> bool {
    // SAFETY: the selected adapter remains retained through device creation;
    // this read-only format query has no resource or queue side effects. Query
    // sRGB separately because UNORM facts must never imply it.
    unsafe { adapter.texture_format_capabilities(wgt::TextureFormat::Rgba8UnormSrgb) }
        .contains(wgpu_hal::TextureFormatCapabilities::SAMPLED_LINEAR)
}

pub(crate) fn rgba8_unorm_storage_read<A: wgpu_hal::Adapter>(adapter: &A) -> bool {
    // SAFETY: this is the same read-only adapter-format query used for
    // filterability; its result is retained as an unmodified native fact.
    unsafe { adapter.texture_format_capabilities(wgt::TextureFormat::Rgba8Unorm) }
        .contains(wgpu_hal::TextureFormatCapabilities::STORAGE_READ_ONLY)
}

pub(crate) fn rgba8_unorm_storage_write<A: wgpu_hal::Adapter>(adapter: &A) -> bool {
    // SAFETY: same side-effect-free adapter query as the read capability.
    unsafe { adapter.texture_format_capabilities(wgt::TextureFormat::Rgba8Unorm) }
        .contains(wgpu_hal::TextureFormatCapabilities::STORAGE_WRITE_ONLY)
}

pub(crate) fn storage_texture_features(
    available: wgt::Features,
    rgba8_read: bool,
) -> wgt::Features {
    let read_feature = wgt::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES;
    if rgba8_read && available.contains(read_feature) {
        read_feature
    } else {
        wgt::Features::empty()
    }
}

pub(crate) fn capabilities(
    enabled_features: wgt::Features,
    capabilities: &wgpu_hal::Capabilities,
    rgba8_unorm_filterable: bool,
    rgba8_unorm_srgb_filterable: bool,
    rgba8_unorm_storage_read: bool,
    rgba8_unorm_storage_write: bool,
) -> HardwareCapabilities {
    HardwareCapabilities {
        rgba8_unorm_filterable,
        rgba8_unorm_srgb_filterable,
        rgba8_unorm_storage_read,
        rgba8_unorm_storage_write,
        rgba8_unorm_storage_read_enabled: rgba8_unorm_storage_read
            && enabled_features.contains(wgt::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES),
        max_texture_dimension_2d: capabilities.limits.max_texture_dimension_2d,
        max_bind_groups: capabilities.limits.max_bind_groups,
        min_uniform_buffer_offset_alignment: capabilities
            .limits
            .min_uniform_buffer_offset_alignment,
        min_storage_buffer_offset_alignment: capabilities
            .limits
            .min_storage_buffer_offset_alignment,
        max_storage_buffer_binding_size: capabilities.limits.max_storage_buffer_binding_size,
        max_compute_workgroups_per_dimension: [capabilities
            .limits
            .max_compute_workgroups_per_dimension; 3],
        max_compute_workgroup_size: [
            capabilities.limits.max_compute_workgroup_size_x,
            capabilities.limits.max_compute_workgroup_size_y,
            capabilities.limits.max_compute_workgroup_size_z,
        ],
        max_compute_invocations_per_workgroup: capabilities
            .limits
            .max_compute_invocations_per_workgroup,
    }
}
