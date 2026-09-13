//! Native compute-pass recording.

use super::*;

/// Rechecks the range immediately before constructing an unchecked HAL binding.
pub(crate) fn validate_native_compute_binding_range(
    offset: u64,
    size: u64,
    buffer_size: u64,
    minimum_storage_offset_alignment: u32,
    maximum_storage_binding_size: u64,
) -> Result<(), String> {
    let required_offset_alignment = u64::from(minimum_storage_offset_alignment.max(4));
    let end = offset
        .checked_add(size)
        .ok_or_else(|| "compute storage binding range overflows".to_owned())?;
    if size == 0
        || !offset.is_multiple_of(required_offset_alignment)
        || !size.is_multiple_of(4)
        || size > maximum_storage_binding_size
        || end > buffer_size
    {
        return Err("invalid compute storage binding range".into());
    }
    Ok(())
}

pub(crate) fn begin_compute(encoder: &mut CopyEncoder, label: &str) -> Result<(), String> {
    match encoder.native.as_mut().expect("live encoder") {
        #[cfg(feature = "dx12")]
        NativeEncoder::Dx12(encoder) => unsafe {
            // SAFETY: ExecutionBackend serializes pass scopes and invokes this
            // only while the encoder is recording and outside another pass.
            encoder.begin_compute_pass(&wgpu_hal::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
        },
        #[cfg(feature = "vulkan")]
        NativeEncoder::Vulkan(encoder) => unsafe {
            // SAFETY: same recording and pass-scope invariant as DX12.
            encoder.begin_compute_pass(&wgpu_hal::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
        },
    }
    Ok(())
}

pub(crate) fn end_compute(encoder: &mut CopyEncoder) -> Result<(), String> {
    match encoder.native.as_mut().expect("live encoder") {
        #[cfg(feature = "dx12")]
        NativeEncoder::Dx12(encoder) => unsafe {
            // SAFETY: ExecutionBackend pairs every successful begin with one end.
            encoder.end_compute_pass();
        },
        #[cfg(feature = "vulkan")]
        NativeEncoder::Vulkan(encoder) => unsafe {
            // SAFETY: ExecutionBackend pairs every successful begin with one end.
            encoder.end_compute_pass();
        },
    }
    Ok(())
}

pub(crate) fn set_compute_pipeline(
    encoder: &mut CopyEncoder,
    pipeline: &NativeComputePipeline,
) -> Result<(), String> {
    if !Arc::ptr_eq(&encoder.owner, &pipeline.0.owner) {
        return Err("compute pipeline belongs to another native device".into());
    }
    match (
        encoder.native.as_mut().expect("live encoder"),
        pipeline.0.native.as_ref(),
    ) {
        #[cfg(feature = "dx12")]
        (NativeEncoder::Dx12(encoder), Some(NativeComputePipelineInner::Dx12 { pipeline, .. })) => unsafe {
            // SAFETY: both objects are created by the matching device; caller
            // has opened a compute pass before selecting the pipeline.
            encoder.set_compute_pipeline(pipeline);
        },
        #[cfg(feature = "vulkan")]
        (
            NativeEncoder::Vulkan(encoder),
            Some(NativeComputePipelineInner::Vulkan { pipeline, .. }),
        ) => unsafe {
            // SAFETY: same device and active-pass invariant as DX12.
            encoder.set_compute_pipeline(pipeline);
        },
        _ => return Err("compute pipeline belongs to another native backend".into()),
    }
    Ok(())
}

pub(crate) fn set_compute_bindings(
    encoder: &mut CopyEncoder,
    bindings: &NativeComputeBindings,
) -> Result<(), String> {
    if !Arc::ptr_eq(&encoder.owner, &bindings.pipeline.0.owner) {
        return Err("compute bindings belong to another native device".into());
    }
    match (
        encoder.native.as_mut().expect("live encoder"),
        bindings.pipeline.0.native.as_ref(),
        bindings.native.as_ref(),
    ) {
        #[cfg(feature = "dx12")]
        (
            NativeEncoder::Dx12(encoder),
            Some(NativeComputePipelineInner::Dx12 {
                pipeline_layout, ..
            }),
            Some(NativeComputeBindingsInner::Dx12(group)),
        ) => unsafe {
            // SAFETY: the group was made from this pipeline's sole BGL; there
            // are no dynamic offsets and the active compute pass is tracked by
            // the safe execution layer.
            encoder.set_bind_group(pipeline_layout, 0, group, &[]);
        },
        #[cfg(feature = "dx12")]
        (
            NativeEncoder::Dx12(encoder),
            Some(NativeComputePipelineInner::Dx12 {
                pipeline_layout, ..
            }),
            Some(NativeComputeBindingsInner::Dx12Texture { group, .. }),
        ) => unsafe {
            encoder.set_bind_group(pipeline_layout, 0, group, &[]);
        },
        #[cfg(feature = "vulkan")]
        (
            NativeEncoder::Vulkan(encoder),
            Some(NativeComputePipelineInner::Vulkan {
                pipeline_layout, ..
            }),
            Some(NativeComputeBindingsInner::Vulkan(group)),
        ) => unsafe {
            // SAFETY: same fixed-layout and active-pass invariant as DX12.
            encoder.set_bind_group(pipeline_layout, 0, group, &[]);
        },
        #[cfg(feature = "vulkan")]
        (
            NativeEncoder::Vulkan(encoder),
            Some(NativeComputePipelineInner::Vulkan {
                pipeline_layout, ..
            }),
            Some(NativeComputeBindingsInner::VulkanTexture { group, .. }),
        ) => unsafe {
            encoder.set_bind_group(pipeline_layout, 0, group, &[]);
        },
        _ => return Err("compute bindings belong to another native backend".into()),
    }
    Ok(())
}

/// Selects the closed X01 two-entry group. The group itself retains its view,
/// pipeline and safe-layer resource leases, so no raw object can be destroyed
/// while the encoder records or the accepted submission is pending.
pub(crate) fn set_texture_pack_bindings(
    encoder: &mut CopyEncoder,
    bindings: &NativeTexturePackBindings,
) -> Result<(), String> {
    if !Arc::ptr_eq(&encoder.owner, &bindings.pipeline.0.owner) {
        return Err("texture-pack bindings belong to another native device".into());
    }
    match (
        encoder.native.as_mut().expect("live encoder"),
        bindings.pipeline.0.native.as_ref(),
        bindings.native.as_ref(),
    ) {
        #[cfg(feature = "dx12")]
        (
            NativeEncoder::Dx12(encoder),
            Some(NativeComputePipelineInner::Dx12 {
                pipeline_layout, ..
            }),
            Some(NativeTexturePackBindingsInner::Dx12 { group, .. }),
        ) => unsafe {
            // SAFETY: matching device/layout and active compute-pass are
            // enforced by the safe execution layer.
            encoder.set_bind_group(pipeline_layout, 0, group, &[]);
        },
        #[cfg(feature = "vulkan")]
        (
            NativeEncoder::Vulkan(encoder),
            Some(NativeComputePipelineInner::Vulkan {
                pipeline_layout, ..
            }),
            Some(NativeTexturePackBindingsInner::Vulkan { group, .. }),
        ) => unsafe {
            // SAFETY: same fixed-layout and active-pass proof as DX12.
            encoder.set_bind_group(pipeline_layout, 0, group, &[]);
        },
        _ => return Err("texture-pack bindings belong to another native backend".into()),
    }
    Ok(())
}

pub(crate) fn dispatch(encoder: &mut CopyEncoder, workgroups: [u32; 3]) -> Result<(), String> {
    if workgroups.contains(&0) {
        return Err("compute dispatch dimensions must be non-zero".into());
    }
    match encoder.native.as_mut().expect("live encoder") {
        #[cfg(feature = "dx12")]
        NativeEncoder::Dx12(encoder) => unsafe {
            // SAFETY: caller has established active pass, matching pipeline and
            // bindings, and has validated all dimensions against device limits.
            encoder.dispatch_workgroups(workgroups);
        },
        #[cfg(feature = "vulkan")]
        NativeEncoder::Vulkan(encoder) => unsafe {
            // SAFETY: same active-pass and prevalidated-dimensions invariant.
            encoder.dispatch_workgroups(workgroups);
        },
    }
    Ok(())
}
