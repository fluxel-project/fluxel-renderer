//! Shared constructors and declaration-validation test groups.

use fluxel_rendergraph::*;

fn texture() -> TextureDesc {
    TextureDesc {
        dimension: TextureDimension::D2,
        extent: Extent3d {
            width: 4,
            height: 4,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        sample_count: 1,
        format: TextureFormat::Rgba8Unorm,
    }
}

fn capabilities(
    compute: bool,
    storage_buffers: bool,
    storage_textures: bool,
    surface_color: bool,
) -> DeviceCapabilities {
    DeviceCapabilities::builder()
        .queue(QueueDescriptor::new(
            QueueId::new(0),
            QueueCapabilities::new(true, compute, true, true),
        ))
        .limits(DeviceLimits::new(4, 1))
        .buffers(BufferCapabilities::new(
            storage_buffers,
            storage_buffers,
            true,
        ))
        .texture_format(
            TextureFormatCapabilities::builder(TextureFormat::Rgba8Unorm)
                .sampled(true, true)
                .storage(storage_textures, storage_textures)
                .attachments(true, false, vec![1])
                .copies(true, true)
                .build(),
        )
        .surface(SurfaceCapabilities::new(
            vec![TextureFormat::Rgba8Unorm],
            surface_color,
            true,
        ))
        .build()
}

fn assert_error(result: CompileResult, expected: CompileErrorKind) {
    let error = result.expect_err("graph should be rejected");
    assert_eq!(error.kind, expected);
    assert!(
        error.context.unsupported.is_none(),
        "non-capability errors must not carry capability evidence"
    );
}

fn assert_unsupported(
    result: CompileResult,
    capabilities: &DeviceCapabilities,
    expected: CapabilityRequirement,
) {
    let error = result.expect_err("graph should be rejected");
    assert_eq!(error.kind, CompileErrorKind::UnsupportedSemanticRequirement);
    let unsupported = error
        .context
        .unsupported
        .expect("capability rejection must include structured evidence");
    assert_eq!(unsupported.requirement, expected);
    assert_eq!(unsupported.observed.as_ref(), capabilities);
}

fn import_buffer(contents: InitialContents) -> ImportBufferContract {
    ImportBufferContract {
        descriptor: BufferDesc { size: 16 },
        initial_state: ResourceAccessState::ShaderStorageRead,
        ownership: ExternalOwnership::Caller,
        initial_contents: contents,
    }
}

mod boundaries;
mod capabilities;
mod ranges;
