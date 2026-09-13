//! Fixed compute object registry.

use super::*;
use crate::execution::helpers::require_device;

/// Registry for the fixed compute pipeline and binding recipes.
///
/// Registration is explicit so graph IDs never become implicit native handles.
pub struct ComputeObjectProvider {
    owner: Device,
    device: fluxel_rendergraph::DeviceIdentity,
    pipelines: HashMap<fluxel_rendergraph::ComputePipelineId, ComputePipeline>,
    bindings: HashMap<
        fluxel_rendergraph::BindingSetId,
        (fluxel_rendergraph::ComputePipelineId, ComputePipeline),
    >,
}

impl ComputeObjectProvider {
    /// Creates an empty registry for `device`.
    pub fn new(device: &Device) -> Self {
        Self {
            owner: device.clone(),
            device: device.identity(),
            pipelines: HashMap::new(),
            bindings: HashMap::new(),
        }
    }

    /// Registers a graph pipeline identity with a device-affine fixed artifact.
    pub fn register_pipeline(
        &mut self,
        id: fluxel_rendergraph::ComputePipelineId,
        pipeline: ComputePipeline,
    ) -> Result<(), NativeExecutionError> {
        require_device(
            pipeline.device_identity(),
            self.device,
            NativeExecutionError::ForeignResource,
        )?;
        self.pipelines.insert(id, pipeline);
        Ok(())
    }

    /// Registers a binding recipe and the pipeline layout it must resolve against.
    pub fn register_bindings(
        &mut self,
        id: fluxel_rendergraph::BindingSetId,
        expected_pipeline: fluxel_rendergraph::ComputePipelineId,
    ) -> Result<(), NativeExecutionError> {
        let pipeline = self
            .pipelines
            .get(&expected_pipeline)
            .ok_or(NativeExecutionError::ComputeBindingMismatch)?;
        self.bindings
            .insert(id, (expected_pipeline, pipeline.clone()));
        Ok(())
    }
}

pub(crate) fn provider_error(
    kind: fluxel_rendergraph::RecordingErrorKind,
    detail: impl Into<String>,
) -> fluxel_rendergraph::RecordingError {
    fluxel_rendergraph::RecordingError {
        kind,
        context: fluxel_rendergraph::DiagnosticContext {
            passes: vec![],
            resource: None,
            texture_slot: None,
            buffer_slot: None,
            detail: detail.into(),
            unsupported: None,
        },
    }
}

pub(crate) fn compute_binding_error_kind(
    error: &ComputeCreateError,
) -> fluxel_rendergraph::RecordingErrorKind {
    match error {
        ComputeCreateError::ForeignDevice
        | ComputeCreateError::InvalidBindingRange
        | ComputeCreateError::StorageUsageRequired
        | ComputeCreateError::BindingRecipeMismatch
        | ComputeCreateError::UnsupportedComputeLimits => {
            fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe
        }
        ComputeCreateError::ShaderValidation(_)
        | ComputeCreateError::ShaderCompilation(_)
        | ComputeCreateError::NativeObjectCreation(_)
        | ComputeCreateError::NativeFailure(_) => {
            fluxel_rendergraph::RecordingErrorKind::BackendObjectCreation
        }
    }
}

impl fluxel_rendergraph::RenderObjectProvider<ComputeBackend> for ComputeObjectProvider {
    fn raster_pipeline(
        &self,
        _: fluxel_rendergraph::RasterPipelineId,
    ) -> Result<
        fluxel_rendergraph::BoundRasterPipeline<UnsupportedRasterPipeline, ResourceLease>,
        fluxel_rendergraph::RecordingError,
    > {
        Err(provider_error(
            fluxel_rendergraph::RecordingErrorKind::MissingFrameBinding,
            "raster is outside the compute slice",
        ))
    }

    fn compute_pipeline(
        &self,
        id: fluxel_rendergraph::ComputePipelineId,
    ) -> Result<
        fluxel_rendergraph::BoundComputePipeline<ComputePipeline, ResourceLease>,
        fluxel_rendergraph::RecordingError,
    > {
        let pipeline = self.pipelines.get(&id).ok_or_else(|| {
            provider_error(
                fluxel_rendergraph::RecordingErrorKind::MissingFrameBinding,
                "unknown compute pipeline",
            )
        })?;
        if pipeline.device_identity() != self.device {
            return Err(provider_error(
                fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe,
                "foreign compute pipeline",
            ));
        }
        Ok(fluxel_rendergraph::BoundComputePipeline {
            device: self.device,
            physical: pipeline.clone(),
            lease: pipeline.lease().into(),
        })
    }

    fn bindings(
        &self,
        id: fluxel_rendergraph::BindingSetId,
        resources: &[fluxel_rendergraph::ResolvedBindingResource<'_, Texture, Buffer>],
        dynamic_offsets: &[u32],
    ) -> Result<
        fluxel_rendergraph::BoundBindings<ComputeBindings, ResourceLease>,
        fluxel_rendergraph::RecordingError,
    > {
        if !dynamic_offsets.is_empty() || resources.len() != 1 {
            return Err(provider_error(
                fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe,
                "compute fixture requires exactly one RW storage buffer and no dynamic offsets",
            ));
        }
        let (_, pipeline) = self.bindings.get(&id).ok_or_else(|| {
            provider_error(
                fluxel_rendergraph::RecordingErrorKind::MissingFrameBinding,
                "unknown compute binding recipe",
            )
        })?;
        let fluxel_rendergraph::ResolvedBindingResource::Buffer {
            physical: buffer,
            range,
            semantic,
        } = &resources[0]
        else {
            return Err(provider_error(
                fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe,
                "compute fixture requires a buffer",
            ));
        };
        if !matches!(
            semantic,
            fluxel_rendergraph::BindingResourceSemantic::BufferReadWrite(
                fluxel_rendergraph::BufferReadWriteUse::Storage
            )
        ) {
            return Err(provider_error(
                fluxel_rendergraph::RecordingErrorKind::DeclaredUseMismatch,
                "compute fixture requires BufferReadWrite(Storage)",
            ));
        }
        if buffer.device_identity() != self.device || pipeline.device_identity() != self.device {
            return Err(provider_error(
                fluxel_rendergraph::RecordingErrorKind::IncompatibleBindingRecipe,
                "compute binding is foreign to provider device",
            ));
        }
        let (offset, size) = match *range {
            BufferRange::Whole => (0, buffer.descriptor().buffer.size),
            BufferRange::Bytes { offset, size } => (offset, size),
        };
        let bindings = self
            .owner
            .create_compute_bindings(pipeline, buffer, offset, size)
            .map_err(|error| {
                provider_error(compute_binding_error_kind(&error), error.to_string())
            })?;
        Ok(fluxel_rendergraph::BoundBindings {
            device: self.device,
            physical: bindings.clone(),
            lease: bindings.lease().into(),
        })
    }
}
