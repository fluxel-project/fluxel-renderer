//! Implements the closed raster `ExecutionBackend` adapter.
//!
//! This module implements the portable `ExecutionBackend` contract while
//! delegating every native operation to the private `imp` boundary. Encoder
//! state is pass-local: beginning a pass clears prior bindings, and changing a
//! stream-dependent pipeline invalidates its binding and vertex-input proof.
//! Resource leases remain retained through submission, queue operations stay
//! serialized by the owning device, and no HAL or native handle escapes here.

use super::*;
use crate::execution::helpers::require_device;

impl ExecutionBackend for RasterBackend {
    type Texture = Texture;
    type Buffer = Buffer;
    type RasterPipeline = RasterPipeline;
    type ComputePipeline = ComputePipeline;
    type Bindings = RasterBindings;
    type Encoder = CopyEncoder;
    type CommandBuffer = CopyCommandBuffer;
    type Completion = NativeCompletion;
    type PresentationToken = PresentationToken;
    type Lease = ResourceLease;
    type Error = NativeExecutionError;

    fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }
    fn device_identity(&self) -> fluxel_rendergraph::DeviceIdentity {
        self.device.identity()
    }
    fn create_transient_texture(
        &mut self,
        descriptor: TextureDesc,
        usage: TextureUsage,
    ) -> Result<BoundTexture<Texture, ResourceLease>, Self::Error> {
        let texture = self.device.create_texture(TextureDescriptor {
            texture: descriptor,
            usage,
            memory: MemoryPolicy::DeviceOnly,
        })?;
        Ok(BoundTexture {
            device: self.device.identity(),
            identity: texture.identity(),
            physical: texture.clone(),
            descriptor,
            usage: texture.allowed_usage(),
            initial_state: ResourceAccessState::Undefined,
            lease: texture.lease().into(),
        })
    }
    fn create_transient_buffer(
        &mut self,
        descriptor: BufferDesc,
        usage: BufferUsage,
    ) -> Result<BoundBuffer<Buffer, ResourceLease>, Self::Error> {
        let buffer = self.device.create_buffer(BufferDescriptor {
            buffer: descriptor,
            usage,
            memory: MemoryPolicy::DeviceOnly,
        })?;
        Ok(BoundBuffer {
            device: self.device.identity(),
            identity: buffer.identity(),
            physical: buffer.clone(),
            descriptor,
            usage: buffer.allowed_usage(),
            initial_state: ResourceAccessState::Undefined,
            lease: buffer.lease().into(),
        })
    }
    fn begin_encoder(&mut self, queue: QueueId) -> Result<CopyEncoder, Self::Error> {
        if queue != QueueId::new(0) {
            return Err(NativeExecutionError::UnknownQueue(queue));
        }
        crate::imp::begin_copy_encoder(&self.device.inner)
            .map(|native| CopyEncoder {
                native,
                device: self.device.identity(),
                leases: Vec::new(),
                active_compute: None,
                bound_compute_pipeline: None,
                active_raster: None,
                bound_raster_uniform: None,
                bound_raster_texture: None,
                bound_raster_uv_texture: None,
                bound_raster_normal: None,
                bound_raster_vertex_color: None,
                raster_uv_epoch: 0,
                raster_uv_binding_epoch: None,
                raster_uv_vertex_slots: [None, None],
                raster_uv_index_ready: false,
                raster_normal_epoch: 0,
                raster_normal_binding_epoch: None,
                raster_normal_vertex_slots: [None, None],
                raster_normal_index_ready: false,
                raster_vertex_color_epoch: 0,
                raster_vertex_color_binding_epoch: None,
                raster_vertex_color_slots: [None, None],
                raster_vertex_color_index_ready: false,
                vertex_buffer: None,
                index_buffer: None,
                raster_extent: None,
                compute_open: false,
                copy_open: false,
            })
            .map_err(NativeExecutionError::Recording)
    }
    fn transition_texture(
        &mut self,
        encoder: &mut CopyEncoder,
        texture: &Texture,
        range: TextureRange,
        before: ResourceAccessState,
        after: ResourceAccessState,
    ) -> Result<(), Self::Error> {
        self.copy()
            .transition_texture(encoder, texture, range, before, after)
    }
    fn transition_buffer(
        &mut self,
        encoder: &mut CopyEncoder,
        buffer: &Buffer,
        range: BufferRange,
        before: ResourceAccessState,
        after: ResourceAccessState,
    ) -> Result<(), Self::Error> {
        self.copy()
            .transition_buffer(encoder, buffer, range, before, after)
    }
    fn begin_raster(
        &mut self,
        encoder: &mut CopyEncoder,
        descriptor: &RasterPassDescriptor<'_, Texture>,
    ) -> Result<(), Self::Error> {
        self.check_encoder(encoder)?;
        if encoder.raster_extent.is_some()
            || encoder.compute_open
            || encoder.copy_open
            || descriptor.colors.len() != 1
        {
            return Err(NativeExecutionError::RasterStateMismatch);
        }
        let color = &descriptor.colors[0];
        if color.index != 0 || color.range != TextureRange::Whole {
            return Err(NativeExecutionError::RasterStateMismatch);
        }
        self.check_texture(color.texture)?;
        if !color
            .texture
            .allowed_usage()
            .contains(TextureUsageKind::ColorAttachment)
        {
            return Err(NativeExecutionError::RasterStateMismatch);
        }
        let texture = color.texture.descriptor().texture;
        if texture.dimension != fluxel_rendergraph::TextureDimension::D2
            || texture.format != TextureFormat::Rgba8Unorm
            || texture.mip_levels != 1
            || texture.array_layers != 1
            || texture.sample_count != 1
            || texture.extent.depth != 1
        {
            return Err(NativeExecutionError::RasterStateMismatch);
        }
        let (clear, load) = match color.operations.load {
            LoadOp::Clear(value) => (Some(value), false),
            LoadOp::Load => (None, true),
            LoadOp::DontCare => return Err(NativeExecutionError::RasterStateMismatch),
        };
        if color.operations.store != StoreOp::Store {
            return Err(NativeExecutionError::RasterStateMismatch);
        }
        let depth = descriptor
            .depth_stencil
            .as_ref()
            .map(|attachment| {
                self.check_texture(attachment.texture)?;
                if attachment.range != TextureRange::Whole
                    || attachment.stencil.is_some()
                    || !attachment
                        .texture
                        .allowed_usage()
                        .contains(TextureUsageKind::DepthStencilAttachment)
                {
                    return Err(NativeExecutionError::RasterStateMismatch);
                }
                let texture = attachment.texture.descriptor().texture;
                if texture.dimension != fluxel_rendergraph::TextureDimension::D2
                    || texture.format != TextureFormat::Depth32Float
                    || texture.mip_levels != 1
                    || texture.array_layers != 1
                    || texture.sample_count != 1
                    || texture.extent.depth != 1
                    || texture.extent.width != color.texture.descriptor().texture.extent.width
                    || texture.extent.height != color.texture.descriptor().texture.extent.height
                {
                    return Err(NativeExecutionError::RasterStateMismatch);
                }
                let operations = attachment
                    .depth
                    .ok_or(NativeExecutionError::RasterStateMismatch)?;
                if let LoadOp::Clear(value) = operations.load {
                    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                        return Err(NativeExecutionError::RasterStateMismatch);
                    }
                }
                if operations.store != StoreOp::Store
                    || matches!(operations.load, LoadOp::DontCare)
                        && operations.write_coverage != fluxel_rendergraph::WriteCoverage::Full
                {
                    return Err(NativeExecutionError::RasterStateMismatch);
                }
                let (clear, load) = match operations.load {
                    LoadOp::Clear(value) => (Some(value), false),
                    LoadOp::Load => (None, true),
                    LoadOp::DontCare => (None, false),
                };
                Ok((attachment.texture, texture, clear, load))
            })
            .transpose()?;
        crate::imp::begin_raster(
            &mut encoder.native,
            color.texture.native(),
            texture,
            clear,
            load,
            true,
            depth.map(|(texture, descriptor, clear, load)| {
                (texture.native(), descriptor, clear, load)
            }),
            descriptor.label,
        )
        .map_err(NativeExecutionError::Recording)?;
        // Raster state is pass-local. Retaining any pipeline, binding, or
        // vertex/index readiness here could let a later pass satisfy its
        // validation from native state recorded for the previous pass.
        encoder.leases.push(color.texture.lease().into());
        if let Some((texture, _, _, _)) = depth {
            encoder.leases.push(texture.lease().into());
        }
        encoder.raster_extent = Some((texture.extent.width, texture.extent.height));
        encoder.active_raster = None;
        encoder.vertex_buffer = None;
        encoder.index_buffer = None;
        encoder.bound_raster_uv_texture = None;
        encoder.bound_raster_normal = None;
        encoder.bound_raster_vertex_color = None;
        encoder.raster_uv_binding_epoch = None;
        encoder.raster_uv_vertex_slots = [None, None];
        encoder.raster_uv_index_ready = false;
        encoder.raster_normal_binding_epoch = None;
        encoder.raster_normal_vertex_slots = [None, None];
        encoder.raster_normal_index_ready = false;
        encoder.raster_vertex_color_binding_epoch = None;
        encoder.raster_vertex_color_slots = [None, None];
        encoder.raster_vertex_color_index_ready = false;
        Ok(())
    }
    fn end_raster(&mut self, encoder: &mut CopyEncoder) -> Result<(), Self::Error> {
        self.check_encoder(encoder)?;
        if encoder.raster_extent.is_none() {
            return Err(NativeExecutionError::RasterStateMismatch);
        }
        crate::imp::end_raster(&mut encoder.native).map_err(NativeExecutionError::Recording)?;
        encoder.raster_extent = None;
        encoder.active_raster = None;
        encoder.vertex_buffer = None;
        encoder.index_buffer = None;
        encoder.bound_raster_uv_texture = None;
        encoder.bound_raster_normal = None;
        encoder.bound_raster_vertex_color = None;
        encoder.raster_uv_binding_epoch = None;
        encoder.raster_uv_vertex_slots = [None, None];
        encoder.raster_uv_index_ready = false;
        encoder.raster_normal_binding_epoch = None;
        encoder.raster_normal_vertex_slots = [None, None];
        encoder.raster_normal_index_ready = false;
        encoder.raster_vertex_color_binding_epoch = None;
        encoder.raster_vertex_color_slots = [None, None];
        encoder.raster_vertex_color_index_ready = false;
        Ok(())
    }
    fn begin_compute(&mut self, encoder: &mut CopyEncoder, label: &str) -> Result<(), Self::Error> {
        self.check_encoder(encoder)?;
        if encoder.raster_extent.is_some() || encoder.compute_open || encoder.copy_open {
            return Err(NativeExecutionError::RasterStateMismatch);
        }
        crate::imp::begin_compute(&mut encoder.native, label)
            .map_err(NativeExecutionError::Recording)?;
        encoder.active_compute = None;
        encoder.bound_compute_pipeline = None;
        encoder.compute_open = true;
        Ok(())
    }
    fn end_compute(&mut self, encoder: &mut CopyEncoder) -> Result<(), Self::Error> {
        self.check_encoder(encoder)?;
        if !encoder.compute_open {
            return Err(NativeExecutionError::ComputeBindingMismatch);
        }
        crate::imp::end_compute(&mut encoder.native).map_err(NativeExecutionError::Recording)?;
        encoder.active_compute = None;
        encoder.bound_compute_pipeline = None;
        encoder.compute_open = false;
        Ok(())
    }
    fn begin_copy(&mut self, encoder: &mut CopyEncoder, label: &str) -> Result<(), Self::Error> {
        self.copy().begin_copy(encoder, label)
    }
    fn end_copy(&mut self, encoder: &mut CopyEncoder) -> Result<(), Self::Error> {
        self.copy().end_copy(encoder)
    }
    fn set_raster_pipeline(
        &mut self,
        encoder: &mut CopyEncoder,
        pipeline: &RasterPipeline,
    ) -> Result<(), Self::Error> {
        self.check_encoder(encoder)?;
        self.check_texture_pipeline(pipeline)?;
        if encoder.raster_extent.is_none() {
            return Err(NativeExecutionError::RasterStateMismatch);
        }
        crate::imp::set_raster_pipeline(&mut encoder.native, pipeline.native())
            .map_err(NativeExecutionError::Recording)?;
        let resets_uv_epoch = encoder
            .active_raster
            .as_ref()
            .is_some_and(|active| Self::is_uv_kernel(active.kernel()))
            || Self::is_uv_kernel(pipeline.kernel());
        let resets_normal_epoch = encoder
            .active_raster
            .as_ref()
            .is_some_and(|active| Self::is_normal_kernel(active.kernel()))
            || Self::is_normal_kernel(pipeline.kernel());
        let resets_vertex_color_epoch = encoder
            .active_raster
            .as_ref()
            .is_some_and(|active| Self::is_vertex_color_kernel(active.kernel()))
            || Self::is_vertex_color_kernel(pipeline.kernel());
        encoder.active_raster = Some(pipeline.clone());
        encoder.bound_raster_uniform = None;
        encoder.bound_raster_texture = None;
        // Entering or leaving a stream-dependent recipe starts a new binding
        // generation. This prevents an A -> B -> A switch from reusing the
        // bindings and input-readiness proof established for the first A.
        if resets_uv_epoch {
            encoder.raster_uv_epoch = encoder.raster_uv_epoch.wrapping_add(1);
            encoder.bound_raster_uv_texture = None;
            encoder.raster_uv_binding_epoch = None;
            encoder.raster_uv_vertex_slots = [None, None];
            encoder.raster_uv_index_ready = false;
        }
        if resets_normal_epoch {
            encoder.raster_normal_epoch = encoder.raster_normal_epoch.wrapping_add(1);
            encoder.bound_raster_normal = None;
            encoder.raster_normal_binding_epoch = None;
            encoder.raster_normal_vertex_slots = [None, None];
            encoder.raster_normal_index_ready = false;
        }
        if resets_vertex_color_epoch {
            encoder.raster_vertex_color_epoch = encoder.raster_vertex_color_epoch.wrapping_add(1);
            encoder.bound_raster_vertex_color = None;
            encoder.raster_vertex_color_binding_epoch = None;
            encoder.raster_vertex_color_slots = [None, None];
            encoder.raster_vertex_color_index_ready = false;
        }
        encoder.leases.push(pipeline.lease().into());
        Ok(())
    }
    fn set_compute_pipeline(
        &mut self,
        encoder: &mut CopyEncoder,
        pipeline: &ComputePipeline,
    ) -> Result<(), Self::Error> {
        self.check_encoder(encoder)?;
        if !encoder.compute_open {
            return Err(NativeExecutionError::ComputeBindingMismatch);
        }
        require_device(
            pipeline.device_identity(),
            self.device.identity(),
            NativeExecutionError::ForeignResource,
        )?;
        crate::imp::set_compute_pipeline(&mut encoder.native, pipeline.native())
            .map_err(NativeExecutionError::Recording)?;
        encoder.active_compute = Some(pipeline.clone());
        encoder.bound_compute_pipeline = None;
        encoder.leases.push(pipeline.lease().into());
        Ok(())
    }
    fn set_bindings(
        &mut self,
        encoder: &mut CopyEncoder,
        bindings: &RasterBindings,
    ) -> Result<(), Self::Error> {
        self.check_encoder(encoder)?;
        if encoder.raster_extent.is_some() && !encoder.compute_open {
            let pipeline = encoder
                .active_raster
                .as_ref()
                .ok_or(NativeExecutionError::RasterStateMismatch)?;
            // Kernel/layout equality is insufficient: bindings are created
            // for one concrete pipeline object. The epoch checks below also
            // make each binding establish exactly one input-identity contract
            // for the current pipeline generation.
            match bindings {
                RasterBindings::RasterUvTexture(value)
                    if pipeline.kernel()
                        == crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUv
                        && value.device_identity() == self.device.identity()
                        && value.pipeline().same_object(pipeline) =>
                {
                    if encoder.raster_uv_binding_epoch == Some(encoder.raster_uv_epoch) {
                        return Err(NativeExecutionError::RasterBindingsAlreadySet);
                    }
                    crate::imp::set_raster_uv_texture_bindings(&mut encoder.native, value.native())
                        .map_err(NativeExecutionError::Recording)?;
                    encoder.bound_raster_uv_texture = Some(RasterUvBindings::Texture(value.clone()));
                    encoder.raster_uv_binding_epoch = Some(encoder.raster_uv_epoch);
                    encoder.leases.push(value.lease().into());
                    return Ok(());
                }
                RasterBindings::RasterUvLinearClampTexture(value)
                    if pipeline.kernel()
                        == crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClamp
                        && value.device_identity() == self.device.identity()
                        && value.pipeline().same_object(pipeline) =>
                {
                    if encoder.raster_uv_binding_epoch == Some(encoder.raster_uv_epoch) {
                        return Err(NativeExecutionError::RasterBindingsAlreadySet);
                    }
                    crate::imp::set_raster_uv_linear_clamp_texture_bindings(
                        &mut encoder.native,
                        value.native(),
                    )
                    .map_err(NativeExecutionError::Recording)?;
                    encoder.bound_raster_uv_texture =
                        Some(RasterUvBindings::LinearClamp(value.clone()));
                    encoder.raster_uv_binding_epoch = Some(encoder.raster_uv_epoch);
                    encoder.leases.push(value.lease().into());
                    return Ok(());
                }
                RasterBindings::RasterUvLinearClampSrgbTexture(value)
                    if pipeline.kernel()
                        == crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTextureUvLinearClampSrgb
                        && value.device_identity() == self.device.identity()
                        && value.pipeline().same_object(pipeline) =>
                {
                    if encoder.raster_uv_binding_epoch == Some(encoder.raster_uv_epoch) {
                        return Err(NativeExecutionError::RasterBindingsAlreadySet);
                    }
                    crate::imp::set_raster_uv_linear_clamp_srgb_texture_bindings(
                        &mut encoder.native,
                        value.native(),
                    )
                    .map_err(NativeExecutionError::Recording)?;
                    encoder.bound_raster_uv_texture =
                        Some(RasterUvBindings::LinearClampSrgb(value.clone()));
                    encoder.raster_uv_binding_epoch = Some(encoder.raster_uv_epoch);
                    encoder.leases.push(value.lease().into());
                    return Ok(());
                }
                RasterBindings::RasterUniform(value)
                    if pipeline.kernel()
                        == crate::RasterKernel::IndexedPositionFloat32x3CameraMaterial
                        && value.device_identity() == self.device.identity()
                        && value.pipeline().same_object(pipeline) =>
                {
                    crate::imp::set_raster_uniform_bindings(&mut encoder.native, value.native())
                        .map_err(NativeExecutionError::Recording)?;
                    encoder.bound_raster_uniform = Some(value.clone());
                    encoder.leases.push(value.lease().into());
                    return Ok(());
                }
                RasterBindings::RasterNormal(value)
                    if pipeline.kernel()
                        == crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialNormalLambert
                        && value.device_identity() == self.device.identity()
                        && value.pipeline().same_object(pipeline) =>
                {
                    if encoder.raster_normal_binding_epoch == Some(encoder.raster_normal_epoch) {
                        return Err(NativeExecutionError::RasterBindingsAlreadySet);
                    }
                    crate::imp::set_raster_uniform_bindings(&mut encoder.native, value.native())
                        .map_err(NativeExecutionError::Recording)?;
                    encoder.bound_raster_normal = Some(value.clone());
                    encoder.raster_normal_binding_epoch = Some(encoder.raster_normal_epoch);
                    encoder.leases.push(value.lease().into());
                    return Ok(());
                }
                RasterBindings::RasterVertexColor(value)
                    if pipeline.kernel() == crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialVertexColor
                        && value.device_identity() == self.device.identity()
                        && value.pipeline().same_object(pipeline) =>
                {
                    if encoder.raster_vertex_color_binding_epoch == Some(encoder.raster_vertex_color_epoch) { return Err(NativeExecutionError::RasterBindingsAlreadySet); }
                    crate::imp::set_raster_uniform_bindings(&mut encoder.native, value.native()).map_err(NativeExecutionError::Recording)?;
                    encoder.bound_raster_vertex_color = Some(value.clone());
                    encoder.raster_vertex_color_binding_epoch = Some(encoder.raster_vertex_color_epoch);
                    encoder.leases.push(value.lease().into());
                    return Ok(());
                }
                RasterBindings::RasterTexture(value)
                    if pipeline.kernel()
                        == crate::RasterKernel::IndexedPositionFloat32x3CameraMaterialTexture
                        && value.device_identity() == self.device.identity()
                        && value.pipeline().same_object(pipeline) =>
                {
                    crate::imp::set_raster_texture_bindings(&mut encoder.native, value.native())
                        .map_err(NativeExecutionError::Recording)?;
                    encoder.bound_raster_texture = Some(value.clone());
                    encoder.leases.push(value.lease().into());
                    return Ok(());
                }
                _ => return Err(NativeExecutionError::RasterStateMismatch),
            }
        }
        if !encoder.compute_open {
            return Err(NativeExecutionError::ComputeBindingMismatch);
        }
        let pipeline = encoder
            .active_compute
            .as_ref()
            .ok_or(NativeExecutionError::ComputeBindingMismatch)?;
        match bindings {
            RasterBindings::Compute(value)
                if value.device_identity() == self.device.identity()
                    && value.pipeline().same_object(pipeline) =>
            {
                crate::imp::set_compute_bindings(&mut encoder.native, value.native())
                    .map_err(NativeExecutionError::Recording)?;
                encoder.leases.push(value.lease().into());
            }
            RasterBindings::TexturePack(value)
                if value.device_identity() == self.device.identity()
                    && value.pipeline().same_object(pipeline) =>
            {
                crate::imp::set_texture_pack_bindings(&mut encoder.native, value.native())
                    .map_err(NativeExecutionError::Recording)?;
                encoder.leases.push(value.lease().into());
            }
            _ => return Err(NativeExecutionError::ComputeBindingMismatch),
        }
        encoder.bound_compute_pipeline = Some(pipeline.clone());
        Ok(())
    }
    fn set_vertex_buffer(
        &mut self,
        encoder: &mut CopyEncoder,
        slot: u32,
        buffer: &Buffer,
        offset: u64,
    ) -> Result<(), Self::Error> {
        self.set_vertex(encoder, slot, buffer, offset)
    }
    fn set_index_buffer(
        &mut self,
        encoder: &mut CopyEncoder,
        buffer: &Buffer,
        offset: u64,
        format: IndexFormat,
    ) -> Result<(), Self::Error> {
        self.set_index(encoder, buffer, offset, format)
    }
    fn set_viewport(
        &mut self,
        encoder: &mut CopyEncoder,
        viewport: Viewport,
    ) -> Result<(), Self::Error> {
        self.set_viewport_checked(encoder, viewport)
    }
    fn set_scissor(
        &mut self,
        encoder: &mut CopyEncoder,
        scissor: ScissorRect,
    ) -> Result<(), Self::Error> {
        self.set_scissor_checked(encoder, scissor)
    }
    fn draw(
        &mut self,
        encoder: &mut CopyEncoder,
        vertices: Range<u32>,
        instances: Range<u32>,
    ) -> Result<(), Self::Error> {
        self.draw_checked(encoder, vertices, instances)
    }
    fn draw_indexed(
        &mut self,
        encoder: &mut CopyEncoder,
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    ) -> Result<(), Self::Error> {
        self.draw_indexed_checked(encoder, indices, base_vertex, instances)
    }
    fn dispatch(&mut self, encoder: &mut CopyEncoder, groups: [u32; 3]) -> Result<(), Self::Error> {
        self.dispatch_checked(encoder, groups)
    }
    fn copy_texture(
        &mut self,
        encoder: &mut CopyEncoder,
        source: &Texture,
        destination: &Texture,
        region: TextureCopyRegion,
    ) -> Result<(), Self::Error> {
        self.copy()
            .copy_texture(encoder, source, destination, region)
    }
    fn copy_buffer(
        &mut self,
        encoder: &mut CopyEncoder,
        source: &Buffer,
        destination: &Buffer,
        region: BufferCopyRegion,
    ) -> Result<(), Self::Error> {
        self.copy()
            .copy_buffer(encoder, source, destination, region)
    }
    fn finish_encoder(&mut self, encoder: CopyEncoder) -> Result<CopyCommandBuffer, Self::Error> {
        self.copy().finish_encoder(encoder)
    }
    fn submit(
        &mut self,
        queue: QueueId,
        command_buffer: CopyCommandBuffer,
        mut presentations: Vec<PresentationSubmission<Self::PresentationToken>>,
    ) -> Result<NativeCompletion, Self::Error> {
        if queue != QueueId::new(0) {
            return Err(NativeExecutionError::UnknownQueue(queue));
        }
        require_device(
            command_buffer.device,
            self.device.identity(),
            NativeExecutionError::ForeignCommandBuffer,
        )?;
        if presentations.is_empty() {
            return self.copy().submit(queue, command_buffer, presentations);
        }
        if presentations.len() != 1 {
            return Err(NativeExecutionError::SubmitRejected(
                "raster presentation requires exactly one acquired image".into(),
            ));
        }
        let token = presentations.pop().expect("checked one presentation").token;
        #[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
        {
            let presentation = token.into_native().ok_or_else(|| {
                NativeExecutionError::SubmitRejected(
                    "presentation token was already consumed".into(),
                )
            })?;
            let CopyCommandBuffer {
                native,
                device: _,
                leases,
            } = command_buffer;
            crate::imp::submit_presented(native, leases, presentation)
                .map(NativeCompletion)
                .map_err(NativeExecutionError::SubmitRejected)
        }
        #[cfg(not(all(windows, any(feature = "dx12", feature = "vulkan"))))]
        {
            let _ = token;
            Err(NativeExecutionError::SubmitRejected(
                "presentation is unavailable on this target/build".into(),
            ))
        }
    }
    fn completion_status(&self, completion: &NativeCompletion) -> CompletionStatus {
        // The trait exposes a total status query. A native query failure cannot
        // safely be reported as Pending or Complete, so collapse it to the
        // terminal conservative failure used by portable execution.
        crate::imp::completion_status(&completion.0)
            .unwrap_or(CompletionStatus::Failed(CompletionFailure::DeviceLost))
    }
    fn retire(&mut self, completion: NativeCompletion, leases: Vec<ResourceLease>) {
        self.retired.push(Retired { completion, leases });
    }
    fn collect_retired(&mut self) -> Result<usize, Self::Error> {
        let before = self.retired.len();
        let mut error = None;
        // A query error proves neither completion nor failure of the submitted
        // GPU work. Keep its leases quarantined until a later query supplies a
        // terminal status; releasing them here could destroy in-flight handles.
        self.retired.retain(
            |entry| match crate::imp::completion_status(&entry.completion.0) {
                Ok(CompletionStatus::Pending) => true,
                Ok(_) => false,
                Err(value) => {
                    error.get_or_insert(value);
                    true
                }
            },
        );
        error.map_or(Ok(before - self.retired.len()), |value| {
            Err(NativeExecutionError::Completion(value))
        })
    }
}
