//! In-memory resource and render-object registry for `TestRhi`.

use std::{cell::RefCell, collections::HashMap};

use super::{
    backend::TestRhi,
    lease::{TestLease, TestLeaseProbe},
    trace::{
        TestBindings, TestBuffer, TestComputePipeline, TestPresentationToken, TestRasterPipeline,
        TestTexture,
    },
};

use crate::{
    backend::{
        BindingResourceSemantic, BoundBindings, BoundBuffer, BoundComputePipeline,
        BoundRasterPipeline, BoundSurfaceTexture, BoundTexture, DeviceIdentity, FrameBindingError,
        FrameBindingErrorKind, FrameResourceProvider, RenderObjectProvider,
        ResolvedBindingResource,
    },
    error::{DiagnosticContext, RecordingError, RecordingErrorKind},
    handles::{
        BindingSetId, BufferBindingId, ComputePipelineId, RasterPipelineId, SurfaceBindingId,
        TextureBindingId,
    },
    plan::{BufferUsage, TextureUsage},
    rhi::{BufferDesc, ResourceAccessState, TextureDesc},
};

struct RegisteredTexture {
    physical: TestTexture,
    descriptor: TextureDesc,
    usage: TextureUsage,
    initial_state: ResourceAccessState,
    lease: TestLease,
}
struct RegisteredBuffer {
    physical: TestBuffer,
    descriptor: BufferDesc,
    usage: BufferUsage,
    initial_state: ResourceAccessState,
    lease: TestLease,
}
struct RegisteredSurface {
    texture: RegisteredTexture,
    presentation: TestPresentationToken,
}
struct RegisteredRaster {
    physical: TestRasterPipeline,
    lease: TestLease,
}
struct RegisteredCompute {
    physical: TestComputePipeline,
    lease: TestLease,
}
struct RegisteredBindings {
    physical: TestBindings,
    lease: TestLease,
    expected_resources: Option<Vec<TestBindingResource>>,
    expected_dynamic_offsets: Option<Vec<u32>>,
}

/// One owned binding resource observed by [`TestRegistry`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TestBindingResource {
    /// A texture together with the subresources and semantic authorized by the graph.
    Texture {
        /// Physical test texture.
        physical: TestTexture,
        /// Selected subresources.
        range: crate::TextureRange,
        /// Graph-declared binding semantic.
        semantic: BindingResourceSemantic,
    },
    /// A buffer together with the byte range and semantic authorized by the graph.
    Buffer {
        /// Physical test buffer.
        physical: TestBuffer,
        /// Selected bytes.
        range: crate::BufferRange,
        /// Graph-declared binding semantic.
        semantic: BindingResourceSemantic,
    },
}

/// A binding-resolution request captured by [`TestRegistry`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TestBindingResolution {
    /// Opaque binding recipe selected by the pass.
    pub recipe: BindingSetId,
    /// Resources, ranges, and semantics delivered to the recipe provider.
    pub resources: Vec<TestBindingResource>,
    /// Dynamic offsets delivered to the recipe provider.
    pub dynamic_offsets: Vec<u32>,
}

/// In-memory registry for imported frame objects and renderer-owned objects.
pub struct TestRegistry {
    device: DeviceIdentity,
    textures: HashMap<TextureBindingId, RegisteredTexture>,
    surfaces: RefCell<HashMap<SurfaceBindingId, RegisteredSurface>>,
    buffers: HashMap<BufferBindingId, RegisteredBuffer>,
    raster: HashMap<RasterPipelineId, RegisteredRaster>,
    compute: HashMap<ComputePipelineId, RegisteredCompute>,
    bindings: HashMap<BindingSetId, RegisteredBindings>,
    binding_resolutions: RefCell<Vec<TestBindingResolution>>,
}

impl TestRegistry {
    /// Creates an empty registry for one test backend device.
    pub fn new(device: DeviceIdentity) -> Self {
        Self {
            device,
            textures: HashMap::new(),
            surfaces: RefCell::new(HashMap::new()),
            buffers: HashMap::new(),
            raster: HashMap::new(),
            compute: HashMap::new(),
            bindings: HashMap::new(),
            binding_resolutions: RefCell::new(Vec::new()),
        }
    }
    /// Registers an imported texture and returns its observable lease probe.
    pub fn register_texture(
        &mut self,
        id: TextureBindingId,
        physical: TestTexture,
        descriptor: TextureDesc,
        usage: TextureUsage,
        initial_state: ResourceAccessState,
    ) -> TestLeaseProbe {
        let (lease, probe) = TestLease::fresh();
        self.textures.insert(
            id,
            RegisteredTexture {
                physical,
                descriptor,
                usage,
                initial_state,
                lease,
            },
        );
        probe
    }
    /// Registers an acquired presentation image and returns its observable lease probe.
    pub fn register_surface(
        &mut self,
        id: SurfaceBindingId,
        physical: TestTexture,
        descriptor: TextureDesc,
        usage: TextureUsage,
        initial_state: ResourceAccessState,
        presentation: TestPresentationToken,
    ) -> TestLeaseProbe {
        let (lease, probe) = TestLease::fresh();
        self.surfaces.get_mut().insert(
            id,
            RegisteredSurface {
                texture: RegisteredTexture {
                    physical,
                    descriptor,
                    usage,
                    initial_state,
                    lease,
                },
                presentation,
            },
        );
        probe
    }
    /// Registers an imported buffer and returns its observable lease probe.
    pub fn register_buffer(
        &mut self,
        id: BufferBindingId,
        physical: TestBuffer,
        descriptor: BufferDesc,
        usage: BufferUsage,
        initial_state: ResourceAccessState,
    ) -> TestLeaseProbe {
        let (lease, probe) = TestLease::fresh();
        self.buffers.insert(
            id,
            RegisteredBuffer {
                physical,
                descriptor,
                usage,
                initial_state,
                lease,
            },
        );
        probe
    }
    /// Registers a raster pipeline and returns its observable lease probe.
    pub fn register_raster_pipeline(
        &mut self,
        id: RasterPipelineId,
        physical: TestRasterPipeline,
    ) -> TestLeaseProbe {
        let (lease, probe) = TestLease::fresh();
        self.raster.insert(id, RegisteredRaster { physical, lease });
        probe
    }
    /// Registers a compute pipeline and returns its observable lease probe.
    pub fn register_compute_pipeline(
        &mut self,
        id: ComputePipelineId,
        physical: TestComputePipeline,
    ) -> TestLeaseProbe {
        let (lease, probe) = TestLease::fresh();
        self.compute
            .insert(id, RegisteredCompute { physical, lease });
        probe
    }
    /// Registers a binding recipe and returns its observable lease probe.
    pub fn register_bindings(
        &mut self,
        id: BindingSetId,
        physical: TestBindings,
    ) -> TestLeaseProbe {
        let (lease, probe) = TestLease::fresh();
        self.bindings.insert(
            id,
            RegisteredBindings {
                physical,
                lease,
                expected_resources: None,
                expected_dynamic_offsets: None,
            },
        );
        probe
    }

    /// Registers a binding recipe with an exact expected per-frame contract.
    ///
    /// This is a conformance-test helper: a mismatch is reported as
    /// [`RecordingErrorKind::IncompatibleBindingRecipe`]. Native providers
    /// normally derive the equivalent requirements from their descriptor and
    /// pipeline metadata.
    pub fn register_bindings_with_contract(
        &mut self,
        id: BindingSetId,
        physical: TestBindings,
        resources: Vec<TestBindingResource>,
        dynamic_offsets: Vec<u32>,
    ) -> TestLeaseProbe {
        let (lease, probe) = TestLease::fresh();
        self.bindings.insert(
            id,
            RegisteredBindings {
                physical,
                lease,
                expected_resources: Some(resources),
                expected_dynamic_offsets: Some(dynamic_offsets),
            },
        );
        probe
    }

    /// Returns all successful binding requests observed so far.
    pub fn binding_resolutions(&self) -> Vec<TestBindingResolution> {
        self.binding_resolutions.borrow().clone()
    }
}

fn recording_error(_kind: RecordingErrorKind, _detail: impl Into<String>) -> RecordingError {
    RecordingError {
        kind: _kind,
        context: DiagnosticContext {
            passes: Vec::new(),
            resource: None,
            texture_slot: None,
            buffer_slot: None,
            detail: _detail.into(),
            unsupported: None,
        },
    }
}

impl FrameResourceProvider<TestRhi> for TestRegistry {
    fn texture(
        &self,
        _id: TextureBindingId,
    ) -> Result<BoundTexture<TestTexture, TestLease>, FrameBindingError> {
        let item = self.textures.get(&_id).ok_or_else(|| FrameBindingError {
            kind: FrameBindingErrorKind::MissingTexture,
            texture_slot: None,
            buffer_slot: None,
            resource: None,
            surface_binding: None,
            detail: "test texture binding is not registered".to_owned(),
        })?;
        Ok(BoundTexture {
            device: self.device,
            identity: item.physical.identity(),
            physical: item.physical,
            descriptor: item.descriptor,
            usage: item.usage,
            initial_state: item.initial_state,
            lease: item.lease.clone(),
        })
    }
    fn buffer(
        &self,
        _id: BufferBindingId,
    ) -> Result<BoundBuffer<TestBuffer, TestLease>, FrameBindingError> {
        let item = self.buffers.get(&_id).ok_or_else(|| FrameBindingError {
            kind: FrameBindingErrorKind::MissingBuffer,
            texture_slot: None,
            buffer_slot: None,
            resource: None,
            surface_binding: None,
            detail: "test buffer binding is not registered".to_owned(),
        })?;
        Ok(BoundBuffer {
            device: self.device,
            identity: item.physical.identity(),
            physical: item.physical,
            descriptor: item.descriptor,
            usage: item.usage,
            initial_state: item.initial_state,
            lease: item.lease.clone(),
        })
    }
    fn surface(
        &self,
        _id: SurfaceBindingId,
    ) -> Result<BoundSurfaceTexture<TestTexture, TestLease, TestPresentationToken>, FrameBindingError>
    {
        let item = self
            .surfaces
            .borrow_mut()
            .remove(&_id)
            .ok_or_else(|| FrameBindingError {
                kind: FrameBindingErrorKind::MissingSurface,
                texture_slot: None,
                buffer_slot: None,
                resource: None,
                surface_binding: Some(_id),
                detail: "test surface binding is not registered".to_owned(),
            })?;
        Ok(BoundSurfaceTexture {
            texture: BoundTexture {
                device: self.device,
                identity: item.texture.physical.identity(),
                physical: item.texture.physical,
                descriptor: item.texture.descriptor,
                usage: item.texture.usage,
                initial_state: item.texture.initial_state,
                lease: item.texture.lease.clone(),
            },
            presentation: item.presentation,
        })
    }
}

impl RenderObjectProvider<TestRhi> for TestRegistry {
    fn raster_pipeline(
        &self,
        _id: RasterPipelineId,
    ) -> Result<BoundRasterPipeline<TestRasterPipeline, TestLease>, RecordingError> {
        let item = self.raster.get(&_id).ok_or_else(|| {
            recording_error(
                RecordingErrorKind::IncompatibleBindingRecipe,
                "test raster pipeline is not registered",
            )
        })?;
        Ok(BoundRasterPipeline {
            device: self.device,
            physical: item.physical,
            lease: item.lease.clone(),
        })
    }
    fn compute_pipeline(
        &self,
        _id: ComputePipelineId,
    ) -> Result<BoundComputePipeline<TestComputePipeline, TestLease>, RecordingError> {
        let item = self.compute.get(&_id).ok_or_else(|| {
            recording_error(
                RecordingErrorKind::IncompatibleBindingRecipe,
                "test compute pipeline is not registered",
            )
        })?;
        Ok(BoundComputePipeline {
            device: self.device,
            physical: item.physical,
            lease: item.lease.clone(),
        })
    }
    fn bindings(
        &self,
        _id: BindingSetId,
        _resources: &[ResolvedBindingResource<'_, TestTexture, TestBuffer>],
        _dynamic_offsets: &[u32],
    ) -> Result<BoundBindings<TestBindings, TestLease>, RecordingError> {
        let item = self.bindings.get(&_id).ok_or_else(|| {
            recording_error(
                RecordingErrorKind::IncompatibleBindingRecipe,
                "test binding recipe is not registered",
            )
        })?;
        let resources = _resources
            .iter()
            .map(|resource| match resource {
                ResolvedBindingResource::Texture {
                    physical,
                    range,
                    semantic,
                } => TestBindingResource::Texture {
                    physical: **physical,
                    range: *range,
                    semantic: *semantic,
                },
                ResolvedBindingResource::Buffer {
                    physical,
                    range,
                    semantic,
                } => TestBindingResource::Buffer {
                    physical: **physical,
                    range: *range,
                    semantic: *semantic,
                },
            })
            .collect::<Vec<_>>();
        if item
            .expected_resources
            .as_ref()
            .is_some_and(|expected| expected != &resources)
            || item
                .expected_dynamic_offsets
                .as_ref()
                .is_some_and(|expected| expected != _dynamic_offsets)
        {
            return Err(recording_error(
                RecordingErrorKind::IncompatibleBindingRecipe,
                "binding recipe does not match graph-authorized resources or dynamic offsets",
            ));
        }
        self.binding_resolutions
            .borrow_mut()
            .push(TestBindingResolution {
                recipe: _id,
                resources,
                dynamic_offsets: _dynamic_offsets.to_vec(),
            });
        Ok(BoundBindings {
            device: self.device,
            physical: item.physical,
            lease: item.lease.clone(),
        })
    }
}
