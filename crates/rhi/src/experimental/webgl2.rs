//! Browser-private WebGL2 implementation of the retained unlit indexed ABI.
//!
//! This is intentionally a closed execution seam, not a WebGL wrapper.  The
//! only input is already prepared position/index/PVM/color data; all browser
//! objects, including the canvas context, remain owned here.

use core::fmt;
use std::collections::VecDeque;

#[cfg(test)]
use std::cell::Cell;

use js_sys::{Float32Array, Object, Reflect, Uint32Array};
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{
    HtmlCanvasElement, WebGl2RenderingContext as Gl, WebGlBuffer, WebGlFramebuffer, WebGlProgram,
    WebGlSampler, WebGlShader, WebGlSync, WebGlTexture, WebGlUniformLocation,
    WebGlVertexArrayObject,
};

use fluxel_rendergraph::{
    BufferCapabilities, BufferUsageKind, CompileError, CompileResult, CompiledGraph,
    DeviceCapabilities, DeviceLimits, LoadOp, PassKind, QueueCapabilities, QueueDescriptor,
    QueueId, RecordingCapabilities, RecordingModel, RenderGraph, ResourceAccessState,
    ResourceUsageSummary, StoreOp, SurfaceCapabilities, SynchronizationCapabilities, TextureFormat,
    TextureFormatCapabilities, TextureUsageKind, TimestampCapabilities,
    TransientResourceCapabilities, TransitionCapabilities,
};

mod resource_floor;
#[cfg(test)]
mod tests;
use resource_floor::ResourceFloorObjects;
pub use resource_floor::WebGl2ResourceFloorEvidence;

/// Compiles a graph for the closed WebGL2 executor before a canvas context or
/// any browser GPU object is created.
///
/// This is deliberately a capability gate, not a lowering path.  The retained
/// WebGL2 ABI has fixed default-framebuffer raster work only: compute and storage resources are
/// rejected by RenderGraph with its structured
/// [`UnsupportedCapability`](fluxel_rendergraph::UnsupportedCapability)
/// evidence.  Callers must select a raster variant before opening a session;
/// this function never silently lowers a graph.
pub fn compile_for_webgl2<F: 'static>(graph: &RenderGraph<F>) -> CompileResult<F> {
    graph.compile(&webgl2_capabilities())
}

/// Returns the capability facts of the closed WebGL2 executor.
///
/// The facts describe this RHI implementation, rather than an optimistic
/// projection of WebGL extensions: it has one immediate-context default-framebuffer
/// raster/present graph queue and no graph lowering for compute, copy, sampling,
/// storage-buffer, or storage-texture work. [`WebGl2Session::run_resource_floor_fixture`]
/// is separate, closed browser conformance evidence; it does not make those
/// operations available to arbitrary RenderGraph workloads.
pub fn webgl2_capabilities() -> DeviceCapabilities {
    let rgba8 = TextureFormatCapabilities::builder(TextureFormat::Rgba8Unorm)
        .sampled(false, false)
        .storage(false, false)
        .attachments(true, false, vec![1])
        .copies(false, false)
        .build();
    DeviceCapabilities::builder()
        .queue(QueueDescriptor::new(
            QueueId::new(0),
            QueueCapabilities::new(true, false, false, true),
        ))
        .recording(RecordingCapabilities::new(
            RecordingModel::ImmediateContext,
            false,
        ))
        .transitions(TransitionCapabilities::BackendManaged)
        .synchronization(SynchronizationCapabilities::SingleQueueOrdering)
        .timestamps(TimestampCapabilities::Unsupported)
        .transient_resources(TransientResourceCapabilities::new(false, false, false))
        .limits(DeviceLimits::new(4, 256))
        .buffers(BufferCapabilities::new(false, false, false))
        .texture_format(rgba8)
        .surface(SurfaceCapabilities::new(
            vec![TextureFormat::Rgba8Unorm],
            true,
            false,
        ))
        .build()
}

/// Test-only observation of browser actions which must remain absent when the
/// compile-time capability gate rejects a graph.
#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct BrowserSideEffects {
    context_creation: u32,
    resource_creation: u32,
    recording: u32,
    submission: u32,
}

#[cfg(test)]
thread_local! {
    static BROWSER_SIDE_EFFECTS: Cell<BrowserSideEffects> = const { Cell::new(BrowserSideEffects {
        context_creation: 0,
        resource_creation: 0,
        recording: 0,
        submission: 0,
    }) };
}

#[cfg(test)]
#[derive(Clone, Copy)]
enum BrowserSideEffect {
    ContextCreation,
    ResourceCreation,
    Recording,
    Submission,
}

#[cfg(test)]
fn note_browser_side_effect(effect: BrowserSideEffect) {
    BROWSER_SIDE_EFFECTS.with(|cell| {
        let mut observed = cell.get();
        match effect {
            BrowserSideEffect::ContextCreation => observed.context_creation += 1,
            BrowserSideEffect::ResourceCreation => observed.resource_creation += 1,
            BrowserSideEffect::Recording => observed.recording += 1,
            BrowserSideEffect::Submission => observed.submission += 1,
        }
        cell.set(observed);
    });
}

#[cfg(test)]
fn reset_browser_side_effects() {
    BROWSER_SIDE_EFFECTS.with(|cell| cell.set(BrowserSideEffects::default()));
}

#[cfg(test)]
fn browser_side_effects() -> BrowserSideEffects {
    BROWSER_SIDE_EFFECTS.with(Cell::get)
}

/// RHI-owned view of the portable fixed-scene graph contract.
pub struct FixedUnlitGraph<'a> {
    compiled: &'a CompiledGraph<()>,
    draw_count: usize,
    extent: [u32; 2],
}

impl<'a> FixedUnlitGraph<'a> {
    /// Associates a compiled portable graph with its closed physical draw ABI.
    pub const fn new(compiled: &'a CompiledGraph<()>, draw_count: usize, extent: [u32; 2]) -> Self {
        Self {
            compiled,
            draw_count,
            extent,
        }
    }
}

/// The closed physical input consumed by the Stage 1 unlit WebGL recipe.
///
/// It contains no renderer type so RHI never depends on renderer semantics.
#[derive(Clone, Copy, Debug)]
pub struct FixedUnlitDraw<'a> {
    /// Position-only vertices in the portable clip-volume convention.
    pub positions: &'a [[f32; 3]],
    /// `u32` triangle-list indices.
    pub indices: &'a [u32],
    /// Column-major `P * V * M` matrix followed by a linear RGBA color.
    pub pvm_and_color: [f32; 20],
    /// Stable renderer insertion order, retained for diagnostics.
    pub insertion_index: usize,
}

/// Structured browser session state reported to its binding owner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WebGl2SessionState {
    /// Resources are configured and may render.
    Active,
    /// New rendering is paused or has a zero-sized target.
    Suspended,
    /// The browser invalidated the context generation.
    Lost,
    /// Explicit disposal completed.
    Disposed,
    /// A browser operation made safe continuation unknowable.
    Poisoned,
}

/// A closed diagnostic emitted by the browser executor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WebGl2Diagnostic {
    /// Stable machine-readable diagnostic code.
    pub code: &'static str,
    /// Closed severity label.
    pub severity: &'static str,
    /// Stable operation category.
    pub operation: &'static str,
    /// Generation affected by the diagnostic.
    pub generation: u64,
    /// Human-readable browser diagnostic retained without erasing its category.
    pub message: String,
    /// Flat key/value facts safe to forward through the wasm bridge.
    pub context: Vec<(&'static str, String)>,
}

/// A rejected browser executor operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WebGl2SessionError {
    /// The explicit target did not contain a WebGL2 canvas.
    CanvasUnavailable,
    /// The browser did not provide WebGL2 for the supplied canvas.
    ContextUnavailable,
    /// An operation was attempted outside the active state.
    State(WebGl2SessionState),
    /// Three still-live GPU fences enforce bounded frames in flight.
    Busy,
    /// Browser/driver failure while executing a named operation.
    Browser {
        /// Stable machine-readable failure code.
        code: &'static str,
        /// Closed operation category which failed.
        operation: &'static str,
        /// Resource generation affected by the failure.
        generation: u64,
        /// Browser-provided or executor-generated reason.
        message: String,
    },
}

impl fmt::Display for WebGl2SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for WebGl2SessionError {}

impl WebGl2SessionError {
    /// Stable machine-readable failure code.
    pub const fn code(&self) -> &'static str {
        match self {
            Self::CanvasUnavailable => "canvas-unavailable",
            Self::ContextUnavailable => "context-unavailable",
            Self::State(_) => "invalid-state",
            Self::Busy => "frames-in-flight-busy",
            Self::Browser { code, .. } => code,
        }
    }
    /// Closed operation category associated with the failure.
    pub const fn operation(&self) -> &'static str {
        match self {
            Self::CanvasUnavailable => "open-canvas",
            Self::ContextUnavailable => "create-context",
            Self::State(_) => "lifecycle",
            Self::Busy => "wait",
            Self::Browser { operation, .. } => operation,
        }
    }
    /// Resource generation affected by the failure, or zero before creation.
    pub const fn generation(&self) -> u64 {
        match self {
            Self::Browser { generation, .. } => *generation,
            _ => 0,
        }
    }
    /// Human-readable error context.
    pub fn message(&self) -> String {
        match self {
            Self::Browser { message, .. } => message.clone(),
            Self::State(state) => format!("invalid session state: {state:?}"),
            _ => self.to_string(),
        }
    }
}

/// Failure while opening a WebGL2 session for a specific render graph.
///
/// Graph compilation is reported separately so callers can inspect the exact
/// [`UnsupportedCapability`](fluxel_rendergraph::UnsupportedCapability)
/// rather than receiving an unstructured browser-open failure.
#[derive(Clone, Debug)]
pub enum WebGl2GraphOpenError {
    /// The graph is outside the closed WebGL2 capability profile.
    Compile(CompileError),
    /// The graph was legal, but opening the canvas-bound session failed.
    Session(WebGl2SessionError),
}

impl fmt::Display for WebGl2GraphOpenError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compile(error) => write!(formatter, "{error}"),
            Self::Session(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for WebGl2GraphOpenError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Compile(error) => Some(error),
            Self::Session(error) => Some(error),
        }
    }
}

struct Objects {
    program: WebGlProgram,
    position: WebGlBuffer,
    index: WebGlBuffer,
    vertex_array: WebGlVertexArrayObject,
    uniform: WebGlUniformLocation,
    color: WebGlUniformLocation,
}

struct PendingFence {
    sync: WebGlSync,
    generation: u64,
}

/// One explicit canvas-bound, owner-thread WebGL2 session.
pub struct WebGl2Session {
    canvas: HtmlCanvasElement,
    gl: Gl,
    objects: Option<Objects>,
    resource_floor: Option<ResourceFloorObjects>,
    fences: VecDeque<PendingFence>,
    state: WebGl2SessionState,
    generation: u64,
    frame_marker: u64,
    diagnostics: Vec<WebGl2Diagnostic>,
}

impl WebGl2Session {
    /// Compiles `graph` against the closed WebGL2 profile before inspecting
    /// `canvas` or creating a browser context.
    ///
    /// This is the production entry point for graph-backed WebGL2 use.  A
    /// rejected graph therefore cannot create a context, allocate objects,
    /// record commands, or submit work. [`Self::new`] remains available for
    /// the legacy fixed-scene caller that already owns its compatibility
    /// decision.
    pub fn new_for_graph<F: 'static>(
        canvas: JsValue,
        graph: &RenderGraph<F>,
    ) -> Result<Self, WebGl2GraphOpenError> {
        compile_for_webgl2(graph).map_err(WebGl2GraphOpenError::Compile)?;
        Self::new(canvas).map_err(WebGl2GraphOpenError::Session)
    }

    /// Opens WebGL2 only for the explicitly supplied canvas JS value.
    pub fn new(canvas: JsValue) -> Result<Self, WebGl2SessionError> {
        let canvas = canvas
            .dyn_into::<HtmlCanvasElement>()
            .map_err(|_| WebGl2SessionError::CanvasUnavailable)?;
        #[cfg(test)]
        note_browser_side_effect(BrowserSideEffect::ContextCreation);
        let options = Object::new();
        Reflect::set(&options, &"preserveDrawingBuffer".into(), &JsValue::TRUE)
            .map_err(|e| browser("context-options", e))?;
        let value = canvas
            .get_context_with_context_options("webgl2", &options)
            .map_err(|e| browser("create-context", e))?
            .ok_or(WebGl2SessionError::ContextUnavailable)?;
        let gl = value
            .dyn_into::<Gl>()
            .map_err(|_| WebGl2SessionError::ContextUnavailable)?;
        let mut value = Self {
            canvas,
            gl,
            objects: None,
            resource_floor: None,
            fences: VecDeque::new(),
            state: WebGl2SessionState::Suspended,
            generation: 0,
            frame_marker: 0,
            diagnostics: Vec::new(),
        };
        if value.canvas.width() != 0 && value.canvas.height() != 0 {
            value.activate()?;
        }
        Ok(value)
    }

    /// Changes the drawing-buffer extent; zero suspends new work.
    ///
    /// HTML owns and replaces the default drawing buffer when these attributes
    /// change. This executor retains no framebuffer object or presentable-image
    /// identity across that replacement. Existing fences still protect only
    /// submitted commands and reusable program/buffer objects; resize is never
    /// treated as completion and does not retire a fence.
    pub fn resize(
        &mut self,
        width: u32,
        height: u32,
    ) -> Result<WebGl2SessionState, WebGl2SessionError> {
        self.require_not_terminal()?;
        self.canvas.set_width(width);
        self.canvas.set_height(height);
        if width == 0 || height == 0 {
            self.state = WebGl2SessionState::Suspended;
            return Ok(self.state);
        }
        if self.objects.is_some() {
            self.state = WebGl2SessionState::Active;
        } else {
            self.activate()?;
        }
        Ok(self.state)
    }

    /// Stops new rendering without claiming any GPU completion.
    pub fn suspend(&mut self) -> Result<(), WebGl2SessionError> {
        self.require_not_terminal()?;
        self.state = WebGl2SessionState::Suspended;
        Ok(())
    }

    /// Resumes the same configured canvas after an explicit suspension.
    pub fn resume(&mut self) -> Result<WebGl2SessionState, WebGl2SessionError> {
        self.require_not_terminal()?;
        if self.canvas.width() == 0 || self.canvas.height() == 0 {
            return Ok(self.state);
        }
        if self.objects.is_some() {
            self.state = WebGl2SessionState::Active;
        } else {
            self.activate()?;
        }
        Ok(self.state)
    }

    /// Records exactly one clear followed by the retained insertion-order draws.
    pub fn render(
        &mut self,
        graph: &FixedUnlitGraph<'_>,
        draws: &[FixedUnlitDraw<'_>],
    ) -> Result<u64, WebGl2SessionError> {
        self.require_active()?;
        self.validate_graph(graph, draws)?;
        self.validate_draws(draws)?;
        #[cfg(test)]
        note_browser_side_effect(BrowserSideEffect::Recording);
        if self.gl.is_context_lost() {
            self.context_lost();
            return Err(WebGl2SessionError::Browser {
                code: "context-lost",
                operation: "render",
                generation: self.generation,
                message: "WebGL context is lost".into(),
            });
        }
        self.collect_or_backpressure()?;
        let width = match i32::try_from(self.canvas.width()) {
            Ok(value) => value,
            Err(_) => return Err(self.fail("viewport", "width exceeds i32")),
        };
        let height = match i32::try_from(self.canvas.height()) {
            Ok(value) => value,
            Err(_) => return Err(self.fail("viewport", "height exceeds i32")),
        };
        self.gl.viewport(0, 0, width, height);
        self.gl.clear_color(0.0, 0.0, 0.0, 1.0);
        self.gl.clear(Gl::COLOR_BUFFER_BIT);
        let objects = self.objects.as_ref().expect("active sessions have objects");
        self.gl.use_program(Some(&objects.program));
        self.gl.bind_vertex_array(Some(&objects.vertex_array));
        self.gl
            .bind_buffer(Gl::ARRAY_BUFFER, Some(&objects.position));
        self.gl
            .bind_buffer(Gl::ELEMENT_ARRAY_BUFFER, Some(&objects.index));
        for draw in draws {
            let positions: Vec<f32> = draw.positions.iter().flatten().copied().collect();
            let values = Float32Array::from(positions.as_slice());
            self.gl
                .buffer_data_with_array_buffer_view(Gl::ARRAY_BUFFER, &values, Gl::DYNAMIC_DRAW);
            let indices = Uint32Array::from(draw.indices);
            self.gl.buffer_data_with_array_buffer_view(
                Gl::ELEMENT_ARRAY_BUFFER,
                &indices,
                Gl::DYNAMIC_DRAW,
            );
            self.gl.uniform_matrix4fv_with_f32_array(
                Some(&objects.uniform),
                false,
                &draw.pvm_and_color[..16],
            );
            self.gl
                .uniform4fv_with_f32_array(Some(&objects.color), &draw.pvm_and_color[16..]);
            self.gl
                .vertex_attrib_pointer_with_i32(0, 3, Gl::FLOAT, false, 0, 0);
            self.gl.enable_vertex_attrib_array(0);
            let count = i32::try_from(draw.indices.len())
                .expect("draw ABI was fully validated before recording");
            self.gl
                .draw_elements_with_i32(Gl::TRIANGLES, count, Gl::UNSIGNED_INT, 0);
        }
        self.check_error("draw")?;
        let fence = self
            .gl
            .fence_sync(Gl::SYNC_GPU_COMMANDS_COMPLETE, 0)
            .ok_or_else(|| self.fail("fence", "browser returned no sync object"))?;
        self.gl.flush();
        #[cfg(test)]
        note_browser_side_effect(BrowserSideEffect::Submission);
        self.fences.push_back(PendingFence {
            sync: fence,
            generation: self.generation,
        });
        self.frame_marker = self
            .frame_marker
            .checked_add(1)
            .ok_or_else(|| self.fail("frame", "frame marker exhausted"))?;
        Ok(self.frame_marker)
    }

    /// Executes and retains one fixed WebGL2 common-resource conformance recipe.
    ///
    /// The recipe uploads indexed vertex data, an immutable RGBA8 texture, and
    /// a uniform tint; draws it to an RGBA8 offscreen target with a
    /// `DEPTH_COMPONENT32F` attachment; copies that color texture; samples it
    /// in a second pass to the default
    /// framebuffer, and readbacks deterministic witnesses for the buffer and
    /// texture copy paths. It intentionally accepts no caller-selected shader,
    /// format, extent, sampler, or resource handle. The session owns its lease
    /// until [`Self::dispose`] or [`Self::context_lost`]; after a restored
    /// context, invoking this method creates a fresh lease rather than reusing
    /// an invalid browser object.
    pub fn run_resource_floor_fixture(
        &mut self,
    ) -> Result<WebGl2ResourceFloorEvidence, WebGl2SessionError> {
        self.require_active()?;
        if let Some(fixture) = &self.resource_floor {
            return Ok(fixture.evidence);
        }
        if self.gl.is_context_lost() {
            self.context_lost();
            return Err(self.reject("context-lost", "resource-floor", "WebGL context is lost"));
        }
        #[cfg(test)]
        note_browser_side_effect(BrowserSideEffect::ResourceCreation);
        let fixture = resource_floor::create(&self.gl)
            .map_err(|message| self.fail("resource-floor-create", message))?;
        self.resource_floor = Some(fixture);
        Ok(self
            .resource_floor
            .as_ref()
            .expect("fixture was just retained")
            .evidence)
    }

    /// Stops drawing because the adapter observed a context-loss event.
    pub fn context_lost(&mut self) {
        if !matches!(self.state, WebGl2SessionState::Disposed) {
            self.fences.clear();
            self.objects = None;
            self.resource_floor = None;
            self.state = WebGl2SessionState::Lost;
        }
    }

    /// Rebuilds closed resources only for this same canvas after restoration.
    pub fn context_restored(&mut self) -> Result<WebGl2SessionState, WebGl2SessionError> {
        if self.state != WebGl2SessionState::Lost {
            return Err(self.reject("invalid-state", "context-restored", "session is not lost"));
        }
        if self.canvas.width() == 0 || self.canvas.height() == 0 {
            self.state = WebGl2SessionState::Suspended;
            return Ok(self.state);
        }
        self.activate()?;
        Ok(self.state)
    }

    /// Explicitly proves normal disposal with `finish` before dropping objects.
    pub fn dispose(&mut self) -> Result<(), WebGl2SessionError> {
        if self.state == WebGl2SessionState::Disposed {
            return Ok(());
        }
        if self.state != WebGl2SessionState::Lost {
            self.gl.finish();
        }
        let finish_error = if self.state == WebGl2SessionState::Lost {
            None
        } else {
            self.check_error("dispose").err()
        };
        while let Some(fence) = self.fences.pop_front() {
            self.gl.delete_sync(Some(&fence.sync));
        }
        if let Some(objects) = self.objects.take() {
            destroy_objects(&self.gl, objects);
        }
        if let Some(fixture) = self.resource_floor.take() {
            resource_floor::destroy(&self.gl, fixture);
        }
        if let Some(error) = finish_error {
            return Err(error);
        }
        self.state = WebGl2SessionState::Disposed;
        Ok(())
    }
    /// Returns a non-draining diagnostic snapshot.
    pub fn diagnostics(&self) -> &[WebGl2Diagnostic] {
        &self.diagnostics
    }
    /// Returns the closed state.
    pub const fn state(&self) -> WebGl2SessionState {
        self.state
    }
    /// Returns the monotonically allocated browser resource generation.
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    fn activate(&mut self) -> Result<(), WebGl2SessionError> {
        if self.gl.is_context_lost() {
            self.state = WebGl2SessionState::Lost;
            return Err(self.reject("context-lost", "create-resources", "WebGL context is lost"));
        }
        #[cfg(test)]
        note_browser_side_effect(BrowserSideEffect::ResourceCreation);
        let objects = create_objects(&self.gl).map_err(|e| self.fail("create-resources", e))?;
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| self.fail("generation", "generation exhausted"))?;
        self.objects = Some(objects);
        self.state = WebGl2SessionState::Active;
        Ok(())
    }
    fn collect_or_backpressure(&mut self) -> Result<(), WebGl2SessionError> {
        if self.fences.len() < 3 {
            return Ok(());
        }
        let fence = self.fences.front().expect("checked nonempty");
        if fence.generation != self.generation {
            self.state = WebGl2SessionState::Poisoned;
            return Err(self.fail("wait", "fence belongs to another resource generation"));
        }
        let result = self.gl.client_wait_sync_with_u32(&fence.sync, 0, 0);
        if result == Gl::ALREADY_SIGNALED || result == Gl::CONDITION_SATISFIED {
            let fence = self.fences.pop_front().expect("front exists");
            self.gl.delete_sync(Some(&fence.sync));
            Ok(())
        } else if result == Gl::TIMEOUT_EXPIRED {
            Err(WebGl2SessionError::Busy)
        } else {
            self.state = WebGl2SessionState::Poisoned;
            Err(self.fail("wait", "clientWaitSync failed"))
        }
    }
    fn require_not_terminal(&mut self) -> Result<(), WebGl2SessionError> {
        if matches!(
            self.state,
            WebGl2SessionState::Disposed | WebGl2SessionState::Poisoned
        ) {
            Err(self.reject("invalid-state", "lifecycle", "session is terminal"))
        } else {
            Ok(())
        }
    }
    fn require_active(&mut self) -> Result<(), WebGl2SessionError> {
        if self.state == WebGl2SessionState::Active {
            Ok(())
        } else {
            Err(self.reject("invalid-state", "render", "session is not active"))
        }
    }
    fn check_error(&mut self, operation: &'static str) -> Result<(), WebGl2SessionError> {
        let error = self.gl.get_error();
        if error == Gl::NO_ERROR {
            Ok(())
        } else {
            self.state = WebGl2SessionState::Poisoned;
            Err(self.fail(operation, format!("WebGL error 0x{error:04x}")))
        }
    }
    fn fail(&mut self, operation: &'static str, message: impl Into<String>) -> WebGl2SessionError {
        self.reject("webgl-operation-failed", operation, message)
    }
    fn reject(
        &mut self,
        code: &'static str,
        operation: &'static str,
        message: impl Into<String>,
    ) -> WebGl2SessionError {
        let message = message.into();
        self.diagnostics.push(WebGl2Diagnostic {
            code,
            severity: "error",
            operation,
            generation: self.generation,
            message: message.clone(),
            context: vec![("state", format!("{:?}", self.state))],
        });
        WebGl2SessionError::Browser {
            code,
            operation,
            generation: self.generation,
            message,
        }
    }

    fn validate_graph(
        &mut self,
        graph: &FixedUnlitGraph<'_>,
        draws: &[FixedUnlitDraw<'_>],
    ) -> Result<(), WebGl2SessionError> {
        let plan = graph.compiled.execution_plan();
        let valid_pass = plan.passes().len() == 1
            && plan.passes()[0].kind == PassKind::Raster
            && plan.passes()[0].raster.as_ref().is_some_and(|raster| {
                raster.depth_stencil.is_none()
                    && raster.colors.len() == 1
                    && raster.colors[0].descriptor.index == 0
                    && raster.colors[0].descriptor.operations.load
                        == LoadOp::Clear([0.0, 0.0, 0.0, 1.0])
                    && raster.colors[0].descriptor.operations.store == StoreOp::Store
            });
        let presents = plan
            .final_transitions()
            .iter()
            .any(|transition| transition.after == ResourceAccessState::Present);
        let mut vertex_buffers = 0;
        let mut index_buffers = 0;
        let mut uniform_buffers = 0;
        let mut presentable_textures = 0;
        let mut unexpected_usage = false;
        for requirement in plan.resource_requirements() {
            match requirement.usage {
                ResourceUsageSummary::Buffer(usage) => {
                    vertex_buffers += usize::from(usage.contains(BufferUsageKind::Vertex));
                    index_buffers += usize::from(usage.contains(BufferUsageKind::Index));
                    uniform_buffers += usize::from(usage.contains(BufferUsageKind::Uniform));
                    unexpected_usage |= usage.contains(BufferUsageKind::StorageRead)
                        || usage.contains(BufferUsageKind::StorageWrite)
                        || usage.contains(BufferUsageKind::Indirect);
                }
                ResourceUsageSummary::Texture(usage) => {
                    presentable_textures += usize::from(
                        usage.contains(TextureUsageKind::ColorAttachment)
                            && usage.contains(TextureUsageKind::Present),
                    );
                    unexpected_usage |= usage.contains(TextureUsageKind::StorageRead)
                        || usage.contains(TextureUsageKind::StorageWrite)
                        || usage.contains(TextureUsageKind::DepthStencilAttachment);
                }
            }
        }
        let valid_resources = vertex_buffers == graph.draw_count
            && index_buffers == graph.draw_count
            && uniform_buffers == graph.draw_count
            && presentable_textures == 1
            && plan.resource_requirements().len() == graph.draw_count * 3 + 1
            && !unexpected_usage;
        let canvas_extent = [self.canvas.width(), self.canvas.height()];
        if !valid_pass
            || !presents
            || !valid_resources
            || graph.draw_count == 0
            || graph.draw_count != draws.len()
            || graph.extent != canvas_extent
        {
            return Err(self.reject(
                "fixed-graph-contract-rejected",
                "validate-graph",
                "compiled graph, draw count, or drawing-buffer extent violated the fixed ABI",
            ));
        }
        Ok(())
    }

    fn validate_draws(&mut self, draws: &[FixedUnlitDraw<'_>]) -> Result<(), WebGl2SessionError> {
        for (expected_index, draw) in draws.iter().enumerate() {
            let valid_indices = !draw.indices.is_empty()
                && draw.indices.len().is_multiple_of(3)
                && i32::try_from(draw.indices.len()).is_ok()
                && draw
                    .indices
                    .iter()
                    .all(|index| usize::try_from(*index).is_ok_and(|i| i < draw.positions.len()));
            let valid_values = draw
                .positions
                .iter()
                .flatten()
                .all(|value| value.is_finite())
                && draw.pvm_and_color.iter().all(|value| value.is_finite());
            if draw.insertion_index != expected_index || !valid_indices || !valid_values {
                return Err(self.reject(
                    "fixed-draw-contract-rejected",
                    "validate-draws",
                    "draw order, index topology, or finite-value ABI was rejected",
                ));
            }
        }
        Ok(())
    }
}

fn browser(operation: &'static str, error: JsValue) -> WebGl2SessionError {
    WebGl2SessionError::Browser {
        code: "webgl-operation-failed",
        operation,
        generation: 0,
        message: format!("{error:?}"),
    }
}
fn create_objects(gl: &Gl) -> Result<Objects, String> {
    let vertex = compile(
        gl,
        Gl::VERTEX_SHADER,
        "#version 300 es\nlayout(location=0) in vec3 a_position; uniform mat4 u_pvm; void main(){ gl_Position=u_pvm*vec4(a_position,1.0); }",
    )?;
    let fragment = compile(
        gl,
        Gl::FRAGMENT_SHADER,
        "#version 300 es\nprecision mediump float; uniform vec4 u_color; out vec4 out_color; void main(){ out_color=u_color; }",
    )?;
    let program = match gl.create_program() {
        Some(program) => program,
        None => {
            gl.delete_shader(Some(&vertex));
            gl.delete_shader(Some(&fragment));
            return Err("createProgram returned null".into());
        }
    };
    gl.attach_shader(&program, &vertex);
    gl.attach_shader(&program, &fragment);
    gl.link_program(&program);
    if !gl
        .get_program_parameter(&program, Gl::LINK_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        let message = gl
            .get_program_info_log(&program)
            .unwrap_or_else(|| "program link failed".into());
        gl.delete_program(Some(&program));
        gl.delete_shader(Some(&vertex));
        gl.delete_shader(Some(&fragment));
        return Err(message);
    }
    gl.detach_shader(&program, &vertex);
    gl.detach_shader(&program, &fragment);
    gl.delete_shader(Some(&vertex));
    gl.delete_shader(Some(&fragment));
    let position = match gl.create_buffer() {
        Some(value) => value,
        None => {
            gl.delete_program(Some(&program));
            return Err("create position buffer returned null".into());
        }
    };
    let index = match gl.create_buffer() {
        Some(value) => value,
        None => {
            gl.delete_buffer(Some(&position));
            gl.delete_program(Some(&program));
            return Err("create index buffer returned null".into());
        }
    };
    let vertex_array = match gl.create_vertex_array() {
        Some(value) => value,
        None => {
            gl.delete_buffer(Some(&position));
            gl.delete_buffer(Some(&index));
            gl.delete_program(Some(&program));
            return Err("createVertexArray returned null".into());
        }
    };
    let uniform = match gl.get_uniform_location(&program, "u_pvm") {
        Some(value) => value,
        None => {
            destroy_partial_objects(gl, &program, &position, &index, &vertex_array);
            return Err("u_pvm missing".into());
        }
    };
    let color = match gl.get_uniform_location(&program, "u_color") {
        Some(value) => value,
        None => {
            destroy_partial_objects(gl, &program, &position, &index, &vertex_array);
            return Err("u_color missing".into());
        }
    };
    Ok(Objects {
        program,
        position,
        index,
        vertex_array,
        uniform,
        color,
    })
}

fn destroy_partial_objects(
    gl: &Gl,
    program: &WebGlProgram,
    position: &WebGlBuffer,
    index: &WebGlBuffer,
    vertex_array: &WebGlVertexArrayObject,
) {
    gl.delete_vertex_array(Some(vertex_array));
    gl.delete_buffer(Some(position));
    gl.delete_buffer(Some(index));
    gl.delete_program(Some(program));
}

fn destroy_objects(gl: &Gl, objects: Objects) {
    gl.bind_vertex_array(None);
    gl.use_program(None);
    gl.bind_buffer(Gl::ARRAY_BUFFER, None);
    gl.bind_buffer(Gl::ELEMENT_ARRAY_BUFFER, None);
    gl.delete_vertex_array(Some(&objects.vertex_array));
    gl.delete_buffer(Some(&objects.position));
    gl.delete_buffer(Some(&objects.index));
    gl.delete_program(Some(&objects.program));
}

fn compile(gl: &Gl, kind: u32, source: &str) -> Result<WebGlShader, String> {
    let shader = gl.create_shader(kind).ok_or("createShader returned null")?;
    gl.shader_source(&shader, source);
    gl.compile_shader(&shader);
    if gl
        .get_shader_parameter(&shader, Gl::COMPILE_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(shader)
    } else {
        let message = gl
            .get_shader_info_log(&shader)
            .unwrap_or_else(|| "shader compile failed".into());
        gl.delete_shader(Some(&shader));
        Err(message)
    }
}
