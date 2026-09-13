//! Browser-private WebGPU executor for the retained fixed unlit ABI.
//!
//! It accepts only the prepared fixed graph/draw shape; all `GPU*` objects,
//! listeners and Promise continuations remain inside this RHI boundary.

use std::{
    cell::{Cell, RefCell},
    collections::VecDeque,
    rc::Rc,
};

use js_sys::{Array, Float32Array, Object, Promise, Uint32Array};
use wasm_bindgen::{JsCast, JsValue, closure::Closure};
use wasm_bindgen_futures::{JsFuture, future_to_promise};
use web_sys::HtmlCanvasElement;

pub use super::webgpu_state::WebGpuSessionState;
use super::webgpu_state::{
    TerminalDisposalRoots, candidate_transaction_committable, recovery_attempt_current,
    terminal_dispose_ready,
};

mod accessors;
mod contract;
mod error;
mod format;
mod js;
mod resource_conformance;
mod resource_floor;
mod resources;
#[cfg(test)]
mod tests;

use contract::{validate, validate_resident};
use js::{
    BrowserRequestProvider, ProductionBrowserRequests, call0, call1, call2, call3, get, js_message,
    request_browser, set, to_js,
};
use resources::{
    FrameDrawResources, ResidentLease, ResidentRegistry, create_frame_resources,
    destroy_frame_resources, pipeline, unregister_uncaptured_error, write_buffer,
};

use resource_floor::{
    ResourceLease, ResourceRegistry, commit as commit_resource_floor,
    submit as submit_resource_floor,
};
#[cfg(test)]
use resource_floor::{observe_pending, prepare as prepare_resource_floor};

use fluxel_rendergraph::CompiledGraph;

const MAX_FRAMES_IN_FLIGHT: usize = 3;

/// A closed WebGPU canvas format accepted by the fixed presentation recipe.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WebGpuCanvasFormat {
    /// `rgba8unorm`.
    Rgba8Unorm,
    /// `bgra8unorm`.
    Bgra8Unorm,
}
/// Adapter facts retained for evidence; no browser adapter object escapes RHI.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct WebGpuAdapterInfo {
    /// Browser-reported vendor string.
    pub vendor: String,
    /// Browser-reported architecture string.
    pub architecture: String,
    /// Browser-reported device string.
    pub device: String,
}
/// Closed device-loss reason retained for lifecycle evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WebGpuLossReason {
    /// Explicit `device.destroy()` evidence loss.
    Destroyed,
    /// Any non-destroyed browser loss.
    Other,
}

/// RHI-owned view of the portable fixed-scene graph contract.
pub struct FixedUnlitGraph<'a> {
    compiled: &'a CompiledGraph<()>,
    draw_count: usize,
    extent: [u32; 2],
    format: WebGpuCanvasFormat,
}
impl<'a> FixedUnlitGraph<'a> {
    /// Associates a compiled graph with its fixed physical draw ABI.
    pub const fn new(
        compiled: &'a CompiledGraph<()>,
        draw_count: usize,
        extent: [u32; 2],
        format: WebGpuCanvasFormat,
    ) -> Self {
        Self {
            compiled,
            draw_count,
            extent,
            format,
        }
    }
}

/// One renderer-prepared indexed draw.  It deliberately contains no GPU type.
#[derive(Clone, Copy, Debug)]
pub struct FixedUnlitDraw<'a> {
    /// Portable clip-volume positions.
    pub positions: &'a [[f32; 3]],
    /// Triangle-list u32 indices.
    pub indices: &'a [u32],
    /// Column-major P*V*M followed by linear RGBA.
    pub pvm_and_color: [f32; 20],
    /// Renderer insertion order.
    pub insertion_index: usize,
}

/// One closed draw that binds a previously uploaded resident mesh. Its opaque
/// token retains the browser buffers even after a replacement removes lookup.
#[derive(Clone, Copy)]
pub struct FixedResidentUnlitDraw<'a> {
    /// Previously acquired mesh token for this exact device generation.
    pub mesh: &'a WebGpuResidentMesh,
    /// Column-major P*V*M followed by linear RGBA.
    pub pvm_and_color: [f32; 20],
    /// Renderer insertion order.
    pub insertion_index: usize,
}

/// Opaque caller-owned identity for one retained fixed-resource set.
///
/// The same key reuses its physical browser resources until device loss or
/// disposal; it never exposes a browser object.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct WebGpuResourceKey(pub(crate) u64);

impl WebGpuResourceKey {
    /// Creates an application-stable retained resource identity.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }
}

/// Exact logical asset revision used by the closed browser residency seam.
/// The renderer supplies stable logical and immutable-content generations;
/// the session adds its current device generation internally.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct WebGpuAssetKey {
    logical: u64,
    content: u64,
}
impl WebGpuAssetKey {
    /// Creates an exact logical asset/content key.
    pub const fn new(logical: u64, content: u64) -> Self {
        Self { logical, content }
    }
}

/// Opaque resident indexed mesh token. It contains no browser object.
#[derive(Clone)]
pub struct WebGpuResidentMesh {
    #[expect(
        dead_code,
        reason = "identity is retained for renderer-owned replacement diagnostics"
    )]
    key: WebGpuAssetKey,
    generation: u64,
    index_count: u32,
    lease: ResidentLease,
}

/// Opaque resident RGBA8 image token. It contains no browser object.
#[derive(Clone)]
pub struct WebGpuResidentImage {
    #[expect(
        dead_code,
        reason = "identity is retained for renderer-owned replacement diagnostics"
    )]
    key: WebGpuAssetKey,
    generation: u64,
    #[expect(
        dead_code,
        reason = "opaque lease keeps the image alive for future fixed textured recipes"
    )]
    lease: ResidentLease,
}

impl WebGpuResidentMesh {
    /// Device generation for diagnostics; no native identity is exposed.
    pub const fn generation(&self) -> u64 {
        self.generation
    }
    /// Fixed index count retained by this token.
    pub const fn index_count(&self) -> u32 {
        self.index_count
    }
}
impl WebGpuResidentImage {
    /// Device generation for diagnostics; no native identity is exposed.
    pub const fn generation(&self) -> u64 {
        self.generation
    }
}

/// CPU inputs for the closed Raster→Compute→Copy browser resource recipe.
///
/// This is not a command list: resource descriptors, usage flags, shader
/// modules, bind layouts, attachment formats and copy regions are fixed by
/// the RHI implementation.
pub struct FixedResourceFrame<'a> {
    /// Stable key selecting the retained physical resource set.
    pub resources: WebGpuResourceKey,
    /// Exactly three clip-space `f32x2` triangle positions.
    pub positions: &'a [[f32; 2]],
    /// Exactly `[0, 1, 2]` triangle-list indices.
    pub indices: &'a [u32],
    /// Fixed fragment multiplier for the sampled texel.
    pub tint: [f32; 4],
    /// Whole immutable one-pixel RGBA8 sampled-image contents.
    pub sampled_rgba8: [u8; 4],
}

/// Structured executor diagnostic. Normal backpressure and loss are states,
/// not diagnostics; a diagnostic represents an actual rejected operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WebGpuDiagnostic {
    /// Stable machine code.
    pub code: &'static str,
    /// Closed operation category.
    pub operation: &'static str,
    /// Device generation.
    pub generation: u64,
    /// Browser reason.
    pub message: String,
}

/// Closed result of a render attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WebGpuRenderOutcome {
    /// A frame was accepted with its monotonic marker.
    Submitted(u64),
    /// Three submitted frames still await completion.
    Backpressure,
    /// The target is non-drawable or explicitly suspended.
    Suspended,
    /// The device generation is lost or recovering.
    Lost,
}

/// Browser session rejection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum WebGpuSessionError {
    /// Input was not an HTML canvas.
    CanvasUnavailable,
    /// WebGPU was unavailable.
    Unavailable,
    /// Preferred format was outside the closed renderer profile.
    UnsupportedFormat(String),
    /// Operation was invalid for the current lifecycle.
    State(WebGpuSessionState),
    /// A closed graph/draw ABI check rejected the frame before acquisition.
    Contract(&'static str),
    /// Browser operation failed.
    Browser {
        /// Stable machine-readable reason.
        code: &'static str,
        /// Closed failing operation category.
        operation: &'static str,
        /// Device generation affected by the operation.
        generation: u64,
        /// Browser-provided reason retained for evidence.
        message: String,
    },
}

struct Shared {
    state: WebGpuSessionState,
    generation: u64,
    token: u64,
    next_marker: u64,
    diagnostics: Vec<WebGpuDiagnostic>,
    // A callback only mutates this shared terminal bit. It never captures a
    // mutable Session nor owns a GPU object, preventing old generations from
    // becoming a producer after recovery.
    lost: bool,
    loss_reason: Option<WebGpuLossReason>,
}
#[derive(Clone, Copy)]
struct InstallAttempt {
    format: WebGpuCanvasFormat,
    generation: u64,
    token: u64,
    expected_state: WebGpuSessionState,
}

impl Default for Shared {
    fn default() -> Self {
        Self {
            state: WebGpuSessionState::Suspended,
            generation: 0,
            token: 0,
            next_marker: 0,
            diagnostics: Vec::new(),
            lost: false,
            loss_reason: None,
        }
    }
}
struct Ticket {
    generation: u64,
    marker: u64,
    settled: Rc<RefCell<Option<Result<(), String>>>>,
    ok: Closure<dyn FnMut(JsValue)>,
    err: Closure<dyn FnMut(JsValue)>,
    promise: Promise,
    resources: Vec<FrameDrawResources>,
    resource_leases: Vec<ResourceLease>,
    resident_leases: Vec<ResidentLease>,
    browser_roots: Vec<JsValue>,
}
struct Objects {
    device: JsValue,
    queue: JsValue,
    context: JsValue,
    pipeline: JsValue,
    layout: JsValue,
    lost_promise: Promise,
    lost_settled: Rc<Cell<bool>>,
    lost_ok: Closure<dyn FnMut(JsValue)>,
    lost_err: Closure<dyn FnMut(JsValue)>,
    uncaptured: Closure<dyn FnMut(JsValue)>,
}

/// A short-lived clone of the browser handles needed to encode one frame.
///
/// This deliberately contains no callback roots.  Taking this snapshot drops
/// the `objects` `RefCell` borrow before entering browser code, whose methods
/// are allowed to synchronously re-enter wasm.
struct RenderObjects {
    device: JsValue,
    queue: JsValue,
    context: JsValue,
    pipeline: JsValue,
    layout: JsValue,
}

/// One owner-thread, canvas-bound WebGPU session.
///
/// `new` and `recover` are asynchronous because requestAdapter/requestDevice
/// are browser Promises. No raw browser GPU object is exposed by this type.
pub struct WebGpuSession {
    canvas: HtmlCanvasElement,
    desired_extent: Rc<Cell<[u32; 2]>>,
    canvas_epoch: Rc<Cell<u64>>,
    format: Rc<Cell<WebGpuCanvasFormat>>,
    shared: Rc<RefCell<Shared>>,
    objects: Rc<RefCell<Option<Objects>>>,
    // Keep recovered `device.lost` closures alive through settlement.
    retired_objects: Rc<RefCell<Vec<Objects>>>,
    tickets: Rc<RefCell<VecDeque<Ticket>>>,
    detached: Rc<RefCell<VecDeque<Ticket>>>,
    quarantined: Rc<RefCell<Vec<FrameDrawResources>>>,
    recovery_promise: Rc<RefCell<Option<Promise>>>,
    dispose_promise: Rc<RefCell<Option<Promise>>>,
    adapter_info: Rc<RefCell<WebGpuAdapterInfo>>,
    requests: Rc<dyn BrowserRequestProvider>,
    resource_registry: Rc<RefCell<ResourceRegistry>>,
    resident_registry: Rc<RefCell<ResidentRegistry>>,
}

impl WebGpuSession {
    /// Requests adapter/device and creates the closed fixed recipe for canvas.
    pub async fn new(canvas: JsValue) -> Result<Self, WebGpuSessionError> {
        Self::new_with_requests(canvas, Rc::new(ProductionBrowserRequests)).await
    }

    async fn new_with_requests(
        canvas: JsValue,
        requests: Rc<dyn BrowserRequestProvider>,
    ) -> Result<Self, WebGpuSessionError> {
        let canvas = canvas
            .dyn_into::<HtmlCanvasElement>()
            .map_err(|_| WebGpuSessionError::CanvasUnavailable)?;
        let (format, info, device, queue, context) =
            request_browser(&canvas, requests.as_ref()).await?;
        let shared = Rc::new(RefCell::new(Shared {
            state: WebGpuSessionState::Suspended,
            generation: 1,
            ..Shared::default()
        }));
        let mut value = Self {
            desired_extent: Rc::new(Cell::new([canvas.width(), canvas.height()])),
            canvas,
            canvas_epoch: Rc::new(Cell::new(0)),
            format: Rc::new(Cell::new(format)),
            shared,
            objects: Rc::new(RefCell::new(None)),
            retired_objects: Rc::new(RefCell::new(Vec::new())),
            tickets: Rc::new(RefCell::new(VecDeque::new())),
            detached: Rc::new(RefCell::new(VecDeque::new())),
            quarantined: Rc::new(RefCell::new(Vec::new())),
            recovery_promise: Rc::new(RefCell::new(None)),
            dispose_promise: Rc::new(RefCell::new(None)),
            adapter_info: Rc::new(RefCell::new(info)),
            requests,
            resource_registry: Rc::new(RefCell::new(ResourceRegistry::default())),
            resident_registry: Rc::new(RefCell::new(ResidentRegistry::default())),
        };
        let objects = match value
            .install(
                device.clone(),
                queue,
                context,
                InstallAttempt {
                    format,
                    generation: 1,
                    token: 0,
                    expected_state: WebGpuSessionState::Suspended,
                },
            )
            .await
        {
            Ok(objects) => objects,
            Err(error) => {
                let _ = call0(&device, "destroy");
                return Err(error);
            }
        };
        *value.objects.borrow_mut() = Some(objects);
        if value.desired_extent.get()[0] != 0 && value.desired_extent.get()[1] != 0 {
            if let Err(error) = value.configure() {
                let objects = value.objects.borrow_mut().take();
                if let Some(objects) = objects {
                    Self::destroy_objects(objects).await;
                }
                return Err(error);
            }
        }
        Ok(value)
    }

    /// Returns the selected closed preferred format.
    pub fn format(&self) -> WebGpuCanvasFormat {
        self.format.get()
    }
    /// Returns device-generation state, never a browser object.
    pub fn state(&self) -> WebGpuSessionState {
        self.shared.borrow().state
    }
    /// Returns current device generation.
    pub fn generation(&self) -> u64 {
        self.shared.borrow().generation
    }
    /// Returns a non-draining diagnostic snapshot.
    pub fn diagnostics(&self) -> Vec<WebGpuDiagnostic> {
        self.shared.borrow().diagnostics.clone()
    }

    /// Executes the private, fixed WebGPU resource-floor witness.
    ///
    /// This is deliberately test-only: it proves the browser implementation's
    /// closed resource recipes without turning the session into a public
    /// command, shader, or pipeline API.
    #[cfg(test)]
    pub(super) async fn resource_conformance(&mut self) -> Result<(), WebGpuSessionError> {
        self.collect();
        if self.state() != WebGpuSessionState::Active {
            return Err(WebGpuSessionError::State(self.state()));
        }
        let objects = self.render_objects()?;
        let frame = FixedResourceFrame {
            resources: WebGpuResourceKey::new(u64::MAX),
            positions: &[[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]],
            indices: &[0, 1, 2],
            tint: [1.0; 4],
            sampled_rgba8: [12, 34, 56, 255],
        };
        call1(&objects.device, "pushErrorScope", &"validation".into())
            .map_err(|error| self.fail("resource-conformance-scope", error))?;
        let prepared = prepare_resource_floor(
            &objects.device,
            &objects.queue,
            &mut self.resource_registry.borrow_mut(),
            self.generation(),
            frame,
        );
        // Pop the scope on both prepare and observation failure. Otherwise a
        // rejected fixture could leak a device-global diagnostic scope into a
        // later, unrelated browser operation.
        let observed = match prepared {
            Ok(pending) => observe_pending(&objects.queue, pending)
                .await
                .map_err(|error| ("resource-conformance", error)),
            Err(error) => Err(("resource-conformance-prepare", error)),
        };
        let scope = JsFuture::from(Promise::from(
            call0(&objects.device, "popErrorScope")
                .map_err(|error| self.fail("resource-conformance-scope", error))?,
        ))
        .await
        .map_err(|error| self.fail("resource-conformance-scope", error))?;
        if !scope.is_null() && !scope.is_undefined() {
            return Err(self.fail(
                "resource-conformance-validation",
                JsValue::from_str(&format!("WebGPU validation error: {scope:?}")),
            ));
        }
        observed.map_err(|(operation, error)| self.fail(operation, error))
    }

    /// Records the closed retained resource recipe on the active WebGPU device.
    ///
    /// This normal wasm production path retains physical resources by key and
    /// generation. It intentionally accepts no shader text, pipeline, browser
    /// handle, arbitrary binding layout, or arbitrary copy command.
    pub fn render_fixed_resources(
        &mut self,
        frame: FixedResourceFrame<'_>,
    ) -> Result<WebGpuRenderOutcome, WebGpuSessionError> {
        self.collect();
        match self.state() {
            WebGpuSessionState::Active => {}
            WebGpuSessionState::Suspended => return Ok(WebGpuRenderOutcome::Suspended),
            WebGpuSessionState::Lost | WebGpuSessionState::Recovering => {
                return Ok(WebGpuRenderOutcome::Lost);
            }
            state => return Err(WebGpuSessionError::State(state)),
        }
        if self.tickets.borrow().len() >= MAX_FRAMES_IN_FLIGHT {
            return Ok(WebGpuRenderOutcome::Backpressure);
        }
        let objects = self.render_objects()?;
        let (command, pending) = submit_resource_floor(
            &objects.device,
            &objects.queue,
            &mut self.resource_registry.borrow_mut(),
            self.generation(),
            frame,
        )
        .map_err(|error| self.fail("fixed-resource-record", error))?;
        let commands = Array::new();
        commands.push(&command);
        call1(&objects.queue, "submit", &commands)
            .map_err(|error| self.fail("fixed-resource-submit", error))?;
        commit_resource_floor(&pending);
        let completion = call0(&objects.queue, "onSubmittedWorkDone")
            .map_err(|error| self.fail("fixed-resource-completion", error))?;
        let marker = {
            let mut shared = self.shared.borrow_mut();
            shared.next_marker += 1;
            shared.next_marker
        };
        self.ticket_with_leases(
            marker,
            completion,
            Vec::new(),
            pending.leases,
            pending.roots,
            Vec::new(),
        );
        Ok(WebGpuRenderOutcome::Submitted(marker))
    }

    /// Retires one closed fixed-resource identity.
    ///
    /// Commands already accepted by the browser retain their ticket-owned
    /// leases; a subsequent submission for this key receives fresh resources.
    pub fn retire_fixed_resources(&mut self, key: WebGpuResourceKey) {
        self.collect();
        self.resource_registry.borrow_mut().retire_key(key);
    }

    /// Uploads or reuses one exact indexed-mesh revision for this device
    /// generation. The returned token is opaque and invalidated by replacement,
    /// retirement, or device recovery.
    pub fn resident_mesh(
        &mut self,
        key: WebGpuAssetKey,
        positions: &[[f32; 3]],
        indices: &[u32],
    ) -> Result<WebGpuResidentMesh, WebGpuSessionError> {
        self.collect();
        if self.state() != WebGpuSessionState::Active
            || positions.is_empty()
            || indices.is_empty()
            || !indices.len().is_multiple_of(3)
            || positions.iter().flatten().any(|value| !value.is_finite())
            || indices
                .iter()
                .any(|index| (*index as usize) >= positions.len())
        {
            return Err(WebGpuSessionError::Contract(
                "resident-mesh-contract-rejected",
            ));
        }
        let objects = self.render_objects()?;
        let lease = self
            .resident_registry
            .borrow_mut()
            .mesh(&objects.device, &objects.queue, key, positions, indices)
            .map_err(|error| self.fail("resident-mesh-upload", error))?;
        Ok(WebGpuResidentMesh {
            key,
            generation: self.generation(),
            index_count: indices.len() as u32,
            lease,
        })
    }

    /// Uploads or reuses one exact linear-RGBA8 image revision. The fixed
    /// unlit browser recipe does not sample this token yet, but its ownership
    /// and retirement rules are identical to mesh residency.
    pub fn resident_image(
        &mut self,
        key: WebGpuAssetKey,
        extent: [u32; 2],
        pixels: &[u8],
    ) -> Result<WebGpuResidentImage, WebGpuSessionError> {
        self.collect();
        let required = extent[0]
            .checked_mul(extent[1])
            .and_then(|v| v.checked_mul(4));
        if self.state() != WebGpuSessionState::Active
            || extent.contains(&0)
            || required.and_then(|v| usize::try_from(v).ok()) != Some(pixels.len())
        {
            return Err(WebGpuSessionError::Contract(
                "resident-image-contract-rejected",
            ));
        }
        let objects = self.render_objects()?;
        let lease = self
            .resident_registry
            .borrow_mut()
            .image(&objects.device, &objects.queue, key, extent, pixels)
            .map_err(|error| self.fail("resident-image-upload", error))?;
        Ok(WebGpuResidentImage {
            key,
            generation: self.generation(),
            lease,
        })
    }

    /// Retires one exact content revision from future lookup. Existing tokens
    /// become unusable, while submitted tickets keep physical resources alive.
    pub fn retire_resident_asset(&mut self, key: WebGpuAssetKey) {
        self.collect();
        self.resident_registry.borrow_mut().retire(key);
    }

    /// Retires every content revision of one logical asset, used for atomic
    /// replacement by a higher renderer-owned residency table.
    pub fn replace_resident_asset(&mut self, logical: u64) {
        self.collect();
        self.resident_registry.borrow_mut().retire_logical(logical);
    }

    /// Reports whether this opaque mesh token remains current for this device.
    pub fn resident_mesh_current(&self, value: &WebGpuResidentMesh) -> bool {
        self.state() == WebGpuSessionState::Active && value.generation == self.generation()
    }

    /// Reports whether this opaque image token remains current for this device.
    pub fn resident_image_current(&self, value: &WebGpuResidentImage) -> bool {
        self.state() == WebGpuSessionState::Active && value.generation == self.generation()
    }

    /// Merges the latest desired extent; active nonzero extents reconfigure.
    pub fn resize(
        &mut self,
        width: u32,
        height: u32,
    ) -> Result<WebGpuSessionState, WebGpuSessionError> {
        self.require_nonterminal()?;
        self.desired_extent.set([width, height]);
        self.canvas.set_width(width);
        self.canvas.set_height(height);
        if width == 0 || height == 0 {
            if matches!(
                self.state(),
                WebGpuSessionState::Active | WebGpuSessionState::Suspended
            ) {
                self.shared.borrow_mut().state = WebGpuSessionState::Suspended;
            }
            return Ok(self.state());
        }
        // Lost/recovering generations must never be reconfigured.
        if matches!(
            self.state(),
            WebGpuSessionState::Active | WebGpuSessionState::Suspended
        ) && self.objects.borrow().is_some()
        {
            self.configure()?;
        }
        Ok(self.state())
    }

    /// Records exactly black clear then retained ordered indexed draws.
    pub fn render(
        &mut self,
        graph: &FixedUnlitGraph<'_>,
        draws: &[FixedUnlitDraw<'_>],
    ) -> Result<WebGpuRenderOutcome, WebGpuSessionError> {
        self.collect();
        match self.state() {
            WebGpuSessionState::Active => {}
            WebGpuSessionState::Suspended => return Ok(WebGpuRenderOutcome::Suspended),
            WebGpuSessionState::Lost | WebGpuSessionState::Recovering => {
                return Ok(WebGpuRenderOutcome::Lost);
            }
            state => return Err(WebGpuSessionError::State(state)),
        }
        if self.tickets.borrow().len() >= MAX_FRAMES_IN_FLIGHT {
            return Ok(WebGpuRenderOutcome::Backpressure);
        }
        validate(graph, draws, self.format.get(), self.desired_extent.get())?;
        let objects = self.render_objects()?;
        let texture =
            call0(&objects.context, "getCurrentTexture").map_err(|e| self.fail("acquire", e))?;
        let view = call0(&texture, "createView").map_err(|e| self.fail("create-view", e))?;
        let encoder =
            call0(&objects.device, "createCommandEncoder").map_err(|e| self.fail("encoder", e))?;
        let attachment = Object::new();
        set(&attachment, "view", &view).map_err(|e| self.fail("attachment-descriptor", e))?;
        set(&attachment, "loadOp", &"clear".into())
            .map_err(|e| self.fail("attachment-descriptor", e))?;
        set(&attachment, "storeOp", &"store".into())
            .map_err(|e| self.fail("attachment-descriptor", e))?;
        let clear = Object::new();
        set(&clear, "r", &0.into()).map_err(|e| self.fail("clear-descriptor", e))?;
        set(&clear, "g", &0.into()).map_err(|e| self.fail("clear-descriptor", e))?;
        set(&clear, "b", &0.into()).map_err(|e| self.fail("clear-descriptor", e))?;
        set(&clear, "a", &1.into()).map_err(|e| self.fail("clear-descriptor", e))?;
        set(&attachment, "clearValue", &clear)
            .map_err(|e| self.fail("attachment-descriptor", e))?;
        let pass_desc = Object::new();
        let colors = Array::new();
        colors.push(&attachment);
        set(&pass_desc, "colorAttachments", &colors)
            .map_err(|e| self.fail("pass-descriptor", e))?;
        let pass = call1(&encoder, "beginRenderPass", &pass_desc)
            .map_err(|e| self.fail("begin-pass", e))?;
        call1(&pass, "setPipeline", &objects.pipeline).map_err(|e| self.fail("set-pipeline", e))?;
        let mut resources = Vec::with_capacity(draws.len());
        for draw in draws {
            let positions: Vec<f32> = draw.positions.iter().flatten().copied().collect();
            let resource = match create_frame_resources(
                &objects.device,
                &objects.layout,
                positions.len(),
                draw.indices.len(),
            ) {
                Ok(resource) => resource,
                Err(error) => {
                    destroy_frame_resources(resources);
                    return Err(self.fail("create-frame-resources", error));
                }
            };
            let result = (|| -> Result<(), WebGpuSessionError> {
                write_buffer(
                    &objects.queue,
                    &resource.position,
                    &Float32Array::from(positions.as_slice()).into(),
                )
                .map_err(|e| self.fail("write-position", e))?;
                write_buffer(
                    &objects.queue,
                    &resource.index,
                    &Uint32Array::from(draw.indices).into(),
                )
                .map_err(|e| self.fail("write-index", e))?;
                write_buffer(
                    &objects.queue,
                    &resource.uniform,
                    &Float32Array::from(draw.pvm_and_color.as_slice()).into(),
                )
                .map_err(|e| self.fail("write-uniform", e))?;
                call2(&pass, "setBindGroup", &0.into(), &resource.bind_group)
                    .map_err(|e| self.fail("set-bind-group", e))?;
                call3(
                    &pass,
                    "setVertexBuffer",
                    &0.into(),
                    &resource.position,
                    &0.into(),
                )
                .map_err(|e| self.fail("set-vertex", e))?;
                call3(
                    &pass,
                    "setIndexBuffer",
                    &resource.index,
                    &"uint32".into(),
                    &0.into(),
                )
                .map_err(|e| self.fail("set-index", e))?;
                call3(
                    &pass,
                    "drawIndexed",
                    &(draw.indices.len() as u32).into(),
                    &1.into(),
                    &0.into(),
                )
                .map_err(|e| self.fail("draw", e))?;
                Ok(())
            })();
            if let Err(error) = result {
                destroy_frame_resources(vec![resource]);
                destroy_frame_resources(resources);
                return Err(error);
            }
            resources.push(resource);
        }
        if let Err(error) = call0(&pass, "end") {
            return Err(self.abort_frame(resources, "end-pass", error));
        }
        let commands = match call0(&encoder, "finish") {
            Ok(commands) => commands,
            Err(error) => return Err(self.abort_frame(resources, "finish", error)),
        };
        let command_list = Array::new();
        command_list.push(&commands);
        if let Err(error) = call1(&objects.queue, "submit", &command_list) {
            return Err(self.abort_frame(resources, "submit", error));
        }
        let marker = {
            let mut s = self.shared.borrow_mut();
            s.next_marker += 1;
            s.next_marker
        };
        let completion = match call0(&objects.queue, "onSubmittedWorkDone") {
            Ok(value) => value,
            Err(error) => {
                self.quarantined.borrow_mut().extend(resources);
                let _ = call0(&objects.device, "destroy");
                return Err(self.fail("completion", error));
            }
        };
        self.ticket(marker, completion, resources);
        Ok(WebGpuRenderOutcome::Submitted(marker))
    }

    /// Records the same closed unlit pass using registry-owned mesh buffers.
    /// It allocates only per-frame uniforms; positions and indices are never
    /// re-uploaded. Replaced tokens remain valid until their generation ends.
    pub fn render_resident(
        &mut self,
        graph: &FixedUnlitGraph<'_>,
        draws: &[FixedResidentUnlitDraw<'_>],
    ) -> Result<WebGpuRenderOutcome, WebGpuSessionError> {
        self.collect();
        if self.state() != WebGpuSessionState::Active {
            return Ok(match self.state() {
                WebGpuSessionState::Suspended => WebGpuRenderOutcome::Suspended,
                WebGpuSessionState::Lost | WebGpuSessionState::Recovering => {
                    WebGpuRenderOutcome::Lost
                }
                state => return Err(WebGpuSessionError::State(state)),
            });
        }
        if self.tickets.borrow().len() >= MAX_FRAMES_IN_FLIGHT {
            return Ok(WebGpuRenderOutcome::Backpressure);
        }
        validate_resident(graph, draws, self.format.get(), self.desired_extent.get())?;
        if draws
            .iter()
            .any(|draw| !self.resident_mesh_current(draw.mesh))
        {
            return Err(WebGpuSessionError::Contract(
                "resident-draw-contract-rejected",
            ));
        }
        let objects = self.render_objects()?;
        let texture =
            call0(&objects.context, "getCurrentTexture").map_err(|e| self.fail("acquire", e))?;
        let view = call0(&texture, "createView").map_err(|e| self.fail("create-view", e))?;
        let encoder =
            call0(&objects.device, "createCommandEncoder").map_err(|e| self.fail("encoder", e))?;
        let attachment = Object::new();
        set(&attachment, "view", &view).map_err(|e| self.fail("attachment", e))?;
        set(&attachment, "loadOp", &"clear".into()).map_err(|e| self.fail("attachment", e))?;
        set(&attachment, "storeOp", &"store".into()).map_err(|e| self.fail("attachment", e))?;
        let clear = Object::new();
        set(&clear, "r", &0.into()).map_err(|e| self.fail("clear", e))?;
        set(&clear, "g", &0.into()).map_err(|e| self.fail("clear", e))?;
        set(&clear, "b", &0.into()).map_err(|e| self.fail("clear", e))?;
        set(&clear, "a", &1.into()).map_err(|e| self.fail("clear", e))?;
        set(&attachment, "clearValue", &clear).map_err(|e| self.fail("clear", e))?;
        let colors = Array::new();
        colors.push(&attachment);
        let pass_desc = Object::new();
        set(&pass_desc, "colorAttachments", &colors).map_err(|e| self.fail("pass", e))?;
        let pass = call1(&encoder, "beginRenderPass", &pass_desc)
            .map_err(|e| self.fail("begin-pass", e))?;
        call1(&pass, "setPipeline", &objects.pipeline).map_err(|e| self.fail("set-pipeline", e))?;
        let mut resources = Vec::with_capacity(draws.len());
        let mut leases = Vec::with_capacity(draws.len());
        for draw in draws {
            let resource = create_frame_resources(&objects.device, &objects.layout, 1, 1)
                .map_err(|e| self.fail("create-frame-uniform", e))?;
            let (position, index) = draw
                .mesh
                .lease
                .mesh()
                .ok_or(WebGpuSessionError::Contract("resident-mesh-token-invalid"))?;
            let recorded = (|| -> Result<(), WebGpuSessionError> {
                write_buffer(
                    &objects.queue,
                    &resource.uniform,
                    &Float32Array::from(draw.pvm_and_color.as_slice()).into(),
                )
                .map_err(|e| self.fail("write-uniform", e))?;
                call2(&pass, "setBindGroup", &0.into(), &resource.bind_group)
                    .map_err(|e| self.fail("set-bind-group", e))?;
                call3(&pass, "setVertexBuffer", &0.into(), position, &0.into())
                    .map_err(|e| self.fail("set-vertex", e))?;
                call3(&pass, "setIndexBuffer", index, &"uint32".into(), &0.into())
                    .map_err(|e| self.fail("set-index", e))?;
                call3(
                    &pass,
                    "drawIndexed",
                    &draw.mesh.index_count.into(),
                    &1.into(),
                    &0.into(),
                )
                .map(|_| ())
                .map_err(|e| self.fail("draw", e))
            })();
            if let Err(error) = recorded {
                destroy_frame_resources(resources);
                destroy_frame_resources(vec![resource]);
                return Err(error);
            }
            resources.push(resource);
            leases.push(draw.mesh.lease.clone());
        }
        if let Err(error) = call0(&pass, "end") {
            return Err(self.abort_frame(resources, "end-pass", error));
        }
        let commands = match call0(&encoder, "finish") {
            Ok(value) => value,
            Err(error) => return Err(self.abort_frame(resources, "finish", error)),
        };
        let command_list = Array::new();
        command_list.push(&commands);
        if let Err(error) = call1(&objects.queue, "submit", &command_list) {
            return Err(self.abort_frame(resources, "submit", error));
        }
        let marker = {
            let mut state = self.shared.borrow_mut();
            state.next_marker += 1;
            state.next_marker
        };
        let completion = match call0(&objects.queue, "onSubmittedWorkDone") {
            Ok(value) => value,
            Err(error) => {
                self.quarantined.borrow_mut().extend(resources);
                let _ = call0(&objects.device, "destroy");
                return Err(self.fail("completion", error));
            }
        };
        self.ticket_with_leases(
            marker,
            completion,
            resources,
            Vec::new(),
            Vec::new(),
            leases,
        );
        Ok(WebGpuRenderOutcome::Submitted(marker))
    }

    /// Starts (or returns) the one cached recovery operation for this session.
    ///
    /// The returned Promise owns only cloned shared cells; it never holds a
    /// Rust borrow of this session across adapter/device awaits.
    pub fn recover(&mut self) -> Result<Promise, WebGpuSessionError> {
        if let Some(promise) = self.recovery_promise.borrow().as_ref() {
            return Ok(promise.clone());
        }
        if !matches!(
            self.state(),
            WebGpuSessionState::Lost | WebGpuSessionState::Recovering
        ) {
            return Err(WebGpuSessionError::State(self.state()));
        }
        {
            let mut s = self.shared.borrow_mut();
            s.state = WebGpuSessionState::Recovering;
            s.token += 1;
        }
        self.detached
            .borrow_mut()
            .append(&mut self.tickets.borrow_mut());
        // Old-generation native resources are removed from the importable
        // registry now, while detached tickets keep accepted work alive until
        // its completion Promise settles.
        self.resource_registry.borrow_mut().retire_generation();
        self.resident_registry.borrow_mut().retire_generation();
        let quarantined = std::mem::take(&mut *self.quarantined.borrow_mut());
        destroy_frame_resources(quarantined);
        if let Some(objects) = self.objects.borrow_mut().take() {
            unregister_uncaptured_error(&objects);
            self.retired_objects.borrow_mut().push(objects);
        }
        self.collect_retired_objects();
        let token = self.shared.borrow().token;
        let session = self.shared_clone();
        let promise = future_to_promise(async move {
            let (format, info, device, queue, context) =
                match request_browser(&session.canvas, session.requests.as_ref()).await {
                    Ok(value) => value,
                    Err(error) => {
                        // A terminal operation may finish while adapter/device
                        // acquisition is pending. Its failure is stale and
                        // must not overwrite Disposing/Disposed diagnostics.
                        if session.shared.borrow().token != token
                            || session.state() != WebGpuSessionState::Recovering
                        {
                            session.recovery_promise.borrow_mut().take();
                            return Ok(JsValue::UNDEFINED);
                        }
                        let error = session.recover_failure(error);
                        session.recovery_promise.borrow_mut().take();
                        return Err(to_js(error));
                    }
                };
            if !session.recovery_attempt_current(token) {
                let _ = call0(&device, "destroy");
                session.recovery_promise.borrow_mut().take();
                return Ok(JsValue::UNDEFINED);
            }
            // Candidate facts remain local until its pipeline, listeners and
            // canvas configuration have all succeeded. A stale/failed attempt
            // must not alter the observable generation, format or adapter.
            let generation = match session.generation().checked_add(1) {
                Some(generation) => generation,
                None => {
                    let error = WebGpuSessionError::Browser {
                        code: "device-generation-exhausted",
                        operation: "recover-generation",
                        generation: session.generation(),
                        message: "device generation exhausted".into(),
                    };
                    // The browser candidate already exists, but no generation
                    // can represent it. It must never escape this attempt.
                    let _ = call0(&device, "destroy");
                    session.recovery_promise.borrow_mut().take();
                    return Err(to_js(session.recover_failure(error)));
                }
            };
            let objects = match session
                .install(
                    device.clone(),
                    queue,
                    context,
                    InstallAttempt {
                        format,
                        generation,
                        token,
                        expected_state: WebGpuSessionState::Recovering,
                    },
                )
                .await
            {
                Ok(objects) => objects,
                Err(error) => {
                    let _ = call0(&device, "destroy");
                    session.recovery_promise.borrow_mut().take();
                    if session.shared.borrow().token == token
                        && !matches!(
                            session.state(),
                            WebGpuSessionState::Disposing | WebGpuSessionState::Disposed
                        )
                    {
                        return Err(to_js(session.recover_failure(error)));
                    }
                    return Err(to_js(error));
                }
            };
            // `install` awaited validation before committing objects. A
            // concurrent dispose/recovery may have changed this state while
            // it was suspended, so do not configure or publish a stale device.
            if !session.recovery_attempt_current(token) {
                Self::destroy_objects(objects).await;
                session.recovery_promise.borrow_mut().take();
                return Ok(JsValue::UNDEFINED);
            }
            let configured = if !session.desired_extent.get().contains(&0) {
                match session.configure_candidate(&objects, format) {
                    Ok(()) => true,
                    Err(error) => {
                        Self::destroy_objects(objects).await;
                        if session.shared.borrow().token == token
                            && !matches!(
                                session.state(),
                                WebGpuSessionState::Disposing | WebGpuSessionState::Disposed
                            )
                        {
                            let error = session.recover_failure(error);
                            session.recovery_promise.borrow_mut().take();
                            return Err(to_js(error));
                        }
                        session.recovery_promise.borrow_mut().take();
                        return Ok(JsValue::UNDEFINED);
                    }
                }
            } else {
                false
            };
            // `configure_candidate` is synchronous, but browser calls can be
            // re-entrant. Recheck immediately before this single publication
            // point; only this branch makes candidate facts observable.
            if !session.candidate_transaction_committable(token, generation) {
                Self::destroy_objects(objects).await;
                session.recovery_promise.borrow_mut().take();
                return Ok(JsValue::UNDEFINED);
            }
            session.format.set(format);
            *session.adapter_info.borrow_mut() = info;
            *session.objects.borrow_mut() = Some(objects);
            {
                let mut state = session.shared.borrow_mut();
                state.generation = generation;
                state.state = if configured {
                    WebGpuSessionState::Active
                } else {
                    WebGpuSessionState::Suspended
                };
            }
            if configured {
                session.canvas_epoch.set(session.canvas_epoch.get() + 1);
            }
            session.collect_retired_objects();
            session.recovery_promise.borrow_mut().take();
            Ok(JsValue::UNDEFINED)
        });
        *self.recovery_promise.borrow_mut() = Some(promise.clone());
        Ok(promise)
    }

    /// Evidence-only controlled destruction seam; normal code must use dispose.
    #[doc(hidden)]
    pub fn controlled_destroy_for_evidence(&mut self) -> Result<(), WebGpuSessionError> {
        let device = self
            .objects
            .borrow()
            .as_ref()
            .map(|objects| objects.device.clone())
            .ok_or(WebGpuSessionError::State(self.state()))?;
        call0(&device, "destroy").map_err(|e| self.fail("controlled-destroy", e))?;
        // `device.lost` is the authoritative state transition.
        Ok(())
    }

    /// Starts (or returns) the cached terminal cleanup Promise.
    pub fn dispose(&mut self) -> Result<Promise, WebGpuSessionError> {
        if let Some(promise) = self.dispose_promise.borrow().as_ref() {
            return Ok(promise.clone());
        }
        if self.state() == WebGpuSessionState::Disposed {
            return Ok(Promise::resolve(&JsValue::UNDEFINED));
        }
        {
            let mut state = self.shared.borrow_mut();
            state.state = WebGpuSessionState::Disposing;
            state.token += 1;
        }
        // Token invalidation makes a pending recovery stale, but its Promise
        // may still own candidate browser objects and continuations. Terminal
        // cleanup must join that exact operation before it publishes Disposed.
        let recovery = self.recovery_promise.borrow().clone();
        let objects = self.objects.borrow_mut().take();
        self.resource_registry.borrow_mut().retire_generation();
        self.resident_registry.borrow_mut().retire_generation();
        let session = self.shared_clone();
        let promise = future_to_promise(async move {
            if let Some(recovery) = recovery {
                let _ = JsFuture::from(recovery).await;
            }
            if let Some(objects) = objects {
                // The lost continuations are owned for exactly the device lifetime.
                // Dropping them here makes a post-dispose browser callback inert.
                let _continuations = (&objects.lost_ok, &objects.lost_err);
                let _ = call0(&objects.device, "destroy");
                let _ = JsFuture::from(objects.lost_promise.clone()).await;
                unregister_uncaptured_error(&objects);
            }
            let retired_objects = std::mem::take(&mut *session.retired_objects.borrow_mut());
            for objects in retired_objects {
                let _continuations = (&objects.lost_ok, &objects.lost_err);
                let _ = JsFuture::from(objects.lost_promise.clone()).await;
                unregister_uncaptured_error(&objects);
            }
            session.settle_all_tickets().await;
            let quarantined = std::mem::take(&mut *session.quarantined.borrow_mut());
            destroy_frame_resources(quarantined);
            if !session.terminal_dispose_ready() {
                return Err(JsValue::from_str(
                    "WebGPU disposal retained an asynchronous ownership root",
                ));
            }
            session.shared.borrow_mut().state = WebGpuSessionState::Disposed;
            Ok(JsValue::UNDEFINED)
        });
        *self.dispose_promise.borrow_mut() = Some(promise.clone());
        Ok(promise)
    }

    // Never exposed: callbacks cannot obtain a second producer.
    fn shared_clone(&self) -> Self {
        Self {
            canvas: self.canvas.clone(),
            desired_extent: Rc::clone(&self.desired_extent),
            canvas_epoch: Rc::clone(&self.canvas_epoch),
            format: Rc::clone(&self.format),
            shared: Rc::clone(&self.shared),
            objects: Rc::clone(&self.objects),
            retired_objects: Rc::clone(&self.retired_objects),
            tickets: Rc::clone(&self.tickets),
            detached: Rc::clone(&self.detached),
            quarantined: Rc::clone(&self.quarantined),
            recovery_promise: Rc::clone(&self.recovery_promise),
            dispose_promise: Rc::clone(&self.dispose_promise),
            adapter_info: Rc::clone(&self.adapter_info),
            requests: Rc::clone(&self.requests),
            resource_registry: Rc::clone(&self.resource_registry),
            resident_registry: Rc::clone(&self.resident_registry),
        }
    }

    /// Reads the production recovery contract without retaining a `RefCell`
    /// guard beyond the pure predicate.
    fn recovery_attempt_current(&self, token: u64) -> bool {
        let shared = self.shared.borrow();
        recovery_attempt_current(shared.state, shared.token, token)
    }

    /// The only guard used at the candidate device transaction's publication
    /// point. Its generation check prevents a partial/stale candidate commit.
    fn candidate_transaction_committable(&self, token: u64, candidate_generation: u64) -> bool {
        let shared = self.shared.borrow();
        candidate_transaction_committable(
            shared.state,
            shared.token,
            token,
            shared.generation,
            candidate_generation,
        )
    }

    /// Consults the production terminal contract after the disposal future has
    /// joined recovery, device observers, completion tickets and quarantine.
    fn terminal_dispose_ready(&self) -> bool {
        terminal_dispose_ready(
            self.state(),
            TerminalDisposalRoots {
                recovery_pending: self.recovery_promise.borrow().is_some(),
                live_objects: self.objects.borrow().is_some(),
                retired_objects: !self.retired_objects.borrow().is_empty(),
                active_tickets: !self.tickets.borrow().is_empty(),
                detached_tickets: !self.detached.borrow().is_empty(),
                quarantined_resources: !self.quarantined.borrow().is_empty(),
                // `Objects` owns the only RHI listener/callback roots, which
                // are already reflected by the two object predicates above.
                callback_or_listener_producer: false,
            },
        )
    }

    async fn install(
        &self,
        device: JsValue,
        queue: JsValue,
        context: JsValue,
        attempt: InstallAttempt,
    ) -> Result<Objects, WebGpuSessionError> {
        call1(&device, "pushErrorScope", &"validation".into())
            .map_err(|e| self.fail("push-error-scope", e))?;
        // Always pop the scope, including a synchronous construction failure.
        // Otherwise a failed first install leaves an unowned browser scope.
        let construction = pipeline(&device, attempt.format);
        let scope = Promise::from(
            call0(&device, "popErrorScope").map_err(|e| self.fail("pop-error-scope", e))?,
        );
        let scope_result = match JsFuture::from(scope).await {
            Ok(value) => value,
            Err(error) => {
                if self.shared.borrow().token != attempt.token
                    || self.state() != attempt.expected_state
                {
                    let _ = call0(&device, "destroy");
                    return Err(WebGpuSessionError::State(self.state()));
                }
                return Err(self.fail("pop-error-scope", error));
            }
        };
        // Nothing beyond this point may install browser callbacks or Objects
        // unless the operation which started this await still owns the session.
        // In particular, dispose increments `token` before it observes objects.
        if self.shared.borrow().token != attempt.token || self.state() != attempt.expected_state {
            let _ = call0(&device, "destroy");
            return Err(WebGpuSessionError::State(self.state()));
        }
        let (pipeline, layout) =
            construction.map_err(|error| self.fail("create-pipeline", error))?;
        if !scope_result.is_null() && !scope_result.is_undefined() {
            return Err(self.fail("validation-scope", scope_result));
        }
        // Install the removable listener before registering `device.lost`.
        // There are no fallible operations after the non-cancellable Promise
        // continuations are attached.
        let shared = Rc::clone(&self.shared);
        let uncaptured = Closure::wrap(Box::new(move |value: JsValue| {
            let mut s = shared.borrow_mut();
            if s.generation != attempt.generation || s.token != attempt.token {
                return;
            }
            s.diagnostics.push(WebGpuDiagnostic {
                code: "uncaptured-error",
                operation: "uncaptured-error",
                generation: attempt.generation,
                message: js_message(&value),
            });
            s.state = WebGpuSessionState::Poisoned;
        }) as Box<dyn FnMut(JsValue)>);
        call2(
            &device,
            "addEventListener",
            &"uncapturederror".into(),
            uncaptured.as_ref(),
        )
        .map_err(|e| self.fail("install-uncaptured-error", e))?;
        let lost_promise = match get(&device, "lost") {
            Some(value) => Promise::from(value),
            None => {
                let _ = call2(
                    &device,
                    "removeEventListener",
                    &"uncapturederror".into(),
                    uncaptured.as_ref(),
                );
                return Err(WebGpuSessionError::Unavailable);
            }
        };
        let lost_settled = Rc::new(Cell::new(false));
        let settled = Rc::clone(&lost_settled);
        let shared = Rc::clone(&self.shared);
        let lost_ok = Closure::wrap(Box::new(move |value: JsValue| {
            settled.set(true);
            let mut s = shared.borrow_mut();
            if s.generation == attempt.generation
                && s.token == attempt.token
                && matches!(
                    s.state,
                    WebGpuSessionState::Active | WebGpuSessionState::Suspended
                )
            {
                let destroyed = get(&value, "reason")
                    .and_then(|reason| reason.as_string())
                    .is_some_and(|reason| reason == "destroyed");
                s.lost = true;
                s.loss_reason = Some(if destroyed {
                    WebGpuLossReason::Destroyed
                } else {
                    WebGpuLossReason::Other
                });
                if destroyed {
                    s.state = WebGpuSessionState::Lost;
                } else {
                    s.diagnostics.push(WebGpuDiagnostic {
                        code: "device-lost",
                        operation: "device-lost",
                        generation: attempt.generation,
                        message: js_message(&value),
                    });
                    s.state = WebGpuSessionState::Poisoned;
                }
            }
        }) as Box<dyn FnMut(JsValue)>);
        let settled = Rc::clone(&lost_settled);
        let shared = Rc::clone(&self.shared);
        let lost_err = Closure::wrap(Box::new(move |value: JsValue| {
            settled.set(true);
            let mut s = shared.borrow_mut();
            if s.generation == attempt.generation
                && s.token == attempt.token
                && matches!(
                    s.state,
                    WebGpuSessionState::Active | WebGpuSessionState::Suspended
                )
            {
                s.diagnostics.push(WebGpuDiagnostic {
                    code: "device-lost-promise-rejected",
                    operation: "device-lost",
                    generation: attempt.generation,
                    message: js_message(&value),
                });
                s.state = WebGpuSessionState::Poisoned;
            }
        }) as Box<dyn FnMut(JsValue)>);
        let _ = lost_promise.then2(&lost_ok, &lost_err);
        Ok(Objects {
            device,
            queue,
            context,
            pipeline,
            layout,
            lost_promise,
            lost_settled,
            lost_ok,
            lost_err,
            uncaptured,
        })
    }
    /// Configures a fully-built recovery candidate without publishing it as
    /// the session device. The caller owns its state transition and commit.
    fn configure_candidate(
        &self,
        objects: &Objects,
        format: WebGpuCanvasFormat,
    ) -> Result<(), WebGpuSessionError> {
        self.configure_candidate_handles(&objects.device, &objects.context, format)
    }
    /// Configures from cloned browser handles, so an active-session configure
    /// cannot keep an `objects` cell borrow alive across browser FFI.
    fn configure_candidate_handles(
        &self,
        device: &JsValue,
        context: &JsValue,
        format: WebGpuCanvasFormat,
    ) -> Result<(), WebGpuSessionError> {
        let config = Object::new();
        set(&config, "device", device).map_err(|e| self.fail("configure-descriptor", e))?;
        set(&config, "format", &format.as_str().into())
            .map_err(|e| self.fail("configure-descriptor", e))?;
        set(&config, "alphaMode", &"opaque".into())
            .map_err(|e| self.fail("configure-descriptor", e))?;
        call1(context, "configure", &config).map_err(|e| self.fail("configure", e))?;
        Ok(())
    }
    fn configure(&mut self) -> Result<(), WebGpuSessionError> {
        if self.desired_extent.get().contains(&0) {
            self.shared.borrow_mut().state = WebGpuSessionState::Suspended;
            return Ok(());
        }
        let (device, context) = self
            .objects
            .borrow()
            .as_ref()
            .map(|objects| (objects.device.clone(), objects.context.clone()))
            .ok_or(WebGpuSessionError::Unavailable)?;
        self.configure_candidate_handles(&device, &context, self.format.get())?;
        self.canvas_epoch.set(self.canvas_epoch.get() + 1);
        self.shared.borrow_mut().state = WebGpuSessionState::Active;
        Ok(())
    }
    fn ticket(&self, marker: u64, promise: JsValue, resources: Vec<FrameDrawResources>) {
        self.ticket_with_leases(
            marker,
            promise,
            resources,
            Vec::new(),
            Vec::new(),
            Vec::new(),
        );
    }
    fn ticket_with_leases(
        &self,
        marker: u64,
        promise: JsValue,
        resources: Vec<FrameDrawResources>,
        resource_leases: Vec<ResourceLease>,
        browser_roots: Vec<JsValue>,
        resident_leases: Vec<ResidentLease>,
    ) {
        let settled = Rc::new(RefCell::new(None));
        let a = Rc::clone(&settled);
        let ok =
            Closure::wrap(Box::new(move |_v: JsValue| *a.borrow_mut() = Some(Ok(())))
                as Box<dyn FnMut(JsValue)>);
        let b = Rc::clone(&settled);
        let err =
            Closure::wrap(
                Box::new(move |v: JsValue| *b.borrow_mut() = Some(Err(js_message(&v))))
                    as Box<dyn FnMut(JsValue)>,
            );
        let promise = Promise::from(promise);
        let _ = promise.then2(&ok, &err);
        self.tickets.borrow_mut().push_back(Ticket {
            generation: self.generation(),
            marker,
            settled,
            ok,
            err,
            promise,
            resources,
            resource_leases,
            resident_leases,
            browser_roots,
        });
    }
    /// Snapshots just the frame encoding handles. No `RefCell` guard escapes
    /// this function: all following WebGPU calls operate on owned `JsValue`
    /// clones and may therefore re-enter browser/wasm code safely.
    fn render_objects(&self) -> Result<RenderObjects, WebGpuSessionError> {
        self.objects
            .borrow()
            .as_ref()
            .map(|objects| RenderObjects {
                device: objects.device.clone(),
                queue: objects.queue.clone(),
                context: objects.context.clone(),
                pipeline: objects.pipeline.clone(),
                layout: objects.layout.clone(),
            })
            .ok_or(WebGpuSessionError::Unavailable)
    }
    fn collect(&self) {
        self.collect_retired_objects();
        loop {
            let Some(ticket) = ({
                let mut tickets = self.tickets.borrow_mut();
                if tickets
                    .front()
                    .is_some_and(|t| t.settled.borrow().is_some())
                {
                    tickets.pop_front()
                } else {
                    None
                }
            }) else {
                break;
            };
            // These closures are deliberately retained until settlement; this
            // read also documents that their drop is the ticket retirement.
            let _continuations = (
                &ticket.ok,
                &ticket.err,
                ticket.marker,
                &ticket.browser_roots,
            );
            {
                let mut state = self.shared.borrow_mut();
                if ticket.generation == state.generation
                    && ticket.settled.borrow().as_ref().is_some_and(Result::is_err)
                {
                    state.state = WebGpuSessionState::Poisoned;
                    state.diagnostics.push(WebGpuDiagnostic {
                        code: "completion-rejected",
                        operation: "completion",
                        generation: ticket.generation,
                        message: ticket
                            .settled
                            .borrow()
                            .as_ref()
                            .unwrap()
                            .as_ref()
                            .err()
                            .unwrap()
                            .clone(),
                    });
                }
            }
            // No `Shared` borrow crosses the JS FFI destroys below.
            destroy_frame_resources(ticket.resources);
            drop(ticket.resource_leases);
            drop(ticket.resident_leases);
        }
        loop {
            let Some(ticket) = ({
                let mut detached = self.detached.borrow_mut();
                if detached
                    .front()
                    .is_some_and(|t| t.settled.borrow().is_some())
                {
                    detached.pop_front()
                } else {
                    None
                }
            }) else {
                break;
            };
            destroy_frame_resources(ticket.resources);
            drop(ticket.resource_leases);
            drop(ticket.resident_leases);
        }
    }
    /// Drops old device observer roots only after the browser has settled the
    /// corresponding `device.lost` Promise. Their listeners were removed when
    /// retired, and generation/token gates keep their continuations inert.
    fn collect_retired_objects(&self) {
        self.retired_objects
            .borrow_mut()
            .retain(|objects| !objects.lost_settled.get());
    }
    /// Terminally releases one observer root without letting the browser call
    /// dropped wasm closures. This is used by partial construction failures;
    /// normal recovery retains a settled old generation for cheap collection.
    async fn destroy_objects(objects: Objects) {
        unregister_uncaptured_error(&objects);
        let _continuations = (&objects.lost_ok, &objects.lost_err);
        let _ = call0(&objects.device, "destroy");
        let _ = JsFuture::from(objects.lost_promise.clone()).await;
    }
    async fn settle_all_tickets(&self) {
        while let Some(ticket) = {
            self.tickets
                .borrow_mut()
                .pop_front()
                .or_else(|| self.detached.borrow_mut().pop_front())
        } {
            let _ = JsFuture::from(ticket.promise.clone()).await;
            destroy_frame_resources(ticket.resources);
            drop(ticket.resource_leases);
            drop(ticket.resident_leases);
        }
    }
    fn require_nonterminal(&self) -> Result<(), WebGpuSessionError> {
        match self.state() {
            WebGpuSessionState::Disposed
            | WebGpuSessionState::Disposing
            | WebGpuSessionState::Poisoned => Err(WebGpuSessionError::State(self.state())),
            _ => Ok(()),
        }
    }
    fn fail(&self, operation: &'static str, value: JsValue) -> WebGpuSessionError {
        let generation = self.generation();
        let message = js_message(&value);
        let mut state = self.shared.borrow_mut();
        state.diagnostics.push(WebGpuDiagnostic {
            code: "webgpu-operation-failed",
            operation,
            generation,
            message: message.clone(),
        });
        // A rejected browser operation may leave command encoding or surface
        // state unknown. Never let that generation continue producing work.
        if !matches!(
            state.state,
            WebGpuSessionState::Disposed | WebGpuSessionState::Disposing
        ) {
            state.state = WebGpuSessionState::Poisoned;
        }
        WebGpuSessionError::Browser {
            code: "webgpu-operation-failed",
            operation,
            generation,
            message,
        }
    }
    /// Destroys resources that were never submitted before poisoning the
    /// generation for the rejected browser operation.
    fn abort_frame(
        &self,
        resources: Vec<FrameDrawResources>,
        operation: &'static str,
        value: JsValue,
    ) -> WebGpuSessionError {
        destroy_frame_resources(resources);
        self.fail(operation, value)
    }
    /// Converts a failed replacement-device request into a terminal outcome
    /// for the current generation. A recovery may not remain indefinitely in
    /// `Recovering`, and its original browser error was created before the
    /// replacement generation existed, so it must be re-attributed here.
    fn recover_failure(&self, error: WebGpuSessionError) -> WebGpuSessionError {
        let generation = self.generation();
        let message = error.message();
        let mut state = self.shared.borrow_mut();
        state.diagnostics.push(WebGpuDiagnostic {
            code: "recovery-request-failed",
            operation: "recover-request",
            generation,
            message: message.clone(),
        });
        state.state = WebGpuSessionState::Poisoned;
        WebGpuSessionError::Browser {
            code: "recovery-request-failed",
            operation: "recover-request",
            generation,
            message,
        }
    }
}
