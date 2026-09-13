//! Explicit WebGPU capsule for the retained scene.
//!
//! The JavaScript adapter owns RAF and DOM events. This module maps the
//! renderer's closed prepared scene into the RHI-owned WebGPU generation; it
//! never exposes adapter, device, queue, context, texture, or Promise internals.

use fluxel_renderer::adapter::{
    PreparedBasicGraph, PreparedBasicScene, PresentableFormat, PresentationProfile,
};
use fluxel_rhi::adapter::webgpu::{
    FixedResidentUnlitDraw, FixedUnlitGraph, WebGpuAssetKey, WebGpuCanvasFormat, WebGpuLossReason,
    WebGpuRenderOutcome, WebGpuResidentMesh, WebGpuSession as RhiSession, WebGpuSessionError,
    WebGpuSessionState,
};
use js_sys::{Array, Object, Promise, Reflect};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::HtmlCanvasElement;

use crate::{retained_scene, structured_error, wasm_memory_bytes};

/// One explicit canvas-bound WebGPU session.
#[wasm_bindgen]
pub struct WebGpuSession {
    inner: RhiSession,
    scene: PreparedBasicScene,
    graph: Option<PreparedBasicGraph>,
    resident_generation: Option<u64>,
    resident_meshes: Vec<WebGpuResidentMesh>,
}

#[wasm_bindgen]
impl WebGpuSession {
    /// Asynchronously opens the browser adapter/device for this canvas only.
    #[wasm_bindgen(js_name = create)]
    pub async fn create(canvas: HtmlCanvasElement) -> Result<WebGpuSession, JsValue> {
        let scene = retained_scene().map_err(|error| {
            structured_error(
                "scene-preparation-failed",
                "prepare-scene",
                0,
                &error.to_string(),
            )
        })?;
        let mut inner = RhiSession::new(canvas.clone().into())
            .await
            .map_err(webgpu_error)?;
        let extent = [canvas.width(), canvas.height()];
        let graph = match compile_graph(&scene, inner.format(), extent) {
            Ok(graph) => graph,
            Err(error) => {
                if let Ok(disposal) = inner.dispose() {
                    let _ = JsFuture::from(disposal).await;
                }
                return Err(error);
            }
        };
        let resident_meshes = if inner.state() == WebGpuSessionState::Active {
            prepare_resident_meshes(&mut inner, &scene)?
        } else {
            Vec::new()
        };
        let resident_generation =
            (inner.state() == WebGpuSessionState::Active).then_some(inner.generation());
        Ok(Self {
            inner,
            scene,
            graph,
            resident_generation,
            resident_meshes,
        })
    }

    /// Updates the exact drawing-buffer pixel extent.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), JsValue> {
        let graph = compile_graph(&self.scene, self.inner.format(), [width, height])?;
        self.inner
            .resize(width, height)
            .map_err(|error| webgpu_error_at(error, self.inner.generation()))?;
        self.graph = graph;
        Ok(())
    }

    /// Submits one frame; RAF remains owned by the JavaScript adapter.
    pub fn render_once(&mut self) -> Result<JsValue, JsValue> {
        let started = js_sys::Date::now();
        if matches!(
            self.inner.state(),
            WebGpuSessionState::Lost | WebGpuSessionState::Recovering
        ) {
            return Ok(frame_report(
                "device-lost",
                None,
                &self.inner,
                js_sys::Date::now() - started,
            ));
        }
        if self.inner.state() == WebGpuSessionState::Suspended {
            return Ok(frame_report(
                "suspended",
                None,
                &self.inner,
                js_sys::Date::now() - started,
            ));
        }
        self.refresh_graph_for_format()?;
        if self.resident_generation != Some(self.inner.generation()) {
            self.resident_meshes = prepare_resident_meshes(&mut self.inner, &self.scene)?;
            self.resident_generation = Some(self.inner.generation());
        }
        let draws = self
            .scene
            .draws()
            .iter()
            .zip(&self.resident_meshes)
            .map(|(draw, mesh)| FixedResidentUnlitDraw {
                mesh,
                pvm_and_color: draw.pvm_and_color(),
                insertion_index: draw.insertion_index(),
            })
            .collect::<Vec<_>>();
        let graph = self.graph.as_ref().ok_or_else(|| {
            structured_error(
                "invalid-state",
                "render",
                self.inner.generation(),
                "presentable graph is suspended",
            )
        })?;
        let contract = FixedUnlitGraph::new(
            graph.compiled(),
            graph.draw_count(),
            graph.extent(),
            self.inner.format(),
        );
        let generation = self.inner.generation();
        let (outcome, marker) = match self
            .inner
            .render_resident(&contract, &draws)
            .map_err(|error| webgpu_error_at(error, generation))?
        {
            WebGpuRenderOutcome::Submitted(marker) => ("submitted", Some(marker)),
            WebGpuRenderOutcome::Backpressure => ("backpressure", None),
            WebGpuRenderOutcome::Suspended => ("suspended", None),
            WebGpuRenderOutcome::Lost => ("device-lost", None),
        };
        Ok(frame_report(
            outcome,
            marker,
            &self.inner,
            js_sys::Date::now() - started,
        ))
    }

    /// Stops new submissions without claiming GPU completion.
    pub fn suspend(&mut self) -> Result<(), JsValue> {
        let generation = self.inner.generation();
        self.inner
            .suspend()
            .map_err(|error| webgpu_error_at(error, generation))
    }

    /// Resumes only when the latest desired extent is drawable.
    pub fn resume(&mut self) -> Result<(), JsValue> {
        let generation = self.inner.generation();
        self.inner
            .resume()
            .map(|_| ())
            .map_err(|error| webgpu_error_at(error, generation))
    }

    /// Starts or returns the one cached asynchronous device recovery.
    pub fn recover(&mut self) -> Result<Promise, JsValue> {
        let generation = self.inner.generation();
        self.inner
            .recover()
            .map_err(|error| webgpu_error_at(error, generation))
    }

    /// Starts or returns the one cached asynchronous terminal cleanup.
    pub fn dispose(&mut self) -> Result<Promise, JsValue> {
        let generation = self.inner.generation();
        self.inner
            .dispose()
            .map_err(|error| webgpu_error_at(error, generation))
    }

    /// Destroys the active device so the proof harness can observe `device.lost`.
    #[wasm_bindgen(js_name = controlled_destroy_for_evidence)]
    pub fn controlled_destroy_for_evidence(&mut self) -> Result<(), JsValue> {
        let generation = self.inner.generation();
        self.inner
            .controlled_destroy_for_evidence()
            .map_err(|error| webgpu_error_at(error, generation))
    }

    /// Returns copied backend facts without exposing browser GPU objects.
    pub fn backend_snapshot(&self) -> JsValue {
        let info = self.inner.adapter_info();
        let object = Object::new();
        set(&object, "vendor", &info.vendor.into());
        set(&object, "architecture", &info.architecture.into());
        set(&object, "device", &info.device.into());
        set(&object, "format", &self.inner.format().as_str().into());
        set(
            &object,
            "generation",
            &JsValue::from_f64(self.inner.generation() as f64),
        );
        set(
            &object,
            "canvasEpoch",
            &JsValue::from_f64(self.inner.canvas_epoch() as f64),
        );
        set(
            &object,
            "state",
            &format!("{:?}", self.inner.state()).into(),
        );
        let loss_reason = match self.inner.loss_reason() {
            Some(WebGpuLossReason::Destroyed) => "destroyed",
            Some(WebGpuLossReason::Other) => "other",
            None => "none",
        };
        set(&object, "lossReason", &loss_reason.into());
        object.into()
    }

    /// Returns a non-draining snapshot of RHI-generated diagnostics.
    pub fn diagnostics_snapshot(&self) -> Array {
        let result = Array::new();
        for diagnostic in self.inner.diagnostics() {
            let object = Object::new();
            set(&object, "code", &diagnostic.code.into());
            set(&object, "severity", &"error".into());
            set(&object, "operation", &diagnostic.operation.into());
            set(
                &object,
                "generation",
                &JsValue::from_f64(diagnostic.generation as f64),
            );
            set(&object, "message", &diagnostic.message.into());
            set(&object, "context", &Object::new());
            result.push(&object);
        }
        result
    }
}

fn prepare_resident_meshes(
    session: &mut RhiSession,
    scene: &PreparedBasicScene,
) -> Result<Vec<WebGpuResidentMesh>, JsValue> {
    scene
        .draws()
        .iter()
        .enumerate()
        .map(|(index, draw)| {
            session
                .resident_mesh(
                    WebGpuAssetKey::new(index as u64, 1),
                    draw.positions(),
                    draw.indices(),
                )
                .map_err(|error| webgpu_error_at(error, session.generation()))
        })
        .collect()
}

impl WebGpuSession {
    fn refresh_graph_for_format(&mut self) -> Result<(), JsValue> {
        let format = renderer_format(self.inner.format());
        let needs_refresh = self
            .graph
            .as_ref()
            .is_some_and(|graph| graph.format() != format);
        if needs_refresh {
            let extent = self.graph.as_ref().expect("checked graph").extent();
            self.graph = compile_graph(&self.scene, self.inner.format(), extent)?;
        }
        Ok(())
    }
}

fn compile_graph(
    scene: &PreparedBasicScene,
    format: WebGpuCanvasFormat,
    extent: [u32; 2],
) -> Result<Option<PreparedBasicGraph>, JsValue> {
    if extent.contains(&0) {
        return Ok(None);
    }
    scene
        .compile_presentable_graph_for_profile(
            PresentationProfile::for_format(renderer_format(format)),
            extent,
        )
        .map(Some)
        .map_err(|error| {
            structured_error(
                "graph-compile-failed",
                "compile-graph",
                0,
                &error.to_string(),
            )
        })
}

const fn renderer_format(format: WebGpuCanvasFormat) -> PresentableFormat {
    match format {
        WebGpuCanvasFormat::Rgba8Unorm => PresentableFormat::Rgba8Unorm,
        WebGpuCanvasFormat::Bgra8Unorm => PresentableFormat::Bgra8Unorm,
    }
}

fn frame_report(
    outcome: &str,
    marker: Option<u64>,
    session: &RhiSession,
    cpu_submission_ms: f64,
) -> JsValue {
    let report = Object::new();
    set(&report, "outcome", &outcome.into());
    if let Some(marker) = marker {
        set(&report, "frameMarker", &JsValue::from_f64(marker as f64));
    }
    set(
        &report,
        "generation",
        &JsValue::from_f64(session.generation() as f64),
    );
    set(
        &report,
        "canvasEpoch",
        &JsValue::from_f64(session.canvas_epoch() as f64),
    );
    set(
        &report,
        "cpuSubmissionMs",
        &JsValue::from_f64(cpu_submission_ms),
    );
    set(&report, "wasmMemoryBytes", &wasm_memory_bytes());
    set(&report, "state", &format!("{:?}", session.state()).into());
    report.into()
}

fn webgpu_error(error: WebGpuSessionError) -> JsValue {
    webgpu_error_at(error, 0)
}

fn webgpu_error_at(error: WebGpuSessionError, current_generation: u64) -> JsValue {
    match error {
        WebGpuSessionError::CanvasUnavailable => structured_error(
            "canvas-unavailable",
            "open-canvas",
            0,
            "input was not an HTML canvas",
        ),
        WebGpuSessionError::Unavailable => structured_error(
            "webgpu-unavailable",
            "open-device",
            0,
            "WebGPU is unavailable",
        ),
        WebGpuSessionError::UnsupportedFormat(format) => structured_error(
            "unsupported-format",
            "preferred-format",
            0,
            &format!("unsupported preferred canvas format: {format}"),
        ),
        WebGpuSessionError::State(state) => structured_error(
            "invalid-state",
            "lifecycle",
            current_generation,
            &format!("invalid WebGPU session state: {state:?}"),
        ),
        WebGpuSessionError::Contract(code) => {
            structured_error(code, "validate-contract", current_generation, code)
        }
        WebGpuSessionError::Browser {
            code,
            operation,
            generation,
            message,
        } => structured_error(code, operation, generation, &message),
    }
}

fn set(object: &Object, name: &str, value: &JsValue) {
    let _ = Reflect::set(object, &name.into(), value);
}
