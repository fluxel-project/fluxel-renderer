//! Explicit canvas-bound wasm bridge for the retained red/green/blue scene.
//!
//! The JavaScript adapter owns animation frames and DOM events.  This crate
//! owns neither a browser singleton nor a scene API; it only turns those
//! explicit lifecycle calls into the closed renderer/RHI execution path.

#![cfg(target_arch = "wasm32")]

mod webgpu;

pub use webgpu::WebGpuSession;

use fluxel_renderer::adapter::{PreparedBasicGraph, PreparedBasicScene};
use fluxel_renderer::{BasicMaterial, Camera, DrawList, Geometry, Mesh, ModelTransform};
use fluxel_rhi::adapter::webgl2::{
    FixedResidentUnlitDraw, FixedUnlitGraph, WebGl2AssetKey, WebGl2ResidentMesh,
    WebGl2Session as RhiSession, WebGl2SessionError,
};
use js_sys::{Array, Object, Reflect};
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

/// Explicit one-canvas browser rendering session.
#[wasm_bindgen]
pub struct WebGl2Session {
    inner: RhiSession,
    scene: PreparedBasicScene,
    graph: Option<PreparedBasicGraph>,
    resident_generation: Option<u64>,
    resident_meshes: Vec<WebGl2ResidentMesh>,
}

#[wasm_bindgen]
impl WebGl2Session {
    /// Creates a session for this canvas only; no document/global lookup occurs.
    #[wasm_bindgen(constructor)]
    pub fn new(canvas: HtmlCanvasElement) -> Result<Self, JsValue> {
        let scene = retained_scene().map_err(error)?;
        let extent = [canvas.width(), canvas.height()];
        let graph = if extent[0] == 0 || extent[1] == 0 {
            None
        } else {
            Some(scene.compile_presentable_graph(extent).map_err(error)?)
        };
        let mut inner = RhiSession::new(canvas.into()).map_err(error)?;
        let resident_meshes =
            if inner.state() == fluxel_rhi::adapter::webgl2::WebGl2SessionState::Active {
                prepare_webgl2_resident_meshes(&mut inner, &scene)?
            } else {
                Vec::new()
            };
        let resident_generation = (inner.state()
            == fluxel_rhi::adapter::webgl2::WebGl2SessionState::Active)
            .then_some(inner.generation());
        Ok(Self {
            resident_generation,
            resident_meshes,
            inner,
            scene,
            graph,
        })
    }
    /// Changes the exact drawing-buffer pixel extent.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), JsValue> {
        let graph = if width == 0 || height == 0 {
            None
        } else {
            Some(
                self.scene
                    .compile_presentable_graph([width, height])
                    .map_err(error)?,
            )
        };
        self.inner.resize(width, height).map_err(error)?;
        self.graph = graph;
        Ok(())
    }
    /// Renders once. The caller, not this session, owns RAF scheduling.
    pub fn render_once(&mut self) -> Result<JsValue, JsValue> {
        let started = js_sys::Date::now();
        if self.inner.state() != fluxel_rhi::adapter::webgl2::WebGl2SessionState::Active {
            return Err(structured_error(
                "invalid-state",
                "render",
                self.inner.generation(),
                "browser session is not active",
            ));
        }
        if self.resident_generation != Some(self.inner.generation()) {
            self.resident_meshes = prepare_webgl2_resident_meshes(&mut self.inner, &self.scene)?;
            self.resident_generation = Some(self.inner.generation());
        }
        let draws: Vec<_> = self
            .scene
            .draws()
            .iter()
            .zip(&self.resident_meshes)
            .map(|(draw, mesh)| FixedResidentUnlitDraw {
                mesh,
                pvm_and_color: draw.pvm_and_color(),
                insertion_index: draw.insertion_index(),
            })
            .collect();
        let graph = self.graph.as_ref().ok_or_else(|| {
            structured_error(
                "invalid-state",
                "render",
                self.inner.generation(),
                "presentable graph is suspended",
            )
        })?;
        let contract = FixedUnlitGraph::new(graph.compiled(), graph.draw_count(), graph.extent());
        let frame_marker = match self.inner.render_resident(&contract, &draws) {
            Ok(marker) => marker,
            Err(WebGl2SessionError::Busy) => {
                return Ok(frame_report(
                    "backpressure",
                    None,
                    self.inner.generation(),
                    js_sys::Date::now() - started,
                    self.inner.state(),
                ));
            }
            Err(error_value) if error_value.code() == "context-lost" => {
                return Ok(frame_report(
                    "context-lost",
                    None,
                    self.inner.generation(),
                    js_sys::Date::now() - started,
                    self.inner.state(),
                ));
            }
            Err(error_value) => return Err(error(error_value)),
        };
        Ok(frame_report(
            "submitted",
            Some(frame_marker),
            self.inner.generation(),
            js_sys::Date::now() - started,
            self.inner.state(),
        ))
    }
    /// Stops new frames without claiming completion.
    pub fn suspend(&mut self) -> Result<(), JsValue> {
        self.inner.suspend().map_err(error)
    }
    /// Resumes the fixed same-canvas session.
    pub fn resume(&mut self) -> Result<(), JsValue> {
        self.inner.resume().map(|_| ()).map_err(error)
    }
    /// Applies an adapter-observed context-loss event.
    pub fn context_lost(&mut self) {
        self.inner.context_lost();
    }
    /// Rebuilds resources after restoration of the same canvas only.
    pub fn context_restored(&mut self) -> Result<(), JsValue> {
        self.inner.context_restored().map(|_| ()).map_err(error)
    }
    /// Explicit shutdown with browser completion proof where available.
    pub fn dispose(&mut self) -> Result<(), JsValue> {
        self.inner.dispose().map_err(error)
    }
    /// Returns a non-draining snapshot of RHI-generated structured diagnostics.
    pub fn diagnostics_snapshot(&self) -> Array {
        let result = Array::new();
        for item in self.inner.diagnostics() {
            let object = Object::new();
            let _ = Reflect::set(&object, &"code".into(), &item.code.into());
            let _ = Reflect::set(&object, &"severity".into(), &item.severity.into());
            let _ = Reflect::set(&object, &"operation".into(), &item.operation.into());
            let _ = Reflect::set(
                &object,
                &"generation".into(),
                &JsValue::from_f64(item.generation as f64),
            );
            let _ = Reflect::set(&object, &"message".into(), &item.message.clone().into());
            let context = Object::new();
            for (key, value) in &item.context {
                let _ = Reflect::set(&context, &(*key).into(), &value.clone().into());
            }
            let _ = Reflect::set(&object, &"context".into(), &context);
            result.push(&object);
        }
        result
    }
}

fn prepare_webgl2_resident_meshes(
    session: &mut RhiSession,
    scene: &PreparedBasicScene,
) -> Result<Vec<WebGl2ResidentMesh>, JsValue> {
    scene
        .draws()
        .iter()
        .enumerate()
        .map(|(index, draw)| {
            session
                .resident_mesh(
                    WebGl2AssetKey::new(index as u64, 1),
                    draw.positions(),
                    draw.indices(),
                )
                .map_err(error)
        })
        .collect()
}

fn frame_report(
    outcome: &str,
    frame_marker: Option<u64>,
    generation: u64,
    cpu_submission_ms: f64,
    state: fluxel_rhi::adapter::webgl2::WebGl2SessionState,
) -> JsValue {
    let report = Object::new();
    let _ = Reflect::set(&report, &"outcome".into(), &outcome.into());
    if let Some(frame_marker) = frame_marker {
        let _ = Reflect::set(
            &report,
            &"frameMarker".into(),
            &JsValue::from_f64(frame_marker as f64),
        );
    }
    let _ = Reflect::set(&report, &"wasmMemoryBytes".into(), &wasm_memory_bytes());
    let _ = Reflect::set(
        &report,
        &"generation".into(),
        &JsValue::from_f64(generation as f64),
    );
    let _ = Reflect::set(
        &report,
        &"cpuSubmissionMs".into(),
        &JsValue::from_f64(cpu_submission_ms),
    );
    let _ = Reflect::set(&report, &"state".into(), &format!("{state:?}").into());
    report.into()
}

pub(crate) fn retained_scene()
-> Result<PreparedBasicScene, fluxel_renderer::adapter::PreparedBasicSceneError> {
    let triangle = |color| {
        Mesh::new(
            Geometry::from_positions(vec![
                [-0.16, -0.16, 0.5],
                [0.16, -0.16, 0.5],
                [0.0, 0.16, 0.5],
            ])
            .with_indices(vec![0, 1, 2])
            .expect("fixed valid triangle"),
            BasicMaterial::new(color).expect("fixed unit color"),
        )
    };
    let red = triangle([1.0, 0.0, 0.0, 1.0]);
    let green = triangle([0.0, 1.0, 0.0, 1.0]);
    let blue = triangle([0.0, 0.35, 1.0, 1.0]);
    let camera = Camera::default();
    let mut draws = DrawList::new(&camera);
    for (mesh, x, y) in [
        (&red, -0.45, -0.35),
        (&green, 0.45, -0.35),
        (&blue, 0.0, 0.42),
    ] {
        let mut matrix = ModelTransform::IDENTITY.world_from_model().to_owned();
        matrix[3][0] = x;
        matrix[3][1] = y;
        draws.push_transformed(
            mesh,
            ModelTransform::from_column_major(matrix).expect("finite affine placement"),
        );
    }
    PreparedBasicScene::prepare(&draws)
}
fn error(value: impl IntoBrowserError) -> JsValue {
    value.into_browser_error()
}

trait IntoBrowserError {
    fn into_browser_error(self) -> JsValue;
}

impl IntoBrowserError for WebGl2SessionError {
    fn into_browser_error(self) -> JsValue {
        structured_error(
            self.code(),
            self.operation(),
            self.generation(),
            &self.message(),
        )
    }
}

impl IntoBrowserError for fluxel_renderer::adapter::PreparedBasicSceneError {
    fn into_browser_error(self) -> JsValue {
        structured_error(
            "scene-preparation-failed",
            "prepare-scene",
            0,
            &self.to_string(),
        )
    }
}

impl IntoBrowserError for fluxel_renderer::adapter::PreparedBasicGraphError {
    fn into_browser_error(self) -> JsValue {
        structured_error(
            "graph-compile-failed",
            "compile-graph",
            0,
            &self.to_string(),
        )
    }
}

pub(crate) fn structured_error(
    code: &str,
    operation: &str,
    generation: u64,
    message: &str,
) -> JsValue {
    let object = Object::new();
    let _ = Reflect::set(&object, &"code".into(), &code.into());
    let _ = Reflect::set(&object, &"severity".into(), &"error".into());
    let _ = Reflect::set(&object, &"operation".into(), &operation.into());
    let _ = Reflect::set(
        &object,
        &"generation".into(),
        &JsValue::from_f64(generation as f64),
    );
    let _ = Reflect::set(&object, &"message".into(), &message.into());
    let _ = Reflect::set(&object, &"context".into(), &Object::new());
    object.into()
}

pub(crate) fn wasm_memory_bytes() -> JsValue {
    let memory = wasm_bindgen::memory();
    let buffer = js_sys::Reflect::get(&memory, &"buffer".into()).unwrap_or(JsValue::UNDEFINED);
    js_sys::Reflect::get(&buffer, &"byteLength".into()).unwrap_or(JsValue::from_f64(0.0))
}
