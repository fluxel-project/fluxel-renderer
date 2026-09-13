//! Small JavaScript reflection boundary used by the browser-private executor.
//!
//! This module does not model WebGPU lifecycle or resources. It only centralizes
//! dynamic property access and method dispatch so the session remains readable.

use js_sys::{Function, Object, Reflect};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::HtmlCanvasElement;

use super::{WebGpuAdapterInfo, WebGpuCanvasFormat, WebGpuSessionError};

/// RHI-private boundary around the two asynchronous browser device requests.
///
/// Production delegates directly to WebGPU. Browser contract tests replace
/// only these calls so they can control settlement while the real
/// `request_browser` recovery path remains under test.
pub(super) trait BrowserRequestProvider {
    fn request_adapter(&self, gpu: &JsValue) -> Result<js_sys::Promise, JsValue>;
    fn request_device(&self, adapter: &JsValue) -> Result<js_sys::Promise, JsValue>;
}

pub(super) struct ProductionBrowserRequests;

impl BrowserRequestProvider for ProductionBrowserRequests {
    fn request_adapter(&self, gpu: &JsValue) -> Result<js_sys::Promise, JsValue> {
        call0(gpu, "requestAdapter").map(js_sys::Promise::from)
    }

    fn request_device(&self, adapter: &JsValue) -> Result<js_sys::Promise, JsValue> {
        call0(adapter, "requestDevice").map(js_sys::Promise::from)
    }
}

pub(super) fn global() -> JsValue {
    js_sys::global().into()
}

/// Acquires browser objects without letting them escape the RHI-private
/// executor. Lifecycle installation/configuration remains in the parent.
pub(super) async fn request_browser(
    canvas: &HtmlCanvasElement,
    requests: &dyn BrowserRequestProvider,
) -> Result<
    (
        WebGpuCanvasFormat,
        WebGpuAdapterInfo,
        JsValue,
        JsValue,
        JsValue,
    ),
    WebGpuSessionError,
> {
    let gpu = get(&global(), "navigator")
        .and_then(|navigator| get(&navigator, "gpu"))
        .ok_or(WebGpuSessionError::Unavailable)?;
    let format_name = call0(&gpu, "getPreferredCanvasFormat")
        .map_err(|error| browser("preferred-format", 0, error))?
        .as_string()
        .ok_or(WebGpuSessionError::Unavailable)?;
    let format = WebGpuCanvasFormat::parse(&format_name)
        .ok_or(WebGpuSessionError::UnsupportedFormat(format_name))?;
    let context: JsValue = canvas
        .get_context("webgpu")
        .map_err(|error| browser("canvas-context", 0, error))?
        .ok_or(WebGpuSessionError::Unavailable)?
        .into();
    let adapter = JsFuture::from(
        requests
            .request_adapter(&gpu)
            .map_err(|error| browser("request-adapter", 0, error))?,
    )
    .await
    .map_err(|error| browser("request-adapter", 0, error))?;
    if adapter.is_null() || adapter.is_undefined() {
        return Err(WebGpuSessionError::Unavailable);
    }
    let info = get(&adapter, "info")
        .map(|value| WebGpuAdapterInfo {
            vendor: get(&value, "vendor")
                .and_then(|value| value.as_string())
                .unwrap_or_default(),
            architecture: get(&value, "architecture")
                .and_then(|value| value.as_string())
                .unwrap_or_default(),
            device: get(&value, "device")
                .and_then(|value| value.as_string())
                .unwrap_or_default(),
        })
        .unwrap_or_default();
    let device = JsFuture::from(
        requests
            .request_device(&adapter)
            .map_err(|error| browser("request-device", 0, error))?,
    )
    .await
    .map_err(|error| browser("request-device", 0, error))?;
    let queue = match get(&device, "queue") {
        Some(queue) => queue,
        None => {
            // The browser handed us an owned device but not the queue needed
            // to use it. Do not leak that partial generation on this path.
            let _ = call0(&device, "destroy");
            return Err(WebGpuSessionError::Unavailable);
        }
    };
    Ok((format, info, device, queue, context))
}

pub(super) fn to_js(error: WebGpuSessionError) -> JsValue {
    // Promise rejection is part of the binding contract. Do not collapse a
    // typed RHI failure into a string: JS must retain its closed fields.
    let value = Object::new();
    let context = Object::new();
    let _ = Reflect::set(&value, &"code".into(), &error.code().into());
    let _ = Reflect::set(&value, &"severity".into(), &"error".into());
    let _ = Reflect::set(&value, &"operation".into(), &error.operation().into());
    let _ = Reflect::set(&value, &"generation".into(), &error.generation().into());
    let _ = Reflect::set(&value, &"message".into(), &error.message().into());
    let _ = Reflect::set(&context, &"source".into(), &"fluxel-rhi-webgpu".into());
    let _ = Reflect::set(&value, &"context".into(), &context.into());
    value.into()
}

pub(super) fn get(value: &JsValue, name: &str) -> Option<JsValue> {
    Reflect::get(value, &name.into())
        .ok()
        .filter(|value| !value.is_null() && !value.is_undefined())
}

/// Extracts a browser Error-like message, retaining Debug as a non-error
/// object fallback.
pub(super) fn js_message(value: &JsValue) -> String {
    get(value, "message")
        .and_then(|message| message.as_string())
        .filter(|message| !message.is_empty())
        .unwrap_or_else(|| format!("{value:?}"))
}

pub(super) fn set(value: &JsValue, name: &str, field: &JsValue) -> Result<(), JsValue> {
    Reflect::set(value, &name.into(), field).and_then(|ok| {
        ok.then_some(())
            .ok_or_else(|| JsValue::from_str("Reflect.set rejected"))
    })
}

fn method(value: &JsValue, name: &str) -> Result<Function, JsValue> {
    get(value, name)
        .ok_or_else(|| JsValue::from_str("missing method"))
        .and_then(|value| value.dyn_into::<Function>())
}

pub(super) fn call0(value: &JsValue, name: &str) -> Result<JsValue, JsValue> {
    method(value, name)?.call0(value)
}

pub(super) fn call1(value: &JsValue, name: &str, a: &JsValue) -> Result<JsValue, JsValue> {
    method(value, name)?.call1(value, a)
}

pub(super) fn call2(
    value: &JsValue,
    name: &str,
    a: &JsValue,
    b: &JsValue,
) -> Result<JsValue, JsValue> {
    method(value, name)?.call2(value, a, b)
}

pub(super) fn call3(
    value: &JsValue,
    name: &str,
    a: &JsValue,
    b: &JsValue,
    c: &JsValue,
) -> Result<JsValue, JsValue> {
    method(value, name)?.call3(value, a, b, c)
}

/// Calls a four-argument browser method while retaining its receiver.
pub(super) fn call4(
    value: &JsValue,
    name: &str,
    a: &JsValue,
    b: &JsValue,
    c: &JsValue,
    d: &JsValue,
) -> Result<JsValue, JsValue> {
    method(value, name)?.call4(value, a, b, c, d)
}

/// Calls a five-argument browser method while retaining its receiver.
pub(super) fn call5(
    value: &JsValue,
    name: &str,
    a: &JsValue,
    b: &JsValue,
    c: &JsValue,
    d: &JsValue,
    e: &JsValue,
) -> Result<JsValue, JsValue> {
    method(value, name)?.call5(value, a, b, c, d, e)
}

pub(super) fn browser(
    operation: &'static str,
    generation: u64,
    value: JsValue,
) -> WebGpuSessionError {
    WebGpuSessionError::Browser {
        code: "webgpu-operation-failed",
        operation,
        generation,
        message: js_message(&value),
    }
}

pub(super) fn set_js(value: &JsValue, name: &str, field: &JsValue) -> Result<(), JsValue> {
    Reflect::set(value, &name.into(), field).map(|_| ())
}

pub(super) fn set_raw<T: Into<JsValue>>(
    object: &JsValue,
    name: &str,
    value: T,
) -> Result<(), JsValue> {
    Reflect::set(object, &name.into(), &value.into()).map(|_| ())
}
