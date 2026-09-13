//! Production recording core for the closed WebGPU resource-floor recipe.
//!
//! It intentionally owns no session state and exports no resource abstraction.
//! The normal wasm path supplies registry-owned physical resources and retains
//! every returned root in its completion ticket. Tests add MAP_READ copies to
//! the same pending encoder; shader text and browser objects remain private.

use js_sys::{Array, Float32Array, Object, Uint8Array, Uint32Array};
use wasm_bindgen::JsValue;

use super::js::{call0, call1, call2, call3, call4, call5, set};

const RENDER_WGSL: &str = r#"
struct Tint { value: vec4<f32> };
@group(0) @binding(0) var<uniform> tint: Tint;
@group(1) @binding(0) var sampled_texture: texture_2d<f32>;
@group(1) @binding(1) var sampled_sampler: sampler;
struct VertexOut { @builtin(position) position: vec4<f32> };
@vertex fn vs(@location(0) position: vec2<f32>) -> VertexOut {
  var out: VertexOut;
  out.position = vec4<f32>(position, 0.0, 1.0);
  return out;
}
@fragment fn fs() -> @location(0) vec4<f32> {
  return textureSampleLevel(sampled_texture, sampled_sampler, vec2<f32>(0.5, 0.5), 0.0) * tint.value;
}
"#;

const COMPUTE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> output_buffer: array<u32>;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(1) fn cs() {
  output_buffer[0] = 0x11223344u;
  textureStore(output_texture, vec2<i32>(0, 0), vec4<f32>(1.0, 0.0, 0.0, 1.0));
}
"#;

/// Records the complete fixed resource recipe without observing its result.
/// Production calls this function; the test-only `run` path adds MAP_READ
/// copies and exact byte checks after the same raster/compute sequence.
pub(super) struct RecordedResourceRecipe {
    pub(super) encoder: JsValue,
    #[cfg(test)]
    pub(super) storage: JsValue,
    #[cfg(test)]
    pub(super) color: JsValue,
    #[cfg(test)]
    pub(super) storage_texture: JsValue,
    pub(super) roots: Vec<JsValue>,
}

pub(super) fn record_full(
    device: &JsValue,
    queue: &JsValue,
    resources: &[JsValue],
    positions: &[[f32; 2]],
    tint: [f32; 4],
    sampled_rgba8: [u8; 4],
) -> Result<RecordedResourceRecipe, JsValue> {
    if resources.len() != 11 {
        return Err(JsValue::from_str(
            "fixed resource registry role count mismatch",
        ));
    }
    let upload_vertex = resources[0].clone();
    let vertex = resources[1].clone();
    let index = resources[2].clone();
    let uniform = resources[3].clone();
    let storage = resources[4].clone();
    let uploaded_texture = resources[6].clone();
    let sampled_texture = resources[7].clone();
    let color = resources[8].clone();
    let depth = resources[9].clone();
    let storage_texture = resources[10].clone();
    let sampler = sampler(device)?;
    let positions: Vec<f32> = positions.iter().flatten().copied().collect();
    write_buffer(queue, &upload_vertex, &Float32Array::from(&positions[..]))?;
    write_buffer(queue, &index, &Uint32Array::from(&[0, 1, 2][..]))?;
    write_buffer(queue, &uniform, &Float32Array::from(&tint[..]))?;
    write_texture(queue, &uploaded_texture, &sampled_rgba8)?;
    let render_module = shader(device, RENDER_WGSL)?;
    let render_pipeline = render_pipeline(device, &render_module)?;
    let uniform_group = bind_group(
        device,
        &call1(&render_pipeline, "getBindGroupLayout", &0.into())?,
        &[buffer_entry(0, &uniform)?],
    )?;
    let sampled_group = bind_group(
        device,
        &call1(&render_pipeline, "getBindGroupLayout", &1.into())?,
        &[
            texture_entry(0, &call0(&sampled_texture, "createView")?)?,
            raw_entry(1, &sampler)?,
        ],
    )?;
    let compute_module = shader(device, COMPUTE_WGSL)?;
    let compute_pipeline = compute_pipeline(device, &compute_module)?;
    let compute_group = bind_group(
        device,
        &call1(&compute_pipeline, "getBindGroupLayout", &0.into())?,
        &[
            buffer_entry(0, &storage)?,
            texture_entry(1, &call0(&storage_texture, "createView")?)?,
        ],
    )?;
    let encoder = call0(device, "createCommandEncoder")?;
    call5(
        &encoder,
        "copyBufferToBuffer",
        &upload_vertex,
        &0.into(),
        &vertex,
        &0.into(),
        &24.into(),
    )?;
    copy_texture(&encoder, &uploaded_texture, &sampled_texture)?;
    render(
        &encoder,
        &color,
        &depth,
        &render_pipeline,
        &uniform_group,
        &sampled_group,
        &vertex,
        &index,
    )?;
    compute(&encoder, &compute_pipeline, &compute_group)?;
    Ok(RecordedResourceRecipe {
        encoder,
        #[cfg(test)]
        storage: storage.clone(),
        #[cfg(test)]
        color: color.clone(),
        #[cfg(test)]
        storage_texture: storage_texture.clone(),
        roots: vec![
            upload_vertex,
            vertex,
            index,
            uniform,
            storage,
            uploaded_texture,
            sampled_texture,
            color,
            depth,
            storage_texture,
            sampler,
            render_module,
            render_pipeline,
            uniform_group,
            sampled_group,
            compute_module,
            compute_pipeline,
            compute_group,
        ],
    })
}

fn sampler(device: &JsValue) -> Result<JsValue, JsValue> {
    let descriptor = Object::new();
    set(&descriptor, "minFilter", &"nearest".into())?;
    set(&descriptor, "magFilter", &"nearest".into())?;
    call1(device, "createSampler", &descriptor.into())
}

fn shader(device: &JsValue, source: &str) -> Result<JsValue, JsValue> {
    let descriptor = Object::new();
    set(&descriptor, "code", &source.into())?;
    call1(device, "createShaderModule", &descriptor.into())
}

fn render_pipeline(device: &JsValue, module: &JsValue) -> Result<JsValue, JsValue> {
    let attribute = Object::new();
    set(&attribute, "format", &"float32x2".into())?;
    set(&attribute, "offset", &0.into())?;
    set(&attribute, "shaderLocation", &0.into())?;
    let attributes = Array::new();
    attributes.push(&attribute);
    let vertex_buffer = Object::new();
    set(&vertex_buffer, "arrayStride", &8.into())?;
    set(&vertex_buffer, "attributes", &attributes.into())?;
    let buffers = Array::new();
    buffers.push(&vertex_buffer);
    let vertex = Object::new();
    set(&vertex, "module", module)?;
    set(&vertex, "entryPoint", &"vs".into())?;
    set(&vertex, "buffers", &buffers.into())?;
    let target = Object::new();
    set(&target, "format", &"rgba8unorm".into())?;
    let targets = Array::new();
    targets.push(&target);
    let fragment = Object::new();
    set(&fragment, "module", module)?;
    set(&fragment, "entryPoint", &"fs".into())?;
    set(&fragment, "targets", &targets.into())?;
    let primitive = Object::new();
    set(&primitive, "topology", &"triangle-list".into())?;
    let depth = Object::new();
    set(&depth, "format", &"depth32float".into())?;
    set(&depth, "depthWriteEnabled", &true.into())?;
    set(&depth, "depthCompare", &"always".into())?;
    let descriptor = Object::new();
    set(&descriptor, "layout", &"auto".into())?;
    set(&descriptor, "vertex", &vertex.into())?;
    set(&descriptor, "fragment", &fragment.into())?;
    set(&descriptor, "primitive", &primitive.into())?;
    set(&descriptor, "depthStencil", &depth.into())?;
    call1(device, "createRenderPipeline", &descriptor.into())
}

fn compute_pipeline(device: &JsValue, module: &JsValue) -> Result<JsValue, JsValue> {
    let compute = Object::new();
    set(&compute, "module", module)?;
    set(&compute, "entryPoint", &"cs".into())?;
    let descriptor = Object::new();
    set(&descriptor, "layout", &"auto".into())?;
    set(&descriptor, "compute", &compute.into())?;
    call1(device, "createComputePipeline", &descriptor.into())
}

fn raw_entry(binding: u32, resource: &JsValue) -> Result<JsValue, JsValue> {
    let entry = Object::new();
    set(&entry, "binding", &binding.into())?;
    set(&entry, "resource", resource)?;
    Ok(entry.into())
}

fn buffer_entry(binding: u32, buffer: &JsValue) -> Result<JsValue, JsValue> {
    let resource = Object::new();
    set(&resource, "buffer", buffer)?;
    raw_entry(binding, &resource.into())
}

fn texture_entry(binding: u32, view: &JsValue) -> Result<JsValue, JsValue> {
    raw_entry(binding, view)
}

fn bind_group(device: &JsValue, layout: &JsValue, entries: &[JsValue]) -> Result<JsValue, JsValue> {
    let values = Array::new();
    for entry in entries {
        values.push(entry);
    }
    let descriptor = Object::new();
    set(&descriptor, "layout", layout)?;
    set(&descriptor, "entries", &values.into())?;
    call1(device, "createBindGroup", &descriptor.into())
}

fn write_buffer(queue: &JsValue, buffer: &JsValue, values: &JsValue) -> Result<(), JsValue> {
    call3(queue, "writeBuffer", buffer, &0.into(), values).map(|_| ())
}

fn write_texture(queue: &JsValue, texture: &JsValue, bytes: &[u8]) -> Result<(), JsValue> {
    let destination = Object::new();
    set(&destination, "texture", texture)?;
    let layout = Object::new();
    set(&layout, "bytesPerRow", &256.into())?;
    set(&layout, "rowsPerImage", &1.into())?;
    let size = Object::new();
    set(&size, "width", &1.into())?;
    set(&size, "height", &1.into())?;
    set(&size, "depthOrArrayLayers", &1.into())?;
    let data = Uint8Array::from(bytes);
    call4(
        queue,
        "writeTexture",
        &destination.into(),
        &data.into(),
        &layout.into(),
        &size.into(),
    )?;
    Ok(())
}

fn copy_texture(encoder: &JsValue, source: &JsValue, destination: &JsValue) -> Result<(), JsValue> {
    let src = Object::new();
    set(&src, "texture", source)?;
    let dst = Object::new();
    set(&dst, "texture", destination)?;
    let extent = extent();
    call3(
        encoder,
        "copyTextureToTexture",
        &src.into(),
        &dst.into(),
        &extent.into(),
    )?;
    Ok(())
}

fn extent() -> Object {
    let extent = Object::new();
    set(&extent, "width", &1.into()).expect("plain object accepts width");
    set(&extent, "height", &1.into()).expect("plain object accepts height");
    set(&extent, "depthOrArrayLayers", &1.into()).expect("plain object accepts depth");
    extent
}

#[allow(
    clippy::too_many_arguments,
    reason = "the closed render witness names every resource role"
)]
fn render(
    encoder: &JsValue,
    color: &JsValue,
    depth: &JsValue,
    pipeline: &JsValue,
    uniform_group: &JsValue,
    sampled_group: &JsValue,
    vertex: &JsValue,
    index: &JsValue,
) -> Result<(), JsValue> {
    let color_attachment = Object::new();
    set(&color_attachment, "view", &call0(color, "createView")?)?;
    set(&color_attachment, "loadOp", &"clear".into())?;
    set(&color_attachment, "storeOp", &"store".into())?;
    let clear = Object::new();
    set(&clear, "r", &0.into())?;
    set(&clear, "g", &0.into())?;
    set(&clear, "b", &0.into())?;
    set(&clear, "a", &1.into())?;
    set(&color_attachment, "clearValue", &clear.into())?;
    let colors = Array::new();
    colors.push(&color_attachment);
    let depth_attachment = Object::new();
    set(&depth_attachment, "view", &call0(depth, "createView")?)?;
    set(&depth_attachment, "depthLoadOp", &"clear".into())?;
    set(&depth_attachment, "depthStoreOp", &"store".into())?;
    set(&depth_attachment, "depthClearValue", &1.into())?;
    let descriptor = Object::new();
    set(&descriptor, "colorAttachments", &colors.into())?;
    set(
        &descriptor,
        "depthStencilAttachment",
        &depth_attachment.into(),
    )?;
    let pass = call1(encoder, "beginRenderPass", &descriptor.into())?;
    call1(&pass, "setPipeline", pipeline)?;
    call2(&pass, "setBindGroup", &0.into(), uniform_group)?;
    call2(&pass, "setBindGroup", &1.into(), sampled_group)?;
    call3(&pass, "setVertexBuffer", &0.into(), vertex, &0.into())?;
    call2(&pass, "setIndexBuffer", index, &"uint32".into())?;
    call3(&pass, "drawIndexed", &3.into(), &1.into(), &0.into())?;
    call0(&pass, "end")?;
    Ok(())
}

fn compute(encoder: &JsValue, pipeline: &JsValue, group: &JsValue) -> Result<(), JsValue> {
    let pass = call1(encoder, "beginComputePass", &Object::new().into())?;
    call1(&pass, "setPipeline", pipeline)?;
    call2(&pass, "setBindGroup", &0.into(), group)?;
    call3(&pass, "dispatchWorkgroups", &1.into(), &1.into(), &1.into())?;
    call0(&pass, "end")?;
    Ok(())
}
