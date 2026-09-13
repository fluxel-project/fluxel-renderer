//! Per-frame GPU resource and fixed-pipeline construction for WebGPU.
//!
//! Resource ownership remains with the parent session's tickets; this module
//! only constructs and destroys the JavaScript objects it is handed.

use std::{collections::HashMap, rc::Rc};

use js_sys::{Array, Object, Reflect, Uint8Array};
use wasm_bindgen::JsValue;

use super::js::{call0, call1, call2, call3, call4, set_js, set_raw};
use super::{Objects, WebGpuAssetKey, WebGpuCanvasFormat};

pub(super) struct FrameDrawResources {
    pub(super) position: JsValue,
    pub(super) index: JsValue,
    pub(super) uniform: JsValue,
    pub(super) bind_group: JsValue,
}

/// Completion-safe ownership of a closed resident asset. Browser objects never
/// escape this module; a session ticket retains this lease until completion.
#[derive(Clone)]
pub(super) struct ResidentLease(Rc<ResidentPhysical>);

struct ResidentPhysical {
    position: Option<JsValue>,
    index: Option<JsValue>,
    image: Option<JsValue>,
}

impl Drop for ResidentPhysical {
    fn drop(&mut self) {
        for object in [&self.position, &self.index, &self.image]
            .into_iter()
            .flatten()
        {
            let _ = call0(object, "destroy");
        }
    }
}

/// Generation-local closed asset lookup. Retirement only removes lookup
/// authority; accepted work keeps its `ResidentLease` alive.
#[derive(Default)]
pub(super) struct ResidentRegistry {
    meshes: HashMap<WebGpuAssetKey, ResidentLease>,
    images: HashMap<WebGpuAssetKey, ResidentLease>,
}

impl ResidentRegistry {
    pub(super) fn mesh(
        &mut self,
        device: &JsValue,
        queue: &JsValue,
        key: WebGpuAssetKey,
        positions: &[[f32; 3]],
        indices: &[u32],
    ) -> Result<ResidentLease, JsValue> {
        if let Some(value) = self.meshes.get(&key) {
            return Ok(value.clone());
        }
        let position = buffer(device, bytes_rounded(positions.len() * 3 * 4), 0x20 | 0x8)?;
        let index = match buffer(device, bytes_rounded(indices.len() * 4), 0x10 | 0x8) {
            Ok(value) => value,
            Err(error) => {
                let _ = call0(&position, "destroy");
                return Err(error);
            }
        };
        let values: Vec<f32> = positions.iter().flatten().copied().collect();
        write_buffer(
            queue,
            &position,
            &js_sys::Float32Array::from(values.as_slice()).into(),
        )?;
        if let Err(error) = write_buffer(queue, &index, &js_sys::Uint32Array::from(indices).into())
        {
            let _ = call0(&position, "destroy");
            let _ = call0(&index, "destroy");
            return Err(error);
        }
        let value = ResidentLease(Rc::new(ResidentPhysical {
            position: Some(position),
            index: Some(index),
            image: None,
        }));
        self.meshes.insert(key, value.clone());
        Ok(value)
    }

    pub(super) fn image(
        &mut self,
        device: &JsValue,
        queue: &JsValue,
        key: WebGpuAssetKey,
        extent: [u32; 2],
        pixels: &[u8],
    ) -> Result<ResidentLease, JsValue> {
        if let Some(value) = self.images.get(&key) {
            return Ok(value.clone());
        }
        let size = Object::new();
        set_raw(&size, "width", extent[0])?;
        set_raw(&size, "height", extent[1])?;
        set_raw(&size, "depthOrArrayLayers", 1_u32)?;
        let descriptor = Object::new();
        set_js(&descriptor, "size", &size)?;
        set_raw(&descriptor, "format", "rgba8unorm")?;
        set_raw(&descriptor, "usage", 0x04_u32 | 0x02_u32)?;
        let image = call1(device, "createTexture", &descriptor)?;
        let destination = Object::new();
        set_js(&destination, "texture", &image)?;
        let row_bytes = usize::try_from(extent[0])
            .unwrap_or(usize::MAX)
            .saturating_mul(4);
        let padded_row = row_bytes.next_multiple_of(256);
        let mut padded = vec![0_u8; padded_row.saturating_mul(extent[1] as usize)];
        for row in 0..extent[1] as usize {
            let source = row * row_bytes;
            let target = row * padded_row;
            padded[target..target + row_bytes].copy_from_slice(&pixels[source..source + row_bytes]);
        }
        let layout = Object::new();
        set_raw(
            &layout,
            "bytesPerRow",
            u32::try_from(padded_row).unwrap_or(u32::MAX),
        )?;
        set_raw(&layout, "rowsPerImage", extent[1])?;
        let copy_extent = Object::new();
        set_raw(&copy_extent, "width", extent[0])?;
        set_raw(&copy_extent, "height", extent[1])?;
        set_raw(&copy_extent, "depthOrArrayLayers", 1_u32)?;
        if let Err(error) = call4(
            queue,
            "writeTexture",
            &destination,
            &Uint8Array::from(padded.as_slice()).into(),
            &layout,
            &copy_extent,
        ) {
            let _ = call0(&image, "destroy");
            return Err(error);
        }
        let value = ResidentLease(Rc::new(ResidentPhysical {
            position: None,
            index: None,
            image: Some(image),
        }));
        self.images.insert(key, value.clone());
        Ok(value)
    }

    pub(super) fn retire(&mut self, key: WebGpuAssetKey) {
        self.meshes.remove(&key);
        self.images.remove(&key);
    }
    pub(super) fn retire_logical(&mut self, logical: u64) {
        self.meshes.retain(|key, _| key.logical != logical);
        self.images.retain(|key, _| key.logical != logical);
    }
    pub(super) fn retire_generation(&mut self) {
        self.meshes.clear();
        self.images.clear();
    }
}

impl ResidentLease {
    pub(super) fn mesh(&self) -> Option<(&JsValue, &JsValue)> {
        Some((self.0.position.as_ref()?, self.0.index.as_ref()?))
    }
}

fn buffer(device: &JsValue, size: u32, usage: u32) -> Result<JsValue, JsValue> {
    let descriptor = Object::new();
    Reflect::set(&descriptor, &"size".into(), &size.into())?;
    Reflect::set(&descriptor, &"usage".into(), &usage.into())?;
    call1(device, "createBuffer", &descriptor)
}

fn bytes_rounded(bytes: usize) -> u32 {
    u32::try_from(bytes.max(4).next_multiple_of(4)).unwrap_or(u32::MAX)
}

pub(super) fn create_frame_resources(
    device: &JsValue,
    layout: &JsValue,
    position_bytes: usize,
    index_bytes: usize,
) -> Result<FrameDrawResources, JsValue> {
    let position = buffer(device, bytes_rounded(position_bytes * 4), 0x20 | 0x8)?;
    let index = match buffer(device, bytes_rounded(index_bytes * 4), 0x10 | 0x8) {
        Ok(value) => value,
        Err(error) => {
            let _ = call0(&position, "destroy");
            return Err(error);
        }
    };
    let uniform = match buffer(device, 256, 0x40 | 0x8) {
        Ok(value) => value,
        Err(error) => {
            let _ = call0(&position, "destroy");
            let _ = call0(&index, "destroy");
            return Err(error);
        }
    };
    let bind_group = match frame_bind_group(device, layout, &uniform) {
        Ok(value) => value,
        Err(error) => {
            let _ = call0(&position, "destroy");
            let _ = call0(&index, "destroy");
            let _ = call0(&uniform, "destroy");
            return Err(error);
        }
    };
    Ok(FrameDrawResources {
        position,
        index,
        uniform,
        bind_group,
    })
}

fn frame_bind_group(
    device: &JsValue,
    layout: &JsValue,
    uniform: &JsValue,
) -> Result<JsValue, JsValue> {
    let resource = Object::new();
    set_raw(&resource, "buffer", uniform.clone())?;
    let entry = Object::new();
    set_raw(&entry, "binding", 0)?;
    set_raw(&entry, "resource", resource)?;
    let entries = Array::new();
    entries.push(&entry);
    let descriptor = Object::new();
    set_raw(&descriptor, "layout", layout.clone())?;
    set_raw(&descriptor, "entries", entries)?;
    call1(device, "createBindGroup", &descriptor)
}

pub(super) fn destroy_frame_resources(resources: Vec<FrameDrawResources>) {
    for resource in resources {
        let _ = call0(&resource.position, "destroy");
        let _ = call0(&resource.index, "destroy");
        let _ = call0(&resource.uniform, "destroy");
        drop(resource.bind_group);
    }
}

pub(super) fn unregister_uncaptured_error(objects: &Objects) {
    let _ = call2(
        &objects.device,
        "removeEventListener",
        &"uncapturederror".into(),
        objects.uncaptured.as_ref(),
    );
}

pub(super) fn pipeline(
    device: &JsValue,
    format: WebGpuCanvasFormat,
) -> Result<(JsValue, JsValue), JsValue> {
    let module = Object::new();
    set_raw(
        &module,
        "code",
        "struct U { pvm: mat4x4<f32>, color: vec4<f32> }; @group(0) @binding(0) var<uniform> u: U; struct O { @builtin(position) p: vec4<f32> }; @vertex fn vs(@location(0) p: vec3<f32>) -> O { var o: O; o.p = u.pvm * vec4<f32>(p,1.0); return o; } @fragment fn fs() -> @location(0) vec4<f32> { return u.color; }",
    )?;
    let shader = call1(device, "createShaderModule", &module)?;
    let bind = Object::new();
    set_raw(&bind, "binding", 0)?;
    // The fixed uniform is read by both vertex PVM and fragment color.
    set_raw(&bind, "visibility", 3)?;
    let buffer = Object::new();
    set_raw(&buffer, "type", "uniform")?;
    set_js(&bind, "buffer", &buffer)?;
    let layout = Object::new();
    let entries = Array::new();
    entries.push(&bind);
    set_js(&layout, "entries", &entries)?;
    let bgl = call1(device, "createBindGroupLayout", &layout)?;
    let pipeline_layout = Object::new();
    let layouts = Array::new();
    layouts.push(&bgl);
    set_js(&pipeline_layout, "bindGroupLayouts", &layouts)?;
    let pipeline_layout = call1(device, "createPipelineLayout", &pipeline_layout)?;
    let vertex = Object::new();
    set_js(&vertex, "module", &shader)?;
    set_raw(&vertex, "entryPoint", "vs")?;
    let attribute = Object::new();
    set_raw(&attribute, "shaderLocation", 0)?;
    set_raw(&attribute, "offset", 0)?;
    set_raw(&attribute, "format", "float32x3")?;
    let attributes = Array::new();
    attributes.push(&attribute);
    let vertex_buffer = Object::new();
    set_raw(&vertex_buffer, "arrayStride", 12)?;
    set_js(&vertex_buffer, "attributes", &attributes)?;
    let vertex_buffers = Array::new();
    vertex_buffers.push(&vertex_buffer);
    set_js(&vertex, "buffers", &vertex_buffers)?;
    let fragment = Object::new();
    set_js(&fragment, "module", &shader)?;
    set_raw(&fragment, "entryPoint", "fs")?;
    let target = Object::new();
    set_raw(&target, "format", format.as_str())?;
    let targets = Array::new();
    targets.push(&target);
    set_js(&fragment, "targets", &targets)?;
    let primitive = Object::new();
    set_raw(&primitive, "topology", "triangle-list")?;
    let descriptor = Object::new();
    set_js(&descriptor, "layout", &pipeline_layout)?;
    set_js(&descriptor, "vertex", &vertex)?;
    set_js(&descriptor, "fragment", &fragment)?;
    set_js(&descriptor, "primitive", &primitive)?;
    Ok((call1(device, "createRenderPipeline", &descriptor)?, bgl))
}

pub(super) fn write_buffer(
    queue: &JsValue,
    buffer: &JsValue,
    data: &JsValue,
) -> Result<(), JsValue> {
    call3(queue, "writeBuffer", buffer, &0.into(), data).map(|_| ())
}
