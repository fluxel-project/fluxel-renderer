//! Retained browser resources for the closed WebGPU resource recipe.
//!
//! This module is compiled in normal wasm builds. It owns the narrow resource
//! registry and does not expose WebGPU objects, shader text, or command APIs.

use std::{cell::Cell, collections::HashMap, rc::Rc};

use js_sys::Object;
use wasm_bindgen::JsValue;

use super::js::{call0, call1, set};
#[cfg(test)]
use super::js::{call2, call3, call5};
use super::{FixedResourceFrame, WebGpuResourceKey};

const COPY_SRC: u32 = 0x0004;
const COPY_DST: u32 = 0x0008;
const INDEX: u32 = 0x0010;
const VERTEX: u32 = 0x0020;
const UNIFORM: u32 = 0x0040;
const STORAGE: u32 = 0x0080;
const TEXTURE_COPY_SRC: u32 = 0x01;
const TEXTURE_COPY_DST: u32 = 0x02;
const TEXTURE_BINDING: u32 = 0x04;
const TEXTURE_STORAGE: u32 = 0x08;
const RENDER_ATTACHMENT: u32 = 0x10;

/// Completion-safe strong reference to a retained browser resource.
#[derive(Clone)]
pub(super) struct ResourceLease(Rc<Physical>);

impl ResourceLease {
    pub(super) fn native(&self) -> JsValue {
        self.0.native.clone()
    }
}

struct Physical {
    native: JsValue,
    texture: bool,
    generation: u64,
    descriptor: &'static str,
    usage: u32,
    state: Cell<&'static str>,
}

impl Drop for Physical {
    fn drop(&mut self) {
        // The registry and every accepted ticket own an Rc. Consequently this
        // is called only after the last completion-safe lease is gone.
        let _ = call0(&self.native, "destroy");
    }
}

/// Generation-scoped retained physical resources indexed by a closed role.
#[derive(Default)]
pub(super) struct ResourceRegistry {
    entries: HashMap<(WebGpuResourceKey, Role), ResourceLease>,
}

impl ResourceRegistry {
    /// Makes all previous-generation resources non-importable. Outstanding
    /// submission tickets retain their leases and therefore defer destruction.
    pub(super) fn retire_generation(&mut self) {
        self.entries.clear();
    }

    /// Removes one closed recipe identity. Submitted tickets keep their leases
    /// until completion, while a later use of this key creates fresh objects.
    pub(super) fn retire_key(&mut self, key: WebGpuResourceKey) {
        self.entries.retain(|(entry_key, _), _| *entry_key != key);
    }

    fn resource(
        &mut self,
        device: &JsValue,
        key: WebGpuResourceKey,
        role: Role,
        generation: u64,
    ) -> Result<ResourceLease, JsValue> {
        if let Some(value) = self.entries.get(&(key, role)) {
            if value.0.generation == generation {
                return Ok(value.clone());
            }
        }
        let (native, texture, descriptor, usage) = match role {
            Role::UploadVertex => (
                buffer(device, 24, COPY_SRC | COPY_DST)?,
                false,
                "buffer:upload-vertex",
                COPY_SRC | COPY_DST,
            ),
            Role::Vertex => (
                buffer(device, 24, VERTEX | COPY_DST)?,
                false,
                "buffer:vertex",
                VERTEX | COPY_DST,
            ),
            Role::Index => (
                buffer(device, 12, INDEX | COPY_DST)?,
                false,
                "buffer:index",
                INDEX | COPY_DST,
            ),
            Role::Uniform => (
                buffer(device, 16, UNIFORM | COPY_DST)?,
                false,
                "buffer:uniform",
                UNIFORM | COPY_DST,
            ),
            Role::Storage => (
                buffer(device, 4, STORAGE | COPY_SRC | COPY_DST)?,
                false,
                "buffer:storage",
                STORAGE | COPY_SRC | COPY_DST,
            ),
            Role::Copy => (
                buffer(device, 4, COPY_SRC | COPY_DST)?,
                false,
                "buffer:copy",
                COPY_SRC | COPY_DST,
            ),
            Role::UploadedTexture => (
                texture(device, "rgba8unorm", TEXTURE_COPY_SRC | TEXTURE_COPY_DST)?,
                true,
                "texture:uploaded-rgba8",
                TEXTURE_COPY_SRC | TEXTURE_COPY_DST,
            ),
            Role::SampledTexture => (
                texture(
                    device,
                    "rgba8unorm",
                    TEXTURE_COPY_SRC | TEXTURE_COPY_DST | TEXTURE_BINDING,
                )?,
                true,
                "texture:sampled-rgba8",
                TEXTURE_COPY_SRC | TEXTURE_COPY_DST | TEXTURE_BINDING,
            ),
            Role::Color => (
                texture(device, "rgba8unorm", RENDER_ATTACHMENT | TEXTURE_COPY_SRC)?,
                true,
                "texture:color-rgba8",
                RENDER_ATTACHMENT | TEXTURE_COPY_SRC,
            ),
            Role::Depth => (
                texture(device, "depth32float", RENDER_ATTACHMENT)?,
                true,
                "texture:depth32float",
                RENDER_ATTACHMENT,
            ),
            Role::StorageTexture => (
                texture(
                    device,
                    "rgba8unorm",
                    TEXTURE_STORAGE | TEXTURE_COPY_SRC | TEXTURE_COPY_DST,
                )?,
                true,
                "texture:storage-rgba8",
                TEXTURE_STORAGE | TEXTURE_COPY_SRC | TEXTURE_COPY_DST,
            ),
        };
        let value = ResourceLease(Rc::new(Physical {
            native,
            texture,
            generation,
            descriptor,
            usage,
            state: Cell::new("Undefined"),
        }));
        self.entries.insert((key, role), value.clone());
        Ok(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum Role {
    UploadVertex,
    Vertex,
    Index,
    Uniform,
    Storage,
    Copy,
    UploadedTexture,
    SampledTexture,
    Color,
    Depth,
    StorageTexture,
}

/// A recorded registry-backed recipe before its encoder is finished.
pub(super) struct PendingRecipe {
    #[cfg(test)]
    pub(super) device: JsValue,
    pub(super) encoder: JsValue,
    #[cfg(test)]
    pub(super) storage: JsValue,
    #[cfg(test)]
    pub(super) color: JsValue,
    #[cfg(test)]
    pub(super) storage_texture: JsValue,
    pub(super) leases: Vec<ResourceLease>,
    pub(super) roots: Vec<JsValue>,
    post_states: [&'static str; 11],
}

/// Records the fixed production recipe using the registry's physical objects.
pub(super) fn prepare(
    device: &JsValue,
    queue: &JsValue,
    registry: &mut ResourceRegistry,
    generation: u64,
    frame: FixedResourceFrame<'_>,
) -> Result<PendingRecipe, JsValue> {
    if frame.positions.len() != 3
        || frame.indices != [0, 1, 2]
        || frame.tint.iter().any(|v| !v.is_finite())
    {
        return Err(JsValue::from_str("fixed resource frame ABI rejected"));
    }
    let roles = [
        Role::UploadVertex,
        Role::Vertex,
        Role::Index,
        Role::Uniform,
        Role::Storage,
        Role::Copy,
        Role::UploadedTexture,
        Role::SampledTexture,
        Role::Color,
        Role::Depth,
        Role::StorageTexture,
    ];
    let leases: Vec<_> = roles
        .into_iter()
        .map(|role| registry.resource(device, frame.resources, role, generation))
        .collect::<Result<_, _>>()?;
    let native: Vec<_> = leases.iter().map(ResourceLease::native).collect();
    let recorded = super::resource_conformance::record_full(
        device,
        queue,
        &native,
        frame.positions,
        frame.tint,
        frame.sampled_rgba8,
    )?;
    for lease in &leases {
        let physical = &lease.0;
        let _metadata = (
            physical.texture,
            physical.descriptor,
            physical.usage,
            physical.state.get(),
        );
    }
    let post_states = [
        "CopySource",
        "CopyDestination",
        "CopyDestination",
        "CopyDestination",
        "ShaderStorageWrite",
        "CopyDestination",
        "CopySource",
        "ShaderSampledRead",
        "ColorAttachmentWrite",
        "DepthStencilWrite",
        "ShaderStorageWrite",
    ];
    Ok(PendingRecipe {
        #[cfg(test)]
        device: device.clone(),
        encoder: recorded.encoder,
        #[cfg(test)]
        storage: recorded.storage,
        #[cfg(test)]
        color: recorded.color,
        #[cfg(test)]
        storage_texture: recorded.storage_texture,
        leases,
        roots: recorded.roots,
        post_states,
    })
}

/// Applies state transitions only after the browser accepted `queue.submit`.
pub(super) fn commit(pending: &PendingRecipe) {
    commit_leases(&pending.leases, &pending.post_states);
}

fn commit_leases(leases: &[ResourceLease], states: &[&'static str]) {
    for (lease, state) in leases.iter().zip(states) {
        lease.0.state.set(state);
    }
}

/// Finishes a production pending recipe and returns its completion-safe roots.
pub(super) fn submit(
    device: &JsValue,
    queue: &JsValue,
    registry: &mut ResourceRegistry,
    generation: u64,
    frame: FixedResourceFrame<'_>,
) -> Result<(JsValue, PendingRecipe), JsValue> {
    let pending = prepare(device, queue, registry, generation, frame)?;
    let command = call0(&pending.encoder, "finish")?;
    Ok((command, pending))
}

fn buffer(device: &JsValue, size: u32, usage: u32) -> Result<JsValue, JsValue> {
    let descriptor = Object::new();
    set(&descriptor, "size", &size.into())?;
    set(&descriptor, "usage", &usage.into())?;
    call1(device, "createBuffer", &descriptor.into())
}
fn texture(device: &JsValue, format: &str, usage: u32) -> Result<JsValue, JsValue> {
    let size = Object::new();
    set(&size, "width", &1.into())?;
    set(&size, "height", &1.into())?;
    set(&size, "depthOrArrayLayers", &1.into())?;
    let descriptor = Object::new();
    set(&descriptor, "size", &size.into())?;
    set(&descriptor, "format", &format.into())?;
    set(&descriptor, "usage", &usage.into())?;
    call1(device, "createTexture", &descriptor.into())
}
#[cfg(test)]
pub(super) async fn observe_pending(
    queue: &JsValue,
    pending: PendingRecipe,
) -> Result<(), JsValue> {
    use js_sys::{Array, Promise, Uint8Array};
    use wasm_bindgen_futures::JsFuture;
    let storage_readback = buffer(&pending.device, 4, 1 | COPY_DST)?;
    let color_readback = buffer(&pending.device, 256, 1 | COPY_DST)?;
    let texture_readback = buffer(&pending.device, 256, 1 | COPY_DST)?;
    call5(
        &pending.encoder,
        "copyBufferToBuffer",
        &pending.storage,
        &0.into(),
        &storage_readback,
        &0.into(),
        &4.into(),
    )?;
    copy_texture_to_buffer(&pending.encoder, &pending.color, &color_readback)?;
    copy_texture_to_buffer(
        &pending.encoder,
        &pending.storage_texture,
        &texture_readback,
    )?;
    let command = call0(&pending.encoder, "finish")?;
    let commands = Array::new();
    commands.push(&command);
    call1(queue, "submit", &commands.into())?;
    commit(&pending);
    JsFuture::from(Promise::from(call0(queue, "onSubmittedWorkDone")?)).await?;
    JsFuture::from(Promise::from(call2(
        &storage_readback,
        "mapAsync",
        &1.into(),
        &0.into(),
    )?))
    .await?;
    let storage = Uint8Array::new(&call2(
        &storage_readback,
        "getMappedRange",
        &0.into(),
        &4.into(),
    )?)
    .to_vec();
    call0(&storage_readback, "unmap")?;
    let color = map_first_four(&color_readback).await?;
    let texture = map_first_four(&texture_readback).await?;
    for buffer in [&storage_readback, &color_readback, &texture_readback] {
        let _ = call0(buffer, "destroy");
    }
    if storage != [0x44, 0x33, 0x22, 0x11]
        || color != [12, 34, 56, 255]
        || texture != [255, 0, 0, 255]
    {
        return Err(JsValue::from_str("same-registry MAP_READ oracle mismatch"));
    }
    Ok(())
}

#[cfg(test)]
async fn map_first_four(buffer: &JsValue) -> Result<Vec<u8>, JsValue> {
    use js_sys::{Promise, Uint8Array};
    use wasm_bindgen_futures::JsFuture;
    JsFuture::from(Promise::from(call2(
        buffer,
        "mapAsync",
        &1.into(),
        &0.into(),
    )?))
    .await?;
    let bytes = Uint8Array::new(&call2(buffer, "getMappedRange", &0.into(), &4.into())?).to_vec();
    call0(buffer, "unmap")?;
    Ok(bytes)
}

#[cfg(test)]
fn copy_texture_to_buffer(
    encoder: &JsValue,
    texture: &JsValue,
    buffer: &JsValue,
) -> Result<(), JsValue> {
    let source = Object::new();
    set(&source, "texture", texture)?;
    let destination = Object::new();
    set(&destination, "buffer", buffer)?;
    set(&destination, "bytesPerRow", &256.into())?;
    set(&destination, "rowsPerImage", &1.into())?;
    let extent = Object::new();
    set(&extent, "width", &1.into())?;
    set(&extent, "height", &1.into())?;
    set(&extent, "depthOrArrayLayers", &1.into())?;
    call3(
        encoder,
        "copyTextureToBuffer",
        &source.into(),
        &destination.into(),
        &extent.into(),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lease(state: &'static str) -> ResourceLease {
        ResourceLease(Rc::new(Physical {
            native: JsValue::NULL,
            texture: false,
            generation: 1,
            descriptor: "test",
            usage: 0,
            state: Cell::new(state),
        }))
    }

    #[test]
    fn state_transitions_require_explicit_commit() {
        let leases = vec![lease("Undefined")];
        assert_eq!(leases[0].0.state.get(), "Undefined");
        commit_leases(&leases, &["Written"]);
        assert_eq!(leases[0].0.state.get(), "Written");
    }

    #[test]
    fn retiring_key_preserves_outstanding_lease_and_recreates_identity() {
        let key = WebGpuResourceKey::new(7);
        let held = lease("Undefined");
        let original = Rc::as_ptr(&held.0);
        let weak = Rc::downgrade(&held.0);
        let mut registry = ResourceRegistry::default();
        registry.entries.insert((key, Role::Storage), held.clone());
        registry.retire_key(key);
        assert!(weak.upgrade().is_some());
        drop(held);
        assert!(weak.upgrade().is_none());
        let replacement = lease("Undefined");
        registry
            .entries
            .insert((key, Role::Storage), replacement.clone());
        assert_ne!(Rc::as_ptr(&replacement.0), original);
    }
}
