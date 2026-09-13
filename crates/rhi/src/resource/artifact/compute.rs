//! Fixed compute artifacts and their device-affine handles.
use super::super::*;
/// Why creation of a fixed compute artifact was rejected.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ComputeCreateError {
    /// The device cannot support the fixed shader's declared workgroup.
    UnsupportedComputeLimits,
    /// The device did not enable the exact RGBA8 storage access required by this recipe.
    UnsupportedStorageTexture,
    /// The pipeline, buffer, or binding operation crossed device identities.
    ForeignDevice,
    /// The requested storage-buffer view is empty, unaligned, overflowing, or out of bounds.
    InvalidBindingRange,
    /// The buffer's actual native usage cannot provide read-write storage access.
    StorageUsageRequired,
    /// The selected fixed artifact does not accept this binding recipe.
    BindingRecipeMismatch,
    /// The fixed WGSL artifact failed Naga parsing or validation.
    ShaderValidation(String),
    /// The backend rejected shader lowering or compute-pipeline compilation.
    ShaderCompilation(String),
    /// The backend could not create a fixed layout or another non-shader object.
    NativeObjectCreation(String),
    /// Native creation of a validated fixed binding object failed.
    NativeFailure(String),
}

impl fmt::Display for ComputeCreateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedComputeLimits => {
                f.write_str("device does not support this fixed compute workgroup")
            }
            Self::UnsupportedStorageTexture => {
                f.write_str("device does not support the fixed RGBA8 storage-texture recipe")
            }
            Self::ForeignDevice => f.write_str("compute objects belong to different devices"),
            Self::InvalidBindingRange => f.write_str("invalid compute storage-buffer range"),
            Self::StorageUsageRequired => {
                f.write_str("compute bindings require read-write storage usage")
            }
            Self::BindingRecipeMismatch => {
                f.write_str("compute bindings do not match the fixed artifact recipe")
            }
            Self::ShaderValidation(reason) => {
                write!(f, "compute shader validation failed: {reason}")
            }
            Self::ShaderCompilation(reason) => {
                write!(f, "compute shader compilation failed: {reason}")
            }
            Self::NativeObjectCreation(reason) => {
                write!(f, "native compute object creation failed: {reason}")
            }
            Self::NativeFailure(reason) => {
                write!(f, "native compute binding creation failed: {reason}")
            }
        }
    }
}
impl std::error::Error for ComputeCreateError {}

#[cfg(all(windows, any(feature = "dx12", feature = "vulkan")))]
pub(in crate::resource) fn map_compute_pipeline_create_error(
    error: crate::imp::ComputePipelineCreateError,
) -> ComputeCreateError {
    match error {
        crate::imp::ComputePipelineCreateError::ShaderValidation(reason) => {
            ComputeCreateError::ShaderValidation(reason)
        }
        crate::imp::ComputePipelineCreateError::ShaderCompilation(reason) => {
            ComputeCreateError::ShaderCompilation(reason)
        }
        crate::imp::ComputePipelineCreateError::NativeObjectCreation(reason) => {
            ComputeCreateError::NativeObjectCreation(reason)
        }
    }
}

/// The fixed, deterministic compute artifacts supported by this milestone.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ComputeKernel {
    /// Adds one to every addressed `u32` with wrapping arithmetic.
    WrappingAdd,
    /// Multiplies every addressed `u32` by three with wrapping arithmetic.
    WrappingMultiply,
    /// Packs every texel of one `Rgba8Unorm` texture into row-major `u32`s.
    TexturePackRgba8,
    /// Stores a fixed RGBA8 value into every texel of a storage texture.
    TextureStoreRgba8,
    /// Loads RGBA8 storage texels and packs them into a RW storage buffer.
    TextureLoadRgba8,
}

/// Portable identity of one fixed compute artifact.
///
/// This identifies the validated source-level artifact shared by every native
/// backend. It deliberately does not identify backend-specific DXIL or SPIR-V
/// binaries.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ComputeArtifactIdentity {
    /// Hash of the complete embedded WGSL module.
    pub module_source_hash: u64,
    /// Selected entry point within that module.
    pub entry_point: &'static str,
    /// Declared local workgroup shape.
    pub workgroup_size: [u32; 3],
    /// Version of the fixed one-RW-storage-buffer binding recipe.
    pub binding_recipe_version: u32,
}

impl ComputeKernel {
    /// Returns the fixed WGSL entry point for this artifact.
    pub const fn entry_point(self) -> &'static str {
        match self {
            Self::WrappingAdd => "wrapping_add",
            Self::WrappingMultiply => "wrapping_multiply",
            Self::TexturePackRgba8 => "pack_rgba8",
            Self::TextureStoreRgba8 => "store_rgba8",
            Self::TextureLoadRgba8 => "load_rgba8",
        }
    }

    /// Returns the fixed workgroup shape baked into this artifact.
    pub const fn workgroup_size(self) -> [u32; 3] {
        match self {
            Self::WrappingAdd | Self::WrappingMultiply => [64, 1, 1],
            Self::TexturePackRgba8 | Self::TextureStoreRgba8 | Self::TextureLoadRgba8 => [8, 8, 1],
        }
    }

    /// Returns the source hash recorded by conformance fixtures.
    pub const fn source_hash(self) -> u64 {
        // FNV-1a is an artifact identifier, not a security primitive. Keeping
        // the calculation adjacent to the embedded source makes accidental
        // source/hash drift impossible.
        fnv1a64(self.wgsl_source().as_bytes())
    }

    /// Returns the portable identity shared by DX12 and Vulkan lowering.
    pub const fn portable_identity(self) -> ComputeArtifactIdentity {
        ComputeArtifactIdentity {
            module_source_hash: self.source_hash(),
            entry_point: self.entry_point(),
            workgroup_size: self.workgroup_size(),
            binding_recipe_version: self.binding_recipe_version(),
        }
    }

    /// Returns the version of this slice's one read-write storage binding recipe.
    pub const fn binding_recipe_version(self) -> u32 {
        match self {
            Self::WrappingAdd | Self::WrappingMultiply => 1,
            Self::TexturePackRgba8 => 2,
            Self::TextureStoreRgba8 => 3,
            Self::TextureLoadRgba8 => 4,
        }
    }

    pub(crate) const fn wgsl_source(self) -> &'static str {
        match self {
            Self::WrappingAdd | Self::WrappingMultiply => {
                "@group(0) @binding(0) var<storage, read_write> values: array<u32>;\n\
         @compute @workgroup_size(64)\n\
         fn wrapping_add(@builtin(global_invocation_id) id: vec3<u32>) {\n\
             if (id.x < arrayLength(&values)) { values[id.x] = values[id.x] + 1u; }\n\
         }\n\
         @compute @workgroup_size(64)\n\
         fn wrapping_multiply(@builtin(global_invocation_id) id: vec3<u32>) {\n\
             if (id.x < arrayLength(&values)) { values[id.x] = values[id.x] * 3u; }\n\
         }"
            }
            Self::TexturePackRgba8 => {
                "@group(0) @binding(0) var source: texture_2d<f32>;\n\
         @group(0) @binding(1) var<storage, read_write> destination: array<u32>;\n\
         @compute @workgroup_size(8, 8, 1)\n\
         fn pack_rgba8(@builtin(global_invocation_id) id: vec3<u32>) {\n\
             let dimensions = textureDimensions(source);\n\
             if (id.x >= dimensions.x || id.y >= dimensions.y) { return; }\n\
             let pixel = textureLoad(source, vec2<i32>(id.xy), 0);\n\
             let r = u32(round(pixel.r * 255.0));\n\
             let g = u32(round(pixel.g * 255.0));\n\
             let b = u32(round(pixel.b * 255.0));\n\
             let a = u32(round(pixel.a * 255.0));\n\
             destination[id.y * dimensions.x + id.x] = r | (g << 8u) | (b << 16u) | (a << 24u);\n\
         }"
            }
            Self::TextureStoreRgba8 => {
                "@group(0) @binding(0) var output_image: texture_storage_2d<rgba8unorm, write>;\n\
         @compute @workgroup_size(8, 8, 1)\n\
         fn store_rgba8(@builtin(global_invocation_id) id: vec3<u32>) {\n\
             let dimensions = textureDimensions(output_image);\n\
             if (id.x < dimensions.x && id.y < dimensions.y) {\n\
                 textureStore(output_image, vec2<i32>(id.xy), vec4<f32>(0.25, 0.5, 0.75, 1.0));\n\
             }\n\
         }"
            }
            Self::TextureLoadRgba8 => {
                "@group(0) @binding(0) var source: texture_storage_2d<rgba8unorm, read>;\n\
         @group(0) @binding(1) var<storage, read_write> destination: array<u32>;\n\
         @compute @workgroup_size(8, 8, 1)\n\
         fn load_rgba8(@builtin(global_invocation_id) id: vec3<u32>) {\n\
             let dimensions = textureDimensions(source);\n\
             if (id.x >= dimensions.x || id.y >= dimensions.y) { return; }\n\
             let pixel = textureLoad(source, vec2<i32>(id.xy));\n\
             let r = u32(round(pixel.r * 255.0)); let g = u32(round(pixel.g * 255.0));\n\
             let b = u32(round(pixel.b * 255.0)); let a = u32(round(pixel.a * 255.0));\n\
             destination[id.y * dimensions.x + id.x] = r | (g << 8u) | (b << 16u) | (a << 24u);\n\
         }"
            }
        }
    }
}

pub(crate) const fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut index = 0;
    while index < bytes.len() {
        hash ^= bytes[index] as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        index += 1;
    }
    hash
}

pub(in crate::resource) struct ComputePipelineShared {
    pub(in crate::resource) _native: crate::imp::NativeComputePipeline,
    pub(in crate::resource) kernel: ComputeKernel,
    pub(in crate::resource) device: fluxel_rendergraph::DeviceIdentity,
}

/// An opaque, device-affine pipeline for one fixed compute artifact.
#[derive(Clone)]
pub struct ComputePipeline(pub(in crate::resource) Arc<ComputePipelineShared>);

/// A cloneable strong lease for a compute pipeline and its native layout/module.
#[derive(Clone)]
pub struct ComputePipelineLease(pub(in crate::resource) Arc<ComputePipelineShared>);

pub(in crate::resource) struct ComputeBindingsShared {
    pub(in crate::resource) _native: crate::imp::NativeComputeBindings,
    pub(in crate::resource) pipeline: ComputePipeline,
    pub(in crate::resource) _buffer: Option<BufferLease>,
    pub(in crate::resource) _texture: Option<TextureLease>,
    pub(in crate::resource) offset: u64,
    pub(in crate::resource) size: u64,
    pub(in crate::resource) device: fluxel_rendergraph::DeviceIdentity,
}

/// An opaque binding object for one closed fixed-compute recipe.
///
/// It retains the recipe's pipeline and exact buffer/texture inputs, without
/// exposing a bind-group layout or arbitrary shader resource interface.
#[derive(Clone)]
pub struct ComputeBindings(pub(in crate::resource) Arc<ComputeBindingsShared>);

/// A cloneable strong lease for a binding object and every resource it retains.
#[derive(Clone)]
pub struct ComputeBindingsLease(pub(in crate::resource) Arc<ComputeBindingsShared>);
