//! Headless renderer domain types for Fluxel.
//!
//! This crate models cameras, geometry, meshes, basic materials, and ordered
//! draw lists. Its opt-in `gpu-upload` feature publishes immutable indexed-mesh
//! and RGBA8 (linear or sRGB-encoded) texture snapshots after native completion.
//! Its opt-in fixed-frame slice lowers closed constant-color, mip-zero
//! `textureLoad`, or fixed linear-clamp `textureSampleLevel` indexed draws
//! without exposing native resources or a configurable sampler.
//!
//! The default domain model validates material input before it enters a frame:
//!
//! ```
//! use fluxel_renderer::{BasicMaterial, Camera, DrawList, Geometry, Mesh};
//!
//! let geometry = Geometry::from_positions(vec![
//!     [-0.5, -0.5, 0.0],
//!     [0.5, -0.5, 0.0],
//!     [0.0, 0.5, 0.0],
//! ])
//! .with_indices(vec![0, 1, 2])?;
//! let material = BasicMaterial::new([0.2, 0.7, 1.0, 1.0])?;
//! let mesh = Mesh::new(geometry, material);
//! let camera = Camera::default();
//! let mut draws = DrawList::new(&camera);
//! draws.push(&mesh);
//! assert_eq!(draws.len(), 1);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![deny(missing_docs)]

#[cfg(feature = "gpu-upload")]
pub(crate) mod fixed_frame;
#[cfg(feature = "gpu-upload")]
pub(crate) mod frame_uniform;
#[cfg(feature = "gpu-upload")]
mod prepared_scene;
#[cfg(feature = "gpu-residency")]
mod residency;
mod shader;
#[cfg(feature = "gpu-upload")]
mod upload;

use core::fmt;

#[cfg(all(test, windows, feature = "gpu-upload"))]
pub(crate) fn native_fixture_guard() -> std::sync::MutexGuard<'static, ()> {
    use std::sync::{Mutex, OnceLock};

    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(feature = "gpu-upload")]
pub use fixed_frame::{
    DrawStartError, FixedFrameExecutionError, FixedFrameFailure, FixedFrameRasterObservationError,
    FixedFrameRenderer, FixedFrameStatus, FixedFrameSubmission, FixedFrameUniformObservationError,
    FrameImage, RenderPacket, RenderPacketBuildError, RenderPacketDrawBuildError,
    RenderPacketFailure, RenderPacketReservationError, RenderPacketStartError, RenderPacketStatus,
    RenderPacketSubmission,
};
#[cfg(all(windows, feature = "gpu-upload"))]
pub use fixed_frame::{VisibleFrameStartError, VisibleFrameStatus, VisibleFrameSubmission};
#[cfg(feature = "gpu-residency")]
pub use residency::{
    AssetResidencyError, AssetResidencyFailure, ImageAsset, MeshAsset, ResidencyRecreation,
    ResidentAssetPair, ResidentAssetStatus,
};
#[cfg(feature = "gpu-upload")]
pub use upload::{
    BaseColorTextureSnapshot, BaseColorTextureUpload, BaseColorTextureUploadFailure,
    BaseColorTextureUploadStartError, BaseColorTextureUploadStatus, IndexedMeshSnapshot,
    IndexedMeshUpload, IndexedMeshUploadFailure, IndexedMeshUploadStartError,
    IndexedMeshUploadStatus, NormalGeometry, NormalGeometryError, NormalGeometryStream,
    NormalIndexedMeshSnapshot, NormalIndexedMeshUpload, NormalIndexedMeshUploadFailure,
    NormalIndexedMeshUploadStartError, NormalIndexedMeshUploadStatus, Rgba8Image, Rgba8ImageError,
    SrgbBaseColorTextureSnapshot, SrgbBaseColorTextureUpload, SrgbBaseColorTextureUploadFailure,
    SrgbBaseColorTextureUploadStartError, SrgbBaseColorTextureUploadStatus,
    SrgbTexturedBasicMaterial, Srgba8Image, Srgba8ImageError, TexturedBasicMaterial,
    TexturedGeometry, TexturedGeometryError, TexturedGeometryStream, TexturedIndexedMeshSnapshot,
    TexturedIndexedMeshUpload, TexturedIndexedMeshUploadFailure,
    TexturedIndexedMeshUploadStartError, TexturedIndexedMeshUploadStatus, VertexColorGeometry,
    VertexColorGeometryError, VertexColorGeometryStream, VertexColorIndexedMeshSnapshot,
    VertexColorIndexedMeshUpload, VertexColorIndexedMeshUploadFailure,
    VertexColorIndexedMeshUploadStartError, VertexColorIndexedMeshUploadStatus,
    VertexColorMaterial, VertexColorMaterialError,
};
#[cfg(feature = "gpu-upload")]
/// Closed adapter-facing preparation contract for retained renderer paths.
///
/// This surface exists for sibling native/browser execution adapters. It is
/// not a general scene, resource, pipeline, or backend extension API.
pub mod adapter {
    pub use crate::prepared_scene::{
        PreparedBasicDraw, PreparedBasicGraph, PreparedBasicGraphError, PreparedBasicScene,
        PreparedBasicSceneError, PresentableFormat, PresentationProfile,
    };
}

/// A camera described by view and projection matrices.
#[derive(Clone, Debug, PartialEq)]
pub struct Camera {
    view: [[f32; 4]; 4],
    projection: [[f32; 4]; 4],
}

impl Camera {
    /// Creates a camera from column-major view and projection matrices.
    #[must_use]
    pub const fn new(view: [[f32; 4]; 4], projection: [[f32; 4]; 4]) -> Self {
        Self { view, projection }
    }

    /// Returns the camera view matrix.
    #[must_use]
    pub const fn view(&self) -> &[[f32; 4]; 4] {
        &self.view
    }

    /// Returns the camera projection matrix.
    #[must_use]
    pub const fn projection(&self) -> &[[f32; 4]; 4] {
        &self.projection
    }
}

impl Default for Camera {
    fn default() -> Self {
        const IDENTITY: [[f32; 4]; 4] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        Self::new(IDENTITY, IDENTITY)
    }
}

/// A finite affine transform from model space to world space.
///
/// Matrices use column-major storage and multiply column vectors. This value
/// deliberately permits singular transforms, reflections, and shear: those
/// are valid placements for the current unlit draw path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelTransform {
    column_major: [[f32; 4]; 4],
}

impl ModelTransform {
    /// The identity model-to-world transform.
    pub const IDENTITY: Self = Self {
        column_major: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    };

    /// Validates a column-major finite affine model-to-world matrix.
    pub fn from_column_major(column_major: [[f32; 4]; 4]) -> Result<Self, ModelTransformError> {
        if !column_major.iter().flatten().all(|value| value.is_finite()) {
            return Err(ModelTransformError::NonFinite);
        }
        if column_major[0][3] != 0.0
            || column_major[1][3] != 0.0
            || column_major[2][3] != 0.0
            || column_major[3][3] != 1.0
        {
            return Err(ModelTransformError::NonAffine);
        }
        Ok(Self { column_major })
    }

    /// Returns the validated column-major matrix mapping model space to world space.
    #[must_use]
    pub const fn world_from_model(&self) -> &[[f32; 4]; 4] {
        &self.column_major
    }
}

impl Default for ModelTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Why a model transform cannot be scheduled for drawing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ModelTransformError {
    /// One or more matrix elements were NaN or infinite.
    NonFinite,
    /// The matrix did not have the affine final row `[0, 0, 0, 1]`.
    NonAffine,
}

impl fmt::Display for ModelTransformError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite => formatter.write_str("model transform contains a non-finite value"),
            Self::NonAffine => formatter.write_str("model transform is not affine"),
        }
    }
}

impl std::error::Error for ModelTransformError {}

#[cfg(all(test, feature = "gpu-upload", not(windows)))]
mod non_windows_gpu_upload_tests {
    use super::*;

    #[test]
    fn gpu_upload_keeps_the_headless_renderer_api_off_windows() {
        let _: fn(fluxel_rhi::Device) -> FixedFrameRenderer = FixedFrameRenderer::new;
    }
}

/// Vertex positions and optional triangle indices for one mesh.
#[derive(Clone, Debug, PartialEq)]
pub struct Geometry {
    positions: Vec<[f32; 3]>,
    indices: Vec<u32>,
}

impl Geometry {
    /// Creates non-indexed geometry from vertex positions.
    #[must_use]
    pub fn from_positions(positions: impl Into<Vec<[f32; 3]>>) -> Self {
        Self {
            positions: positions.into(),
            indices: Vec::new(),
        }
    }

    /// Adds indices after checking that every index refers to a vertex.
    pub fn with_indices(mut self, indices: impl Into<Vec<u32>>) -> Result<Self, GeometryError> {
        let indices = indices.into();
        if let Some(&index) = indices
            .iter()
            .find(|&&index| index as usize >= self.positions.len())
        {
            return Err(GeometryError::IndexOutOfBounds {
                index,
                vertex_count: self.positions.len(),
            });
        }
        self.indices = indices;
        Ok(self)
    }

    /// Returns the vertex positions.
    #[must_use]
    pub fn positions(&self) -> &[[f32; 3]] {
        &self.positions
    }

    /// Returns the triangle indices, or an empty slice for non-indexed geometry.
    #[must_use]
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// Reports whether this geometry uses an index buffer.
    #[must_use]
    pub fn is_indexed(&self) -> bool {
        !self.indices.is_empty()
    }
}

/// An invalid geometry construction request.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum GeometryError {
    /// An index referred to a vertex outside the position array.
    IndexOutOfBounds {
        /// The invalid index.
        index: u32,
        /// The number of available positions.
        vertex_count: usize,
    },
}

impl fmt::Display for GeometryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IndexOutOfBounds {
                index,
                vertex_count,
            } => write!(
                formatter,
                "geometry index {index} is outside its {vertex_count} vertex positions"
            ),
        }
    }
}

impl std::error::Error for GeometryError {}

/// A minimal unlit material with a linear RGBA base color.
#[derive(Clone, Debug, PartialEq)]
pub struct BasicMaterial {
    base_color: [f32; 4],
}

impl BasicMaterial {
    /// Creates a basic material after checking each linear RGBA component is finite and in `[0, 1]`.
    pub fn new(base_color: [f32; 4]) -> Result<Self, BasicMaterialError> {
        for (component, value) in base_color.into_iter().enumerate() {
            if !value.is_finite() {
                return Err(BasicMaterialError::NonFinite { component });
            }
            if !(0.0..=1.0).contains(&value) {
                return Err(BasicMaterialError::OutOfRange { component });
            }
        }
        Ok(Self { base_color })
    }

    /// Returns the linear RGBA base color.
    #[must_use]
    pub const fn base_color(&self) -> [f32; 4] {
        self.base_color
    }
}

impl Default for BasicMaterial {
    fn default() -> Self {
        Self::new([1.0, 1.0, 1.0, 1.0]).expect("unit RGBA is a valid basic material")
    }
}

/// Why [`BasicMaterial`] cannot be constructed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum BasicMaterialError {
    /// A base-color component was NaN or infinite.
    NonFinite {
        /// Zero-based RGBA component index.
        component: usize,
    },
    /// A finite base-color component was outside the unit interval.
    OutOfRange {
        /// Zero-based RGBA component index.
        component: usize,
    },
}
impl fmt::Display for BasicMaterialError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { component } => {
                write!(
                    f,
                    "basic-material base-color component {component} is non-finite"
                )
            }
            Self::OutOfRange { component } => write!(
                f,
                "basic-material base-color component {component} is outside [0, 1]"
            ),
        }
    }
}
impl std::error::Error for BasicMaterialError {}

/// A renderable geometry and material pair.
#[derive(Clone, Debug, PartialEq)]
pub struct Mesh {
    geometry: Geometry,
    material: BasicMaterial,
}

impl Mesh {
    /// Creates a mesh from geometry and a basic material.
    #[must_use]
    pub fn new(geometry: Geometry, material: BasicMaterial) -> Self {
        Self { geometry, material }
    }

    /// Returns this mesh's geometry.
    #[must_use]
    pub const fn geometry(&self) -> &Geometry {
        &self.geometry
    }

    /// Returns this mesh's material.
    #[must_use]
    pub const fn material(&self) -> &BasicMaterial {
        &self.material
    }
}

/// A borrowed mesh scheduled for a future renderer submission.
#[derive(Clone, Copy, Debug)]
pub struct DrawItem<'a> {
    mesh: &'a Mesh,
    transform: ModelTransform,
}

impl<'a> DrawItem<'a> {
    /// Returns the mesh selected by this draw item.
    #[must_use]
    pub const fn mesh(self) -> &'a Mesh {
        self.mesh
    }

    /// Returns this draw's model-to-world placement.
    #[must_use]
    pub const fn transform(self) -> ModelTransform {
        self.transform
    }
}

/// An insertion-ordered list of meshes to draw for one camera.
#[derive(Clone, Debug)]
pub struct DrawList<'a> {
    camera: &'a Camera,
    items: Vec<DrawItem<'a>>,
}

impl<'a> DrawList<'a> {
    /// Creates an empty draw list for `camera`.
    #[must_use]
    pub const fn new(camera: &'a Camera) -> Self {
        Self {
            camera,
            items: Vec::new(),
        }
    }

    /// Returns the camera supplied when this list was created.
    #[must_use]
    pub const fn camera(&self) -> &'a Camera {
        self.camera
    }

    /// Appends `mesh` after all existing items.
    pub fn push(&mut self, mesh: &'a Mesh) {
        self.push_transformed(mesh, ModelTransform::IDENTITY);
    }

    /// Appends `mesh` with a finite affine model-to-world placement.
    pub fn push_transformed(&mut self, mesh: &'a Mesh, transform: ModelTransform) {
        self.items.push(DrawItem { mesh, transform });
    }

    /// Returns the scheduled items in submission order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = DrawItem<'a>> + '_ {
        self.items.iter().copied()
    }

    /// Returns the number of scheduled items.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Reports whether no meshes are scheduled.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

#[cfg(test)]
mod tests;
