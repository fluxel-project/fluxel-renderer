//! Coordinates the renderer's closed immutable upload domains and snapshot gates.
//!
//! Payload policy and publication live here; native allocation, submission, and
//! completion remain RHI responsibilities.

mod indexed;
mod normal;
mod shared;
mod texture;
mod textured;
mod vertex_color;

#[cfg(test)]
mod tests;

pub use indexed::*;
pub use normal::*;
pub use texture::*;
pub use textured::*;
pub use vertex_color::*;

pub(crate) use shared::{SnapshotDrawReservation, SnapshotUseError, completion_requires_retention};
