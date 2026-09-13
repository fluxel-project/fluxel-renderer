//! Compiles a retained graph into immutable validation and execution artifacts.
//!
//! `model` owns public compile outputs, `dependency` derives execution order,
//! and `validation` proves that a graph can execute on the selected device.

use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::atomic::{AtomicU64, Ordering},
};

use crate::{
    access::{
        BufferRange, BufferReadUse, BufferReadWriteUse, BufferWriteUse, TextureAspect,
        TextureRange, TextureReadUse, TextureReadWriteUse, TextureWriteUse, WriteCoverage,
    },
    error::{
        CapabilityRequirement, CompileError, CompileErrorKind, DiagnosticContext,
        SurfaceCapabilityOperation, UnsupportedCapability,
    },
    graph::RenderGraph,
    handles::{PassId, ResourceId},
    internal::{
        AccessSemantic, DeclRange, PassDecl, ResourceDecl, ResourceKind, ResourceOrigin, RootDecl,
    },
    pass::PassKind,
    plan::{ExecutionPlan, build_execution_plan},
    resource::InitialContents,
    rhi::{
        DeviceCapabilities, ExternalOwnership, ResourceAccessState, TextureDimension, TextureFormat,
    },
};

mod dependency;
mod diagnostics;
mod model;
mod orchestrate;
mod validation;

static NEXT_COMPILED_ID: AtomicU64 = AtomicU64::new(1);

pub use model::{
    CapabilityFallback, CompileOutput, CompileReport, CompileResult, CompiledGraph, ExplicitOrder,
    PassDependency, RetainedSideEffect,
};

use dependency::{push_dep, topo};
use diagnostics::{err, unsupported, unsupported_root};
use model::VersionKey;
pub(crate) use orchestrate::compile_graph;
use validation::{
    buffer_boundary_state, buffer_state_supported, texture_boundary_state, texture_state_supported,
    validate_capabilities, validate_initialization, validate_references, validate_roots,
};
