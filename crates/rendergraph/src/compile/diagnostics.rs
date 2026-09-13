//! Builds consistent compiler diagnostics for validation failures.

use super::*;

pub(super) fn unsupported(
    p: PassId,
    r: Option<ResourceId>,
    d: &str,
    requirement: CapabilityRequirement,
    c: &DeviceCapabilities,
) -> CompileError {
    err(
        CompileErrorKind::UnsupportedSemanticRequirement,
        vec![p],
        r,
        d,
        Some(Box::new(UnsupportedCapability {
            requirement,
            observed: Box::new(c.clone()),
        })),
    )
}
pub(super) fn unsupported_root(
    r: ResourceId,
    d: &str,
    requirement: CapabilityRequirement,
    c: &DeviceCapabilities,
) -> CompileError {
    err(
        CompileErrorKind::UnsupportedSemanticRequirement,
        Vec::new(),
        Some(r),
        d,
        Some(Box::new(UnsupportedCapability {
            requirement,
            observed: Box::new(c.clone()),
        })),
    )
}
pub(super) fn err(
    kind: CompileErrorKind,
    passes: Vec<PassId>,
    resource: Option<ResourceId>,
    detail: impl Into<String>,
    unsupported: Option<Box<UnsupportedCapability>>,
) -> CompileError {
    CompileError {
        kind,
        context: DiagnosticContext {
            passes,
            resource,
            texture_slot: None,
            buffer_slot: None,
            detail: detail.into(),
            unsupported,
        },
    }
}
