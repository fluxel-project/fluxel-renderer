//! Compile-only diagnostics example; run with `cargo run --example 07_compile_diagnostics`.
//! Execute callbacks are retained but are not invoked by `compile`.
//!
//! This example proves that callers can type-match stable diagnostic kinds and
//! inspect structured capability requirements without parsing error strings.

mod common;

use fluxel_rendergraph::*;

fn inspect(result: CompileResult) {
    match result {
        Ok(output) => {
            let _report: CompileReport = output.report;
        }
        Err(error) => match error.kind {
            CompileErrorKind::StaleOrForeignVersion
            | CompileErrorKind::ReadBeforeInitialization
            | CompileErrorKind::InvalidResourceBranch
            | CompileErrorKind::ConflictingAccess
            | CompileErrorKind::InvalidSubresourceRange
            | CompileErrorKind::DependencyCycle
            | CompileErrorKind::MissingImportContract
            | CompileErrorKind::UnsupportedSemanticRequirement => {
                if let Some(unsupported) = error.context.unsupported {
                    match unsupported.requirement {
                        CapabilityRequirement::Queue { .. }
                        | CapabilityRequirement::QueueConfiguration
                        | CapabilityRequirement::ColorAttachmentCount { .. }
                        | CapabilityRequirement::BufferState { .. }
                        | CapabilityRequirement::TextureFormat { .. }
                        | CapabilityRequirement::TextureState { .. }
                        | CapabilityRequirement::Surface { .. } => {}
                        _ => {}
                    }
                }
            }
            CompileErrorKind::InvalidExportOrPresent => {}
            _ => {}
        },
    }
}

fn main() {
    let graph = RenderGraph::new();
    inspect(graph.compile(&common::single_queue_capabilities()));
}
