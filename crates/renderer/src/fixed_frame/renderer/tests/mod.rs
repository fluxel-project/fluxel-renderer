//! Verifies that public fixed-frame errors retain matchable causes and source chains.

use std::error::Error as _;

use fluxel_rendergraph::{CompileError, CompileErrorKind, DiagnosticContext};
use fluxel_rhi::{
    BufferUploadError, NativeExecutionError, adapter::fixed_artifacts::RasterCreateError,
};

use super::{
    DrawStartError, FixedFrameExecutionError, FixedFrameFailure, FixedFrameUniformObservationError,
};
use crate::{RenderPacketFailure, RenderPacketStartError};

fn compile_error() -> CompileError {
    CompileError {
        kind: CompileErrorKind::InvalidExportOrPresent,
        context: DiagnosticContext {
            passes: Vec::new(),
            resource: None,
            texture_slot: None,
            buffer_slot: None,
            detail: "test compile cause".into(),
            unsupported: None,
        },
    }
}

#[test]
fn draw_start_sources_preserve_compile_create_provider_and_upload_types() {
    let graph = DrawStartError::Graph(compile_error());
    assert_eq!(
        graph
            .source()
            .unwrap()
            .downcast_ref::<CompileError>()
            .unwrap()
            .kind,
        CompileErrorKind::InvalidExportOrPresent
    );

    let pipeline = DrawStartError::Pipeline(RasterCreateError::ForeignDevice);
    assert!(
        pipeline
            .source()
            .unwrap()
            .downcast_ref::<RasterCreateError>()
            .is_some()
    );

    let provider = DrawStartError::Provider(NativeExecutionError::ForeignResource);
    assert_eq!(
        provider
            .source()
            .unwrap()
            .downcast_ref::<NativeExecutionError>(),
        Some(&NativeExecutionError::ForeignResource)
    );

    let upload = DrawStartError::UniformStart(BufferUploadError::ForeignDevice);
    assert_eq!(
        upload.source().unwrap().downcast_ref::<BufferUploadError>(),
        Some(&BufferUploadError::ForeignDevice)
    );
}

#[test]
fn terminal_and_packet_sources_retain_observation_and_execution_layers() {
    let observation = FixedFrameUniformObservationError::Status(BufferUploadError::ForeignDevice);
    assert!(observation.source().unwrap().is::<BufferUploadError>());

    let execution = FixedFrameExecutionError::Backend(NativeExecutionError::ForeignResource);
    assert!(execution.source().unwrap().is::<NativeExecutionError>());

    let failure = FixedFrameFailure::RasterStart(execution.clone());
    assert!(failure.source().unwrap().is::<FixedFrameExecutionError>());

    let packet_start = RenderPacketStartError::Graph(compile_error());
    assert!(packet_start.source().unwrap().is::<CompileError>());

    let packet_failure = RenderPacketFailure::RasterStart { cause: execution };
    assert!(
        packet_failure
            .source()
            .unwrap()
            .is::<FixedFrameExecutionError>()
    );
}
