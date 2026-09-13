//! Fixed graph and draw ABI validation for the browser WebGPU executor.
//!
//! This module validates only the portable retained recipe. It does not own
//! browser objects, lifecycle state, or GPU resource allocation.

use fluxel_rendergraph::{
    BufferUsageKind, LoadOp, PassKind, ResourceAccessState, ResourceUsageSummary, StoreOp,
    TextureUsageKind,
};

use super::{
    FixedResidentUnlitDraw, FixedUnlitDraw, FixedUnlitGraph, WebGpuCanvasFormat, WebGpuSessionError,
};

pub(super) fn validate(
    graph: &FixedUnlitGraph<'_>,
    draws: &[FixedUnlitDraw<'_>],
    format: WebGpuCanvasFormat,
    extent: [u32; 2],
) -> Result<(), WebGpuSessionError> {
    validate_graph(graph, draws.len(), format, extent)?;
    for (i, draw) in draws.iter().enumerate() {
        if draw.insertion_index != i
            || draw.indices.is_empty()
            || !draw.indices.len().is_multiple_of(3)
            || draw
                .indices
                .iter()
                .any(|&x| x as usize >= draw.positions.len())
            || draw.positions.iter().flatten().any(|x| !x.is_finite())
            || draw.pvm_and_color.iter().any(|x| !x.is_finite())
        {
            return Err(WebGpuSessionError::Contract("fixed-draw-contract-rejected"));
        }
    }
    Ok(())
}

pub(super) fn validate_resident(
    graph: &FixedUnlitGraph<'_>,
    draws: &[FixedResidentUnlitDraw<'_>],
    format: WebGpuCanvasFormat,
    extent: [u32; 2],
) -> Result<(), WebGpuSessionError> {
    validate_graph(graph, draws.len(), format, extent)?;
    if draws.iter().enumerate().any(|(index, draw)| {
        draw.insertion_index != index || draw.pvm_and_color.iter().any(|v| !v.is_finite())
    }) {
        return Err(WebGpuSessionError::Contract(
            "resident-draw-contract-rejected",
        ));
    }
    Ok(())
}

fn validate_graph(
    graph: &FixedUnlitGraph<'_>,
    draw_count: usize,
    format: WebGpuCanvasFormat,
    extent: [u32; 2],
) -> Result<(), WebGpuSessionError> {
    let plan = graph.compiled.execution_plan();
    let pass = plan.passes().first();
    let pass_ok = plan.passes().len() == 1
        && pass.is_some_and(|p| {
            p.kind == PassKind::Raster
                && p.raster.as_ref().is_some_and(|r| {
                    r.depth_stencil.is_none()
                        && r.colors.len() == 1
                        && r.colors[0].descriptor.operations.load == LoadOp::Clear([0., 0., 0., 1.])
                        && r.colors[0].descriptor.operations.store == StoreOp::Store
                })
        });
    let present = plan
        .final_transitions()
        .iter()
        .any(|t| t.after == ResourceAccessState::Present);
    let mut vb = 0;
    let mut ib = 0;
    let mut ub = 0;
    let mut presentable = 0;
    let mut unexpected = false;
    for r in plan.resource_requirements() {
        match r.usage {
            ResourceUsageSummary::Buffer(u) => {
                vb += usize::from(u.contains(BufferUsageKind::Vertex));
                ib += usize::from(u.contains(BufferUsageKind::Index));
                ub += usize::from(u.contains(BufferUsageKind::Uniform));
                unexpected |= u.contains(BufferUsageKind::StorageRead)
                    | u.contains(BufferUsageKind::StorageWrite)
                    | u.contains(BufferUsageKind::Indirect);
            }
            ResourceUsageSummary::Texture(u) => {
                presentable += usize::from(
                    u.contains(TextureUsageKind::ColorAttachment)
                        && u.contains(TextureUsageKind::Present),
                );
                unexpected |= u.contains(TextureUsageKind::StorageRead)
                    | u.contains(TextureUsageKind::StorageWrite)
                    | u.contains(TextureUsageKind::DepthStencilAttachment);
            }
        }
    }
    if graph.format != format
        || !pass_ok
        || !present
        || unexpected
        || vb != graph.draw_count
        || ib != graph.draw_count
        || ub != graph.draw_count
        || presentable != 1
        || graph.draw_count != draw_count
        || graph.draw_count == 0
        || graph.extent != extent
    {
        return Err(WebGpuSessionError::Contract(
            "fixed-graph-contract-rejected",
        ));
    }
    Ok(())
}
