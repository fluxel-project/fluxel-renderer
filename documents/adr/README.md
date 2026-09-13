# Architecture Decision Records

This directory preserves the small set of decisions that constrain future
work. Plans and reviews record a release's investigation and evidence; an ADR
records only a decision that remains relevant after that release.

The design documents describe the current system and how it works. ADRs record
why a durable boundary was chosen, its rejected alternatives, and its cost;
they link to designs instead of duplicating implementation detail. An ADR may
later be marked `Superseded by: ADR-NNNN`.

## Index

- [ADR-0010: Keep fixed-asset residency renderer-private](0010-renderer-private-fixed-asset-residency.md)
- [ADR-0009: Keep the resource floor closed and reuse stateful](0009-resource-floor-and-reuse-safety.md)
- [ADR-0008: Execute platform-specific test paths natively](0008-native-platform-test-gates.md)
- [ADR-0007: Keep fixed renderer recipes closed](0007-closed-fixed-renderer-recipes.md)
- [ADR-0006: Do not expose a general pipeline abstraction yet](0006-no-general-pipeline-yet.md)
- [ADR-0005: Require GPU conformance evidence for native correctness claims](0005-gpu-conformance-evidence.md)
- [ADR-0004: Quarantine accepted-unknown GPU work](0004-accepted-unknown-quarantine.md)
- [ADR-0003: Keep execution lowering serial until evidence justifies more](0003-serial-execution-lowering.md)
- [ADR-0002: Contain native handles and unsafe in the RHI implementation](0002-rhi-unsafe-containment.md)
- [ADR-0001: Keep assets and renderer policy outside RenderGraph](0001-assets-outside-rendergraph.md)

## Release mapping

Newest first. “Confirmed” means a release retained or strengthened an existing
constraint; it does not create a new ADR.

| Release | ADR decision recorded or confirmed |
| --- | --- |
| 0.2.8 | ADR-0007 confirmed; ADR-0008 added. |
| 0.2.7 | ADR-0007 confirmed. |
| 0.2.6 | ADR-0006 confirmed. |
| 0.2.5 | ADR-0003 and ADR-0006 confirmed. |
| 0.2.4 | ADR-0006 confirmed. |
| 0.2.3 | ADR-0001 and ADR-0006 confirmed. |
| 0.2.2 | ADR-0004 and ADR-0006 confirmed. |
| 0.2.1 | ADR-0004 and ADR-0006 confirmed. |
| 0.2.0 | ADR-0001 and ADR-0004 confirmed. |
| 0.1.4 | ADR-0002, ADR-0003, and ADR-0005 confirmed. |
| 0.1.3 | ADR-0003, ADR-0004, and ADR-0005 confirmed. |
| 0.1.2 | ADR-0003, ADR-0004, and ADR-0005 confirmed. |
| 0.1.1 | ADR-0002 confirmed; no submission decision existed yet. |
| 0.1.0 | ADR-0001 and ADR-0002 added. |
