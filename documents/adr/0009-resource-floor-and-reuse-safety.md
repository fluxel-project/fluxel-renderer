# ADR-0009: Keep the resource floor closed and reuse stateful

**Status:** Accepted

## Context

The 0.12 resource closure needs resources to cross frames and, for compatible
compiled graphs, permits transient allocation reuse. Those two conveniences can
silently become unsafe if a caller treats a logical name as a physical object,
forgets a resource's real outgoing state, or frees an object after a submission
whose acceptance or completion is not known.

The same closure is needed on native and browser backends, but their useful
capabilities differ. A portable declaration must report an unsupported
operation precisely and before it creates browser or native work. None of this
requires a general shader, descriptor, or pipeline API.

## Decision

Keep the 0.12 resource surface a fixed, evidence-backed floor. It consists of
the common fixed resource paths plus only the named closed compute/storage
recipes. Backend capability reports are observed facts; unsupported semantics
return structured `UnsupportedCapability` diagnostics and do not fall back to
emulation.

Persistent imports bind a provider-selected physical identity and generation,
actual incoming state, descriptor/usage facts, and a lease. Exports report the
actual outgoing state. A consumer must use that state rather than inventing a
convenient one.

Transient reuse is private executor policy. A slot is keyed by device and
compiled-graph generation as well as logical resource identity. It is eligible
only when whole-resource state can be carried exactly. On reuse, lowering uses
the remembered real outgoing state as the next frame's incoming state; partial
or mixed-state resources remain ineligible until the graph has a complete
subresource-state reuse model. Graph/device invalidation stops future checkout,
but does not destroy old slots until their work has terminal completion.

Pending and accepted-unknown work is quarantined. In particular, changing a
device or compiled-graph generation cannot free a slot still owned by pending
or unknown work. Browser resource registries apply the same rule per opaque
key and device generation: tickets retain their resource leases until safe
retirement.

## Alternatives

- Reset every reused resource to `Undefined` and let the next plan transition
  from that fabricated state.
- Reuse allocations across arbitrary subresource histories without tracking
  their state.
- Destroy all old-generation resources immediately during graph/device loss or
  browser recovery.
- Advertise a broad portable resource profile and emulate unsupported WebGL2
  compute/storage operations.
- Generalize shader, pipeline, layout, or binding construction to express the
  new recipes.

## Consequences

Reuse can save compatible allocations without weakening RenderGraph state
semantics, but it deliberately excludes some otherwise-compatible resources.
Providers and browser registries carry generation/lease bookkeeping that a
stateless allocator would avoid. Capability matrices remain intentionally
asymmetric: WebGL2 rejects compute and storage with zero side effects, and a
storage-texture read is not promised on a backend unless that exact closed path
is supported.

This decision reinforces, rather than replaces,
[ADR-0004](0004-accepted-unknown-quarantine.md) for unknown completion and
[ADR-0006](0006-no-general-pipeline-yet.md) for the closed-artifact boundary.

## Evidence

Portable contract tests cover capability diagnostics, identity/state/lease
validation, reuse eligibility, carry-over, invalidation, and pending/unknown
quarantine. Browser and native conformance fixtures are separate tests of the
implemented fixed paths; they do not turn the resource floor into a general
graphics API.
