# Thoughtform implementation plan

This repository owns named individual model definitions and model-neutral agency.
The [complete Cohesion Block B plan](https://github.com/wispsy/wispsy/blob/main/docs/cohesion-block-b/IMPLEMENTATION_PLAN.md)
records architecture, contracts, cross-library reuse, all milestones and validation.
The [Project](https://github.com/orgs/wispsy/projects/28) and issues hold current status.

## Foundation — TFORM-01

Define immutable `ModelDefinition`, `GenerationConfig`, `Being`, `Observation`,
`Action` and `Policy`. A being and a model have independent identity. Many beings
may share a model, and a deterministic policy needs no model. Observation/action
values carry no world authority; consuming applications validate proposals.

Provide a narrow `ThoughtformulaModel` adapter over caller-owned real
`ModernMicroGPT` and `BPETokenizer` instances. It uses the existing synchronous
stream generator, rejects incompatible vocabulary/context budgets, handles EOS
and closes on early exit. A zero output budget performs no computation. Do not
import model math into core, train/download/load artifacts on import, or create
another tokenizer or trainer. Hypnosis supplies the async execution boundary.

Acceptance: strict immutable values; model-neutral policy; real tiny Transformer
integration; empty/zero/unknown/context/EOS/close checks; installed base/model
profiles from wheel and source on Linux/macOS/Windows. The tiny model is random
and untrained and makes no language-quality claim.

## Named artifact resolution — TFORM-02

Resolve explicit checkpoints, tokenizer identity and optional specialization
adapters with model lineage. Depend on Thoughtformula's boundary qualification.
Call existing checkpoint APIs and require caller-selected source/destination;
no implicit downloads, mutable global model registry or automatic training.
Verify compatibility before generation. Share base models only through a
qualified serialized access/execution adapter.

## Domain memory — TFORM-MEM-01

Add versioned goals, experience and relationship records plus optional Backpack
codecs. Keep identity stable across restart. Bounded recall is a policy, not a
second storage engine. Memory updates do not imply weight updates or world
mutation. Verify close/reopen and independent identities in separate processes.

## Collective policy — TFORM-COLLECTIVE-01

Compose individual, group and environment roles through explicit observations,
proposed actions and bounded delegation. Require Hypnosis orchestration and
memory foundations. Perspectives, model access and action authority stay
separate. World rules remain authoritative even for a habitat or world mind.

## Integration and release

Thoughtformula owns computation/training/checkpoints. Hypnosis owns execution,
context/tools, evaluation and training workflow policy. Paracosm owns worlds and
accepted state. Adapters are optional; avoid circular core imports. Python 3.14+
remains the project floor. Use immutable source profiles until independent
compatible releases are selected. Public source is authorized; package/weight
publication and deployment require a separately selected concrete operation.
