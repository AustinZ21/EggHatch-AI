<!--
Sync Impact Report
- Version change: template placeholders -> 1.0.0 initial project constitution
- Updated sections: project principles, constraints, workflow, governance
- Artifacts reviewed for alignment: README.md, ROADMAP.md, .specify templates
- No template changes required for this initial draft
-->

# EggHatch-AI Constitution

## Core Principles

### I. Prototype Honesty Over Product Theater
EggHatch-AI is an open-source AI shopping agent prototype, not a production ecommerce platform. All specifications, plans, tasks, README updates, and implementation decisions must preserve that boundary. Static fixtures, simulated workflows, and planned modules are acceptable, but they must be labeled clearly. No contribution may imply live commerce reliability, current pricing accuracy, or enterprise readiness unless the repository actually provides those guarantees.

### II. Explainable Agent Workflows Over Black-Box Answers
The project exists to demonstrate how a shopping assistant can route a user request through structured analysis steps before responding. Query understanding, review analysis, sentiment signals, topic modeling, filtering logic, and final synthesis should remain legible in the codebase and explainable in the UI or documentation. New features should improve reasoning transparency rather than hide it behind increasingly opaque prompts or uncontrolled model behavior.

### III. Local-First Defaults With Graceful Degradation
The default development and demo experience must work locally with project fixtures and a local Ollama runtime. Optional heavy NLP components are allowed, but the project must continue to degrade gracefully when those components are unavailable. Fallback logic, especially for sentiment analysis and recommendation support flows, is a first-class requirement rather than an afterthought.

### IV. Source Integrity And Freshness Must Be Explicit
EggHatch-AI may combine product fixtures, review fixtures, extracted features, inferred relationships, and model-generated synthesis, but those categories must not be silently conflated. If a module uses static data, that should be documented. If a feature is simulated, that should be documented. If a recommendation depends on stale fixtures rather than live adapters, the code and docs should not present it as current market truth.

### V. Keep The System Small, Testable, And Worth Reading
This repository should stay understandable to a single contributor or a small open-source audience. Favor focused modules, bounded responsibilities, and lightweight tests around deterministic logic. Avoid framework sprawl, speculative abstractions, and production-style infrastructure that does not materially improve the core demo. Architecture should serve clarity and experimentation, not resume-driven complexity.

## Technical And Product Constraints

- The canonical runtime is Python plus Streamlit, with a local Ollama-backed response path.
- The repository should remain usable without hosted infrastructure or private services.
- Planned modules such as product knowledge, pricing availability, and build recommendation may stay stubbed until they are implemented with clear value.
- Any live adapters, scraping integrations, or benchmark lookups added later must document source, update strategy, failure behavior, and data freshness limits.
- The project must remain explicit that it is independent and not affiliated with Newegg or any retailer.
- Telemetry, secret handling, and generated artifacts must remain opt-in, documented, and easy to audit.

## Development Workflow And Quality Gates

- Functional changes should preserve the current POC promise: a small, locally runnable, explainable shopping agent demo.
- User-facing behavior changes should update README or docs when they alter setup, capability scope, or limitation boundaries.
- Deterministic logic should receive lightweight test coverage where practical, especially in the data pipeline and repo hygiene layers.
- When a change depends on unavailable local model weights, external APIs, or expensive runtime setup, the limitation must be stated in validation notes.
- Generated outputs such as `graphify-out/` or other tooling artifacts should be intentionally managed rather than accidentally committed.

## Governance

This constitution supersedes ad hoc preferences in specs, plans, tasks, and reviews. When there is tension between adding a flashy capability and preserving prototype honesty, local-first usability, or source integrity, the constitution wins.

Amendment rules:

- Amendments must be explicit edits to this file.
- Each amendment must preserve the project's open-source prototype identity unless the repository scope is intentionally redefined.
- Major principle changes should also be reflected in README positioning and roadmap language.
- Future `/speckit.plan` and `/speckit.tasks` outputs must treat these principles as binding constraints, not optional guidance.

**Version**: 1.0.0 | **Ratified**: 2026-06-06 | **Last Amended**: 2026-06-06
