# EggHatch-AI Roadmap

EggHatch-AI is intentionally small. The roadmap focuses on making the prototype easier to run, easier to evaluate, and more credible as an open-source AI agent demo.

## Phase 1: Open-Source Polish

- [x] Remove exploratory notebooks from the main repo surface
- [x] Rewrite README around the project value and runnable demo
- [x] Add license, contribution guide, security policy, and code of conduct
- [x] Add GitHub Pages project site
- [x] Add lightweight smoke tests and CI

## Phase 2: Evaluation Harness

- [ ] Add a small set of benchmark shopping queries
- [ ] Record expected structured outputs for deterministic parts of the flow
- [ ] Add regression tests for query classification and task decomposition
- [ ] Add example responses for offline review without requiring Ollama

## Phase 3: Agent Capability Expansion

- [ ] Implement a bounded benchmark lookup adapter
- [ ] Implement a pricing/availability fixture adapter
- [ ] Add product comparison explanations
- [ ] Add recommendation rationale scoring

## Phase 4: Interaction Polish

- [ ] Improve Streamlit layout and response formatting
- [ ] Add visible trace of agent steps
- [ ] Add a sidebar showing detected budget, use case, and constraints
- [ ] Add exportable recommendation summaries

## Phase 5: Production-Like Architecture Notes

- [ ] Document a production architecture variant
- [ ] Add cost/latency tradeoff notes for LLM and embedding choices
- [ ] Add privacy and data freshness considerations
- [ ] Add a clear path for replacing fixture data with real adapters
