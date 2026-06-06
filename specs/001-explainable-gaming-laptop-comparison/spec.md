# Feature Spec: Explainable Gaming Laptop Comparison

## Summary

Add an explainable comparison flow for gaming laptop questions so users can compare 2-3 recommendations and see a clear rationale across performance, value, portability, and review signal tradeoffs.

## Problem

EggHatch-AI can surface laptop candidates, but it does not yet explain *why* one option is better than another in a structured, reviewable way. For a repo that aims to demonstrate agentic product reasoning, this leaves a gap between recommendation and trust.

## Goals

- Support explicit comparison-style questions such as `compare`, `vs`, `versus`, and `between`
- Produce a deterministic comparison object in addition to the free-form response
- Show a recommended option, key reasons, tradeoffs, and dimension winners
- Render the comparison clearly in the Streamlit UI

## Non-Goals

- Live pricing adapters
- Full benchmark ingestion
- Production-grade ranking models
- Personalized user profiles

## Functional Requirements

1. The system must detect when a query is asking for a comparison.
2. The analysis layer must build a structured comparison for the top 2-3 candidate laptops.
3. The comparison must include:
   - recommended laptop
   - short rationale
   - tradeoffs
   - per-candidate strengths and cautions
   - winners for core dimensions
4. The UI must display the structured comparison when available.
5. The fallback behavior must remain usable even if the LLM response is weak or unavailable.

## Success Criteria

- Users can ask a comparison question and receive both a conversational answer and a visible comparison breakdown.
- The explanation is grounded in existing fixture data and deterministic heuristics.
- The feature stays within EggHatch-AI's POC boundaries and does not pretend to use live commerce data.
