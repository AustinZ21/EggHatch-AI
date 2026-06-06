# Implementation Plan: Explainable Gaming Laptop Comparison

## Scope

Keep the feature inside the current POC architecture:

- `app/agents/comparison.py`
- `app/agents/trend_analysis.py`
- `app/prompts.py`
- `dashboard_app.py`
- `tests/`

## Approach

1. Add a lightweight deterministic comparison helper.
2. Use existing laptop fixture fields to score candidates on:
   - performance
   - value
   - reviews
   - portability
3. Attach the comparison object to `trend_insights`.
4. Update response synthesis guidance so the LLM can reference the structured explanation.
5. Render the comparison in Streamlit as a table plus recommendation summary.

## Constraints

- No live APIs
- No new heavy model requirements
- No hidden ranking logic; the comparison must stay explainable
- Prefer deterministic heuristics over opaque prompt-only reasoning

## Validation

- Lightweight tests for comparison query detection
- Lightweight tests for deterministic recommendation output
- Existing data pipeline tests must continue to pass
