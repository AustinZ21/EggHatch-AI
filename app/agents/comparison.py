"""Deterministic laptop comparison helpers for EggHatch-AI.

This module keeps comparison logic lightweight and explainable so the
repository can offer structured recommendation rationale even when the
LLM response layer is unavailable or terse.
"""

from __future__ import annotations

from math import isfinite
from typing import Any


COMPARISON_QUERY_TERMS = (
    "compare",
    "comparison",
    " versus ",
    " vs ",
    "better than",
    "which is better",
    "which one is better",
)


def looks_like_comparison_query(query: str | None) -> bool:
    """Return True when the user is explicitly asking for a comparison."""
    if not query:
        return False

    haystack = f" {query.strip().lower()} "
    if "between" in haystack and any(term in haystack for term in ("which", "better", "choose", "pick")):
        return True

    return any(term in haystack for term in COMPARISON_QUERY_TERMS)


def _to_float(value: Any, default: float = 0.0) -> float:
    """Coerce arbitrary values into finite floats."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    return number if isfinite(number) else default


def _normalize(values: list[float], *, reverse: bool = False) -> list[float]:
    """Normalize values into the [0, 1] range."""
    if not values:
        return []

    floor = min(values)
    ceiling = max(values)

    if ceiling == floor:
        return [0.5 for _ in values]

    normalized = [(value - floor) / (ceiling - floor) for value in values]
    if reverse:
        normalized = [1.0 - value for value in normalized]
    return normalized


def _comparison_mode(query: str | None, use_case: str | None = None) -> str:
    """Infer which tradeoff profile best fits the comparison request."""
    haystack = " ".join(part for part in [query or "", use_case or ""]).lower()

    if any(term in haystack for term in ("competitive", "fps", "ray tracing", "aaa", "performance")):
        return "performance"
    if any(term in haystack for term in ("budget", "value", "cheapest", "under $", "under ", "bang for the buck")):
        return "value"
    if any(term in haystack for term in ("portable", "travel", "light", "lightweight")):
        return "portability"
    return "balanced"


def _weights_for_mode(mode: str) -> dict[str, float]:
    """Return comparison weights for the inferred mode."""
    if mode == "performance":
        return {"performance": 0.45, "value": 0.2, "reviews": 0.2, "portability": 0.15}
    if mode == "value":
        return {"performance": 0.25, "value": 0.4, "reviews": 0.2, "portability": 0.15}
    if mode == "portability":
        return {"performance": 0.2, "value": 0.2, "reviews": 0.2, "portability": 0.4}
    return {"performance": 0.35, "value": 0.25, "reviews": 0.25, "portability": 0.15}


def _candidate_metrics(candidate: dict[str, Any]) -> dict[str, float]:
    """Compute raw comparison metrics for a laptop candidate."""
    price = _to_float(candidate.get("price"), 0.0)
    gpu_tier = _to_float(candidate.get("gpu_tier"), 0.0)
    processor_tier = _to_float(candidate.get("processor_tier"), 0.0)
    rating_value = _to_float(candidate.get("rating_value"), 0.0)
    review_count = _to_float(candidate.get("review_count"), 0.0)
    display_size = _to_float(candidate.get("display_size"), 15.6)

    performance = gpu_tier * 0.65 + processor_tier * 0.35
    value = (performance / max(price, 1.0)) * 1000.0
    reviews = rating_value + min(review_count, 500.0) / 100.0
    portability = 1.0 / max(display_size, 1.0)

    return {
        "performance": performance,
        "value": value,
        "reviews": reviews,
        "portability": portability,
    }


def _strengths(candidate: dict[str, Any], candidates: list[dict[str, Any]]) -> list[str]:
    """Generate short strengths for a candidate relative to its peers."""
    strengths: list[str] = []

    prices = [_to_float(item.get("price")) for item in candidates]
    gpu_tiers = [_to_float(item.get("gpu_tier")) for item in candidates]
    ratings = [_to_float(item.get("rating_value")) for item in candidates]
    display_sizes = [_to_float(item.get("display_size"), 15.6) for item in candidates]

    if _to_float(candidate.get("price")) == min(prices):
        strengths.append("Lowest price among the compared options")
    if _to_float(candidate.get("gpu_tier")) == max(gpu_tiers):
        strengths.append("Strongest GPU tier in this comparison")
    if _to_float(candidate.get("rating_value")) == max(ratings):
        strengths.append("Best average review rating in this comparison")
    if _to_float(candidate.get("display_size"), 15.6) == min(display_sizes):
        strengths.append("Most portable footprint in the comparison set")

    if not strengths:
        strengths.append("Balanced option without a major weak spot")

    return strengths[:3]


def _cautions(candidate: dict[str, Any], candidates: list[dict[str, Any]]) -> list[str]:
    """Generate short cautions for a candidate relative to its peers."""
    cautions: list[str] = []

    prices = [_to_float(item.get("price")) for item in candidates]
    gpu_tiers = [_to_float(item.get("gpu_tier")) for item in candidates]
    ratings = [_to_float(item.get("rating_value")) for item in candidates]
    display_sizes = [_to_float(item.get("display_size"), 15.6) for item in candidates]

    if _to_float(candidate.get("price")) == max(prices):
        cautions.append("Highest price in the compared set")
    if _to_float(candidate.get("gpu_tier")) == min(gpu_tiers):
        cautions.append("Weakest GPU tier among the compared options")
    if _to_float(candidate.get("rating_value")) == min(ratings):
        cautions.append("Softest review rating signal in this comparison")
    if _to_float(candidate.get("display_size"), 15.6) == max(display_sizes):
        cautions.append("Largest chassis, so likely less portable")

    return cautions[:3]


def _dimension_reason(metric: str) -> str:
    """Return a concise explanation for a comparison dimension."""
    reasons = {
        "performance": "weighted from GPU and CPU tiers for gaming workloads",
        "value": "performance relative to current listed price",
        "reviews": "rating signal plus review volume confidence",
        "portability": "smaller display size as a proxy for easier carry",
    }
    return reasons[metric]


def build_laptop_comparison(
    candidates: list[dict[str, Any]],
    *,
    query: str | None = None,
    use_case: str | None = None,
) -> dict[str, Any] | None:
    """Create a deterministic comparison report for 2-3 laptops."""
    if len(candidates) < 2:
        return None

    trimmed = candidates[:3]
    mode = _comparison_mode(query, use_case)
    weights = _weights_for_mode(mode)

    raw_metrics = [_candidate_metrics(candidate) for candidate in trimmed]
    normalized_metrics: dict[str, list[float]] = {}
    for metric in ("performance", "value", "reviews", "portability"):
        values = [metrics[metric] for metrics in raw_metrics]
        reverse = False
        normalized_metrics[metric] = _normalize(values, reverse=reverse)

    enriched: list[dict[str, Any]] = []
    for index, candidate in enumerate(trimmed):
        scores = {
            metric: round(normalized_metrics[metric][index], 3)
            for metric in normalized_metrics
        }
        overall = round(sum(scores[name] * weights[name] for name in weights), 3)
        enriched.append(
            {
                **candidate,
                "scores": scores,
                "overall_score": overall,
                "strengths": _strengths(candidate, trimmed),
                "cautions": _cautions(candidate, trimmed),
            }
        )

    ranked = sorted(enriched, key=lambda item: item["overall_score"], reverse=True)
    recommended = ranked[0]
    runner_up = ranked[1]

    top_metric_names = sorted(
        recommended["scores"],
        key=recommended["scores"].get,
        reverse=True,
    )[:2]
    reasons = [
        f"Best {metric} score in this comparison set"
        for metric in top_metric_names
        if recommended["scores"][metric] >= runner_up["scores"][metric]
    ]
    if not reasons:
        reasons = recommended["strengths"][:2]

    tradeoffs: list[str] = []
    price_gap = round(_to_float(recommended.get("price")) - _to_float(runner_up.get("price")), 2)
    if price_gap > 0:
        tradeoffs.append(f"Costs ${price_gap:.0f} more than {runner_up['name']}")
    if recommended["scores"]["portability"] < runner_up["scores"]["portability"]:
        tradeoffs.append(f"Less portable than {runner_up['name']}")
    if recommended["scores"]["value"] < runner_up["scores"]["value"]:
        tradeoffs.append(f"Runner-up offers a stronger value-per-dollar signal")

    dimensions = []
    for metric in ("performance", "value", "reviews", "portability"):
        winner = max(enriched, key=lambda item: item["scores"][metric])
        dimensions.append(
            {
                "name": metric.title(),
                "winner": winner["name"],
                "reason": _dimension_reason(metric),
            }
        )

    return {
        "mode": mode,
        "recommended": {
            "name": recommended["name"],
            "price": recommended["price"],
            "summary": (
                f"{recommended['name']} is the best overall fit for this {mode} comparison "
                f"because it leads on {top_metric_names[0]} and stays competitive on {top_metric_names[1]}."
            ),
            "reasons": reasons[:3],
            "tradeoffs": tradeoffs[:3],
        },
        "candidates": ranked,
        "dimensions": dimensions,
    }
