"""Specialized agent modules for EggHatch-AI.

Modules in this package may load data or ML models when imported directly.
The package initializer intentionally stays lightweight so test discovery and
documentation tooling do not trigger expensive model initialization.
"""

__all__ = [
    "data_pipeline",
    "trend_analysis",
    "sentiment_analysis",
    "product_knowledge",
    "pricing_availability",
    "build_recommendation",
]
