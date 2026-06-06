from app.agents.comparison import build_laptop_comparison, looks_like_comparison_query


def test_detects_explicit_comparison_queries():
    assert looks_like_comparison_query("Compare these gaming laptops under $2000")
    assert looks_like_comparison_query("RTX 4070 laptop vs RTX 4060 laptop")
    assert looks_like_comparison_query("Which is better between these gaming laptops?")
    assert not looks_like_comparison_query("I want a gaming laptop under $2000")
    assert not looks_like_comparison_query("Best gaming laptop between $1000 and $1500")


def test_builds_explainable_comparison_report():
    candidates = [
        {
            "name": "Laptop A",
            "price": 1799.0,
            "processor": "i9",
            "gpu": "RTX 4070",
            "rating_value": 4.6,
            "review_count": 220,
            "display_size": 15.6,
            "processor_tier": 4,
            "gpu_tier": 3,
            "total_storage": 1024,
        },
        {
            "name": "Laptop B",
            "price": 1499.0,
            "processor": "i7",
            "gpu": "RTX 4060",
            "rating_value": 4.3,
            "review_count": 190,
            "display_size": 16.0,
            "processor_tier": 3,
            "gpu_tier": 2,
            "total_storage": 1024,
        },
        {
            "name": "Laptop C",
            "price": 1399.0,
            "processor": "i7",
            "gpu": "RTX 4050",
            "rating_value": 4.1,
            "review_count": 140,
            "display_size": 14.0,
            "processor_tier": 3,
            "gpu_tier": 1,
            "total_storage": 512,
        },
    ]

    comparison = build_laptop_comparison(
        candidates,
        query="Compare gaming laptops for competitive FPS under $2000",
    )

    assert comparison is not None
    assert comparison["mode"] == "performance"
    assert comparison["recommended"]["name"] == "Laptop A"
    assert len(comparison["candidates"]) == 3
    assert any(dimension["name"] == "Performance" for dimension in comparison["dimensions"])
    assert comparison["recommended"]["reasons"]


def test_budget_filter_keeps_engineered_candidate_fields():
    candidates = [
        {
            "name": "Budget Winner",
            "price": 1299.0,
            "processor": "i7",
            "gpu": "RTX 4060",
            "rating_value": 4.5,
            "review_count": 260,
            "display_size": 15.6,
            "processor_tier": 3,
            "gpu_tier": 2,
            "total_storage": 1024,
        },
        {
            "name": "Premium Pick",
            "price": 1899.0,
            "processor": "i9",
            "gpu": "RTX 4070",
            "rating_value": 4.7,
            "review_count": 210,
            "display_size": 16.0,
            "processor_tier": 4,
            "gpu_tier": 3,
            "total_storage": 1024,
        },
    ]

    comparison = build_laptop_comparison(
        candidates,
        query="Compare the best gaming laptops under $1500",
    )

    assert comparison is not None
    assert comparison["recommended"]["name"] == "Budget Winner"
    assert comparison["candidates"][0]["scores"]["reviews"] > 0
    assert "performance" in comparison["candidates"][0]["scores"]
    assert comparison["candidates"][0]["overall_score"] > comparison["candidates"][1]["overall_score"]
