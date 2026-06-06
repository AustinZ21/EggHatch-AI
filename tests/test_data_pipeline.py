from app.agents.data_pipeline import DataPipeline


def test_data_pipeline_loads_fixture_data():
    pipeline = DataPipeline()

    laptops = pipeline.get_laptop_data()
    reviews = pipeline.get_processed_reviews()

    assert laptops is not None
    assert len(laptops) > 0
    assert len(reviews) > 0
    assert {"laptop_name", "price", "brand_name"}.issubset(laptops.columns)


def test_data_pipeline_filters_reviews_by_query():
    pipeline = DataPipeline()

    gaming_reviews = pipeline.filter_reviews_by_query("gaming")

    assert gaming_reviews
    assert all("gaming" in review["text"].lower() for review in gaming_reviews)


def test_preprocess_data_adds_recommendation_features():
    pipeline = DataPipeline()

    processed = pipeline.preprocess_data()

    assert "rating_value" in processed.columns
    assert "display_size" in processed.columns
    assert "processor_tier" in processed.columns
    assert "gpu_tier" in processed.columns
