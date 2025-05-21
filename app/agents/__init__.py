"""EggHatch AI Agents Package.

This package contains specialized agents for the EggHatch AI system.

- data_pipeline: Data loading and preprocessing module
- trend_analysis: Topic modeling and feature identification module
- sentiment_analysis: Sentiment classification module
- product_knowledge: Product specifications and benchmarks module (future development)
- pricing_availability: Pricing and availability module (future development)
- build_recommendation: PC build recommendation module (future development)
"""

# Import the agent functions to make them available from the package
from app.agents.trend_analysis import TrendAnalysisAgent
from app.agents.data_pipeline import get_data_pipeline
from app.agents.sentiment_analysis import get_sentiment_analyzer

# These are planned for future development
# from app.agents.product_knowledge import product_knowledge_agent
# from app.agents.pricing_availability import pricing_availability_agent
# from app.agents.build_recommendation import build_recommendation_agent
