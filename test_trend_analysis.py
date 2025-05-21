"""
Test script for the trend analysis and sentiment analysis agents.
"""

import json
from app.agents.trend_analysis import TrendAnalysisAgent

def main():
    # Create a test query
    test_query = "What's the best gaming laptop under $1500?"
    
    print(f"Testing trend analysis with query: '{test_query}'")
    
    # Initialize the trend analysis agent
    agent = TrendAnalysisAgent()
    
    # Run the analysis
    results = agent.analyze_trends(test_query)
    
    # Print the results in a readable format
    print("\n=== TREND ANALYSIS RESULTS ===\n")
    
    # Print topics
    print("TOPICS:")
    for topic in results.get('topics', []):
        print(f"  - {topic['name']} (Sentiment: {topic['sentiment']})")
        print(f"    Keywords: {', '.join(topic['keywords'][:5])}")
    print()
    
    # Print popular features
    print("POPULAR FEATURES:")
    for feature in results.get('popular_features', [])[:5]:
        if 'category' in feature:
            print(f"  - {feature['category']} (Mentions: {feature.get('mentions', 0)}, Sentiment: {feature.get('sentiment', 'neutral')})")
    print()
    
    # Print sentiment overview
    print("SENTIMENT OVERVIEW:")
    sentiment = results.get('sentiment_overview', {})
    print(f"  - Average Rating: {sentiment.get('average_rating', 'N/A')}")
    print(f"  - Positive Reviews: {sentiment.get('positive_percentage', 'N/A')}%")
    print(f"  - Negative Reviews: {sentiment.get('negative_percentage', 'N/A')}%")
    print()
    
    # Print recommendations
    print("RECOMMENDATIONS:")
    for rec in results.get('recommendations', []):
        print(f"  - {rec}")
    print()
    
    # Print top laptop recommendations
    print("TOP LAPTOP RECOMMENDATIONS:")
    for laptop in results.get('top_laptops', [])[:3]:
        print(f"  - {laptop['name']}")
        print(f"    Price: ${laptop['price']}")
        print(f"    Brand: {laptop['brand']}")
        print(f"    Specs: {laptop['processor']}, {laptop['gpu']}")
        print(f"    Rating: {laptop['rating']}")
        print(f"    Key Features: {', '.join([f['name'] for f in laptop.get('key_features', [])])}")
        print()
    
    # Save the full results to a JSON file for further inspection
    with open('trend_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Full results saved to 'trend_analysis_results.json'")

if __name__ == "__main__":
    main()
