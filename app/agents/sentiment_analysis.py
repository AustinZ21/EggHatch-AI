"""
Sentiment Analysis Module for EggHatch AI.

This module specializes in sentiment classification for tech product reviews.
It uses pre-trained DistilBERT for sentiment analysis with a fallback mechanism.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import re
import numpy as np

# For Sentiment Analysis
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Import data pipeline
from app.agents.data_pipeline import get_data_pipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("sentiment_analysis")

# Model constants
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"  # Pre-trained sentiment model

class SentimentAnalyzer:
    """Analyzes sentiment in tech product reviews."""
    
    def __init__(self):
        self.sentiment_analyzer = None
        self.data_pipeline = None
        
        # Initialize the sentiment analyzer
        self._initialize_sentiment_analyzer()
        
        # Initialize data pipeline
        try:
            self.data_pipeline = get_data_pipeline()
            logger.info("Data pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing data pipeline: {str(e)}")
    
    def _initialize_sentiment_analyzer(self):
        """Initialize the sentiment analysis model."""
        try:
            logger.info("Initializing sentiment analysis model...")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=SENTIMENT_MODEL,
                return_all_scores=True
            )
            logger.info("Sentiment analysis model initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzer: {str(e)}")
            logger.info("Using simple sentiment analyzer as fallback.")
    
    def _simple_sentiment_analyzer(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Simple rule-based sentiment analyzer as fallback.
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            List of sentiment scores for each text
        """
        results = []
        
        # Simple sentiment words
        positive_words = [
            'good', 'great', 'excellent', 'amazing', 'awesome', 'fantastic',
            'wonderful', 'best', 'love', 'perfect', 'recommend', 'impressive',
            'satisfied', 'happy', 'fast', 'powerful', 'solid', 'reliable'
        ]
        
        negative_words = [
            'bad', 'poor', 'terrible', 'awful', 'horrible', 'worst',
            'disappointing', 'disappointed', 'slow', 'issue', 'problem',
            'fail', 'broken', 'waste', 'expensive', 'overpriced', 'hot',
            'loud', 'noisy', 'cheap', 'uncomfortable', 'avoid'
        ]
        
        for text in texts:
            text_lower = text.lower()
            
            # Count positive and negative words
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            # Calculate sentiment score (simple ratio)
            total = pos_count + neg_count
            if total == 0:
                score = 0.5  # Neutral
            else:
                score = pos_count / total
            
            # Create result in the same format as the transformer pipeline
            if score > 0.6:
                label = "POSITIVE"
            elif score < 0.4:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"
            
            results.append([
                {"label": "NEGATIVE", "score": 1.0 - score},
                {"label": "POSITIVE", "score": score}
            ])
        
        return results
    
    def analyze_sentiment(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for a list of texts.
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            List of sentiment results with labels and scores
        """
        results = []
        
        try:
            # Use transformer-based sentiment analyzer if available
            if self.sentiment_analyzer:
                # Process in batches to avoid memory issues
                batch_size = 8
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_results = self.sentiment_analyzer(batch_texts)
                    results.extend(batch_results)
            else:
                # Use simple fallback analyzer
                results = self._simple_sentiment_analyzer(texts)
            
            # Process results to get sentiment labels
            processed_results = []
            for result in results:
                # Find the label with the highest score
                if isinstance(result, list):
                    # Get the highest scoring label
                    max_score_item = max(result, key=lambda x: x['score'])
                    sentiment = max_score_item['label']
                    score = max_score_item['score']
                else:
                    sentiment = result['label']
                    score = result['score']
                
                processed_results.append({
                    'sentiment': sentiment,
                    'score': score,
                    'label': 'positive' if sentiment == 'POSITIVE' else 'negative' if sentiment == 'NEGATIVE' else 'neutral'
                })
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            # Use simple fallback analyzer in case of error
            return self._simple_sentiment_analyzer(texts)
    
    def get_sentiment_overview(self, reviews: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get overall sentiment overview for all reviews.
        
        Args:
            reviews: List of processed reviews with text and ratings. If None, fetches from data pipeline.
            
        Returns:
            Dictionary with sentiment distribution and statistics
        """
        
        # If no reviews provided, fetch from data pipeline
        if reviews is None:
            if self.data_pipeline:
                reviews = self.data_pipeline.get_processed_reviews()
                logger.info(f"Fetched {len(reviews)} reviews from data pipeline.")
            else:
                logger.error("No reviews provided and data pipeline not available.")
                return {
                    'error': 'No reviews available',
                    'overall_sentiment': 'Unknown'
                }
        try:
            # Extract review texts
            review_texts = [review['text'] for review in reviews if 'text' in review and review['text']]
            
            # Get sentiment for each review
            sentiments = self.analyze_sentiment(review_texts)
            
            # Count sentiment distribution
            sentiment_counts = {
                'positive': sum(1 for s in sentiments if s['label'] == 'positive'),
                'neutral': sum(1 for s in sentiments if s['label'] == 'neutral'),
                'negative': sum(1 for s in sentiments if s['label'] == 'negative')
            }
            
            # Calculate percentages
            total = len(sentiments)
            sentiment_percentages = {
                'positive': round(sentiment_counts['positive'] / total * 100, 1) if total > 0 else 0,
                'neutral': round(sentiment_counts['neutral'] / total * 100, 1) if total > 0 else 0,
                'negative': round(sentiment_counts['negative'] / total * 100, 1) if total > 0 else 0
            }
            
            # Calculate average rating if available
            ratings = [review.get('rating', None) for review in reviews]
            ratings = [r for r in ratings if r is not None]
            average_rating = round(sum(ratings) / len(ratings), 1) if ratings else None
            
            # Determine overall sentiment
            if sentiment_percentages['positive'] > 60:
                overall_sentiment = "Very Positive"
            elif sentiment_percentages['positive'] > 40:
                overall_sentiment = "Somewhat Positive"
            elif sentiment_percentages['negative'] > 60:
                overall_sentiment = "Very Negative"
            elif sentiment_percentages['negative'] > 40:
                overall_sentiment = "Somewhat Negative"
            else:
                overall_sentiment = "Mixed or Neutral"
            
            return {
                'sentiment_distribution': sentiment_counts,
                'sentiment_percentages': sentiment_percentages,
                'overall_sentiment': overall_sentiment,
                'average_rating': average_rating,
                'total_reviews': total
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment overview: {str(e)}")
            return {
                'error': str(e),
                'overall_sentiment': 'Unknown'
            }

# Function to get sentiment analyzer instance
def get_sentiment_analyzer():
    """Get a sentiment analyzer instance."""
    return SentimentAnalyzer()

# For testing
if __name__ == "__main__":
    analyzer = get_sentiment_analyzer()
    
    # Test with sample texts
    print("\nTesting with sample texts:")
    test_texts = [
        "This laptop is amazing! Great performance and beautiful display.",
        "Terrible battery life and it runs very hot. Would not recommend.",
        "It's okay for the price, but nothing special."
    ]
    results = analyzer.analyze_sentiment(test_texts)
    for i, result in enumerate(results):
        print(f"Text {i+1}: {result['label']} (Score: {result['score']:.2f})")
    
    # Test with data pipeline
    print("\nTesting with data pipeline:")
    try:
        overview = analyzer.get_sentiment_overview()
        print(f"Overall sentiment: {overview['overall_sentiment']}")
        print(f"Sentiment distribution: {overview['sentiment_distribution']}")
        print(f"Total reviews analyzed: {overview['total_reviews']}")
    except Exception as e:
        print(f"Error testing with data pipeline: {e}")

