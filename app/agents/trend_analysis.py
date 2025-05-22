"""
Trend Analysis Module for EggHatch AI.

This module focuses on topic modeling and feature identification in tech product reviews.
It uses Latent Dirichlet Allocation (LDA) for topic modeling and zero-shot classification
for feature identification.
"""

import json
import numpy as np
import re
from typing import Dict, List, Any, Optional, Tuple
import logging

# For LDA Topic Modeling
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# For NLP tasks
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Import from other modules
from app.agents.data_pipeline import get_data_pipeline
from app.agents.sentiment_analysis import get_sentiment_analyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("trend_analysis")

# Model constants
NUM_TOPICS = 5  # Number of topics for LDA
NUM_WORDS_PER_TOPIC = 10  # Number of words to display per topic
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence embedding model for semantic similarity

class TrendAnalysisAgent:
    """Agent for analyzing trends and feature popularity in tech product reviews."""
    
    def __init__(self):
        self.embedding_model = None
        self.zero_shot_classifier = None
        self.lda_model = None
        self.vectorizer = None
        self.topic_words = []
        self.topic_sentiments = {}
        self.data_pipeline = None
        self.sentiment_analyzer = None
        
        # Common feature categories for tech products (used as reference)
        self.feature_categories = [
            "display", "performance", "cooling", "battery", 
            "build quality", "keyboard", "audio", "value", "portability"
        ]
        
        # Initialize models and data
        self._initialize()
    
    def _initialize(self):
        """Initialize models and load data."""
        logger.info("Initializing Trend Analysis Agent...")
        
        # Get data pipeline instance
        self.data_pipeline = get_data_pipeline()
        
        # Get sentiment analyzer instance
        self.sentiment_analyzer = get_sentiment_analyzer()
        
        # Initialize NLP models
        self._initialize_nlp_models()
        
        # Initialize and train topic model
        self._initialize_topic_model()
        
        logger.info("Trend Analysis Agent initialized successfully.")
    
    def _initialize_nlp_models(self):
        """Initialize NLP models for embeddings and zero-shot classification."""
        try:
            # Initialize embedding model for semantic similarity
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            logger.info(f"Initialized embedding model using {EMBEDDING_MODEL}")
            
            # Initialize zero-shot classifier for feature categorization
            try:
                self.zero_shot_classifier = pipeline(
                    "zero-shot-classification",
                    model="valhalla/distilbart-mnli-12-1")
                logger.info("Initialized zero-shot classifier")
            except Exception as e:
                logger.error(f"Error initializing zero-shot classifier: {str(e)}")
                self.zero_shot_classifier = None
                
        except Exception as e:
            logger.error(f"Error initializing NLP models: {str(e)}")
            # Fallback to a simple rule-based analyzer if models fail to load
            self.sentiment_analyzer = self._simple_sentiment_analyzer
    
    def _simple_sentiment_analyzer(self, texts):
        """Simple rule-based sentiment analyzer as fallback."""
    
    def _initialize_topic_model(self):
        """Initialize and train the LDA topic model."""
        try:
            # Get review texts from data pipeline
            all_reviews = self.data_pipeline.get_review_texts()
            
            logger.info(f"Training topic model on {len(all_reviews)} reviews")
            
            # Create document-term matrix
            self.vectorizer = CountVectorizer(
                max_df=0.95,  # Ignore terms that appear in >95% of documents
                min_df=2,     # Ignore terms that appear in <2 documents
                stop_words='english',
                max_features=1000  # Limit vocabulary size
            )
            
            X = self.vectorizer.fit_transform(all_reviews)
            
            # Train LDA model
            self.lda_model = LatentDirichletAllocation(
                n_components=NUM_TOPICS,
                max_iter=10,
                learning_method='online',
                random_state=42,
                batch_size=128,
                evaluate_every=-1
            )
            
            self.lda_model.fit(X)
            
            # Extract topic words
            feature_names = self.vectorizer.get_feature_names_out()
            self.topic_words = []
            for topic_idx, topic in enumerate(self.lda_model.components_):
                top_words_idx = topic.argsort()[:-NUM_WORDS_PER_TOPIC-1:-1]
                top_words = [feature_names[i] for i in top_words_idx]
                self.topic_words.append(top_words)
                logger.info(f"Topic #{topic_idx}: {', '.join(top_words)}")
            
            # Analyze sentiment for each topic
            self._analyze_topic_sentiments(all_reviews, X)
            
        except Exception as e:
            logger.error(f"Error initializing topic model: {str(e)}")
    
    def _analyze_topic_sentiments(self, all_reviews, document_term_matrix):
        """Analyze sentiment for each topic."""
        try:
            # Get topic distribution for each document
            doc_topic_dist = self.lda_model.transform(document_term_matrix)
            
            # For each topic, find documents where this topic is dominant
            for topic_idx in range(NUM_TOPICS):
                # Get documents where this topic has the highest probability
                topic_docs_indices = [i for i in range(len(all_reviews)) 
                                    if np.argmax(doc_topic_dist[i]) == topic_idx]
                
                if not topic_docs_indices:
                    continue
                
                # Get the actual documents
                topic_docs = [all_reviews[i] for i in topic_docs_indices]
                
                # Sample up to 10 documents for sentiment analysis (to save resources)
                sample_size = min(10, len(topic_docs))
                sampled_docs = np.random.choice(topic_docs, size=sample_size, replace=False)
                
                # Use sentiment analyzer from the sentiment analysis module
                sentiment_results = self.sentiment_analyzer.analyze_sentiment(list(sampled_docs))
                
                # Count sentiment distribution
                positive_count = sum(1 for s in sentiment_results if s['label'] == 'positive')
                negative_count = sum(1 for s in sentiment_results if s['label'] == 'negative')
                neutral_count = sum(1 for s in sentiment_results if s['label'] == 'neutral')
                
                total = len(sentiment_results)
                if total > 0:
                    sentiment_distribution = {
                        'positive': round(positive_count / total, 2),
                        'negative': round(negative_count / total, 2),
                        'neutral': round(neutral_count / total, 2)
                    }
                    
                    # Determine overall sentiment
                    if sentiment_distribution['positive'] > 0.6:
                        overall = 'positive'
                    elif sentiment_distribution['negative'] > 0.6:
                        overall = 'negative'
                    else:
                        overall = 'mixed'
                    
                    self.topic_sentiments[topic_idx] = {
                        'distribution': sentiment_distribution,
                        'overall': overall
                    }
                    
                    logger.info(f"Topic #{topic_idx} sentiment: {overall}")
        except Exception as e:
            logger.error(f"Error analyzing topic sentiments: {str(e)}")
    
    def analyze_trends(self, query=None):
        """
        Analyze trends and sentiment in gaming laptop reviews.
        
        Args:
            query: Optional query to filter results
            
        Returns:
            Dictionary containing trend analysis results
        """
        try:
            # Get processed reviews, filtered by query if provided
            if query:
                reviews = self.data_pipeline.filter_reviews_by_query(query)
            else:
                reviews = self.data_pipeline.get_processed_reviews()
            
            # Get laptop data for specific recommendations
            laptop_data = self.data_pipeline.get_laptop_data()
            
            # Prepare results dictionary
            results = {
                'topics': [],
                'popular_features': [],
                'sentiment_overview': {},
                'recommendations': [],
                'top_laptops': []
            }
            
            # Add topic information
            for topic_idx, words in enumerate(self.topic_words):
                topic_name = self._generate_topic_name(words)
                sentiment = self.topic_sentiments.get(topic_idx, {}).get('overall', 'neutral')
                
                results['topics'].append({
                    'id': topic_idx,
                    'name': topic_name,
                    'keywords': words,
                    'sentiment': sentiment
                })
            
            # Add popular features
            results['popular_features'] = self._identify_popular_features()
            
            # Add sentiment overview using the sentiment analyzer module
            try:
                if self.sentiment_analyzer and hasattr(self.sentiment_analyzer, 'get_sentiment_overview'):
                    results['sentiment_overview'] = self.sentiment_analyzer.get_sentiment_overview(reviews)
                else:
                    logger.warning("Sentiment analyzer not available or missing get_sentiment_overview method")
                    results['sentiment_overview'] = {
                        'overall_sentiment': 'Unknown',
                        'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
                        'average_rating': None,
                        'positive_percentage': 0,
                        'negative_percentage': 0
                    }
            except Exception as e:
                logger.error(f"Error getting sentiment overview: {str(e)}")
                results['sentiment_overview'] = {
                    'error': str(e),
                    'overall_sentiment': 'Unknown'
                }
            
            # Add recommendations
            results['recommendations'] = self._generate_recommendations(query)
            
            # Add specific laptop recommendations with prices
            # Filter laptops based on query if provided
            filtered_laptops = laptop_data
            if query and 'budget' in query.lower() or 'under' in query.lower():
                # Try to extract a price limit from the query
                price_limit = None
                price_matches = re.findall(r'\$?(\d{3,4})', query)
                if price_matches:
                    price_limit = float(price_matches[0])
                    filtered_laptops = laptop_data[laptop_data['price'] <= price_limit]
            
            # Sort by rating and get top 5
            if 'rating' in filtered_laptops.columns:
                top_laptops = filtered_laptops.sort_values(by=['rating'], ascending=False).head(5)
            else:
                top_laptops = filtered_laptops.head(5)
            
            # Convert to list of dictionaries with relevant info
            for _, laptop in top_laptops.iterrows():
                laptop_info = {
                    'name': laptop.get('laptop_name', 'Unknown'),
                    'price': float(laptop.get('price', 0)),
                    'brand': laptop.get('brand_name', 'Unknown'),
                    'processor': laptop.get('processor', 'Unknown'),
                    'gpu': laptop.get('gpu', 'Unknown'),
                    'rating': laptop.get('rating', 'Unknown'),
                    'key_features': []
                }
                
                # Add key features based on popular features
                for feature in results['popular_features'][:3]:
                    if feature.get('category'):
                        laptop_info['key_features'].append({
                            'name': feature.get('category'),
                            'sentiment': feature.get('sentiment', 'neutral')
                        })
                
                results['top_laptops'].append(laptop_info)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {str(e)}")
            return {'error': str(e)}
    
    def _identify_popular_features(self):
        """Identify popular features mentioned in reviews using NLP techniques."""
        try:
            # Initialize feature data
            feature_data = {category: {'mentions': 0, 'sentiment': {}} for category in self.feature_categories}
            
            # Get review texts from data pipeline
            all_reviews = self.data_pipeline.get_review_texts()
            
            # Use zero-shot classification if available
            if self.zero_shot_classifier:
                logger.info("Using zero-shot classification for feature identification")
                
                # Sample reviews for analysis (to save resources)
                sample_size = min(100, len(all_reviews))
                sampled_reviews = np.random.choice(all_reviews, size=sample_size, replace=False)
                
                # Analyze each review
                for review in sampled_reviews:
                    if not review or len(review) < 20:
                        continue
                    
                    try:
                        # Use zero-shot classification to identify features
                        result = self.zero_shot_classifier(
                            review,
                            candidate_labels=self.feature_categories,
                            multi_label=True
                        )
                        
                        # Get features with confidence above threshold
                        for label, score in zip(result['labels'], result['scores']):
                            if score > 0.5:  # Confidence threshold
                                feature_data[label]['mentions'] += 1
                                
                                # Analyze sentiment for this feature mention using sentiment analyzer
                                sentiment_result = self.sentiment_analyzer.analyze_sentiment([review])[0]
                                sentiment = sentiment_result['label']
                                
                                # Update sentiment counts
                                if sentiment not in feature_data[label]['sentiment']:
                                    feature_data[label]['sentiment'][sentiment] = 0
                                feature_data[label]['sentiment'][sentiment] += 1
                    except Exception as e:
                        logger.error(f"Error in zero-shot classification: {str(e)}")
                
                # Sort features by mentions
                sorted_features = sorted(
                    [(k, v) for k, v in feature_data.items() if v['mentions'] > 0],
                    key=lambda x: x[1]['mentions'],
                    reverse=True
                )
                
                # Format results
                results = []
                for category, data in sorted_features:
                    # Determine dominant sentiment
                    sentiment_counts = data['sentiment']
                    if sentiment_counts:
                        max_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]
                        sentiment = max_sentiment.lower()
                    else:
                        sentiment = 'neutral'
                    
                    results.append({
                        'category': category,
                        'mentions': data['mentions'],
                        'sentiment': sentiment
                    })
                
                return results[:5]  # Return top 5 features
            else:
                # Fallback to keyword-based approach
                logger.info("Using keyword-based approach for feature identification")
                return self._identify_features_with_keywords(all_reviews, feature_data)
                
        except Exception as e:
            logger.error(f"Error identifying popular features: {str(e)}")
            return []
    
    def _identify_features_with_keywords(self, reviews, feature_data):
        """Fallback method to identify features using keyword matching."""
        # Define keywords for each feature category
        keywords = {
            'display': ['screen', 'display', 'resolution', 'panel', 'refresh rate', 'hz', 'brightness'],
            'performance': ['performance', 'fps', 'speed', 'fast', 'powerful', 'benchmark'],
            'cooling': ['cooling', 'temperature', 'hot', 'thermal', 'fan', 'heat'],
            'battery': ['battery', 'life', 'hours', 'charge', 'power'],
            'build quality': ['build', 'quality', 'sturdy', 'solid', 'plastic', 'metal', 'chassis'],
            'keyboard': ['keyboard', 'keys', 'typing', 'rgb', 'backlit'],
            'audio': ['audio', 'sound', 'speaker', 'bass', 'volume'],
            'value': ['price', 'value', 'worth', 'expensive', 'cheap', 'cost', 'money'],
            'portability': ['weight', 'portable', 'thin', 'light', 'heavy', 'thick']
        }
        
        # Get processed reviews from data pipeline
        processed_reviews = self.data_pipeline.get_processed_reviews()
        
        # Analyze each review
        for review in processed_reviews:
            review_text = review['text'].lower()
            
            # Get sentiment for this review using sentiment analyzer
            sentiment_result = self.sentiment_analyzer.analyze_sentiment([review_text])[0]
            sentiment = sentiment_result['label']
            
            # Check for keywords
            for category, category_keywords in keywords.items():
                for keyword in category_keywords:
                    if keyword in review_text:
                        feature_data[category]['mentions'] += 1
                        
                        # Update sentiment counts
                        if sentiment not in feature_data[category]['sentiment']:
                            feature_data[category]['sentiment'][sentiment] = 0
                        feature_data[category]['sentiment'][sentiment] += 1
                        break  # Count each category only once per review
        
        # Format results
        results = []
        for category, data in feature_data.items():
            if data['mentions'] > 0:
                # Determine dominant sentiment
                sentiment_counts = data['sentiment']
                if sentiment_counts:
                    max_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]
                    sentiment = max_sentiment.lower()
                else:
                    sentiment = 'neutral'
                
                results.append({
                    'category': category,
                    'mentions': data['mentions'],
                    'sentiment': sentiment
                })
        
        # Sort by mentions and return top 5
        results.sort(key=lambda x: x['mentions'], reverse=True)
        return results[:5]
    
    def _generate_topic_name(self, topic_words):
        """Generate a descriptive name for a topic based on its keywords using NLP."""
        if self.embedding_model is not None:
            try:
                # Convert topic words to a single string
                topic_text = ' '.join(topic_words)
                
                # Define candidate topic names
                candidate_names = [
                    "Gaming Performance", "Display Quality", "Build Construction",
                    "Thermal Management", "Battery Life", "Value for Money",
                    "Keyboard Experience", "Audio Quality", "Portability",
                    "Software Experience", "Connectivity Options", "Storage Performance"
                ]
                
                # Get embeddings
                topic_embedding = self.embedding_model.encode([topic_text])[0]
                candidate_embeddings = self.embedding_model.encode(candidate_names)
                
                # Calculate similarities
                similarities = cosine_similarity([topic_embedding], candidate_embeddings)[0]
                
                # Get the most similar candidate
                best_match_idx = similarities.argmax()
                best_match_score = similarities[best_match_idx]
                
                # If similarity is above threshold, use the candidate name
                if best_match_score > 0.3:  # Threshold for similarity
                    return f"{candidate_names[best_match_idx]} Discussion"
                    
            except Exception as e:
                logger.error(f"Error generating topic name with embeddings: {str(e)}")
        
        # If embedding approach fails or isn't available, use zero-shot classification
        if self.zero_shot_classifier is not None:
            try:
                # Convert topic words to a single string
                topic_text = ' '.join(topic_words)
                
                # Use zero-shot classification to categorize the topic
                result = self.zero_shot_classifier(
                    topic_text,
                    candidate_labels=self.feature_categories,
                    multi_label=False
                )
                
                # Get the top label
                top_label = result['labels'][0]
                top_score = result['scores'][0]
                
                # If confidence is high enough, use this label
                if top_score > 0.5:  # Confidence threshold
                    return f"{top_label.replace('_', ' ').title()} Discussion"
                    
            except Exception as e:
                logger.error(f"Error generating topic name with zero-shot: {str(e)}")
        
        # Default to a simple approach using the top words
        return f"Topic: {', '.join(topic_words[:3])}".title()
    
    def _generate_recommendations(self, query=None):
        """Generate recommendations based on trend analysis."""
        recommendations = []
        
        # Get top positive features
        positive_features = [f for f in self._identify_popular_features() 
                           if 'positive' in f['sentiment'] and f['mentions'] > 5]
        
        if positive_features:
            top_feature = positive_features[0]['category'].lower()
            recommendations.append(f"Focus on {top_feature} in marketing materials as it's highly regarded")
        
        # Get sentiment overview
        try:
            if hasattr(self.sentiment_analyzer, 'get_sentiment_overview'):
                sentiment_overview = self.sentiment_analyzer.get_sentiment_overview()
                sentiment = {'average_rating': sentiment_overview.get('average_rating', 0)}
            else:
                # Fallback to a default value
                sentiment = {'average_rating': 3.5}
        except Exception as e:
            logger.error(f"Error getting sentiment overview in recommendations: {str(e)}")
            sentiment = {'average_rating': 3.5}
        if sentiment['average_rating'] > 4.0:
            recommendations.append("Overall sentiment is very positive - highlight customer satisfaction")
        elif sentiment['average_rating'] < 3.0:
            recommendations.append("Address customer concerns in product improvements")
        
        # Add topic-specific recommendations
        for topic_idx, words in enumerate(self.topic_words):
            topic_sentiment = self.topic_sentiments.get(topic_idx, {}).get('overall', 'neutral')
            topic_name = self._generate_topic_name(words)
            
            if 'performance' in topic_name.lower() and topic_sentiment == 'positive':
                recommendations.append("Gaming performance is viewed positively - emphasize this in marketing")
            elif 'battery' in topic_name.lower() and topic_sentiment == 'negative':
                recommendations.append("Battery life concerns should be addressed in future models")
            elif 'cooling' in topic_name.lower() and topic_sentiment == 'negative':
                recommendations.append("Thermal performance is a concern - consider improved cooling solutions")
        
        # Filter by query if provided
        if query:
            query_lower = query.lower()
            recommendations = [r for r in recommendations if query_lower in r.lower()]
        
        return recommendations[:5]  # Limit to top 5 recommendations

# Function to run the agent and get results
def trend_analysis_agent(state):
    """
    Run the trend analysis agent and update the state.
    
    Args:
        state: Current state object
        
    Returns:
        Updated state with trend insights
    """
    logger.info("Running Trend Analysis Agent...")
    
    # Initialize the agent
    agent = TrendAnalysisAgent()
    
    # Extract query from state if available
    query = None
    if hasattr(state, 'query') and state.query:
        query = state.query
    
    # Run analysis
    trend_insights = agent.analyze_trends(query)
    
    # Update state
    state.trend_insights = trend_insights
    logger.info("Trend Analysis Agent completed")
    
    return state

# For testing
if __name__ == "__main__":
    # Create a simple state object for testing
    class State:
        def __init__(self):
            self.query = "gaming performance"
            self.trend_insights = None
    
    # Run the agent
    state = State()
    updated_state = trend_analysis_agent(state)
    
    # Print results
    print(json.dumps(updated_state.trend_insights, indent=2))
