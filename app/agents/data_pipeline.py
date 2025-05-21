"""
Data Pipeline Module for EggHatch AI.

This module handles data loading, preprocessing, and cleaning for analysis.
It provides standardized data structures for trend and sentiment analysis.
"""

import os
import json
import pandas as pd
import numpy as np
import re
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("data_pipeline")

# Constants
DATA_DIR = Path("data")
CSV_DIR = DATA_DIR / "csv"
REVIEWS_DIR = DATA_DIR / "reviews"
LAPTOPS_CSV = CSV_DIR / "gaming_laptops_with_reviews.csv"

class DataPipeline:
    """Handles data loading and preprocessing for analysis."""
    
    def __init__(self):
        self.laptop_data = None
        self.reviews_data = {}
        self.processed_reviews = []
        self.review_texts = []
        self.preprocessed_data = None
        self.numerical_features = ['price', 'num_to_rate', 'touch']
        self.categorical_features = ['brand_name', 'processor', 'gpu']
        self.text_features = ['laptop_name']
        
        # Preprocessing tools
        self.scaler = StandardScaler()
        self.encoders = {}
        
        # Load the data
        self._load_data()
        
    def _load_data(self):
        """Load laptop and review data."""
        try:
            # Load laptop data
            self.laptop_data = pd.read_csv(LAPTOPS_CSV)
            logger.info(f"Loaded {len(self.laptop_data)} laptops from {LAPTOPS_CSV}")
            
            # Load reviews for each laptop
            for _, row in self.laptop_data.iterrows():
                if pd.notna(row.get('reviews')):
                    review_path = row['reviews']
                    try:
                        with open(review_path, 'r', encoding='utf-8') as f:
                            self.reviews_data[review_path] = json.load(f)
                    except Exception as e:
                        logger.error(f"Error loading reviews from {review_path}: {str(e)}")
            
            logger.info(f"Loaded reviews for {len(self.reviews_data)} laptops")
            
            # Process the reviews
            self._process_reviews()
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
    
    def _process_reviews(self):
        """Process and clean review data."""
        try:
            # Extract all reviews
            for laptop_id, reviews in self.reviews_data.items():
                for review in reviews:
                    # Clean and process the review text
                    if 'text' in review and review['text']:
                        clean_text = self._clean_text(review['text'])
                        
                        # Add to processed reviews with metadata
                        self.processed_reviews.append({
                            'laptop_id': laptop_id,
                            'text': clean_text,
                            'rating': review.get('rating', None),
                            'date': review.get('date', None),
                            'title': review.get('title', None)
                        })
                        
                        # Add to review texts list (for topic modeling)
                        if len(clean_text) > 20:  # Only include substantive reviews
                            self.review_texts.append(clean_text)
            
            logger.info(f"Processed {len(self.processed_reviews)} reviews")
            
        except Exception as e:
            logger.error(f"Error processing reviews: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Convert to string if not already
        if not isinstance(text, str):
            text = str(text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Convert to lowercase for consistency
        text = text.lower()
        
        # Remove special characters if needed
        # text = re.sub(r'[^\w\s]', '', text)
        
        return text
    
    def get_review_texts(self) -> List[str]:
        """Get cleaned review texts for topic modeling."""
        return self.review_texts
    
    def get_processed_reviews(self) -> List[Dict[str, Any]]:
        """Get processed reviews with metadata."""
        return self.processed_reviews
    
    def get_laptop_data(self) -> pd.DataFrame:
        """Get laptop data as DataFrame."""
        return self.laptop_data
    
    def filter_reviews_by_query(self, query: str) -> List[Dict[str, Any]]:
        """Filter reviews by a search query."""
        if not query:
            return self.processed_reviews
        
        query_lower = query.lower()
        return [
            review for review in self.processed_reviews
            if query_lower in review['text'].lower()
        ]
    
    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess laptop data for modeling."""
        if self.laptop_data is None:
            logger.error("No laptop data available for preprocessing")
            return pd.DataFrame()
        
        # Create a copy to avoid modifying original data
        df = self.laptop_data.copy()
        
        # Clean and preprocess features
        try:
            # Extract numerical values from rating (e.g., "Rating + 4" -> 4)
            df['rating_value'] = df['rating'].str.extract(r'Rating \+ (\d+)').astype(float)
            
            # Extract display size as float (e.g., "15.6\"" -> 15.6)
            df['display_size'] = df['display'].str.extract(r'(\d+\.?\d*)').astype(float)
            
            # Extract storage sizes in GB
            df['hdd_size'] = df['hdd'].str.extract(r'(\d+)').astype(float)
            df['ssd_size'] = df['ssd'].str.extract(r'(\d+)').astype(float)
            
            # Fill missing values
            df['hdd_size'] = df['hdd_size'].fillna(0)
            df['ssd_size'] = df['ssd_size'].fillna(0)
            df['total_storage'] = df['hdd_size'] + df['ssd_size']
            
            # Create processor tier feature (higher is better)
            processor_tiers = {'i3': 1, 'i5': 2, 'i7': 3, 'i9': 4}
            df['processor_tier'] = df['processor'].map(processor_tiers).fillna(0).astype(int)
            
            # Create GPU tier feature based on model series
            df['gpu_tier'] = 0
            for idx, gpu in enumerate(df['gpu']):
                if pd.isna(gpu):
                    continue
                if 'RTX' in gpu:
                    df.loc[idx, 'gpu_tier'] = 3
                elif 'GTX 10' in gpu:
                    df.loc[idx, 'gpu_tier'] = 2
                elif 'GTX' in gpu:
                    df.loc[idx, 'gpu_tier'] = 1
            
            # Store preprocessed data
            self.preprocessed_data = df
            logger.info("Data preprocessing completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            return df
    
    def get_feature_vectors(self, normalize: bool = True) -> pd.DataFrame:
        """Get feature vectors ready for modeling."""
        if self.preprocessed_data is None:
            self.preprocess_data()
        
        if self.preprocessed_data is None or self.preprocessed_data.empty:
            logger.error("No preprocessed data available")
            return pd.DataFrame()
        
        try:
            # Select features for modeling
            features = [
                'price', 'rating_value', 'num_to_rate', 'touch', 
                'display_size', 'processor_tier', 'gpu_tier',
                'total_storage'
            ]
            
            # Create feature dataframe
            X = self.preprocessed_data[features].copy()
            
            # Handle missing values
            X = X.fillna(0)
            
            # Normalize if requested
            if normalize:
                X[['price', 'num_to_rate', 'display_size', 'total_storage']] = \
                    self.scaler.fit_transform(X[['price', 'num_to_rate', 'display_size', 'total_storage']])
            
            logger.info(f"Generated feature vectors with {len(features)} features")
            return X
            
        except Exception as e:
            logger.error(f"Error generating feature vectors: {str(e)}")
            return pd.DataFrame()
    
    def get_laptop_by_name(self, name_fragment: str) -> Optional[Dict[str, Any]]:
        """Find a laptop by a fragment of its name."""
        if self.laptop_data is None or self.laptop_data.empty:
            return None
        
        name_fragment = name_fragment.lower()
        matches = self.laptop_data[self.laptop_data['laptop_name'].str.lower().str.contains(name_fragment)]
        
        if matches.empty:
            return None
        
        # Return the first match as a dictionary
        return matches.iloc[0].to_dict()
    
    def get_laptops_by_filter(self, 
                             min_price: Optional[float] = None,
                             max_price: Optional[float] = None,
                             brand: Optional[str] = None,
                             processor: Optional[str] = None,
                             min_rating: Optional[float] = None) -> pd.DataFrame:
        """Filter laptops by various criteria."""
        if self.preprocessed_data is None:
            self.preprocess_data()
            
        if self.preprocessed_data is None or self.preprocessed_data.empty:
            return pd.DataFrame()
        
        df = self.preprocessed_data
        
        # Apply filters
        if min_price is not None:
            df = df[df['price'] >= min_price]
            
        if max_price is not None:
            df = df[df['price'] <= max_price]
            
        if brand is not None:
            df = df[df['brand_name'].str.lower() == brand.lower()]
            
        if processor is not None:
            df = df[df['processor'].str.lower() == processor.lower()]
            
        if min_rating is not None:
            df = df[df['rating_value'] >= min_rating]
            
        return df

# Function to get data pipeline instance
def get_data_pipeline():
    """Get a data pipeline instance."""
    return DataPipeline()

# For testing
if __name__ == "__main__":
    pipeline = get_data_pipeline()
    print(f"Loaded {len(pipeline.get_processed_reviews())} reviews")
    print(f"First review sample: {pipeline.get_processed_reviews()[0]['text'][:100]}...")
   