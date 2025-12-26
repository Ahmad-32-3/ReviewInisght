"""
Script to help download or create sample Amazon reviews data.
Since the original amazon_us_reviews dataset uses deprecated format,
this script provides alternatives.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_sample_data(n_samples=10000, output_path="data/raw/amazon_reviews.csv"):
    """
    Create a realistic sample dataset for testing.
    This simulates real Amazon review data structure.
    """
    print(f"Creating sample dataset with {n_samples} reviews...")
    
    np.random.seed(42)
    
    # Realistic star rating distribution (Amazon typically has more positive reviews)
    # 5 stars: 40%, 4 stars: 25%, 3 stars: 15%, 2 stars: 10%, 1 star: 10%
    rating_probs = [0.10, 0.10, 0.15, 0.25, 0.40]  # 1-5 stars
    star_ratings = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=rating_probs)
    
    # Sample review texts (realistic patterns)
    positive_phrases = [
        "great product", "highly recommend", "excellent quality", "very satisfied",
        "exactly as described", "fast shipping", "love it", "works perfectly",
        "best purchase", "amazing value", "exceeded expectations", "top quality"
    ]
    
    negative_phrases = [
        "poor quality", "not as described", "disappointed", "waste of money",
        "broke quickly", "doesn't work", "terrible", "returned it",
        "cheap material", "not worth it", "defective", "very disappointed"
    ]
    
    neutral_phrases = [
        "okay product", "average quality", "does the job", "nothing special",
        "as expected", "decent", "fine for the price", "mediocre"
    ]
    
    # Generate review texts based on rating
    review_texts = []
    categories = ["Electronics", "Books", "Clothing", "Home & Kitchen", "Sports", "Toys", "Beauty"]
    
    for rating in star_ratings:
        if rating >= 4:
            phrases = np.random.choice(positive_phrases, size=np.random.randint(2, 5))
            text = " ".join(phrases) + ". " + "I would definitely buy this again."
        elif rating <= 2:
            phrases = np.random.choice(negative_phrases, size=np.random.randint(2, 5))
            text = " ".join(phrases) + ". " + "I would not recommend this product."
        else:
            phrases = np.random.choice(neutral_phrases, size=np.random.randint(1, 3))
            text = " ".join(phrases) + ". " + "It's okay but could be better."
        
        # Add some variation
        text += " " * np.random.randint(0, 3)  # Random spacing
        review_texts.append(text)
    
    # Create DataFrame
    dates = pd.date_range(start='2018-01-01', end='2024-12-31', periods=n_samples)
    dates = np.random.choice(dates, size=n_samples, replace=True)
    
    df = pd.DataFrame({
        'review_body': review_texts,
        'star_rating': star_ratings,
        'product_category': np.random.choice(categories, size=n_samples),
        'review_date': dates
    })
    
    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Sample dataset created: {output_path}")
    print(f"Star rating distribution:")
    print(df['star_rating'].value_counts().sort_index())
    
    return df

if __name__ == "__main__":
    create_sample_data(n_samples=50000)

