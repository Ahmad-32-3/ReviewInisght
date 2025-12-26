"""
Data preprocessing module for ReviewInsight project.
Handles data ingestion, cleaning, and text preprocessing.
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
import re
import spacy
from pathlib import Path
import pickle
import json


def load_amazon_reviews(n_samples=200000, random_state=42):
    """
    Load Amazon US reviews dataset from Hugging Face and sample reviews.
    
    Parameters:
    -----------
    n_samples : int
        Number of reviews to sample (between 100k-300k)
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: review_text, star_rating, product_category, review_date
    """
    print(f"Loading Amazon US reviews dataset...")
        
    # Load the real Amazon US Reviews dataset
    # Try the specified dataset path first
    dataset = None
    dataset_name = None
    
    # Try to find a working Amazon reviews dataset
    # Note: Many Amazon review datasets use deprecated script format
    # We'll try multiple approaches to get real data with all star ratings (1-5)
    dataset_options = [
        # Try loading from CSV if user has downloaded it
        ("csv", "data/raw/amazon_reviews.csv"),  # Local CSV file
    ]
    
    # Note: The original amazon_us_reviews dataset uses deprecated format
    # Users should download data manually or use the sample data generator
    
    for ds_option in dataset_options:
        try:
            if isinstance(ds_option, tuple) and ds_option[0] == "csv":
                # Try loading from local CSV file
                csv_path = Path(ds_option[1])
                if csv_path.exists():
                    df = pd.read_csv(csv_path, nrows=None)  # Load all rows
                    # Convert to dataset format for consistency
                    from datasets import Dataset
                    dataset = Dataset.from_pandas(df)
                    dataset_name = "local_csv"
                    print(f"Loaded dataset from CSV: {csv_path} (size: {len(dataset)})")
                    break
                else:
                    continue
            elif isinstance(ds_option, tuple):
                # Load from Hugging Face repository
                if ds_option[1]:
                    dataset = load_dataset(ds_option[0], ds_option[1], split="train")
                    dataset_name = f"{ds_option[0]}/{ds_option[1]}"
                else:
                    dataset = load_dataset(ds_option[0], split="train")
                    dataset_name = ds_option[0]
            else:
                # Load by direct name
                dataset = load_dataset(ds_option, split="train")
                dataset_name = ds_option
            
            print(f"Loaded dataset: {dataset_name} (size: {len(dataset)})")
            break
        except Exception as e:
            print(f"Warning: Could not load {ds_option}: {e}")
            continue
    
    if dataset is None:
        error_msg = (
            f"Failed to load Amazon US Reviews dataset.\n\n"
            f"Tried: {dataset_options}\n\n"
            f"SOLUTION: The original amazon_us_reviews dataset uses deprecated format.\n"
            f"Please download the dataset manually:\n"
            f"1. Visit: https://huggingface.co/datasets/amazon_us_reviews\n"
            f"2. Download the data files\n"
            f"3. Save as CSV: data/raw/amazon_reviews.csv\n"
            f"4. CSV should have columns: review_body, star_rating, product_category, review_date\n\n"
            f"Or use an alternative source like Kaggle Amazon Product Reviews dataset."
        )
        raise RuntimeError(error_msg)
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(dataset)
    # Extract required fields
    # Note: review_body and star_rating are required, others are optional
    required_fields = ['review_body', 'star_rating']
    optional_fields = ['product_category', 'review_date']
    
    # Check available columns and map them
    available_cols = df.columns.tolist()
    print(f"Available columns: {available_cols[:10]}...")
    
    # Common field mappings for Amazon reviews dataset
    # Different datasets have different column names
    field_mapping = {
        'review_body': ['review_body', 'review_text', 'text', 'body', 'content', 'review'],
        'star_rating': ['star_rating', 'rating', 'stars'],
        'product_category': ['product_category', 'category', 'product_category_1', 'product_id'],
        'review_date': ['review_date', 'date', 'review_time']
    }
    
    # No special handling needed - real datasets should have actual star ratings (1-5)
    
    # Find actual column names
    actual_fields = {}
    for target_field, possible_names in field_mapping.items():
        for col in available_cols:
            if col.lower() in [name.lower() for name in possible_names]:
                actual_fields[target_field] = col
                break
    
    # Check if all REQUIRED fields were found
    missing_required = set(required_fields) - set(actual_fields.keys())
    if missing_required:
        error_msg = f"Missing required fields: {missing_required}. Found: {actual_fields}"
        raise ValueError(error_msg)
    
    # Handle missing optional fields by creating dummy columns
    for optional_field in optional_fields:
        if optional_field not in actual_fields:
            if optional_field == 'product_category':
                df['product_category'] = 'unknown'
            elif optional_field == 'review_date':
                df['review_date'] = pd.Timestamp.now()  # Use current date as default
            actual_fields[optional_field] = optional_field
    
    # Select and rename columns
    df_selected = df[list(actual_fields.values())].copy()
    df_selected.columns = ['review_text', 'star_rating', 'product_category', 'review_date']
    
    # Ensure star_rating is numeric
        
    df_selected['star_rating'] = pd.to_numeric(df_selected['star_rating'], errors='coerce')
    
    # Remove rows with missing star ratings
    before_drop = len(df_selected)
    df_selected = df_selected.dropna(subset=['star_rating'])
    after_drop = len(df_selected)
        
    # Stratified sampling by star rating - maintain natural distribution
    print(f"Original dataset size: {len(df_selected)}")
    print(f"Star rating distribution before sampling:\n{df_selected['star_rating'].value_counts().sort_index()}")
    print(f"Sampling {n_samples} reviews with stratified sampling (maintaining natural distribution)...")
    
        
    # True stratified sampling: sample proportionally from each rating
    # This maintains the natural distribution, not equal numbers per rating
    unique_ratings = sorted(df_selected['star_rating'].unique())
    sampled_dfs = []
    
    for rating in unique_ratings:
        rating_df = df_selected[df_selected['star_rating'] == rating]
        # Calculate proportional sample size based on natural distribution
        proportion = len(rating_df) / len(df_selected)
        n_rating = int(n_samples * proportion)
        
        # Ensure we sample at least 1 if possible, but don't force equal numbers
        if n_rating > 0 and len(rating_df) > 0:
            n_to_sample = min(n_rating, len(rating_df))
            sampled = rating_df.sample(n=n_to_sample, random_state=random_state)
            sampled_dfs.append(sampled)
                
    df_sampled = pd.concat(sampled_dfs, ignore_index=True)
    
    # If we need more samples to reach n_samples, randomly sample from remaining
    if len(df_sampled) < n_samples:
        remaining = df_selected[~df_selected.index.isin(df_sampled.index)]
        additional_needed = n_samples - len(df_sampled)
        if len(remaining) > 0:
            # Sample proportionally from remaining to maintain distribution
            additional = remaining.sample(n=min(additional_needed, len(remaining)), random_state=random_state)
            df_sampled = pd.concat([df_sampled, additional], ignore_index=True)
                
    # Shuffle the final dataset
    df_sampled = df_sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"Sampled dataset size: {len(df_sampled)}")
    print(f"Star rating distribution:\n{df_sampled['star_rating'].value_counts().sort_index()}")
    
    # Save raw data
    raw_data_path = Path("data/raw/amazon_reviews_raw.parquet")
    raw_data_path.parent.mkdir(parents=True, exist_ok=True)
    df_sampled.to_parquet(raw_data_path, index=False)
    print(f"Raw data saved to {raw_data_path}")
    
    return df_sampled


def clean_text(text):
    """
    Clean text by lowercasing, removing punctuation, and extra whitespace.
    
    Parameters:
    -----------
    text : str
        Input text to clean
    
    Returns:
    --------
    str
        Cleaned text
    """
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to string
    text = str(text)
    
    # Lowercase
    text = text.lower()
    
    # Remove punctuation (keep alphanumeric and spaces)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def tokenize_text(text, nlp=None):
    """
    Tokenize text using spaCy tokenizer only (no dependency parsing).
    
    Parameters:
    -----------
    text : str
        Input text to tokenize
    nlp : spacy.lang
        spaCy language model (optional, will load if not provided)
    
    Returns:
    --------
    list
        List of tokens
    """
    if not nlp:
        try:
            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
        except OSError:
            print("Warning: spaCy model not found. Using simple tokenization.")
            return text.split() if text else []
    
    if not text or text == '':
        return []
    
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_space]
    return tokens


def preprocess_data(df, save_processed=True):
    """
    Preprocess the dataset: clean text, extract features, handle missing values.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw DataFrame with review data
    save_processed : bool
        Whether to save processed data to disk
    
    Returns:
    --------
    pd.DataFrame
        Preprocessed DataFrame with additional features
    """
    print("Starting data preprocessing...")
        
    df_processed = df.copy()
    
    # Handle missing or empty reviews
    print("Handling missing values...")
        
    df_processed['review_text'] = df_processed['review_text'].fillna('')
    before_len_check = len(df_processed)
    df_processed = df_processed[df_processed['review_text'].str.len() > 0]
    after_len_check = len(df_processed)
        
    # Clean text
    print("Cleaning text...")
    df_processed['review_text_clean'] = df_processed['review_text'].apply(clean_text)
    
    # Remove rows with empty cleaned text
    df_processed = df_processed[df_processed['review_text_clean'].str.len() > 0]
    
    # Tokenize for length calculation
    print("Tokenizing text for length calculation...")
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
    except OSError:
        print("Warning: spaCy model not found. Using simple tokenization for length calculation.")
        nlp = None
    
    df_processed['tokens'] = df_processed['review_text_clean'].apply(
        lambda x: tokenize_text(x, nlp) if nlp else x.split()
    )
    df_processed['review_length'] = df_processed['tokens'].apply(len)
    
    # Extract temporal features
    print("Extracting temporal features...")
        
    df_processed['review_date'] = pd.to_datetime(df_processed['review_date'], errors='coerce')
    df_processed['review_year'] = df_processed['review_date'].dt.year
    df_processed['review_month'] = df_processed['review_date'].dt.month
        
    # Drop token column (not needed for modeling)
    df_processed = df_processed.drop(columns=['tokens'])
    
    print(f"Preprocessing complete. Final dataset size: {len(df_processed)}")
    print(f"Review length stats:\n{df_processed['review_length'].describe()}")
        
    # Save processed data
    if save_processed:
        processed_data_path = Path("data/processed/amazon_reviews_processed.parquet")
                
        processed_data_path.parent.mkdir(parents=True, exist_ok=True)
        df_processed.to_parquet(processed_data_path, index=False)
        print(f"Processed data saved to {processed_data_path}")
            
    return df_processed


if __name__ == "__main__":
    # Example usage
    df_raw = load_amazon_reviews(n_samples=200000)
    df_processed = preprocess_data(df_raw)
    print("\nPreprocessing pipeline complete!")

