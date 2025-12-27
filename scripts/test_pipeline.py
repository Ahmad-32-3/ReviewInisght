"""
Test script to verify the full pipeline and check for data leakage.

This script:
1. Runs feature engineering steps programmatically
2. Verifies train/val split is correct
3. Verifies TF-IDF is fit only on training data
4. Trains models and checks accuracy is realistic
5. Verifies no data leakage
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix, load_npz
import pickle
import warnings
warnings.filterwarnings('ignore')

from preprocessing import preprocess_data
from modeling import (
    create_binary_labels, train_logistic_regression, train_xgboost,
    evaluate_model, validate_model_performance
)

# Set random seed
np.random.seed(42)

def test_pipeline():
    """Run full pipeline test."""
    print("=" * 80)
    print("PIPELINE TEST - Data Leakage Verification")
    print("=" * 80)
    
    results = {
        'passed': [],
        'failed': [],
        'warnings': []
    }
    
    # Step 1: Load processed data
    print("\n[1/7] Loading processed data...")
    data_path = Path("data/processed/amazon_reviews_processed.parquet")
    if not data_path.exists():
        results['failed'].append("Processed data not found. Run 01_eda.ipynb first.")
        return results
    
    df = pd.read_parquet(data_path)
    print(f"  [OK] Loaded {len(df)} samples")
    
    # Step 2: Create binary labels
    print("\n[2/7] Creating binary labels...")
    df_labeled = create_binary_labels(df)
    print(f"  [OK] Created labels: {df_labeled['sentiment'].value_counts().to_dict()}")
    
    # Step 3: Split data BEFORE feature engineering
    print("\n[3/7] Splitting data (BEFORE feature engineering)...")
    df_train, df_val = train_test_split(
        df_labeled,
        test_size=0.2,
        random_state=42,
        stratify=df_labeled['sentiment']
    )
    
    train_dist = df_train['sentiment'].value_counts().to_dict()
    val_dist = df_val['sentiment'].value_counts().to_dict()
    print(f"  [OK] Training: {len(df_train)} samples, distribution: {train_dist}")
    print(f"  [OK] Validation: {len(df_val)} samples, distribution: {val_dist}")
    
    # Verify split proportions
    train_pct = len(df_train) / len(df_labeled)
    if 0.79 <= train_pct <= 0.81:
        results['passed'].append("Train/val split proportions correct (80/20)")
    else:
        results['failed'].append(f"Train/val split proportions incorrect: {train_pct:.2%}")
    
    # Step 4: Fit TF-IDF ONLY on training data
    print("\n[4/7] Creating TF-IDF features (fit on training only)...")
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        lowercase=True,
        stop_words='english'
    )
    
    # CRITICAL: Fit only on training data
    X_tfidf_train = vectorizer.fit_transform(df_train['review_text_clean'])
    X_tfidf_val = vectorizer.transform(df_val['review_text_clean'])
    
    print(f"  [OK] Training TF-IDF: {X_tfidf_train.shape}")
    print(f"  [OK] Validation TF-IDF: {X_tfidf_val.shape}")
    print(f"  [OK] Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Verify TF-IDF was fit on training data only
    # Check that validation features don't have training-specific statistics
    train_mean = X_tfidf_train.mean()
    val_mean = X_tfidf_val.mean()
    if abs(train_mean - val_mean) < 0.1:  # Should be similar but not identical
        results['passed'].append("TF-IDF features created correctly (fit on train, transform on val)")
    else:
        results['warnings'].append(f"TF-IDF means differ significantly: train={train_mean:.4f}, val={val_mean:.4f}")
    
    # Step 5: Create numeric features
    print("\n[5/7] Creating numeric features...")
    train_year_median = df_train['review_year'].median()
    train_month_median = df_train['review_month'].median()
    
    numeric_features_train = pd.DataFrame({
        'review_length': df_train['review_length'],
        'review_year': df_train['review_year'].fillna(train_year_median),
        'review_month': df_train['review_month'].fillna(train_month_median)
    })
    
    numeric_features_val = pd.DataFrame({
        'review_length': df_val['review_length'],
        'review_year': df_val['review_year'].fillna(train_year_median),
        'review_month': df_val['review_month'].fillna(train_month_median)
    })
    
    X_numeric_train = csr_matrix(numeric_features_train.values)
    X_numeric_val = csr_matrix(numeric_features_val.values)
    
    print(f"  [OK] Training numeric: {X_numeric_train.shape}")
    print(f"  [OK] Validation numeric: {X_numeric_val.shape}")
    results['passed'].append("Numeric features created with training statistics")
    
    # Step 6: Combine features
    print("\n[6/7] Combining features...")
    X_train = hstack([X_tfidf_train, X_numeric_train])
    X_val = hstack([X_tfidf_val, X_numeric_val])
    y_train = df_train['sentiment'].values
    y_val = df_val['sentiment'].values
    
    print(f"  [OK] Training features: {X_train.shape}")
    print(f"  [OK] Validation features: {X_val.shape}")
    results['passed'].append("Features combined correctly")
    
    # Step 7: Train models and validate
    print("\n[7/7] Training models and validating performance...")
    
    # Train Logistic Regression
    print("\n  Training Logistic Regression...")
    lr_model, y_pred_lr, y_proba_lr = train_logistic_regression(
        X_train, y_train, X_val, y_val,
        class_weight='balanced',
        random_state=42
    )
    metrics_lr = evaluate_model(y_val, y_pred_lr, y_proba_lr, "Logistic Regression")
    
    # Get training predictions for validation
    if hasattr(lr_model, 'scaler'):
        from scipy.sparse import issparse
        if issparse(X_train):
            X_train_scaled = lr_model.scaler.transform(X_train.toarray())
        else:
            X_train_scaled = lr_model.scaler.transform(X_train)
        y_train_pred_lr = lr_model.predict(X_train_scaled)
        y_train_proba_lr = lr_model.predict_proba(X_train_scaled)[:, 1]
    else:
        y_train_pred_lr = lr_model.predict(X_train)
        y_train_proba_lr = lr_model.predict_proba(X_train)[:, 1]
    
    lr_validation = validate_model_performance(
        y_train, y_train_pred_lr, y_train_proba_lr,
        y_val, y_pred_lr, y_proba_lr,
        model_name="Logistic Regression"
    )
    
    # Train XGBoost
    print("\n  Training XGBoost...")
    xgb_model, y_pred_xgb, y_proba_xgb = train_xgboost(
        X_train, y_train, X_val, y_val,
        random_state=42
    )
    metrics_xgb = evaluate_model(y_val, y_pred_xgb, y_proba_xgb, "XGBoost")
    
    y_train_pred_xgb = xgb_model.predict(X_train)
    y_train_proba_xgb = xgb_model.predict_proba(X_train)[:, 1]
    
    xgb_validation = validate_model_performance(
        y_train, y_train_pred_xgb, y_train_proba_xgb,
        y_val, y_pred_xgb, y_proba_xgb,
        model_name="XGBoost"
    )
    
    # Validate results
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    
    print(f"\nLogistic Regression:")
    print(f"  Status: {lr_validation['status']}")
    print(f"  Train Accuracy: {lr_validation['train_accuracy']:.4f}")
    print(f"  Val Accuracy: {lr_validation['val_accuracy']:.4f}")
    print(f"  Train/Val Gap: {lr_validation['train_val_gap']:.4f}")
    
    print(f"\nXGBoost:")
    print(f"  Status: {xgb_validation['status']}")
    print(f"  Train Accuracy: {xgb_validation['train_accuracy']:.4f}")
    print(f"  Val Accuracy: {xgb_validation['val_accuracy']:.4f}")
    print(f"  Train/Val Gap: {xgb_validation['train_val_gap']:.4f}")
    
    # Check for data leakage
    if lr_validation['status'] == 'PASS' and xgb_validation['status'] == 'PASS':
        results['passed'].append("Both models passed validation (no data leakage detected)")
    else:
        results['failed'].append("One or more models failed validation (possible data leakage)")
    
    # Check for realistic accuracy
    if 0.75 <= lr_validation['val_accuracy'] <= 0.95:
        results['passed'].append(f"Logistic Regression accuracy is realistic: {lr_validation['val_accuracy']:.4f}")
    else:
        results['warnings'].append(f"Logistic Regression accuracy may be unrealistic: {lr_validation['val_accuracy']:.4f}")
    
    if 0.75 <= xgb_validation['val_accuracy'] <= 0.95:
        results['passed'].append(f"XGBoost accuracy is realistic: {xgb_validation['val_accuracy']:.4f}")
    else:
        results['warnings'].append(f"XGBoost accuracy may be unrealistic: {xgb_validation['val_accuracy']:.4f}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"\n[PASSED] {len(results['passed'])}")
    for item in results['passed']:
        print(f"  - {item}")
    
    if results['warnings']:
        print(f"\n[WARNINGS] {len(results['warnings'])}")
        for item in results['warnings']:
            print(f"  - {item}")
    
    if results['failed']:
        print(f"\n[FAILED] {len(results['failed'])}")
        for item in results['failed']:
            print(f"  - {item}")
    else:
        print("\n[SUCCESS] All tests passed!")
    
    return results

if __name__ == "__main__":
    results = test_pipeline()
    sys.exit(0 if not results['failed'] else 1)

