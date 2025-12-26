"""
Modeling utilities for ReviewInsight project.
Handles model training, evaluation, and interpretation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def create_binary_labels(df):
    """
    Create binary sentiment labels from star ratings.
    - Positive: >= 4 stars
    - Negative: <= 2 stars
    - Drop 3-star reviews
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with star_rating column
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with binary labels (1 for positive, 0 for negative)
    """
    df_labeled = df.copy()
    
    # Create binary labels
    df_labeled['sentiment'] = df_labeled['star_rating'].apply(
        lambda x: 1 if x >= 4 else (0 if x <= 2 else None)
    )
    
    # Drop 3-star reviews
    df_labeled = df_labeled.dropna(subset=['sentiment'])
    df_labeled['sentiment'] = df_labeled['sentiment'].astype(int)
    
    print(f"Label distribution:\n{df_labeled['sentiment'].value_counts()}")
    print(f"Positive: {df_labeled['sentiment'].sum()}, Negative: {(~df_labeled['sentiment'].astype(bool)).sum()}")
    
    return df_labeled


def train_logistic_regression(X_train, y_train, X_val, y_val, class_weight='balanced', random_state=42):
    """
    Train Logistic Regression model.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_val : array-like
        Validation features
    y_val : array-like
        Validation labels
    class_weight : str or dict
        Class weight handling for imbalance
    random_state : int
        Random seed
    
    Returns:
    --------
    tuple
        (trained_model, predictions, probabilities)
    """
    print("Training Logistic Regression...")
    
    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train model
    lr_model = LogisticRegression(
        class_weight=class_weight,
        random_state=random_state,
        max_iter=1000,
        solver='lbfgs'
    )
    lr_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = lr_model.predict(X_val_scaled)
    y_proba = lr_model.predict_proba(X_val_scaled)[:, 1]
    
    # Store scaler with model
    lr_model.scaler = scaler
    
    return lr_model, y_pred, y_proba


def train_xgboost(X_train, y_train, X_val, y_val, random_state=42):
    """
    Train XGBoost Classifier.
    
    Parameters:
    -----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_val : array-like
        Validation features
    y_val : array-like
        Validation labels
    random_state : int
        Random seed
    
    Returns:
    --------
    tuple
        (trained_model, predictions, probabilities)
    """
    print("Training XGBoost Classifier...")
    
    # Train model
    xgb_model = xgb.XGBClassifier(
        random_state=random_state,
        eval_metric='logloss',
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1
    )
    xgb_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = xgb_model.predict(X_val)
    y_proba = xgb_model.predict_proba(X_val)[:, 1]
    
    return xgb_model, y_pred, y_proba


def evaluate_model(y_true, y_pred, y_proba, model_name="Model"):
    """
    Evaluate model performance with multiple metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_proba : array-like
        Predicted probabilities
    model_name : str
        Name of the model for display
    
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0
    }
    
    print(f"\n{model_name} Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    print(f"\n{classification_report(y_true, y_pred)}")
    
    return metrics


def plot_roc_curve(y_true, y_proba_lr, y_proba_xgb, save_path=None):
    """
    Plot ROC curves for both models.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_proba_lr : array-like
        Logistic Regression probabilities
    y_proba_xgb : array-like
        XGBoost probabilities
    save_path : str or Path
        Path to save the plot
    """
    fpr_lr, tpr_lr, _ = roc_curve(y_true, y_proba_lr)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_true, y_proba_xgb)
    
    auc_lr = roc_auc_score(y_true, y_proba_lr)
    auc_xgb = roc_auc_score(y_true, y_proba_xgb)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {auc_lr:.3f})')
    plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_xgb:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.close()


def analyze_logistic_coefficients(model, feature_names, top_n=20, save_path=None):
    """
    Analyze and visualize top coefficients from Logistic Regression.
    
    Parameters:
    -----------
    model : sklearn.LogisticRegression
        Trained Logistic Regression model
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to display
    save_path : str or Path
        Path to save the plot
    """
    coefficients = model.coef_[0]
    feature_coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients
    }).sort_values('coefficient', ascending=False)
    
    # Top positive and negative
    top_positive = feature_coef_df.head(top_n)
    top_negative = feature_coef_df.tail(top_n)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Top positive features
    ax1.barh(range(len(top_positive)), top_positive['coefficient'])
    ax1.set_yticks(range(len(top_positive)))
    ax1.set_yticklabels(top_positive['feature'], fontsize=8)
    ax1.set_xlabel('Coefficient Value')
    ax1.set_title(f'Top {top_n} Positive Features')
    ax1.invert_yaxis()
    
    # Top negative features
    ax2.barh(range(len(top_negative)), top_negative['coefficient'])
    ax2.set_yticks(range(len(top_negative)))
    ax2.set_yticklabels(top_negative['feature'], fontsize=8)
    ax2.set_xlabel('Coefficient Value')
    ax2.set_title(f'Top {top_n} Negative Features')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Coefficient analysis saved to {save_path}")
    
    plt.close()
    
    return feature_coef_df


def analyze_xgboost_importance(model, feature_names, top_n=20, save_path=None):
    """
    Analyze and visualize feature importance from XGBoost.
    
    Parameters:
    -----------
    model : xgb.XGBClassifier
        Trained XGBoost model
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to display
    save_path : str or Path
        Path to save the plot
    """
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(feature_importance_df)), feature_importance_df['importance'])
    plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'], fontsize=8)
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} XGBoost Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    plt.close()
    
    return feature_importance_df


def error_analysis(y_true, y_pred, texts, top_n=10):
    """
    Analyze prediction errors: false positives and false negatives.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    texts : array-like
        Review texts
    top_n : int
        Number of examples to show
    
    Returns:
    --------
    dict
        Dictionary with false positive and false negative examples
    """
    errors = {
        'false_positives': [],
        'false_negatives': []
    }
    
    for i, (true, pred, text) in enumerate(zip(y_true, y_pred, texts)):
        if true == 0 and pred == 1:  # False positive
            errors['false_positives'].append((i, text))
        elif true == 1 and pred == 0:  # False negative
            errors['false_negatives'].append((i, text))
    
    print(f"\nError Analysis:")
    print(f"  False Positives: {len(errors['false_positives'])}")
    print(f"  False Negatives: {len(errors['false_negatives'])}")
    
    print(f"\nTop {top_n} False Positive Examples (Predicted Positive, Actually Negative):")
    for idx, (i, text) in enumerate(errors['false_positives'][:top_n]):
        print(f"  {idx+1}. {text[:200]}...")
    
    print(f"\nTop {top_n} False Negative Examples (Predicted Negative, Actually Positive):")
    for idx, (i, text) in enumerate(errors['false_negatives'][:top_n]):
        print(f"  {idx+1}. {text[:200]}...")
    
    return errors


def save_model(model, filepath):
    """
    Save trained model to disk.
    
    Parameters:
    -----------
    model : sklearn or xgb model
        Trained model
    filepath : str or Path
        Path to save the model
    """
    if model is None:
        raise ValueError("Cannot save model: model is None. Make sure models are trained first.")
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load trained model from disk.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to the saved model
    
    Returns:
    --------
    model
        Loaded model
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    return model


def export_model_info(model, model_name, metrics=None, feature_names=None, output_path=None):
    """
    Export model information to human-readable formats (JSON and text).
    
    Parameters:
    -----------
    model : sklearn or xgb model
        Trained model
    model_name : str
        Name of the model (e.g., "logistic_regression", "xgboost")
    metrics : dict, optional
        Dictionary of evaluation metrics
    feature_names : list, optional
        List of feature names for interpretability
    output_path : str or Path, optional
        Directory to save the exported files. If None, uses current directory.
    
    Returns:
    --------
    dict
        Dictionary containing all exported information
    """
    import json
    from datetime import datetime
    
    output_path = Path(output_path) if output_path else Path(".")
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect model information
    model_info = {
        "model_name": model_name,
        "model_type": type(model).__name__,
        "exported_at": datetime.now().isoformat(),
        "metrics": metrics or {},
    }
    
    # Add model-specific information
    if hasattr(model, 'get_params'):
        model_info["hyperparameters"] = model.get_params()
    
    if hasattr(model, 'coef_') and model.coef_ is not None:
        # Logistic Regression coefficients
        if feature_names:
            coefs = model.coef_[0] if model.coef_.shape[0] == 1 else model.coef_
            top_features = sorted(
                zip(feature_names, coefs),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:20]
            model_info["top_features"] = [
                {"feature": feat, "coefficient": float(coef)}
                for feat, coef in top_features
            ]
    
    if hasattr(model, 'feature_importances_'):
        # XGBoost feature importance
        if feature_names:
            importances = model.feature_importances_
            top_features = sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )[:20]
            model_info["top_features"] = [
                {"feature": feat, "importance": float(imp)}
                for feat, imp in top_features
            ]
    
    # Save as JSON
    json_path = output_path / f"{model_name}_info.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    
    # Save as human-readable text
    txt_path = output_path / f"{model_name}_info.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Model Information: {model_name}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Model Type: {model_info['model_type']}\n")
        f.write(f"Exported At: {model_info['exported_at']}\n\n")
        
        if model_info.get('metrics'):
            f.write("Performance Metrics:\n")
            f.write("-" * 80 + "\n")
            for metric, value in model_info['metrics'].items():
                f.write(f"  {metric.capitalize()}: {value:.4f}\n")
            f.write("\n")
        
        if model_info.get('hyperparameters'):
            f.write("Hyperparameters:\n")
            f.write("-" * 80 + "\n")
            for param, value in model_info['hyperparameters'].items():
                f.write(f"  {param}: {value}\n")
            f.write("\n")
        
        if model_info.get('top_features'):
            f.write("Top Features:\n")
            f.write("-" * 80 + "\n")
            for i, feat_info in enumerate(model_info['top_features'], 1):
                if 'coefficient' in feat_info:
                    f.write(f"  {i:2d}. {feat_info['feature']:30s} (coef: {feat_info['coefficient']:+.6f})\n")
                elif 'importance' in feat_info:
                    f.write(f"  {i:2d}. {feat_info['feature']:30s} (importance: {feat_info['importance']:.6f})\n")
            f.write("\n")
    
    print(f"Model information exported to:")
    print(f"  - JSON: {json_path}")
    print(f"  - Text: {txt_path}")
    
    return model_info

