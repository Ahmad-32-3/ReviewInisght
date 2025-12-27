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
from scipy.sparse import issparse
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
    # For sparse matrices (like TF-IDF), we can't center (subtract mean)
    # so we set with_mean=False
    is_sparse = issparse(X_train)
    if is_sparse:
        print("  Detected sparse matrix - using StandardScaler with with_mean=False")
        scaler = StandardScaler(with_mean=False)
    else:
        print("  Detected dense matrix - using StandardScaler with default settings")
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
    Train XGBoost Classifier with regularization to prevent overfitting.
    
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
    print("Training XGBoost Classifier with regularization...")
    
    # Train model with regularization to prevent overfitting
    xgb_model = xgb.XGBClassifier(
        random_state=random_state,
        eval_metric='logloss',
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1,
        # Regularization parameters to prevent overfitting
        max_depth=4,  # Limit tree depth (default 6, reducing to prevent overfitting)
        min_child_weight=3,  # Minimum sum of instance weight needed in a child
        gamma=0.1,  # Minimum loss reduction required to make a split
        reg_alpha=0.1,  # L1 regularization
        reg_lambda=1.0,  # L2 regularization
        subsample=0.8,  # Subsample ratio of training instances
        colsample_bytree=0.8,  # Subsample ratio of columns when constructing each tree
        learning_rate=0.1,  # Step size shrinkage
        n_estimators=100,  # Number of boosting rounds
        early_stopping_rounds=10,  # Early stopping if no improvement
        verbosity=0  # Suppress output
    )
    
    # Use early stopping with validation set
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
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


def compare_models_visually(lr_model, xgb_model, X_val, y_val, feature_names, 
                           val_texts=None, output_dir=None):
    """
    Create comprehensive visualizations comparing linear vs non-linear models.
    Shows how they differ in decision-making, feature importance, and predictions.
    
    Parameters:
    -----------
    lr_model : sklearn.LogisticRegression
        Trained Logistic Regression model
    xgb_model : xgb.XGBClassifier
        Trained XGBoost model
    X_val : array-like or sparse matrix
        Validation features
    y_val : array-like
        Validation labels
    feature_names : list
        List of feature names
    val_texts : array-like, optional
        Validation review texts for examples
    output_dir : str or Path, optional
        Directory to save visualizations. If None, uses "../outputs/figures"
    
    Returns:
    --------
    dict
        Dictionary with comparison summary statistics
    """
    output_dir = Path(output_dir) if output_dir else Path("../outputs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'agreement_rate': 0.0,
        'common_features': [],
        'lr_only_features': [],
        'xgb_only_features': []
    }
    
    try:
        # Get predictions from both models
        if hasattr(lr_model, 'scaler'):
            # Convert sparse to dense if needed for scaling
            if issparse(X_val):
                X_val_lr = lr_model.scaler.transform(X_val.toarray())
            else:
                X_val_lr = lr_model.scaler.transform(X_val)
        else:
            if issparse(X_val):
                X_val_lr = X_val.toarray()
            else:
                X_val_lr = X_val
        
        y_pred_lr = lr_model.predict(X_val_lr)
        y_proba_lr = lr_model.predict_proba(X_val_lr)[:, 1]
        
        # XGBoost can handle sparse matrices directly
        y_pred_xgb = xgb_model.predict(X_val)
        y_proba_xgb = xgb_model.predict_proba(X_val)[:, 1]
        
        # 1. Feature Importance Comparison
        try:
            print("Creating feature importance comparison...")
            
            # Get top features from both models
            if not hasattr(lr_model, 'coef_') or lr_model.coef_ is None:
                print("Warning: Logistic Regression model missing coefficients. Skipping feature comparison.")
                lr_feature_df = pd.DataFrame(columns=['feature', 'importance'])
                lr_top_features = set()
            else:
                lr_coefs = lr_model.coef_[0]
                lr_feature_df = pd.DataFrame({
                    'feature': feature_names[:len(lr_coefs)],
                    'importance': np.abs(lr_coefs)
                }).sort_values('importance', ascending=False).head(20)
                lr_top_features = set(lr_feature_df['feature'].head(10))
            
            if not hasattr(xgb_model, 'feature_importances_'):
                print("Warning: XGBoost model missing feature importances. Skipping feature comparison.")
                xgb_feature_df = pd.DataFrame(columns=['feature', 'importance'])
                xgb_top_features = set()
            else:
                xgb_importance = xgb_model.feature_importances_
                xgb_feature_df = pd.DataFrame({
                    'feature': feature_names[:len(xgb_importance)],
                    'importance': xgb_importance
                }).sort_values('importance', ascending=False).head(20)
                xgb_top_features = set(xgb_feature_df['feature'].head(10))
            
            # Find common top features
            common_features = lr_top_features.intersection(xgb_top_features)
            summary['common_features'] = list(common_features)
            summary['lr_only_features'] = list(lr_top_features - xgb_top_features)
            summary['xgb_only_features'] = list(xgb_top_features - lr_top_features)
            
            if len(lr_feature_df) > 0 or len(xgb_feature_df) > 0:
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                
                # Top features - Logistic Regression
                if len(lr_feature_df) > 0:
                    axes[0, 0].barh(range(len(lr_feature_df)), lr_feature_df['importance'])
                    axes[0, 0].set_yticks(range(len(lr_feature_df)))
                    axes[0, 0].set_yticklabels(lr_feature_df['feature'], fontsize=8)
                    axes[0, 0].set_xlabel('Absolute Coefficient Value')
                    axes[0, 0].set_title('Logistic Regression: Top 20 Features\n(Linear Model)')
                    axes[0, 0].invert_yaxis()
                    axes[0, 0].grid(axis='x', alpha=0.3)
                else:
                    axes[0, 0].text(0.5, 0.5, 'No data available', ha='center', va='center')
                    axes[0, 0].set_title('Logistic Regression: Top 20 Features')
                
                # Top features - XGBoost
                if len(xgb_feature_df) > 0:
                    axes[0, 1].barh(range(len(xgb_feature_df)), xgb_feature_df['importance'])
                    axes[0, 1].set_yticks(range(len(xgb_feature_df)))
                    axes[0, 1].set_yticklabels(xgb_feature_df['feature'], fontsize=8)
                    axes[0, 1].set_xlabel('Feature Importance')
                    axes[0, 1].set_title('XGBoost: Top 20 Features\n(Non-linear Model)')
                    axes[0, 1].invert_yaxis()
                    axes[0, 1].grid(axis='x', alpha=0.3)
                else:
                    axes[0, 1].text(0.5, 0.5, 'No data available', ha='center', va='center')
                    axes[0, 1].set_title('XGBoost: Top 20 Features')
                
                # Feature overlap analysis
                all_top_features = list(lr_top_features.union(xgb_top_features))
                if len(all_top_features) > 0:
                    overlap_data = []
                    for feat in all_top_features:
                        in_lr = feat in lr_top_features
                        in_xgb = feat in xgb_top_features
                        overlap_data.append({
                            'feature': feat,
                            'in_lr': in_lr,
                            'in_xgb': in_xgb,
                            'in_both': in_lr and in_xgb
                        })
                    
                    overlap_df = pd.DataFrame(overlap_data)
                    overlap_df['category'] = overlap_df.apply(
                        lambda x: 'Both Models' if x['in_both'] 
                        else ('LR Only' if x['in_lr'] else 'XGB Only'), axis=1
                    )
                    
                    category_counts = overlap_df['category'].value_counts()
                    if len(category_counts) > 0:
                        axes[1, 0].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
                        axes[1, 0].set_title('Top 10 Features Overlap\n(How models agree on important features)')
                    else:
                        axes[1, 0].text(0.5, 0.5, 'No overlap data', ha='center', va='center')
                        axes[1, 0].set_title('Top 10 Features Overlap')
                else:
                    axes[1, 0].text(0.5, 0.5, 'No features to compare', ha='center', va='center')
                    axes[1, 0].set_title('Top 10 Features Overlap')
                
                # Prediction agreement analysis
                agreement = (y_pred_lr == y_pred_xgb).sum()
                disagreement = (y_pred_lr != y_pred_xgb).sum()
                total = len(y_pred_lr)
                summary['agreement_rate'] = agreement / total if total > 0 else 0.0
                
                agreement_data = {
                    'Agree': agreement,
                    'Disagree': disagreement
                }
                
                axes[1, 1].bar(agreement_data.keys(), agreement_data.values(), 
                               color=['green', 'orange'], alpha=0.7)
                axes[1, 1].set_ylabel('Number of Predictions')
                axes[1, 1].set_title(f'Prediction Agreement\n({agreement/total*100:.1f}% agree)')
                axes[1, 1].grid(axis='y', alpha=0.3)
                for i, (k, v) in enumerate(agreement_data.items()):
                    axes[1, 1].text(i, v, str(v), ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(output_dir / "model_comparison_features.png", dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Feature comparison saved to {output_dir / 'model_comparison_features.png'}")
        except Exception as e:
            print(f"Warning: Feature importance comparison failed: {e}")
            import traceback
            traceback.print_exc()
        
        # 2. Confusion Matrix Comparison
        try:
            print("Creating confusion matrix comparison...")
            
            cm_lr = confusion_matrix(y_val, y_pred_lr)
            cm_xgb = confusion_matrix(y_val, y_pred_xgb)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                        xticklabels=['Negative', 'Positive'],
                        yticklabels=['Negative', 'Positive'])
            axes[0].set_title('Logistic Regression\nConfusion Matrix')
            axes[0].set_ylabel('True Label')
            axes[0].set_xlabel('Predicted Label')
            
            sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                        xticklabels=['Negative', 'Positive'],
                        yticklabels=['Negative', 'Positive'])
            axes[1].set_title('XGBoost\nConfusion Matrix')
            axes[1].set_ylabel('True Label')
            axes[1].set_xlabel('Predicted Label')
            
            plt.tight_layout()
            plt.savefig(output_dir / "model_comparison_confusion.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Confusion matrix comparison saved to {output_dir / 'model_comparison_confusion.png'}")
        except Exception as e:
            print(f"Warning: Confusion matrix comparison failed: {e}")
        
        # 3. Probability Distribution Comparison
        try:
            print("Creating probability distribution comparison...")
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Probability distributions
            if len(y_val) > 0:
                negative_mask = y_val == 0
                positive_mask = y_val == 1
                
                if negative_mask.sum() > 0:
                    axes[0, 0].hist(y_proba_lr[negative_mask], bins=30, alpha=0.6, 
                                   label='Negative Reviews', color='red')
                if positive_mask.sum() > 0:
                    axes[0, 0].hist(y_proba_lr[positive_mask], bins=30, alpha=0.6, 
                                   label='Positive Reviews', color='green')
                axes[0, 0].set_xlabel('Predicted Probability (Positive)')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].set_title('Logistic Regression\nProbability Distribution')
                axes[0, 0].legend()
                axes[0, 0].grid(alpha=0.3)
                
                if negative_mask.sum() > 0:
                    axes[0, 1].hist(y_proba_xgb[negative_mask], bins=30, alpha=0.6, 
                                   label='Negative Reviews', color='red')
                if positive_mask.sum() > 0:
                    axes[0, 1].hist(y_proba_xgb[positive_mask], bins=30, alpha=0.6, 
                                   label='Positive Reviews', color='green')
                axes[0, 1].set_xlabel('Predicted Probability (Positive)')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('XGBoost\nProbability Distribution')
                axes[0, 1].legend()
                axes[0, 1].grid(alpha=0.3)
                
                # Probability scatter plot (how models compare on same samples)
                axes[1, 0].scatter(y_proba_lr, y_proba_xgb, alpha=0.3, s=10)
                axes[1, 0].plot([0, 1], [0, 1], 'r--', label='Perfect Agreement')
                axes[1, 0].set_xlabel('Logistic Regression Probability')
                axes[1, 0].set_ylabel('XGBoost Probability')
                axes[1, 0].set_title('Probability Agreement\n(How similar are predictions?)')
                axes[1, 0].legend()
                axes[1, 0].grid(alpha=0.3)
                
                # Disagreement cases
                disagree_mask = y_pred_lr != y_pred_xgb
                if disagree_mask.sum() > 0:
                    axes[1, 1].scatter(y_proba_lr[disagree_mask], y_proba_xgb[disagree_mask], 
                                      alpha=0.6, s=20, color='red', label='Disagreements')
                    if (~disagree_mask).sum() > 0:
                        axes[1, 1].scatter(y_proba_lr[~disagree_mask], y_proba_xgb[~disagree_mask], 
                                          alpha=0.2, s=10, color='gray', label='Agreements')
                    axes[1, 1].set_xlabel('Logistic Regression Probability')
                    axes[1, 1].set_ylabel('XGBoost Probability')
                    axes[1, 1].set_title(f'Where Models Disagree\n({disagree_mask.sum()} cases)')
                    axes[1, 1].legend()
                    axes[1, 1].grid(alpha=0.3)
                else:
                    axes[1, 1].text(0.5, 0.5, 'Models agree on all predictions!', 
                                   ha='center', va='center', fontsize=14)
                    axes[1, 1].set_title('Where Models Disagree')
            
            plt.tight_layout()
            plt.savefig(output_dir / "model_comparison_probabilities.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Probability comparison saved to {output_dir / 'model_comparison_probabilities.png'}")
        except Exception as e:
            print(f"Warning: Probability distribution comparison failed: {e}")
        
        # 4. Examples where models disagree (if texts provided)
        if val_texts is not None:
            try:
                disagree_mask = y_pred_lr != y_pred_xgb
                if disagree_mask.sum() > 0:
                    print("Creating disagreement examples...")
                    
                    disagree_indices = np.where(disagree_mask)[0]
                    n_examples = min(10, len(disagree_indices))
                    disagree_indices = disagree_indices[:n_examples]
                    
                    with open(output_dir / "model_disagreement_examples.txt", 'w', encoding='utf-8') as f:
                        f.write("=" * 80 + "\n")
                        f.write("EXAMPLES WHERE MODELS DISAGREE\n")
                        f.write("=" * 80 + "\n\n")
                        f.write(f"Total disagreements: {disagree_mask.sum()} out of {len(y_val)} ({disagree_mask.sum()/len(y_val)*100:.1f}%)\n\n")
                        
                        for idx, i in enumerate(disagree_indices, 1):
                            f.write(f"\n{'='*80}\n")
                            f.write(f"Example {idx} (Index {i}):\n")
                            f.write(f"{'='*80}\n")
                            f.write(f"True Label: {'POSITIVE' if y_val[i] == 1 else 'NEGATIVE'}\n")
                            f.write(f"Logistic Regression: {'POSITIVE' if y_pred_lr[i] == 1 else 'NEGATIVE'} "
                                   f"(confidence: {y_proba_lr[i]:.3f})\n")
                            f.write(f"XGBoost: {'POSITIVE' if y_pred_xgb[i] == 1 else 'NEGATIVE'} "
                                   f"(confidence: {y_proba_xgb[i]:.3f})\n\n")
                            if i < len(val_texts):
                                f.write(f"Review Text:\n{val_texts[i]}\n")
                            else:
                                f.write("Review Text: [Index out of range]\n")
                    
                    print(f"Disagreement examples saved to {output_dir / 'model_disagreement_examples.txt'}")
                else:
                    print("Models agree on all predictions - no disagreement examples to save.")
            except Exception as e:
                print(f"Warning: Disagreement examples generation failed: {e}")
        else:
            print("Note: Validation texts not provided. Skipping disagreement examples.")
        
        # Summary statistics
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        agreement = (y_pred_lr == y_pred_xgb).sum()
        total = len(y_pred_lr)
        print(f"\nPrediction Agreement: {agreement}/{total} ({agreement/total*100:.1f}%)")
        print(f"Prediction Disagreement: {(y_pred_lr != y_pred_xgb).sum()}/{total} ({(y_pred_lr != y_pred_xgb).sum()/total*100:.1f}%)")
        print(f"\nCommon Top 10 Features: {len(summary['common_features'])}")
        if len(summary['common_features']) > 0:
            print(f"  Features: {', '.join(list(summary['common_features'])[:5])}...")
        print(f"\nLR-only Top Features: {len(summary['lr_only_features'])}")
        print(f"XGB-only Top Features: {len(summary['xgb_only_features'])}")
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"Error in model comparison: {e}")
        import traceback
        traceback.print_exc()
    
    return summary


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


def validate_model_performance(y_train, y_train_pred, y_train_proba, 
                               y_val, y_val_pred, y_val_proba,
                               model_name="Model"):
    """
    Validate model performance and check for data leakage or unrealistic metrics.
    
    Parameters:
    -----------
    y_train : array-like
        Training true labels
    y_train_pred : array-like
        Training predictions
    y_train_proba : array-like
        Training predicted probabilities
    y_val : array-like
        Validation true labels
    y_val_pred : array-like
        Validation predictions
    y_val_proba : array-like
        Validation predicted probabilities
    model_name : str
        Name of the model for display
    
    Returns:
    --------
    dict
        Validation report with warnings and status
    """
    from sklearn.metrics import accuracy_score
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    train_val_gap = train_acc - val_acc
    
    # Calculate additional metrics for validation
    val_precision = precision_score(y_val, y_val_pred, zero_division=0)
    val_recall = recall_score(y_val, y_val_pred, zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
    
    report = {
        'model_name': model_name,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'train_val_gap': train_val_gap,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_f1': val_f1,
        'warnings': [],
        'errors': [],
        'status': 'PASS'
    }
    
    # Check for suspiciously high accuracy (potential data leakage)
    if val_acc > 0.99:
        report['errors'].append(
            f"CRITICAL: Validation accuracy is {val_acc:.4f} (>0.99). "
            f"This suggests data leakage. The model may have seen validation data during training."
        )
        report['status'] = 'FAIL'
    elif val_acc > 0.98:
        report['warnings'].append(
            f"WARNING: Validation accuracy is {val_acc:.4f} (>0.98). "
            f"This is unusually high and may indicate data leakage."
        )
    
    # Check for unrealistic accuracy (too low might also be suspicious)
    if val_acc < 0.50:
        report['warnings'].append(
            f"WARNING: Validation accuracy is {val_acc:.4f} (<0.50). "
            f"This is worse than random guessing. Check for label issues."
        )
    
    # Check for reasonable accuracy range (typical for sentiment analysis: 0.75-0.95)
    if 0.75 <= val_acc <= 0.95:
        report['warnings'].append(
            f"INFO: Validation accuracy {val_acc:.4f} is in realistic range for sentiment analysis."
        )
    
    # Check train/val gap (should be reasonable, not too large)
    if train_val_gap > 0.10:
        report['warnings'].append(
            f"WARNING: Train/val gap is {train_val_gap:.4f} (>0.10). "
            f"Model may be overfitting significantly."
        )
    elif train_val_gap < -0.05:
        report['warnings'].append(
            f"WARNING: Validation accuracy is higher than training accuracy by {abs(train_val_gap):.4f}. "
            f"This is unusual and may indicate issues with the split or data."
        )
    elif 0.02 <= train_val_gap <= 0.05:
        report['warnings'].append(
            f"INFO: Train/val gap {train_val_gap:.4f} is reasonable (typical overfitting)."
        )
    
    # Check if train accuracy is perfect (another data leakage indicator)
    if train_acc > 0.999:
        report['warnings'].append(
            f"WARNING: Training accuracy is {train_acc:.4f} (near perfect). "
            f"This may indicate the model memorized the training data or data leakage."
        )
    
    # Check for perfect predictions (all correct)
    if val_acc == 1.0:
        report['errors'].append(
            f"CRITICAL: Perfect validation accuracy (1.0000). "
            f"This is almost certainly due to data leakage."
        )
        report['status'] = 'FAIL'
    
    return report

