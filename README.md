# ReviewInsight

Product Review Analysis & Sentiment Modeling

## Overview

ReviewInsight is a comprehensive sentiment analysis project that analyzes large-scale product review text to uncover behavioral patterns and trains supervised models to predict sentiment. The project emphasizes exploratory data analysis (EDA), feature engineering, model comparison, and interpretability.

## Project Structure

```
ReviewInsight/
├── data/
│   ├── raw/              # Raw data (CSV files)
│   └── processed/        # Cleaned and preprocessed data
├── notebooks/
│   ├── 01_eda.ipynb      # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb  # Feature creation
│   └── 03_modeling.ipynb # Model training and evaluation
├── src/
│   ├── preprocessing.py  # Data loading and preprocessing utilities
│   └── modeling.py       # Modeling and evaluation utilities
├── outputs/
│   ├── figures/          # Generated visualizations
│   └── models/           # Saved trained models
├── scripts/
│   └── download_sample_data.py  # Sample data generator
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Features

- **Data Ingestion**: Loads Amazon reviews from CSV with stratified sampling maintaining natural distribution
- **Text Preprocessing**: Lowercasing, punctuation removal, tokenization using spaCy
- **Comprehensive EDA**: Rating distribution, review length analysis, category breakdown, temporal trends, outlier detection
- **Feature Engineering**: TF-IDF vectorization (unigrams + bigrams, 20k features) + numeric features
- **Model Comparison**: Logistic Regression vs XGBoost with detailed evaluation
- **Interpretability**: Coefficient analysis, feature importance, error analysis
- **Human-Readable Exports**: Model information exported as JSON and text files
- **Optional SHAP**: SHAP visualizations for model explainability

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- PowerShell (Windows) or Bash (Linux/Mac)

### Installation

1. **Navigate to Project Directory**
   ```powershell
   cd C:\MiniProjects\ReviewInsight
   ```

2. **Create Virtual Environment**
   ```powershell
   python -m venv venv
   ```

3. **Activate Virtual Environment**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   
   **Note**: If you get an execution policy error, run:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

4. **Install Dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

5. **Download spaCy Language Model**
   ```powershell
   python -m spacy download en_core_web_sm
   ```

6. **Setup Jupyter Kernel (for Cursor IDE)**
   ```powershell
   .\setup_jupyter_kernel.ps1
   ```

## Dataset Setup

The original `amazon_us_reviews` dataset on Hugging Face uses a deprecated format. The project loads data from a local CSV file.

### Option 1: Use Sample Data (Quick Start)

Generate a sample dataset for testing:

```powershell
python scripts\download_sample_data.py
```

This creates `data/raw/amazon_reviews.csv` with 50,000 reviews and all star ratings (1-5) with realistic distribution.

### Option 2: Use Real Amazon Reviews Data

1. **Download from alternative sources:**
   - Kaggle: Search for "Amazon Product Reviews" datasets
   - UCI Machine Learning Repository: Amazon product data
   - Academic datasets with Amazon reviews

2. **Save as CSV** in `data/raw/amazon_reviews.csv` with these columns:
   - `review_body` or `review_text` - The review text
   - `star_rating` or `rating` or `stars` - Star rating (1-5)
   - `product_category` (optional) - Product category
   - `review_date` (optional) - Review date

3. The code will automatically load from CSV if available.

**Note**: The code maintains natural star rating distribution (not equal numbers per rating) and includes ALL ratings (1-5 stars), not just binary positive/negative.

## Running the Project

### Using Jupyter Notebooks in Cursor IDE (Recommended)

1. **Open any notebook** (e.g., `notebooks/01_eda.ipynb`)
2. **Select kernel**: Click "Select Kernel" → Choose "Python (ReviewInsight)"
3. **Run cells**: Use `Shift+Enter` to run cells sequentially
4. **Run notebooks in order**:
   - `01_eda.ipynb` - Exploratory Data Analysis
   - `02_feature_engineering.ipynb` - Feature creation
   - `03_modeling.ipynb` - Model training and evaluation

### Using Jupyter Notebook Server

1. **Activate virtual environment**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. **Start Jupyter**
   ```powershell
   jupyter notebook
   ```

3. **Open and run notebooks** in the order listed above

### Notebook Execution Order

1. **01_eda.ipynb**: Loads and preprocesses data, performs EDA
   - Output: Processed data saved to `data/processed/` and visualizations to `outputs/figures/`
   - Time: 10-30 minutes depending on dataset size

2. **02_feature_engineering.ipynb**: Creates features and labels
   - Output: Feature matrices and labels saved to `outputs/`
   - Time: 5-15 minutes

3. **03_modeling.ipynb**: Trains and evaluates models
   - Output: Trained models saved to `outputs/models/` and evaluation plots to `outputs/figures/`
   - Time: 15-45 minutes

## Models

### Logistic Regression
- Linear baseline model
- High interpretability through coefficient analysis
- Handles class imbalance with balanced class weights
- Exports human-readable model information

### XGBoost Classifier
- Nonlinear model capturing complex patterns
- Feature importance analysis
- Industry-relevant gradient boosting approach
- Exports human-readable model information

## Evaluation Metrics

Both models are evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Positive predictive value
- **Recall**: Sensitivity / True positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

## Output Files

### Data
- `data/raw/amazon_reviews_raw.parquet`: Raw sampled data
- `data/processed/amazon_reviews_processed.parquet`: Cleaned data

### Features
- `outputs/X_combined.npz`: Combined feature matrix
- `outputs/y.npy`: Binary labels
- `outputs/tfidf_vectorizer.pkl`: Fitted vectorizer
- `outputs/feature_names.pkl`: Feature names for interpretability

### Models
- `outputs/models/logistic_regression.pkl`: Trained Logistic Regression (binary)
- `outputs/models/xgboost.pkl`: Trained XGBoost model (binary)
- `outputs/models/logistic_regression_info.json`: Model information (JSON)
- `outputs/models/logistic_regression_info.txt`: Model information (human-readable)
- `outputs/models/xgboost_info.json`: Model information (JSON)
- `outputs/models/xgboost_info.txt`: Model information (human-readable)

### Visualizations
- `outputs/figures/rating_distribution.png`
- `outputs/figures/review_length_by_rating.png`
- `outputs/figures/category_analysis.png`
- `outputs/figures/temporal_trends.png`
- `outputs/figures/outlier_detection.png`
- `outputs/figures/roc_curves.png`
- `outputs/figures/lr_coefficients.png`
- `outputs/figures/xgb_importance.png`
- `outputs/figures/shap_summary_xgb.png` (optional)
- `outputs/figures/shap_bar_xgb.png` (optional)

## Technical Details

### Text Preprocessing
- Lowercasing
- Punctuation removal
- Whitespace normalization
- spaCy tokenization (tokenizer only, no parsing)

### Feature Engineering
- **TF-IDF**: Max 20,000 features, unigrams + bigrams
- **Numeric Features**: Review length, year, month
- **Labels**: Binary sentiment (drop 3-star reviews)

### Model Training
- Train/validation split: 80/20
- Stratified sampling to maintain class distribution
- Class imbalance handling (balanced weights / scale_pos_weight)

## Dependencies

See `requirements.txt` for complete list. Key packages:
- pandas, numpy
- scikit-learn
- xgboost
- matplotlib, seaborn
- jupyter
- datasets (Hugging Face)
- spacy
- shap (optional)

## Troubleshooting

### Kernel Not Found / ModuleNotFoundError
The kernel is not using your virtual environment.

**Solution:**
1. Run: `.\setup_jupyter_kernel.ps1`
2. In notebook, click "Select Kernel" → Choose "Python (ReviewInsight)"
3. Verify: Run `import sys; print(sys.executable)` - should show your venv path

### Dataset Loading Errors
The original Hugging Face dataset uses deprecated format.

**Solution:**
1. Use the sample data generator: `python scripts\download_sample_data.py`
2. Or download data manually and save as `data/raw/amazon_reviews.csv`

### Memory Errors
Try reducing the `n_samples` parameter in the first notebook (e.g., change 200000 to 100000).

## Notes

- The project uses random seeds (42) for reproducibility
- Large datasets may require significant memory
- SHAP analysis is optional and can be computationally expensive
- spaCy model must be downloaded separately
- All output files (models, features, figures) are excluded from git via `.gitignore`

## License

This is a mini project for educational purposes.

## Author

ReviewInsight - Product Review Analysis & Sentiment Modeling
