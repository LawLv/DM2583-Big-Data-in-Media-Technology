import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

print("--- Starting Model Training Script (V6: Memory Fix - GPU Focus) ---")

# --- 1. Load Preprocessed Data ---
try:
    df = pd.read_csv('wine_data_processed_with_tfidf.csv')
    print(f"Successfully loaded preprocessed dataset, total rows: {df.shape[0]}")
except FileNotFoundError:
    print("Error: 'wine_data_processed_with_tfidf.csv' file not found. Please run data_preprocessing.py first.")
    exit()

# --- 2. Filter Data, Focus on Mainstream Price Range ---
df_focused = df[df['price'] <= 80].copy()
print(f"Filtered data to focus on price <= $80, remaining rows: {df_focused.shape[0]}")


# --- 3. Create New, Simplified Price Bins ---
price_bins = [0, 20, 40, 80]
price_labels = ['1. Entry Level (<= $20)', '2. Quality Choice ($21-$40)', '3. Premium Selection ($41-$80)']
df_focused['price_range'] = pd.cut(df_focused['price'], bins=price_bins, labels=price_labels, right=True)

print("\n'price' has been converted into 3 new classification ranges:")
print(df_focused['price_range'].value_counts().sort_index())

# --- 4. Define Features (X) and Target (y) ---
X = df_focused.drop(['price', 'price_range'], axis=1)
y = df_focused['price_range']
categorical_features = ['country', 'variety_simplified', 'winery_simplified']

# --- 5. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nData split: Training set {X_train.shape[0]} rows, Test set {X_test.shape[0]} rows.")

# --- 6. One-Hot Encoding ---
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)],
    remainder='passthrough'
)
# **Memory Optimization**: Convert to float32 to halve memory usage
X_train_processed = preprocessor.fit_transform(X_train).astype(np.float32)
X_test_processed = preprocessor.transform(X_test).astype(np.float32)
print(f"One-Hot Encoding complete! Processed training set feature shape: {X_train_processed.shape}")

# --- 7. Encode Target Variable ---
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# --- 8. V6 Core Improvement: Hyperparameter Tuning ---
print("\n--- Starting Hyperparameter Tuning ---")

# Define parameter search space
param_dist = {
    'n_estimators': [100, 150, 200],
    'learning_rate': [0.05, 0.1, 0.15],
    'max_depth': [5, 7, 9],
    'colsample_bytree': [0.7, 0.8, 0.9], # Feature subsample ratio per tree
    'subsample': [0.7, 0.8, 0.9]       # Data subsample ratio per tree
}

# Initialize XGBoost classifier
xgb_classifier = xgb.XGBClassifier(
    objective='multi:softmax',
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42,
    # **GPU Option**: Enable CUDA acceleration
    device='cuda',
)

# Use Randomized Search for tuning
random_search = RandomizedSearchCV(
    xgb_classifier,
    param_distributions=param_dist,
    n_iter=20,
    cv=3,
    verbose=2,
    random_state=42,
    # **Memory Fix**: Set n_jobs to 1, disable CPU parallelism to avoid memory errors
    n_jobs=1
)

# Execute search on training data
random_search.fit(X_train_processed, y_train_encoded)

print("\nHyperparameter tuning complete!")
print(f"The best parameters found are: {random_search.best_params_}")

# Get and use the best model
best_model = random_search.best_estimator_

# --- 9. Predict and Evaluate on Test Set ---
print("\n--- Evaluating with the Best Model ---")
predictions_encoded = best_model.predict(X_test_processed)
predictions = le.inverse_transform(predictions_encoded)

accuracy = accuracy_score(y_test, predictions)
print(f"Overall accuracy of the tuned model on the test set: {accuracy:.2%}")
print("\n--- Detailed Classification Report by Price Range (Tuned Model) ---")
print(classification_report(y_test, predictions, target_names=price_labels))

# --- 10. Save Best Model and Preprocessor ---
print("\n--- Saving the Best Model and Preprocessor ---")
joblib.dump(best_model, 'wine_price_model_tuned.joblib')
joblib.dump(preprocessor, 'preprocessor_focused.joblib') # Preprocessor remains the same
joblib.dump(le, 'label_encoder_focused.joblib')      # Encoder remains the same
print("Tuned best model, preprocessor, and encoder successfully saved!")
