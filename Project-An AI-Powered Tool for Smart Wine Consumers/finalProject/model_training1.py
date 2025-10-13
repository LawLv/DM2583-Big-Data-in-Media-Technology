import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

print("--- Start executing model training script (V3: XGBoost + Manual Weight Adjustment) ---")

# --- 1. Load preprocessed data ---
try:
    df = pd.read_csv('wine_data_processed_with_tfidf.csv')
    print(f"Successfully loaded preprocessed dataset, shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'wine_data_processed_with_tfidf.csv' file not found. Please run data_preprocessing.py first.")
    exit()

# --- 2. Create price range ---
price_bins = [0, 20, 40, 80, 200, float('inf')]
price_labels = ['1. Entry-level (<= $20)', '2. Quality ($21-$40)', '3. Premium ($41-$80)', '4. Luxury ($81-$200)', '5. Collectible (> $200)']
df['price_range'] = pd.cut(df['price'], bins=price_bins, labels=price_labels, right=True)

print("\nConverted 'price' into categorical 'price_range':")
print(df['price_range'].value_counts().sort_index())

# --- 3. Define features (X) and target (y) ---
X = df.drop(['price', 'price_range'], axis=1)
y = df['price_range']
categorical_features = ['country', 'variety_simplified', 'winery_simplified']

# --- 4. Split data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nData split: training set {X_train.shape[0]} rows, testing set {X_test.shape[0]} rows.")

# --- 5. One-hot encoding ---
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)],
    remainder='passthrough'
)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
print(f"One-hot encoding completed! Processed training feature shape: {X_train_processed.shape}")

# --- 6. Handle class imbalance (manual weight adjustment) ---
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# **V3 Improvement: Manual weight tuning instead of fully automatic calculation**
# Goal: Maintain focus on rare classes while reducing over-penalization for better balance.
manual_weights = {
    '1. Entry-level (<= $20)': 0.6,
    '2. Quality ($21-$40)': 0.7,
    '3. Premium ($41-$80)': 1.0,
    '4. Luxury ($81-$200)': 5.0,     # previously 4.26, slightly increased
    '5. Collectible (> $200)': 20.0   # previously 35.45, significantly reduced
}
# Convert text label weights into numerical label format required by XGBoost
d_class_weights = {i: manual_weights[le.classes_[i]] for i in range(len(le.classes_))}

print("\nManually set class weights for better balance:")
print({le.classes_[i]: weight for i, weight in d_class_weights.items()})

# --- 7. Train XGBoost model ---
print("\n--- Training XGBoost classification model ---")
sample_weights = np.array([d_class_weights[label] for label in y_train_encoded])

model = xgb.XGBClassifier(
    objective='multi:softmax',
    n_estimators=150,
    learning_rate=0.1,
    max_depth=7,
    use_label_encoder=False,
    eval_metric='mlogloss',
    n_jobs=-1,
    random_state=42
)

model.fit(X_train_processed, y_train_encoded, sample_weight=sample_weights)
print("Model training completed.")

# --- 8. Evaluate on the test set ---
print("\n--- Evaluating on the test set ---")
predictions_encoded = model.predict(X_test_processed)
predictions = le.inverse_transform(predictions_encoded)

accuracy = accuracy_score(y_test, predictions)
print(f"Overall test accuracy: {accuracy:.2%}")
print("\n--- Detailed evaluation report by price category (XGBoost with Manual Weights) ---")
print(classification_report(y_test, predictions, target_names=price_labels))

# --- 9. Save model and preprocessors ---
print("\n--- Saving model and preprocessors ---")
joblib.dump(model, 'wine_price_model.joblib')
joblib.dump(preprocessor, 'preprocessor.joblib')
joblib.dump(le, 'label_encoder.joblib')
print("Model, preprocessor, and label encoder successfully saved!")
