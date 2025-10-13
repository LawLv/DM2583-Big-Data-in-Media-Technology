import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("--- Start executing model training script (classification task) ---")

# --- 1. Load preprocessed data ---
try:
    df = pd.read_csv('wine_data_processed_with_tfidf.csv')
    print(f"Successfully loaded preprocessed dataset, shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'wine_data_processed_with_tfidf.csv' file not found. Please run data_preprocessing.py first.")
    exit()

# --- 2. Create price ranges (convert regression problem to classification) ---
price_bins = [0, 20, 40, 80, 200, float('inf')]
price_labels = ['1. Entry-level (<= $20)', '2. Quality ($21-$40)', '3. Premium ($41-$80)', '4. Luxury ($81-$200)', '5. Collectible (> $200)']

df['price_range'] = pd.cut(df['price'], bins=price_bins, labels=price_labels, right=True)

print("\nConverted 'price' into categorical 'price_range':")
print(df['price_range'].value_counts().sort_index())

# --- 3. Define features (X) and new target (y) ---
# Both 'price' and 'price_range' should not be used as features
X = df.drop(['price', 'price_range'], axis=1)
y = df['price_range']

# Identify categorical features for one-hot encoding
categorical_features = ['country', 'variety_simplified', 'winery_simplified']

# --- 4. Split data into training and testing sets ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y # stratify=y ensures balanced class distribution
)
print(f"\nData split: training set {X_train.shape[0]} rows, testing set {X_test.shape[0]} rows.")

# --- 5. Create and apply OneHotEncoder ---
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

print("\nFitting preprocessor (OneHotEncoder) on training data...")
X_train_processed = preprocessor.fit_transform(X_train)
print("Transforming training and testing sets...")
X_test_processed = preprocessor.transform(X_test)

print(f"One-hot encoding and preprocessing completed!")
print(f"Processed training feature shape: {X_train_processed.shape}")

# --- 6. Train Random Forest classification model ---
print("\n--- Training Random Forest classification model ---")
# n_jobs=-1 uses all available CPU cores to speed up training
# verbose=1 prints detailed progress during training
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=1)
model.fit(X_train_processed, y_train)
print("Model training completed.")

# --- 7. Evaluate on the test set ---
print("\n--- Evaluating on the test set ---")
predictions = model.predict(X_test_processed)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Overall test accuracy: {accuracy:.2%}")
print("\nThis means the model correctly predicts approximately {:.0f}% of wine price categories.".format(accuracy * 100))

# Print detailed classification report
print("\n--- Detailed classification report by price category ---")
# precision: proportion of predicted samples that are actually correct
# recall: proportion of true samples correctly predicted
# f1-score: harmonic mean of precision and recall
print(classification_report(y_test, predictions, target_names=price_labels))
