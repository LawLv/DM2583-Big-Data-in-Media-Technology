import pandas as pd
import numpy as np
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

print("--- Start executing data cleaning and feature engineering script (V2: Added TF-IDF saving) ---")

# --- 1. Load Data ---
try:
    # Load dataset
    df = pd.read_csv('winemag-data-130k-v2.csv')
    print(f"Successfully loaded dataset, original shape: {df.shape}")
except FileNotFoundError:
    print("Error: 'winemag-data-130k-v2.csv' file not found. Please make sure the file is in the correct directory.")
    exit()

# --- 2. Initial Data Cleaning ---
print("\n--- Stage 1: Initial Data Cleaning ---")

# Drop unnecessary columns
cols_to_drop = ['Unnamed: 0', 'designation', 'region_2', 'taster_twitter_handle']
df.drop(columns=cols_to_drop, inplace=True)
print(f"Dropped columns: {cols_to_drop}")

# Handle target variable 'price', remove rows with missing prices
initial_rows = len(df)
df.dropna(subset=['price'], inplace=True)
print(f"Processed target variable 'price', removed {initial_rows - len(df)} rows with missing price.")

# Fill missing values for key categorical features
for col in ['country', 'province', 'region_1', 'variety', 'taster_name', 'winery']:
    df[col].fillna('Unknown', inplace=True)
print("Filled missing values for key categorical features.")


# --- 3. Feature Engineering ---
print("\n--- Stage 2: Feature Engineering ---")

# (A) Extract year from 'title'
def extract_year(text):
    """Extract four-digit year from text using regular expression"""
    match = re.search(r'\b(19|20)\d{2}\b', str(text))
    if match:
        return int(match.group(0))
    return np.nan

df['year'] = df['title'].apply(extract_year)

# Fill missing extracted years with median year
median_year = df['year'].median()
df['year'].fillna(median_year, inplace=True)
df['year'] = df['year'].astype(int)
print(f"Extracted year from 'title' and filled missing values with median year {int(median_year)}.")


# (B) Simplify high-cardinality categorical features
def simplify_categorical_feature(df, column_name, top_n=30):
    """Keep top N most frequent categories, others replaced with 'Other'"""
    top_categories = df[column_name].value_counts().nlargest(top_n).index
    new_column_name = f"{column_name}_simplified"
    df[new_column_name] = df[column_name].where(df[column_name].isin(top_categories), 'Other')
    print(f"Simplified '{column_name}' feature, kept Top {top_n}.")
    return df

df = simplify_categorical_feature(df, 'variety', top_n=30)
df = simplify_categorical_feature(df, 'winery', top_n=50) # More wineries, keep top 50


# --- 4. Text Vectorization (TF-IDF) ---
print("\n--- Stage 3: Text Vectorization using TF-IDF ---")

# Ensure 'description' column has no missing values
df['description'].fillna('', inplace=True)

# Initialize TfidfVectorizer, optimized with max_features=2000
tfidf = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_features=2000,
    min_df=5
)

# Fit and transform 'description' column
tfidf_matrix = tfidf.fit_transform(df['description'])
print(f"TF-IDF matrix generated, shape: {tfidf_matrix.shape}")

# **New Step**: Save trained TF-IDF transformer (needed for application phase)
joblib.dump(tfidf, 'tfidf_vectorizer.joblib')
print("Saved TF-IDF vectorizer to 'tfidf_vectorizer.joblib'")


# Convert sparse matrix to DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())
# Add prefix to TF-IDF column names to avoid conflicts
tfidf_df = tfidf_df.add_prefix('tfidf_')

# Reset index before merging
df.reset_index(drop=True, inplace=True)

# Merge TF-IDF features with original data
df_final = pd.concat([df, tfidf_df], axis=1)
print(f"Final dataset shape after merging TF-IDF features: {df_final.shape}")


# --- 5. Final Cleaning and Saving ---
print("\n--- Stage 4: Final Cleaning and Saving ---")

# Drop original processed columns
df_final.drop(columns=['description', 'title', 'variety', 'winery', 'region_1', 'province', 'taster_name'], inplace=True)
print("Dropped original text and high-cardinality columns.")

# Save processed data
output_filename = 'wine_data_processed_with_tfidf.csv'
df_final.to_csv(output_filename, index=False)

print(f"\nProcessing complete! Final dataset shape: {df_final.shape}")
print(f"Cleaned and feature-engineered data saved to '{output_filename}'")
