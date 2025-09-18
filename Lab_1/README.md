# DM2583-Big-Data-in-Media-Technology
https://docs.google.com/document/d/1deF3GDJ26sSi4Q0MlGMRlqhom3KpeBNmh4a3e1BWwOg/edit?tab=t.0
## Part 1: EDA and Preprocessing for Text Classification

### Stage 1: Exploratory Data Analysis (EDA)
This repository contains the Exploratory Data Analysis (EDA) part of Lab 1 for the course *DM2583 Big Data in Media Technology*.  

The EDA includes:
- Dataset overview (size, structure, missing values).  
- Distribution of target variable (balanced positive/negative classes).  
- Review length statistics and distribution.  
- Word frequency analysis (highlighting stopwords, HTML artifacts, mentions, hashtags).  
- Comparison of vocabulary in positive vs. negative reviews.  
- Identification of noise and proposed preprocessing steps.  
- Reflection on challenges and how EDA informs later preprocessing and modeling.  

Results from this analysis guide the upcoming **data cleaning, feature extraction, and model development** steps.  

### Stage 2: Data Cleaning and Feature Extraction

Based on the EDA findings, a robust preprocessing pipeline was developed and applied.

- **Cleaning Pipeline:**
    - Lowercased all text.
    - Removed HTML tags, mentions, hashtags, and other non-alphabetic characters.
    - Applied advanced **Part-of-Speech (POS) aware lemmatization** to accurately reduce words to their root forms.
    - Removed common English stopwords.
- **Feature Extraction:**
    - Converted the cleaned text into numerical features using the **TF-IDF** method.
    - The vocabulary was limited to the top 5,000 unigrams and bi-grams to create an efficient and effective feature set.

✅ **Status:** This phase is complete. The final, processed data required for modeling has been generated and saved. 

The output of the preprocessing stage is the file `tfidf_data_advanced.pkl`. This file contains all the necessary data for the modeling team to proceed.
