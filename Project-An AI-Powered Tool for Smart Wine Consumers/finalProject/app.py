import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import google.generativeai as genai
import asyncio
import json
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="Wine Price Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Securely Configure Gemini API ---
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=gemini_api_key)
    # **Feature Upgrade**: Switch to a multimodal model that supports image recognition
    model = genai.GenerativeModel(model_name="gemini-2.5-flash")
    llm_enabled = True
except (KeyError, FileNotFoundError):
    llm_enabled = False
    st.warning("Gemini API key not configured. LLM features will be unavailable. Please configure GEMINI_API_KEY in .streamlit/secrets.toml.")

# --- Load Models and Preprocessors ---
@st.cache_resource
def load_assets():
    """Load all necessary model and preprocessor files."""
    try:
        assets = {
            "model": joblib.load('wine_price_model_tuned.joblib'),
            "preprocessor": joblib.load('preprocessor_focused.joblib'),
            "label_encoder": joblib.load('label_encoder_focused.joblib'),
            "tfidf_vectorizer": joblib.load('tfidf_vectorizer.joblib')
        }
        return assets
    except FileNotFoundError:
        st.error("Error: Missing necessary model files (.joblib). Please ensure you have successfully run all preprocessing and training scripts.")
        return None

assets = load_assets()

# --- Translation dictionary for model output ---
# The model's label_encoder was trained on Chinese labels. This dictionary translates the output for the UI.
TRANSLATION_MAP = {
    "1. 入门级 (<= $20)": "1. Entry-level (<= $20)",
    "2. 品质之选 ($21-$40)": "2. Quality ($21-$40)",
    "3. 优质佳酿 ($41-$80)": "3. Premium ($41-$80)"
}


# --- Enhanced LLM Function (Supports Image or Text) ---
async def fetch_all_wine_details_with_gemini(image=None, wine_name=None, hints=None):
    """
    Calls the real Gemini API, analyzes images or text, searches the web, and extracts all required features for the model.
    """
    st.info("🤖 Dispatching Gemini AI Assistant to search for wine information online...")
    
    # Build different prompts based on input type
    if image:
        input_content = ["Please identify the wine label in this image and use web search to answer the following questions.", image]
        search_context = "the wine identified from this image"
    elif wine_name:
        # Convert hints dictionary to a more readable string
        hints_str = ", ".join([f"{k}: {v}" for k, v in hints.items()]) if hints else "None"
        input_content = [f"The user wants to find information about: '{wine_name}'.\nThey have provided the following optional hints to help you: {hints_str}"]
        search_context = f"about '{wine_name}'"
    else:
        return None

    prompt = (
        "Act as an expert sommelier's assistant. Your task is to find comprehensive information about a specific wine from the web.\n"
        f"Context: {search_context}\n"
        "Please search the web, prioritizing official reviews from sites like wineenthusiast.com for 'points' and 'description', but use other reliable sources (like winery websites, vivino, etc.) for other details.\n"
        "You must find and extract the following six pieces of information:\n"
        "- 'points': The rating score (integer, e.g., 92).\n"
        "- 'description': The tasting note or review (string).\n"
        "- 'country': The country of origin (string, e.g., 'US').\n"
        "- 'year': The vintage year (integer, e.g., 2018).\n"
        "- 'variety': The primary grape variety (string, e.g., 'Pinot Noir').\n"
        "- 'winery': The name of the winery (string, e.g., 'Williams Selyem').\n"
        "Return the result ONLY in a strict JSON format. If a piece of information cannot be found, use the value 'Unknown'.\n"
        "Example of a perfect output: {\"points\": 91, \"description\": \"...\", \"country\": \"US\", \"year\": 2017, \"variety\": \"Merlot\", \"winery\": \"Duckhorn\"}\n"
        "If you cannot find the wine at all, return exactly this JSON: {\"error\": \"Wine not found\"}"
    )
    
    # Combine the base prompt with the input content
    full_prompt = [prompt] + input_content

    try:
        response = await asyncio.to_thread(model.generate_content, full_prompt)
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        result = json.loads(json_text)

        if "error" in result:
            st.warning("🤖 Gemini could not find the wine you searched for on the web.")
            return None
        
        st.success("🤖 Gemini successfully found the wine's details!")
        return result

    except Exception as e:
        st.error(f"An error occurred while calling the Gemini API: {e}")
        return None

# --- Prediction Function (Handles 'Unknown' values) ---
def predict_price_range(data, assets):
    if not assets: return "Failed to load model files, cannot predict."
    try:
        tfidf_cols = assets["tfidf_vectorizer"].get_feature_names_out()
        tfidf_cols_prefixed = [f'tfidf_{col}' for col in tfidf_cols]
        preprocessor_features = assets["preprocessor"].feature_names_in_
        new_df = pd.DataFrame(columns=preprocessor_features).fillna(0)
        
        # **Key Fix**: If Gemini returns 'Unknown', fill with median values
        new_df.loc[0, 'points'] = data.get('points', 88) if data.get('points', 'Unknown') != 'Unknown' else 88
        new_df.loc[0, 'year'] = data.get('year', 2012) if data.get('year', 'Unknown') != 'Unknown' else 2012
        
        new_df.loc[0, 'country'] = data.get('country', 'Unknown')
        new_df.loc[0, 'variety_simplified'] = data.get('variety', 'Unknown')
        new_df.loc[0, 'winery_simplified'] = data.get('winery', 'Unknown')
        
        if data.get('description') and data.get('description') != 'Unknown':
            tfidf_matrix = assets["tfidf_vectorizer"].transform([data['description']])
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_cols_prefixed)
            for col in tfidf_df.columns:
                if col in new_df.columns: new_df.loc[0, col] = tfidf_df.loc[0, col]
        
        processed_data = assets["preprocessor"].transform(new_df)
        prediction_encoded = assets["model"].predict(processed_data)
        prediction_label = assets["label_encoder"].inverse_transform(prediction_encoded)
        return prediction_label[0]
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return "Prediction Failed"

# --- Streamlit UI Layout (V4 - Smart Version) ---
st.title("🍷 Smart Wine Price Range Predictor (V4)")
st.markdown("Upload a wine label image or enter a wine name. The **AI Assistant (Gemini)** will automatically find the required information, and the **XGBoost Price Analyst** will provide a professional evaluation.")

# --- Sidebar Inputs ---
st.sidebar.header("Please Provide Wine Clues")
st.sidebar.markdown("---")

# Use tabs for user to choose input method
input_method = st.sidebar.radio("Select your input method:", ('Upload Image', 'Enter Text'))

wine_name_input = None
uploaded_file = None

if input_method == 'Upload Image':
    uploaded_file = st.sidebar.file_uploader(
        "**Upload Wine Label Image**", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear photo of the front wine label"
    )
else:
    wine_name_input = st.sidebar.text_input(
        "**Wine Name (Required)**", 
        "Duckhorn Merlot 2017",
        help="Please enter the most complete wine name possible, including winery, variety, and year."
    )
    with st.sidebar.expander("Optional: Provide more clues to improve search accuracy"):
        country_hint = st.selectbox("Country", ['(Any)', 'US', 'France', 'Italy', 'Spain', 'Portugal', 'Chile', 'Argentina', 'Germany', 'Austria', 'Australia', 'New Zealand', 'South Africa', 'Other'])
        winery_hint = st.text_input("Winery")

# --- Main Page ---
if st.sidebar.button("🤖 Smart Find & Predict Price", type="primary", use_container_width=True, disabled=not llm_enabled):
    llm_results = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.sidebar.image(image, caption='Uploaded Image', use_column_width=True)
        llm_results = asyncio.run(fetch_all_wine_details_with_gemini(image=image))

    elif wine_name_input and wine_name_input.strip():
        hints = {
            "country": country_hint if country_hint != '(Any)' else None,
            "winery": winery_hint if winery_hint.strip() else None,
        }
        hints = {k: v for k, v in hints.items() if v is not None}
        llm_results = asyncio.run(fetch_all_wine_details_with_gemini(wine_name=wine_name_input, hints=hints))
        
    else:
        st.sidebar.error("Please provide input based on your selected method!")

    if llm_results and assets:
        st.markdown("---")
        st.subheader("🤖 Here is the information found by the AI Assistant:")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Points", llm_results.get('points', 'N/A'))
        col2.metric("Year", llm_results.get('year', 'N/A'))
        col3.metric("Country", llm_results.get('country', 'N/A'))
        st.text_input("Winery", llm_results.get('winery', 'N/A'), disabled=True)
        st.text_input("Variety", llm_results.get('variety', 'N/A'), disabled=True)
        st.text_area("Description", llm_results.get('description', 'N/A'), height=150, disabled=True)
        
        st.markdown("---")
        st.subheader("📈 Price Analyst's Evaluation Result:")

        with st.spinner('Price Analyst is evaluating...'):
            time.sleep(1)
            prediction = predict_price_range(llm_results, assets)
            
            # **UI Fix**: Translate the prediction output to English before displaying
            prediction_english = TRANSLATION_MAP.get(prediction, prediction) # Fallback to original if not found
            
            st.metric(
                label="Predicted Price Range",
                value=prediction_english.split('(')[0].strip()
            )
            st.success(f"**Detailed Range:** {prediction_english.split('(')[1][:-1]}")

st.sidebar.markdown("---")
st.sidebar.markdown("This is a demo project using a real Gemini API.")

