import streamlit as st
import pickle
import pandas as pd
from src.features.feature_extractor import extract_features_df

# Set up the visual page
st.set_page_config(page_title="Phishing URL Detector", page_icon="🛡️")
st.title("🛡️ Malicious URL & Phishing Detector")
st.write("Enter a URL below to check if it's Safe, Phishing, Malware, or Defacement.")

# Load the trained model
@st.cache_resource
def load_model():
    with open('src/models/artifacts/randomforest.pkl', 'rb') as f:
        return pickle.load(f)

try:
    art = load_model()
except FileNotFoundError:
    st.error("Model not found! Please run the training script first.")
    st.stop()

# Create the user interface
url_input = st.text_input("Enter URL:", placeholder="http://secure-login-update.com")

if st.button("Analyze URL"):
    if url_input:
        with st.spinner("Analyzing lexical features..."):
            try:
                # Extract features just like the demo script
                features_df = extract_features_df([url_input])
                X = features_df[art['feature_names']].fillna(0)
                
                # Make prediction
                pred_idx = int(art['model'].predict(X)[0])
                label = art['label_encoder'].inverse_transform([pred_idx])[0]
                
                # Show visual badges based on the result
                if label == 'benign':
                    st.success(f"✅ **SAFE (benign)** - This URL looks clean.")
                else:
                    st.error(f"🚨 **WARNING: {label.upper()}** - This URL is potentially malicious!")
                    
                # Show the "under the hood" data for recruiters
                st.write("### 🔍 Under the Hood: Extracted Features")
                st.write("These are the numerical features the Machine Learning model used to make its decision:")
                st.dataframe(features_df)
                
            except Exception as e:
                st.error(f"Error analyzing URL: {e}")
    else:
        st.warning("Please enter a URL first.")