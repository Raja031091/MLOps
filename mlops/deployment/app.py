import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import json
import os

# --- Configuration ---
# NOTE: Replace <-----Hugging Face User ID -----> with your actual ID.
# Ensure this ID is correct!
HF_USER_ID = "Sraja0310" 
MODEL_REPO_ID = f"{HF_USER_ID}/churn-model"
MODEL_FILENAME = "best_churn_model_v1.joblib"
LOG_FILENAME = "experiment_log.json"

# --- Model Loading ---
@st.cache_resource
def load_assets():
    """Downloads the model and feature metadata from Hugging Face Model Hub."""
    try:
        # 1. Download model file
        model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
        model = joblib.load(model_path)

        # 2. Download and load feature metadata 
        log_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=LOG_FILENAME)
        with open(log_path, 'r') as f:
            log_data = json.load(f)
            feature_names = log_data.get('feature_names', [])

        return model, feature_names
        
    except Exception as e:
        # The exception block runs when the download or loading fails.
        # This is where your code was previously returning None implicitly.
        
        # Display a clear error message to the user
        st.error(f"Error loading critical assets from Hugging Face Model Hub (Repo: {MODEL_REPO_ID}).")
        st.error(f"Details: {e}")
        
        # We explicitly halt the script execution here.
        st.stop() 
        
        # Failsafe return (should not be reached if st.stop() works correctly)
        return None, []


# Load model and required feature names once
# If load_assets fails, it calls st.stop() and the script should halt here.
model, all_expected_features = load_assets()


# --- WARNING CHECK (Crucial for diagnosis) ---
# If you are still getting the error, it means the model or log file 
# is missing from the Hugging Face repository or the repository ID is wrong.
if not all_expected_features:
    st.error("FATAL: Feature metadata not found in the downloaded log. The prediction logic cannot safely proceed.")
    st.stop()
    

# --- Feature Definition (Rest of code remains the same) ---
# ... (all your existing code for UI and inference follows)

# ... (I've truncated the rest of the code for brevity as it's the same)
