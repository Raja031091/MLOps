import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import json
import os

# --- Configuration ---
# NOTE: Replace <-----Hugging Face User ID -----> with your actual ID.
HF_USER_ID = "Sraja0310" 
MODEL_REPO_ID = f"{HF_USER_ID}/wellness-tourism-predictor"
MODEL_FILENAME = "best_model.joblib"
# The experiment log file from training contains the feature list used by the model
LOG_FILENAME = "experiment_log.json" 

# --- Model Loading ---
@st.cache_resource
def load_assets():
    """Downloads the model and feature metadata from Hugging Face Model Hub."""
    try:
        # 1. Download model file
        model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
        model = joblib.load(model_path)
        
        # 2. Download and load feature metadata (needed for One-Hot Encoding consistency)
        log_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=LOG_FILENAME)
        with open(log_path, 'r') as f:
            log_data = json.load(f)
            # The 'feature_names' were logged in the training script
            feature_names = log_data.get('feature_names', [])

        return model, feature_names
    except Exception as e:
        st.error(f"Error loading model assets from Hugging Face: {e}")
        st.stop() # Stop execution if assets fail to load

# Load model and required feature names once
model, all_expected_features = load_assets()

# --- Feature Definition (Must match training script's feature groups) ---
# These lists are used to define the Streamlit UI inputs
NUMERIC_FEATURES = [
    'Age', 'NumberOfPersonVisiting', 'PreferredPropertyStar', 'NumberOfTrips',
    'Passport', 'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome',
    'PitchSatisfactionScore', 'NumberOfFollowups', 'DurationOfPitch'
]

CATEGORICAL_FEATURES = [
    'TypeofContact', 'CityTier', 'Occupation', 'Gender', 'MaritalStatus',
    'Designation', 'ProductPitched'
]

# --- Streamlit UI for Wellness Tourism Prediction ---
st.title("✈️ Wellness Tourism Package Purchase Predictor")
st.markdown("Use this tool to predict the likelihood of a customer purchasing a wellness tourism package (`ProdTaken`).")

st.subheader("Customer and Interaction Details")

# Collect user input using columns for better layout
col1, col2 = st.columns(2)

with col1:
    # Numeric/Slider Inputs
    age = st.number_input("Age", min_value=18, max_value=90, value=40)
    monthly_income = st.number_input("Monthly Income", min_value=10000.0, value=50000.0, step=5000.0)
    num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    num_children = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
    num_trips = st.number_input("Number of Trips/Year", min_value=0, max_value=30, value=5)
    pref_stars = st.slider("Preferred Property Star", 3, 5, 4)

with col2:
    # Categorical/Selectbox Inputs
    contact = st.selectbox("Type of Contact", ['Self Inquiry', 'Company Invited', 'Unknown'])
    city_tier = st.selectbox("City Tier", ['Tier 1', 'Tier 2', 'Tier 3'])
    occupation = st.selectbox("Occupation", ['Salaried', 'Freelancer', 'Small Business', 'Retired', 'Unemployed'])
    gender = st.selectbox("Gender", ['Male', 'Female'])
    marital = st.selectbox("Marital Status", ['Married', 'Single', 'Divorced'])
    designation = st.selectbox("Designation", ['Manager', 'Analyst', 'Director', 'Associate', 'Executive'])
    product = st.selectbox("Product Pitched", ['Package A', 'Package B', 'Package C'])
    
    # Interaction Numeric Inputs
    pitch_score = st.slider("Pitch Satisfaction Score (1-5)", 1, 5, 4)
    num_followups = st.number_input("Number of Followups", min_value=1, max_value=15, value=3)
    pitch_duration = st.number_input("Duration of Pitch (mins)", min_value=5, max_value=120, value=30)
    
    # Binary Inputs
    passport = st.selectbox("Has Passport?", ["Yes", "No"])
    own_car = st.selectbox("Owns a Car?", ["Yes", "No"])

# Set the classification threshold from the training script
classification_threshold = 0.45 

# --- Inference Logic ---
if st.button("Predict Package Purchase"):
    
    # 1. Initialize input DataFrame with all features used during training, set to zero
    input_df = pd.DataFrame(0, index=[0], columns=all_expected_features)
    
    # 2. Map input values to the DataFrame
    raw_inputs = {
        'Age': age, 'MonthlyIncome': monthly_income, 'NumberOfPersonVisiting': num_persons, 
        'NumberOfChildrenVisiting': num_children, 'NumberOfTrips': num_trips, 
        'PreferredPropertyStar': pref_stars, 'PitchSatisfactionScore': pitch_score, 
        'NumberOfFollowups': num_followups, 'DurationOfPitch': pitch_duration,
        'Passport': 1 if passport == "Yes" else 0,
        'OwnCar': 1 if own_car == "Yes" else 0,
    }
    
    # Apply raw inputs
    for key, value in raw_inputs.items():
        if key in input_df.columns:
            input_df[key] = value

    # Apply one-hot encoding for categorical features
    categorical_inputs = {
        'TypeofContact': contact, 'CityTier': city_tier, 'Occupation': occupation, 
        'Gender': gender, 'MaritalStatus': marital, 'Designation': designation, 
        'ProductPitched': product
    }
    
    # Set the appropriate column to 1 for the selected category
    for col, value in categorical_inputs.items():
        # Column name format is expected to be 'feature_value' (due to OneHotEncoder default)
        feature_name = f'{col}_{value}'
        if feature_name in input_df.columns:
            input_df[feature_name] = 1

    # 3. Predict probability
    try:
        prediction_proba = model.predict_proba(input_df)[0, 1]
        prediction = (prediction_proba >= classification_threshold).astype(int)
        
        # 4. Display Result
        st.subheader("Prediction Result")
        
        if prediction == 1:
            st.success("✅ **LIKELY TO PURCHASE!** (Prediction: 1)")
            st.balloons()
            st.metric("Purchase Probability", f"{prediction_proba * 100:.2f}%")
        else:
            st.warning("❌ **Less Likely to Purchase** (Prediction: 0)")
            st.metric("Purchase Probability", f"{prediction_proba * 100:.2f}%")
            
        st.caption(f"Note: This uses a custom classification threshold of {classification_threshold}.")
        
    except Exception as e:
        st.error(f"Prediction failed. Input data structure might not match the model's required features. Error: {e}")
