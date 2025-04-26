"""
Breast Cancer Prediction Web App using Streamlit

This web application allows users to input medical features related to breast cancer
and predicts whether the tumor is malignant or benign using a trained machine learning model.

Author: Srikanth K
Date: April 2025
Model: Logistic Regression
Deployment: Hugging Face Spaces

"""

# Libraries
import streamlit as st
import numpy as np
import pandas as pd
import pickle 
import sklearn

# Load the model and scaler
with open('breast_cancer_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)






# Set page config and background color
st.set_page_config(page_title="🎀 Breast Cancer Detection", page_icon="💖")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #A9A9A9;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("""
    <style>
        [data-testid="stSidebar"] {
            background-color: #D2B48C;
        }
    </style>
""", unsafe_allow_html=True)
# Load model and scaler
with open('breast_cancer_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

st.sidebar.header("Input Tumor Features")

# Now, sliders explicitly for each feature:
radius_mean = st.sidebar.slider("Radius Mean", 6.0, 30.0, 14.1, 0.01)
texture_mean = st.sidebar.slider("Texture Mean", 9.0, 40.0, 19.4, 0.01)
perimeter_mean = st.sidebar.slider("Perimeter Mean", 40.0, 190.0, 91.7, 0.1)
area_mean = st.sidebar.slider("Area Mean", 140.0, 2500.0, 651.5, 1.0)
smoothness_mean = st.sidebar.slider("Smoothness Mean", 0.05, 0.16, 0.096, 0.001)
compactness_mean = st.sidebar.slider("Compactness Mean", 0.0, 0.5, 0.103, 0.001)
concavity_mean = st.sidebar.slider("Concavity Mean", 0.0, 0.5, 0.087, 0.001)
concave_points_mean = st.sidebar.slider("Concave Points Mean", 0.0, 0.2, 0.048, 0.001)
symmetry_mean = st.sidebar.slider("Symmetry Mean", 0.1, 0.5, 0.181, 0.001)

radius_se = st.sidebar.slider("Radius SE", 0.0, 2.0, 0.40, 0.001)
perimeter_se = st.sidebar.slider("Perimeter SE", 0.0, 10.0, 2.82, 0.01)
area_se = st.sidebar.slider("Area SE", 0.0, 542.2, 39.43, 0.1)
compactness_se = st.sidebar.slider("Compactness SE", 0.0, 0.13, 0.025, 0.001)
concavity_se = st.sidebar.slider("Concavity SE", 0.0, 0.1, 0.031, 0.001)
concave_points_se = st.sidebar.slider("Concave Points SE", 0.0, 0.05, 0.012, 0.001)
fractal_dimension_se = st.sidebar.slider("Fractal Dimension SE", 0.0, 0.02, 0.0037, 0.0001)

radius_worst = st.sidebar.slider("Radius Worst", 7.0, 40.0, 16.18, 0.01)
texture_worst = st.sidebar.slider("Texture Worst", 12.0, 50.0, 25.79, 0.01)
perimeter_worst = st.sidebar.slider("Perimeter Worst", 50.0, 250.0, 106.66, 0.1)
area_worst = st.sidebar.slider("Area Worst", 150.0, 4500.0, 868.23, 1.0)
smoothness_worst = st.sidebar.slider("Smoothness Worst", 0.06, 0.2, 0.132, 0.001)
compactness_worst = st.sidebar.slider("Compactness Worst", 0.0, 1.0, 0.253, 0.001)
concavity_worst = st.sidebar.slider("Concavity Worst", 0.0, 1.0, 0.269, 0.001)
concave_points_worst = st.sidebar.slider("Concave Points Worst", 0.0, 0.3, 0.113, 0.001)
symmetry_worst = st.sidebar.slider("Symmetry Worst", 0.1, 0.7, 0.29, 0.001)

# Create input dataframe for prediction
input_dict = {
    "radius_mean": radius_mean,
    "texture_mean": texture_mean,
    "perimeter_mean": perimeter_mean,
    "area_mean": area_mean,
    "smoothness_mean": smoothness_mean,
    "compactness_mean": compactness_mean,
    "concavity_mean": concavity_mean,
    "concave points_mean": concave_points_mean,
    "symmetry_mean": symmetry_mean,
    "radius_se": radius_se,
    "perimeter_se": perimeter_se,
    "area_se": area_se,
    "compactness_se": compactness_se,
    "concavity_se": concavity_se,
    "concave points_se": concave_points_se,
    "fractal_dimension_se": fractal_dimension_se,
    "radius_worst": radius_worst,
    "texture_worst": texture_worst,
    "perimeter_worst": perimeter_worst,
    "area_worst": area_worst,
    "smoothness_worst": smoothness_worst,
    "compactness_worst": compactness_worst,
    "concavity_worst": concavity_worst,
    "concave points_worst": concave_points_worst,
    "symmetry_worst": symmetry_worst
}

input_df = pd.DataFrame([input_dict])

# Scale input
input_scaled = scaler.transform(input_df)

# Title and instructions
st.title("🎀 Breast Cancer Detection App 💖")
st.write("Adjust the sliders on the left, then click **Predict** to see the diagnosis.")


# Predict button
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    if prediction == 0:
        st.success(f"Prediction: Benign 😊")
        st.write(f"Probability of Benign: {proba[0]*100:.2f}%")
        st.write(f"Probability of Malignant: {proba[1]*100:.2f}%")
    else:
        st.error(f"Prediction: Malignant 😟")
        st.write(f"Probability of Malignant: {proba[1]*100:.2f}%")
        st.write(f"Probability of Benign: {proba[0]*100:.2f}%")
else:
    st.info("Set the tumor features and click **Predict**.")

st.markdown("---")

