# app.py

import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Page config
st.set_page_config(
    page_title="Personality Predictor",
    page_icon="🧠",
    layout="centered",
)

# Custom styling
st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        padding: 0.6em 1.5em;
        border-radius: 10px;
        font-weight: bold;
        border: none;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .stMarkdown h1 {
        font-size: 2.5rem;
        color: #4A4A4A;
    }
    .info-box {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)

# Load model and preprocessing tools
model = load_model('deep_personality_model.keras')
scaler = joblib.load('scaler.pkl')
target_encoder = joblib.load('target_encoder.pkl')

# App Title
st.markdown("<h1 style='text-align: center;'>🧠 Personality Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict whether someone is an <b>Introvert</b> or <b>Extrovert</b> based on personality traits</p>", unsafe_allow_html=True)
st.markdown("---")

# Inputs
feature_names = ['Age', 'Gender', 'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness']
user_input = []

with st.form("prediction_form"):
    st.markdown("<div class='info-box'>", unsafe_allow_html=True)

    cols = st.columns(2)
    for i, feature in enumerate(feature_names):
        with cols[i % 2]:
            val = st.slider(
                label=f"{feature}",
                min_value=0.0, max_value=100.0, value=50.0, step=1.0
            )
            user_input.append(val)

    submitted = st.form_submit_button("🔍 Predict Personality")
    st.markdown("</div>", unsafe_allow_html=True)

# Prediction
if submitted:
    input_array = np.array([user_input])
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    predicted_class = np.argmax(prediction)
    predicted_label = target_encoder.inverse_transform([predicted_class])[0]
    confidence = np.max(prediction) * 100

    st.markdown("<br>", unsafe_allow_html=True)
    st.success(f"🧬 **Prediction: {predicted_label}**")
    st.progress(confidence / 100)
    st.info(f"🔒 Confidence: **{confidence:.2f}%**")

    # Optional: personality traits explanation (optional)
    st.markdown("---")
    st.markdown("💡 **Interpretation Tip:**\nThe model is trained on various traits like Openness and Extraversion. Try adjusting the sliders to see how predictions change.")

