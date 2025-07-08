import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Set Streamlit page config
st.set_page_config(
    page_title="Personality Predictor",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for dark theme overrides and responsiveness
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #fafafa;
}
[data-testid="stAppViewContainer"] {
    background-color: #0e1117;
}
h1, h2, h3 {
    color: #f39c12;
}
</style>
""", unsafe_allow_html=True)

# Load model and label encoder
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model.h5')
    encoder = LabelEncoder()
    encoder.classes_ = np.load('classes.npy', allow_pickle=True)
    return model, encoder

model, encoder = load_model()

# App layout
st.title("🧠 Personality Prediction")
st.write("Enter your details to predict your personality type.")

# Input fields
age = st.slider("Age", 13, 70, 25)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
openness = st.slider("Openness", 0.0, 1.0, 0.5)
neuroticism = st.slider("Neuroticism", 0.0, 1.0, 0.5)
conscientiousness = st.slider("Conscientiousness", 0.0, 1.0, 0.5)
agreeableness = st.slider("Agreeableness", 0.0, 1.0, 0.5)
extraversion = st.slider("Extraversion", 0.0, 1.0, 0.5)

if st.button("Predict Personality"):
    input_data = pd.DataFrame([[
        age, gender, openness, neuroticism,
        conscientiousness, agreeableness, extraversion
    ]], columns=[
        "age", "gender", "openness", "neuroticism",
        "conscientiousness", "agreeableness", "extraversion"
    ])

    input_data["gender"] = encoder.transform(input_data["gender"])
    prediction = model.predict(input_data)
    personality_type = np.argmax(prediction)
    st.success(f"Predicted Personality Type: **{personality_type}**")

