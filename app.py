import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set page config
st.set_page_config(page_title="🧠 Personality Predictor", layout="centered")

# Load model and encoders
@st.cache_resource
def load_model():
    model = joblib.load('rf_personality_model.pkl')
    label_classes = np.load('label_classes.npy', allow_pickle=True)
    feature_columns = joblib.load('feature_columns.pkl')
    return model, label_classes, feature_columns

model, label_classes, feature_columns = load_model()

# Page title
st.title("🧠 Personality Predictor")
st.markdown("Fill out the information below to predict your personality type.")

# User inputs
user_input = {}
user_input['time_spent_alone'] = st.slider("⏱️ Time Spent Alone (hours/day)", 0, 24, 5)
user_input['stage_fear'] = st.selectbox("😨 Do you have stage fear?", ['Yes', 'No'])
user_input['social_event_attendance'] = st.selectbox("🎉 Do you attend social events?", ['Yes', 'No'])
user_input['going_outside'] = st.selectbox("🚶‍♂️ Do you like going outside?", ['Yes', 'No'])
user_input['drained_after_socializing'] = st.selectbox("💤 Do you feel drained after socializing?", ['Yes', 'No'])
user_input['friends_circle_size'] = st.slider("👥 Friends Circle Size", 0, 100, 10)
user_input['post_frequency'] = st.slider("📱 Social Media Post Frequency (posts/week)", 0, 50, 3)

# Predict button
if st.button("🔍 Predict Personality"):
    # Convert input to dataframe
    input_df = pd.DataFrame([user_input])

    # One-hot encode the input (same logic as training)
    input_encoded = pd.get_dummies(input_df)
    
    # Reindex to match training columns
    input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

    # Predict
    prediction = model.predict(input_encoded)[0]
    predicted_label = label_classes[prediction]

    st.success(f"🧠 Predicted Personality Type: **{predicted_label}**")
