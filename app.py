import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load saved model and preprocessing objects
model = load_model('deep_personality_model.keras')
scaler = joblib.load('scaler.pkl')
target_encoder = joblib.load('target_encoder.pkl')

# Page config
st.set_page_config(
    page_title="Personality Classifier",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={"About": "A personality predictor using deep learning."}
)

# Custom dark style
st.markdown("""
    <style>
        body {
            background-color: #0E1117;
            color: white;
        }
        .stApp {
            background-color: #0E1117;
        }
        h1, h2, h3, .stButton>button {
            color: #F3F4F6;
        }
        .stTextInput>div>div>input, .stNumberInput>div>input {
            background-color: #1C1F26;
            color: white;
        }
        .stButton>button {
            background-color: #2563EB;
            color: white;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #1E40AF;
        }
    </style>
""", unsafe_allow_html=True)

# App title
st.title("🧠 Personality Prediction App")
st.subheader("Predict if someone is an Introvert or Extrovert")

# Input fields — make sure these match your features
feature_names = ['Age', 'Gender', 'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness']
inputs = []

with st.form(key="input_form"):
    col1, col2 = st.columns(2)
    with col1:
        inputs.append(st.number_input("Age", 10, 100, value=25))
        inputs.append(st.selectbox("Gender", options=['Male', 'Female'], index=0))
        inputs.append(st.slider("Openness", 1, 10, value=5))
    with col2:
        inputs.append(st.slider("Conscientiousness", 1, 10, value=5))
        inputs.append(st.slider("Extraversion", 1, 10, value=5))
        inputs.append(st.slider("Agreeableness", 1, 10, value=5))

    submit = st.form_submit_button("Predict")

# Map gender to numeric
gender_map = {'Male': 0, 'Female': 1}
if submit:
    try:
        # Prepare input
        input_vals = np.array([
            inputs[0],                          # Age
            gender_map[inputs[1]],             # Gender
            inputs[2], inputs[3], inputs[4], inputs[5]  # Traits
        ]).reshape(1, -1)

        input_scaled = scaler.transform(input_vals)
        prediction_probs = model.predict(input_scaled)
        predicted_class = np.argmax(prediction_probs)
        predicted_label = target_encoder.inverse_transform([predicted_class])[0]
        confidence = np.max(prediction_probs)

        st.markdown("---")
        st.markdown(f"### 🧬 Predicted Personality: **{predicted_label}**")
        st.markdown(f"**🔮 Confidence: {confidence * 100:.2f}%**")
        st.balloons()

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
