import streamlit as st
import joblib
import os

# ----------------------------------
# Compute absolute path of this app file
# ----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model is inside: application/src/models/
MODEL_PATH = os.path.join(BASE_DIR, "models", "emotion_classifier_pipe_lr.pkl")

st.write("Model path inside container:", MODEL_PATH)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model not found at {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()

# ----------------------------------
# Streamlit UI
# ----------------------------------
st.title("Emotion Detection App")
st.write("Enter text below and the model will predict the emotion.")

user_text = st.text_area("Enter your text:", height=150)

if st.button("Predict Emotion"):

    if not user_text.strip():
        st.warning("Please type something before predicting.")
    else:
        # Predict emotion
        prediction = model.predict([user_text])[0]

        # Predict probabilities
        proba = model.predict_proba([user_text])[0]
        classes = model.classes_

        st.subheader("Predicted Emotion:")
        st.success(prediction)

        st.subheader("Confidence Scores:")
        for label, prob in zip(classes, proba):
            st.write(f"**{label}** → {prob*100:.2f}%")
