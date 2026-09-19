import streamlit as st
import numpy as np
import librosa
import pickle
import tempfile
import os
from tensorflow.keras.models import load_model

N_MFCC = 40
MAX_LEN = 174
MODEL_PATH = "emotion_model.keras"
ENCODER_PATH = "label_encoder.pkl"

@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    return model, le


def extract_mfcc(file_path, n_mfcc=N_MFCC, max_len=MAX_LEN):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc.T


def predict_emotion(file_path, model, le):
    features = extract_mfcc(file_path)
    features = np.expand_dims(features, axis=0)
    prediction = model.predict(features)
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_label = le.inverse_transform([predicted_index])[0]
    confidence = float(np.max(prediction))
    all_probs = {label: float(prob) for label, prob in zip(le.classes_, prediction[0])}
    return predicted_label, confidence, all_probs


st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️")
st.title("🎙️ Speech Emotion Recognition")
st.write("Upload a `.wav` audio clip and the model will predict the emotion expressed in the speech.")

model, le = load_artifacts()

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(uploaded_file, format="audio/wav")

    with st.spinner("Analyzing emotion..."):
        try:
            label, confidence, all_probs = predict_emotion(tmp_path, model, le)
            st.success(f"**Predicted Emotion: {label.upper()}**")
            st.write(f"Confidence: {confidence * 100:.1f}%")

            st.subheader("Probability breakdown")
            sorted_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
            st.bar_chart(sorted_probs)
        except Exception as e:
            st.error(f"Error processing audio: {e}")
        finally:
            os.remove(tmp_path)
else:
    st.info("Waiting for a WAV file to be uploaded.")