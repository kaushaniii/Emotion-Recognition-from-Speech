Deep learning model that classifies emotion (angry, calm, disgust, fearful, happy, neutral, sad, surprised) from speech audio, deployed as an interactive web app.

Live demo: https://emotion-recognition-from-speech-tmx6rpvptqphzfy8dvtvkx.streamlit.app/

Overview
Dataset: RAVDESS — 8-class emotional speech corpus
Features: 40-coefficient MFCCs, padded/truncated to 174 time frames
Model: CNN + LSTM — Conv1D blocks extract local spectral patterns, LSTM models their evolution over time
Deployment: Streamlit, hosted on Streamlit Community Cloud
Results
Test accuracy: 85% (576 samples)

Emotion	Precision	Recall	F1
Angry	0.93	0.88	0.91
Calm	0.92	0.87	0.89
Disgust	0.67	0.96	0.79
Fearful	0.93	0.83	0.88
Happy	0.84	0.88	0.86
Neutral	0.90	0.74	0.81
Sad	0.89	0.75	0.82
Surprised	0.84	0.82	0.83
Weighted F1: 0.85

High-arousal emotions (angry, fearful, calm) classify reliably. The main error pattern is disgust being over-predicted when the model is uncertain, and neutral/calm confusion — both consistent with known acoustic overlap in RAVDESS.

Project Structure
├── app.py                  # Streamlit app
├── requirements.txt
├── emotion_model.keras     # Trained model
├── label_encoder.pkl
└── samples/                # Example .wav files for testing

Run Locally
git clone https://github.com/<your-username>/emotion-recognition-from-speech.git
cd emotion-recognition-from-speech
pip install -r requirements.txt
streamlit run app.py
Training Pipeline
Trained in Google Colab: RAVDESS → MFCC extraction → 80/20 split → CNN+LSTM with early stopping → evaluated via classification report and confusion matrix.
