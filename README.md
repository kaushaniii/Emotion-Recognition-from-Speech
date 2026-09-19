Emotion Recognition from Speech
Live Demo: https://emotion-recognition-from-speech-tmx6rpvptqphzfy8dvtvkx.streamlit.app/

1. Project Overview
This project presents a deep learning-based system for recognizing human emotions — angry, calm, disgust, fearful, happy, neutral, sad, and surprised — from speech audio.

The project follows a complete machine learning workflow, starting from audio feature extraction and preprocessing to model training, evaluation, model selection, and deployment through a Streamlit web application.

2. Project Objective
The primary objective of this project is to develop an accurate multi-class classification model that can predict the emotion expressed in a speech clip based on acoustic features extracted from the audio signal.

The project also aims to:

Extract meaningful acoustic features from raw speech audio.
Preprocess and normalize variable-length audio into fixed-size input.
Train a deep learning model combining convolutional and recurrent layers.
Handle class imbalance using weighted training.
Evaluate model performance using standard classification metrics.
Analyze misclassification patterns using a confusion matrix.
Develop an interactive web application using Streamlit.
3. Dataset
The project uses the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.

Dataset Details
Attribute	Description
Number of Samples	~1,440
Number of Classes	8
Target Variable	Emotion (angry, calm, disgust, fearful, happy, neutral, sad, surprised)
Problem Type	Multi-class Classification
Feature Type	Audio (MFCC-based numerical features)
The dataset contains speech recordings from 24 professional actors, each expressing the same statements across all 8 emotion categories at varying intensities.

4. Project Roadmap
emotion-recognition-from-speech/
│
├── app.py
├── emotion_model.keras
├── label_encoder.pkl
├── requirements.txt
├── samples/
│   └── sample_happy.wav
└── README.md
5. Performance Comparison
Emotion	Precision	Recall	F1-Score
Angry	97.00%	91.00%	94.00%
Calm	88.00%	95.00%	91.00%
Disgust	94.00%	95.00%	94.00%
Fearful	79.00%	96.00%	87.00%
Happy	93.00%	90.00%	91.00%
Neutral	96.00%	68.00%	80.00%
Sad	90.00%	92.00%	91.00%
Surprised	91.00%	83.00%	87.00%
Overall Test Accuracy: 90.00% | Weighted F1-Score: 90.00%

Class weighting was applied during training to address the underrepresentation of the neutral class and improve recall on minority classes. This improved overall accuracy from 85% to 90%, with notable gains in fearful (+13% recall) and disgust precision (+27%). Neutral remains the most challenging class, primarily confused with calm — a known acoustic overlap in the RAVDESS dataset, where both categories are recorded with intentionally similar vocal tone.

6. Deep Learning Workflow
The project follows the workflow below:

Dataset (RAVDESS)
   ↓
Audio Loading
   ↓
MFCC Feature Extraction
   ↓
Padding / Truncation to Fixed Length
   ↓
Label Encoding
   ↓
Train-Test Split
   ↓
Class Weight Computation
   ↓
Model Building (CNN + LSTM)
   ↓
Model Training (with Class Weighting)
   ↓
Model Evaluation
   ↓
Model Serialization
   ↓
Streamlit Deployment
7. Sample Test File
A sample audio file is included in the samples/ folder for quick testing on the live demo.

File	True Label	Source
samples/sample_happy.wav	Happy	RAVDESS dataset
To test: download the file from the samples/ folder, upload it on the live demo, and compare the predicted emotion against the true label above.
