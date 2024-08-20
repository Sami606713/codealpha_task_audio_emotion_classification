import streamlit as st
import soundfile as sf
import torch
from transformers import AutoModel, AutoFeatureExtractor
import os

# Get the Hugging Face API token from environment variables
token = "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"

# Load the model and feature extractor using your token
try:
    model = AutoModel.from_pretrained("sami606713/emotion_classification", use_auth_token=token)
    feature_extractor = AutoFeatureExtractor.from_pretrained("sami606713/emotion_classification", use_auth_token=token)
except Exception as e:
    st.write(f"Error loading model: {e}")

# Title and description
st.title("Audio Emotion Classification")
st.write("Upload an audio file and the model will classify the emotion.")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3", "ogg"])

if uploaded_file is not None:
    # Load the audio file
    audio_input, sample_rate = sf.read(uploaded_file)
    sample_rate = 16000  # Ensure the sample rate is 16000

    # Display the audio player
    st.audio(uploaded_file)

    # Perform emotion classification
    if st.button("Classifying"):
        try:
            inputs = feature_extractor(audio_input, sampling_rate=sample_rate, return_tensors="pt")
    
            # Make prediction
            with torch.no_grad():
                outputs = model(**inputs)
    
            embeddings = outputs.pooler_output
    
            # Apply a classification head on top of the embeddings
            id2label={
                0:"angry",
                1:'calm',
                2:'disgust',
                3:'fearful',
                4:'happy',
                5:'neutral',
                6:'sad',
                7:'surprised'
            }
            classifier = torch.nn.Linear(embeddings.shape[-1], len(id2label))
    
            # Pass embeddings through the classifier
            logits = classifier(embeddings)
    
            # Get predicted class
            predicted_class_idx = logits.argmax(-1).item()
            predicted_class = id2label[predicted_class_idx]
    
            st.write(f"Predicted Emotion: {predicted_class}")
        except Exception as e:
            st.write(f"Error during classification: {e}")