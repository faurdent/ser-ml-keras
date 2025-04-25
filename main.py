import streamlit as st

from classifier import SpeechEmotionClassifier


ser_classifier = SpeechEmotionClassifier()

st.header("Speech Emotion Recognition")

tab1, tab2 = st.tabs(["File", "Voice"])

with tab1:
    file = st.file_uploader("Your audio file:")
    if file:
        st.write(f"Mood: {ser_classifier.predict_emotion(file)}")

with tab2:
    audio = st.audio_input("Say something:")

    if audio:
        st.write(f"Audio Mood: {ser_classifier.predict_emotion(audio)}")
