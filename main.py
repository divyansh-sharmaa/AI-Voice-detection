import streamlit as st
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
from keras.models import load_model
from keras.utils import to_categorical
import streamlit as st
from moviepy.editor import *
from pydub import AudioSegment
import streamlit as st
import speech_recognition as sr
recognizer = sr.Recognizer()

import mysql.connector
from mysql.connector import Error


# Constants
SAMPLE_RATE = 44100  # Sample rate of your audio files
DURATION = 3  # Duration of audio clips to be used
N_MELS = 128  # Number of Mel bands for Mel spectrogram
NUM_CLASSES = 2  # Number of classes for classification

# Load model
model = load_model("YOUR-MODEL")

# Function to preprocess audio
def preprocess_audio(audio):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Ensure all spectrograms have the same width (time steps)
    max_time_steps = 109
    if mel_spectrogram.shape[1] < max_time_steps:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, max_time_steps - mel_spectrogram.shape[1])), mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :max_time_steps]

    return mel_spectrogram.reshape(1, mel_spectrogram.shape[0], mel_spectrogram.shape[1], 1)

# Streamlit app
st.title("AI Voice Detection")
st.title('Audio Classification')

def record_audio(duration):
    with st.spinner("Recording..."):
        audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
    return audio.flatten()

uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3', 'flac'])

if st.button("Record"):
    recorded_audio = record_audio(5)
    sf.write("recorded_audio.flac", recorded_audio, SAMPLE_RATE)
    st.audio("recorded_audio.flac", format='audio')
    st.session_state.file_name = "recorded_audio.flac"
    st.session_state.recorded_audio = recorded_audio
    st.session_state.set_stat = "rec"
    
# "Transcript" button
if st.button("Transcript"):
    if "recorded_audio" in st.session_state and st.session_state.set_stat == "rec":
        # Initialize the recognizer
        recognizer = sr.Recognizer()
        audio_data = sr.AudioData(
            st.session_state.recorded_audio.tobytes(),
            SAMPLE_RATE,
            sample_width=st.session_state.recorded_audio.dtype.itemsize
        )
        
        try:
            st.write("Recognizing...")

            # Convert audio to text using Google Speech Recognition
            text = recognizer.recognize_google(audio_data, language="en-IN")

            st.write("Transcribed Text:", text)
            st.session_state.transcribe = text

        except sr.UnknownValueError:
            st.write("Sorry, I couldn't understand the audio.")
        except sr.RequestError as e:
            st.write("Error occurred; {0}".format(e))
    else:
        st.write("No recorded audio found. Please record audio first.")


if uploaded_file is not None:
    upload_data = librosa.load(uploaded_file, sr=SAMPLE_RATE)[0]
    sf.write("uploaded_audio.flac", upload_data, SAMPLE_RATE)
    st.session_state.file_name = "uploaded_audio.flac"
    st.session_state.recorded_audio = upload_data
    st.audio(uploaded_file, format='audio')
    st.session_state.set_stat = "upl"
    st.session_state.transcribe = ""

if st.button("Predict"):
    audio_data = st.session_state.recorded_audio
    # st.success(audio_data)
    mel_spectrogram = preprocess_audio(audio_data)
    prediction = model.predict(mel_spectrogram)
    st.warning(st.session_state.transcribe)
    st.success(prediction)
    class_names = ['Fake', 'Real']
    st.write('Prediction:')
    prediction_label = class_names[np.argmax(prediction)]
    st.write(prediction_label)
    if prediction_label == 'Fake':
        spoof_type = "AI-GEN-AUDIO"
        try:
            with open("response.txt", "a") as file:
                file.write(f"Response: {spoof_type}, Filepath: {st.session_state.file_name}\n")

            st.success("Data appended to response.txt successfully")
        
        except Exception as e:
            st.error(f"Failed to append data to response.txt: {e}")

st.title('')
def convert_mp4_to_mp3(uploaded_video):
    video = VideoFileClip(uploaded_video)
    mp3_file = uploaded_video.replace(".mp4", ".mp3")
    video.audio.write_audiofile(mp3_file)
    return mp3_file

st.checkbox('Video Classification')

uploaded_video = st.file_uploader("Choose an MP4 file", type=["mp4"])

if uploaded_video is not None:
    video_details = st.video(uploaded_video)
    st.session_state.recorded_audio = video_details
    st.write("Uploaded video details:")
    st.write(f"File name: {uploaded_video.name}")
    st.write(f"File size: {uploaded_video.size} bytes")

if st.button('Convert to audio'):
    st.write("File uploaded successfully!")
    st.write("Converting...")
    # Convert MP4 to MP3
    mp3_file = convert_mp4_to_mp3(uploaded_video.name)
    st.session_state.file_name = uploaded_video.name+".mp3"
    st.success(f"Audio file '{mp3_file}' has been created.")
    upload_data = librosa.load(mp3_file, sr=SAMPLE_RATE)[0]
    st.session_state.recorded_audio = upload_data
    st.audio(mp3_file, format='audio')

if st.button("Prediction"):
    mp3_file = convert_mp4_to_mp3(uploaded_video.name)
    upload_data = librosa.load(mp3_file, sr=SAMPLE_RATE)[0]
    st.session_state.recorded_audio = upload_data
    st.session_state.set_stat = "upl"
    st.session_state.class_name = ['Fake', 'Real']
    audio_data = st.session_state.recorded_audio
    mel_spectrogram = preprocess_audio(audio_data)
    prediction = model.predict(mel_spectrogram)
    st.success(prediction)
    class_names = st.session_state.class_name
    st.write('Prediction:')
    prediction_label = class_names[np.argmax(prediction)]
    st.write(prediction_label)
    if prediction_label == 'Fake':
        spoof_type = "AI-VIDEO"
        try:
            with open("response.txt", "a") as file:
                file.write(f"Response: {spoof_type}, Filepath: {st.session_state.file_name}\n")

            st.success("Data appended to response.txt successfully")
        
        except Exception as e:
            st.error(f"Failed to append data to response.txt: {e}")
