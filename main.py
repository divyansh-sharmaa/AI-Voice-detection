import streamlit as st
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
from keras.models import load_model
from keras.utils import to_categorical

# Constants
SAMPLE_RATE = 44100  # Sample rate of your audio files
DURATION = 3  # Duration of audio clips to be used
N_MELS = 128  # Number of Mel bands for Mel spectrogram
NUM_CLASSES = 2  # Number of classes for classification

# Load model
model = load_model('audio_classifier.h5')

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
st.title('Audio Classification')

# Function to record audio
def record_audio(duration):
    with st.spinner("Recording..."):
        audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
    return audio.flatten()

uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3', 'flac'])

if uploaded_file is not None:
    upload_data = librosa.load(uploaded_file, sr=SAMPLE_RATE)[0]
    st.session_state.recorded_audio = upload_data
    st.audio(uploaded_file, format='audio')
    st.session_state.set_stat = "upl"
else:
    st.warning("Please upload an audio file or record one.")

if st.button("Record"):
    recorded_audio = record_audio(3)
    sf.write("recorded_audio.wav", recorded_audio, SAMPLE_RATE)
    st.audio("recorded_audio.wav", format='audio')
    st.session_state.recorded_audio = recorded_audio
    st.session_state.set_stat = "rec"

if st.button("Predict"):
    audio_data = st.session_state.recorded_audio
    # st.success(audio_data)
    mel_spectrogram = preprocess_audio(audio_data)
    prediction = model.predict(mel_spectrogram)
    # st.success(prediction)
    if st.session_state.set_stat == "rec" :
        class_names = ['Real', 'Fake']  
    else:
        class_names = ['Fake', 'Real']
    st.write('Prediction:')
    st.write(class_names[np.argmax(prediction)])