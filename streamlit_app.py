import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import shutil
import streamlit as st
from pytubefix import YouTube
from moviepy import VideoFileClip
from speechbrain.pretrained import EncoderClassifier
import torchaudio
import soundfile as sf
import torch

torchaudio.set_audio_backend("ffmpeg")

# Accent descriptions
ACCENT_DESCRIPTIONS = {
    "african": "Pan-African accent...",
    # Include the rest of your descriptions...
    "us": "General American accent."
}

st.set_page_config(page_title="Accent Classifier", layout="centered")
st.title("🎙️ English-Accent Classifier")

url = st.text_input("Enter a public video URL (e.g., Loom, YouTube)")

if st.button("Analyze") and url:
    # Download video
    with st.spinner("Downloading video…"):
        try:
            yt = YouTube(url)
            video_path = yt.streams.get_lowest_resolution().download()
        except Exception as e:
            st.error(f"Download failed: {e}")
            st.stop()

    # Extract audio
    with st.spinner("Extracting audio…"):
        audio_path = os.path.splitext(video_path)[0] + ".wav"
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path)
        clip.close()

    # Load audio as NumPy array, then convert to Torch tensor
    signal, sr = sf.read(audio_path, dtype='float32')
    signal = torch.from_numpy(signal).unsqueeze(0)  # Add batch dimension

    # Classify accent
    with st.spinner("Classifying accent…"):
        model = EncoderClassifier.from_hparams(
            source="Jzuluaga/accent-id-commonaccent_ecapa",
            run_opts={"device": "cpu"}
        )
        _, pred_prob, _, labels = model.classify_batch(signal)
        accent = labels[0]
        confidence = float(pred_prob[0]) * 100
        description = ACCENT_DESCRIPTIONS.get(accent, "No description available.")

    # Display results
    st.success("✅ Done!")
    st.markdown(f"**Accent:** {accent.capitalize()}")
    st.markdown(f"**Confidence:** {confidence:.1f}%")
    st.markdown(f"**Info:** {description}")

    # Clean up files
    os.remove(video_path)
    os.remove(audio_path)