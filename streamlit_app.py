import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import streamlit as st
from pytubefix import YouTube
from moviepy import VideoFileClip
from speechbrain.pretrained import EncoderClassifier
import torchaudio

# Explicitly set backend
torchaudio.set_audio_backend("ffmpeg")

# Rest of your existing Streamlit code...
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

    # Verify FFMPEG availability (optional, helpful debugging)
    if not shutil.which("ffmpeg"):
        st.error("FFmpeg is not available, cannot process audio.")
        st.stop()

    # Classify accent
    with st.spinner("Classifying accent…"):
        model = EncoderClassifier.from_hparams(
            source="Jzuluaga/accent-id-commonaccent_ecapa",
            run_opts={"device": "cpu"}
        )
        _, pred_prob, _, labels = model.classify_file(audio_path)
        accent = labels[0]
        confidence = float(pred_prob[0]) * 100

    # Display results
    st.success("✅ Done!")
    st.markdown(f"**Accent:** {accent.capitalize()}")
    st.markdown(f"**Confidence:** {confidence:.1f}%")

    # Clean up files
    os.remove(video_path)
    os.remove(audio_path)
