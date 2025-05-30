import os
# Disable Streamlit file-watcher to avoid torch.classes warnings
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import tempfile
import streamlit as st
from pytubefix import YouTube
from moviepy import VideoFileClip
import librosa
import torch
from speechbrain.pretrained import EncoderClassifier

# Descriptions for each accent label
ACCENT_DESCRIPTIONS = {
    "african":     "A pan-African English accent, often influenced by local rhythms and vowel shifts across multiple countries.",
    "australia":   "Australian English, characterized by broad vowel sounds and a distinctive rising inflection.",
    "bermuda":     "Bermudian English, mixing British RP influences with Caribbean and island-local intonations.",
    "canada":      "General Canadian English, similar to General American but with the “eh” tag and some unique vowel qualities.",
    "england":     "Standard British (Received Pronunciation), featuring non-rhoticity (dropping of ‘r’ sounds).",
    "hongkong":    "Hong Kong English, with Cantonese-influenced intonation and syllable timing.",
    "indian":      "Indian English, marked by retroflex consonants and syllable-timed rhythm patterns.",
    "ireland":     "Irish English, often singsongy with distinct diphthongs and rhotic ‘r’s.",
    "malaysia":    "Malaysian English (Manglish), influenced by Malay and Chinese tonal patterns.",
    "newzealand":  "New Zealand English, with a very “flat” vowel space (e.g. the KIT vowel sounds like “ket”).",
    "philippines": "Philippine English, with syllable timing drawn from Tagalog and other local languages.",
    "scotland":    "Scottish English, featuring rolled ‘r’s and Scots vocabulary borrowings.",
    "singapore":   "Singaporean English (Singlish), blending British structure with Cantonese, Malay, and Tamil cadence.",
    "southatlandtic": "South Atlantic (e.g. Falklands) English, a rarer island dialect mixing British and maritime influences.",
    "us":          "General North American (General American), rhotic with flattened “o” sounds.",
    "wales":       "Welsh English, characterized by melodic pitch changes influenced by the Welsh language."
}

# Streamlit UI configuration
st.set_page_config(page_title="Accent Classifier", layout="centered")
st.title("🎙️ English-Accent Classifier")

# Input field for video URL
url = st.text_input("Enter a public video URL (e.g., Loom, YouTube)")

if st.button("Analyze") and url:
    temp_dir = None
    try:
        # Create a temporary workspace
        temp_dir = tempfile.mkdtemp()

        # Download video
        with st.spinner("Downloading video…"):
            yt = YouTube(url)
            video_path = yt.streams.get_highest_resolution().download(
                output_path=temp_dir,
                filename="video.mp4"
            )

        # Extract audio as WAV
        with st.spinner("Extracting audio…"):
            wav_path = os.path.join(temp_dir, "audio.wav")
            clip = VideoFileClip(video_path)
            clip.audio.write_audiofile(wav_path)
            clip.close()

        # Classify accent using librosa
        with st.spinner("Classifying accent…"):
            waveform_np, sr = librosa.load(wav_path, sr=None)
            waveform = torch.from_numpy(waveform_np).float().unsqueeze(0)
            model = EncoderClassifier.from_hparams(
                source="Jzuluaga/accent-id-commonaccent_ecapa",
                run_opts={"device": "cpu"}
            )
            scores, pred_prob, _, labels = model.classify_batch(waveform)
            accent = labels[0]
            confidence = float(pred_prob[0]) * 100
            description = ACCENT_DESCRIPTIONS.get(accent, "No description available.")

        # Display results
        st.success("✅ Done!")
        st.markdown(f"**Accent:** {accent.capitalize()}")
        st.markdown(f"**Confidence:** {confidence:.1f}%")
        st.markdown(f"**Info:** {description}")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
    finally:
        # Clean up temporary files and directory
        if temp_dir:
            try:
                for fname in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, fname))
                os.rmdir(temp_dir)
            except Exception:
                pass
