import os
# Disable Streamlit file-watcher to avoid torch.classes warnings
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import io
import streamlit as st
from pytubefix import YouTube
import requests
from pydub import AudioSegment
import librosa
import torch
from speechbrain.pretrained import EncoderClassifier

# Descriptions for each accent label
ACCENT_DESCRIPTIONS = {
    "african":     "A pan-African English accent, often influenced by local rhythms and vowel shifts across multiple countries.",
    "australia":   "Australian English, characterized by broad vowel sounds and a distinctive rising inflection.",
    "bermuda":     "Bermudian English, mixing British RP influences with Caribbean and island-local intonations.",
    "canada":      "General Canadian English, similar to General American but with the 'eh' tag and some unique vowel qualities.",
    "england":     "Standard British (Received Pronunciation), featuring non-rhoticity (dropping of 'r' sounds).",
    "hongkong":    "Hong Kong English, with Cantonese-influenced intonation and syllable timing.",
    "indian":      "Indian English, marked by retroflex consonants and syllable-timed rhythm patterns.",
    "ireland":     "Irish English, often singsongy with distinct diphthongs and rhotic 'r's.",
    "malaysia":    "Malaysian English (Manglish), influenced by Malay and Chinese tonal patterns.",
    "newzealand":  "New Zealand English, with a very flat vowel space (e.g., KIT sounds like 'ket').",
    "philippines": "Philippine English, with syllable timing drawn from Tagalog and other local languages.",
    "scotland":    "Scottish English, featuring rolled 'r's and Scots vocabulary borrowings.",
    "singapore":   "Singaporean English (Singlish), blending British structure with Cantonese, Malay, and Tamil cadence.",
    "southatlandtic": "South Atlantic (e.g., Falklands) English, a rare island dialect mixing British and maritime influences.",
    "us":          "General North American (General American), rhotic with flattened 'o' sounds.",
    "wales":       "Welsh English, characterized by melodic pitch changes influenced by the Welsh language."
}

# Streamlit UI configuration
st.set_page_config(page_title="Accent Classifier", layout="centered")
st.title("🎙️ English-Accent Classifier")

url = st.text_input("Enter a public video URL (e.g., YouTube, Loom)")

if st.button("Analyze") and url:
    try:
        with st.spinner("Downloading and buffering video…"):
            yt = YouTube(url)
            stream_url = yt.streams.get_lowest_resolution().url
            response = requests.get(stream_url, stream=True)
            video_buffer = io.BytesIO()
            for chunk in response.iter_content(1024*64):
                if chunk:
                    video_buffer.write(chunk)
            video_buffer.seek(0)

        with st.spinner("Extracting audio in-memory…"):
            audio = AudioSegment.from_file(video_buffer, format="mp4")
            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            wav_buffer.seek(0)

        with st.spinner("Processing and classifying accent…"):
            waveform_np, sr = librosa.load(wav_buffer, sr=None)
            waveform = torch.from_numpy(waveform_np).float().unsqueeze(0)
            model = EncoderClassifier.from_hparams(
                source="Jzuluaga/accent-id-commonaccent_ecapa",
                run_opts={"device": "cpu"}
            )
            scores, pred_prob, _, labels = model.classify_batch(waveform)
            accent = labels[0]
            confidence = float(pred_prob[0]) * 100
            description = ACCENT_DESCRIPTIONS.get(accent, "No description available.")

        st.success("✅ Completed!")
        st.markdown(f"**Accent:** {accent.capitalize()}")
        st.markdown(f"**Confidence:** {confidence:.1f}%")
        st.markdown(f"**Info:** {description}")

    except Exception as e:
        st.error(f"Error during processing: {e}")
