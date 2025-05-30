import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import streamlit as st
from pytubefix import YouTube
from moviepy import VideoFileClip
from speechbrain.pretrained import EncoderClassifier

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

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

    # Classify accent
    with st.spinner("Classifying accent…"):
        model = EncoderClassifier.from_hparams(
            source="Jzuluaga/accent-id-commonaccent_ecapa",
            run_opts={"device": "cpu"}
        )
        _, pred_prob, _, labels = model.classify_file(audio_path)
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
