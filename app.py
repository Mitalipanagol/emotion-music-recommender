"""Streamlit app: detect facial emotion from webcam and play a matching
song on Spotify.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from emotion_detector import EMOTIONS, EmotionDetector
from spotify_client import SpotifyClient

EMOTION_EMOJI = {
    "angry":    "😠",
    "disgust":  "🤢",
    "fear":     "😨",
    "happy":    "😄",
    "sad":      "😢",
    "surprise": "😲",
    "neutral":  "😐",
}

st.set_page_config(page_title="Emotion -> Music", page_icon="🎵", layout="centered")
st.title("🎵 Emotion-Based Music Recommender")
st.caption(
    "Detects your facial emotion using a pretrained Vision Transformer "
    "(trpakov/vit-face-expression) and plays a matching song on Spotify."
)


# --------------------------------------------------------------------- #
# Resource caching
# --------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading emotion model (first run downloads weights)...")
def get_detector() -> EmotionDetector:
    return EmotionDetector()


@st.cache_resource(show_spinner="Connecting to Spotify...")
def get_spotify() -> SpotifyClient | None:
    try:
        return SpotifyClient()
    except Exception as e:
        st.session_state["spotify_error"] = str(e)
        return None


detector = get_detector()
spotify = get_spotify()
if spotify is None:
    st.warning(
        "Spotify is not configured: "
        + st.session_state.get("spotify_error", "unknown error")
    )

# --------------------------------------------------------------------- #
# Capture
# --------------------------------------------------------------------- #
st.subheader("1. Capture your face")
mode = st.radio(
    "Input source", ("Webcam", "Upload image"), horizontal=True, label_visibility="collapsed"
)

img_file = (
    st.camera_input("Take a photo")
    if mode == "Webcam"
    else st.file_uploader("Upload an image", type=("jpg", "jpeg", "png"))
)

if img_file is None:
    st.info("Take a photo or upload an image to begin.")
    st.stop()

pil_img = Image.open(io.BytesIO(img_file.getvalue())).convert("RGB")
frame_rgb = np.array(pil_img)
frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

# --------------------------------------------------------------------- #
# Detect
# --------------------------------------------------------------------- #
st.subheader("2. Detected emotion")
with st.spinner("Analyzing emotion..."):
    result = detector.analyze(frame_bgr)

if result is None:
    st.error("No face detected. Please try again with a clearer, well-lit photo.")
    st.stop()

dominant = result["dominant_emotion"]
scores = result["scores"]
region = result["region"]

# Draw bounding box for visual feedback.
annotated = frame_rgb.copy()
if region["w"] and region["h"]:
    cv2.rectangle(
        annotated,
        (region["x"], region["y"]),
        (region["x"] + region["w"], region["y"] + region["h"]),
        (0, 255, 0),
        3,
    )

col_img, col_info = st.columns([1, 1])
with col_img:
    st.image(annotated, caption="Detected face", use_container_width=True)
with col_info:
    st.markdown(
        f"## {EMOTION_EMOJI.get(dominant, '')} {dominant.title()}  \n"
        f"**Confidence:** {scores[dominant] * 100:.1f}%"
    )
    st.markdown("**All scores**")
    for emo in EMOTIONS:
        st.progress(min(max(scores[emo], 0.0), 1.0), text=f"{emo.title()} — {scores[emo] * 100:.1f}%")

# --------------------------------------------------------------------- #
# Recommend + play (7 tracks)
# --------------------------------------------------------------------- #
st.subheader(f"3. 7 songs for your {dominant} mood")
if spotify is None:
    st.error(
        "Spotify is not configured, so no songs can be loaded.\n\n"
        "**Fix:** create a `.env` file in the project root with:\n\n"
        "```\nSPOTIPY_CLIENT_ID=your_client_id\n"
        "SPOTIPY_CLIENT_SECRET=your_client_secret\n"
        "SPOTIPY_REDIRECT_URI=http://127.0.0.1:8888/callback\n```\n\n"
        "Get the ID/secret from https://developer.spotify.com/dashboard, "
        "then restart the app."
    )
    st.stop()

with st.spinner("Finding tracks that match your mood..."):
    try:
        tracks = spotify.recommend_tracks(dominant, limit=7)
    except Exception as e:
        st.error(f"Spotify search failed: {e}")
        st.stop()

if not tracks:
    st.error("Couldn't find matching tracks on Spotify. Try again.")
    st.stop()

for idx, track in enumerate(tracks, start=1):
    with st.container(border=True):
        c1, c2 = st.columns([1, 3])
        with c1:
            if track.image_url:
                st.image(track.image_url, use_container_width=True)
        with c2:
            st.markdown(f"**{idx}. {track.name}**")
            st.caption(f"by {track.artist}")

            if track.preview_url:
                st.audio(track.preview_url)
            else:
                st.caption("_30-sec preview unavailable for this track — use the buttons below._")

            b1, b2 = st.columns(2)
            with b1:
                if st.button("▶ Play full track", key=f"play_{idx}", use_container_width=True):
                    ok, msg = spotify.play_track(track.track_uri)
                    (st.success if ok else st.warning)(msg)
                    if not ok:
                        spotify.open_in_browser(track.track_url)
            with b2:
                st.link_button(
                    "Open in Spotify ↗", track.track_url, use_container_width=True
                )
