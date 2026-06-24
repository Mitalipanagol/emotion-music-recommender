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

EMOTION_GRADIENT = {
    "angry":    ("#ff416c", "#ff4b2b"),
    "disgust":  ("#56ab2f", "#a8e063"),
    "fear":     ("#42275a", "#734b6d"),
    "happy":    ("#f7971e", "#ffd200"),
    "sad":      ("#4b6cb7", "#182848"),
    "surprise": ("#ee0979", "#ff6a00"),
    "neutral":  ("#283048", "#859398"),
}

st.set_page_config(page_title="Moodify", page_icon="🎧", layout="wide")

st.markdown(
    """
    <style>
    #MainMenu, header, footer {visibility: hidden;}
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1180px;}

    .moodify-hero {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 40%, #191414 100%);
        padding: 28px 36px;
        border-radius: 18px;
        color: #fff;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px rgba(29, 185, 84, 0.25);
    }
    .moodify-hero h1 {
        margin: 0;
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .moodify-hero p {
        margin: 6px 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }

    .mood-pill {
        padding: 18px 22px;
        border-radius: 14px;
        color: #fff;
        text-align: center;
        margin-bottom: 12px;
        box-shadow: 0 8px 22px rgba(0,0,0,0.18);
    }
    .mood-pill .emoji {font-size: 3.4rem; line-height: 1;}
    .mood-pill .label {font-size: 1.5rem; font-weight: 700; margin-top: 4px;}
    .mood-pill .conf {opacity: 0.92; font-size: 0.95rem; margin-top: 2px;}

    .score-row {display: flex; align-items: center; gap: 10px; margin: 6px 0;}
    .score-name {width: 90px; font-size: 0.85rem; color: #cfcfcf;}
    .score-bar {flex: 1; height: 8px; background: #2a2a2a; border-radius: 4px; overflow: hidden;}
    .score-fill {height: 100%; background: linear-gradient(90deg, #1DB954, #1ed760); border-radius: 4px;}
    .score-pct {width: 46px; text-align: right; font-size: 0.8rem; color: #9a9a9a;}

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 14px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        background: rgba(255,255,255,0.02);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(29,185,84,0.18);
    }

    .stButton>button, .stLinkButton>a {
        border-radius: 999px !important;
        font-weight: 600 !important;
        border: none !important;
    }
    .stButton>button {background: #1DB954 !important; color: #fff !important;}
    .stButton>button:hover {background: #1ed760 !important;}
    .stLinkButton>a {background: rgba(255,255,255,0.08) !important; color: #fff !important;}

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin: 18px 0 10px 0;
        color: #1DB954;
        letter-spacing: 0.02em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="moodify-hero">
      <h1>🎧 Moodify</h1>
      <p>Your face. Your mood. Your soundtrack — powered by AI &amp; Spotify.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


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

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="section-title">CAPTURE</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Input source",
        ("Webcam", "Upload image"),
        horizontal=True,
        label_visibility="collapsed",
    )
    img_file = (
        st.camera_input("Take a photo", label_visibility="collapsed")
        if mode == "Webcam"
        else st.file_uploader(
            "Upload an image", type=("jpg", "jpeg", "png"), label_visibility="collapsed"
        )
    )

if img_file is None:
    with right:
        st.markdown('<div class="section-title">HOW IT WORKS</div>', unsafe_allow_html=True)
        st.markdown(
            "1. Snap or upload a clear photo of your face.\n"
            "2. A Vision Transformer reads your expression.\n"
            "3. Moodify hand-picks **7 tracks** that match your vibe."
        )
    st.stop()

pil_img = Image.open(io.BytesIO(img_file.getvalue())).convert("RGB")
frame_rgb = np.array(pil_img)
frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

with st.spinner("Analyzing your mood..."):
    result = detector.analyze(frame_bgr)

if result is None:
    with right:
        st.error("No face detected. Please try again with a clearer, well-lit photo.")
    st.stop()

dominant = result["dominant_emotion"]
scores = result["scores"]
region = result["region"]

annotated = frame_rgb.copy()
if region["w"] and region["h"]:
    cv2.rectangle(
        annotated,
        (region["x"], region["y"]),
        (region["x"] + region["w"], region["y"] + region["h"]),
        (29, 185, 84),
        4,
    )

with left:
    st.image(annotated, use_container_width=True)

with right:
    st.markdown('<div class="section-title">DETECTED MOOD</div>', unsafe_allow_html=True)
    g1, g2 = EMOTION_GRADIENT.get(dominant, ("#1DB954", "#191414"))
    st.markdown(
        f"""
        <div class="mood-pill" style="background: linear-gradient(135deg, {g1}, {g2});">
          <div class="emoji">{EMOTION_EMOJI.get(dominant, '')}</div>
          <div class="label">{dominant.title()}</div>
          <div class="conf">Confidence · {scores[dominant] * 100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rows = "".join(
        f"""
        <div class="score-row">
          <div class="score-name">{emo.title()}</div>
          <div class="score-bar"><div class="score-fill" style="width:{scores[emo]*100:.1f}%"></div></div>
          <div class="score-pct">{scores[emo]*100:.0f}%</div>
        </div>
        """
        for emo in EMOTIONS
    )
    st.markdown(rows, unsafe_allow_html=True)

st.markdown(
    f'<div class="section-title">YOUR {dominant.upper()} PLAYLIST · 7 TRACKS</div>',
    unsafe_allow_html=True,
)

if spotify is None:
    st.error(
        "Spotify is not configured. Add `SPOTIPY_CLIENT_ID`, `SPOTIPY_CLIENT_SECRET` "
        "and `SPOTIPY_REDIRECT_URI` to a `.env` file in the project root, then restart."
    )
    st.stop()

with st.spinner("Curating tracks for your mood..."):
    try:
        tracks = spotify.recommend_tracks(dominant, limit=7)
    except Exception as e:
        st.error(f"Spotify search failed: {e}")
        st.stop()

if not tracks:
    st.error("Couldn't find matching tracks on Spotify. Try again.")
    st.stop()

cols = st.columns(2, gap="medium")
for idx, track in enumerate(tracks, start=1):
    with cols[(idx - 1) % 2]:
        with st.container(border=True):
            c1, c2 = st.columns([1, 2.2])
            with c1:
                if track.image_url:
                    st.image(track.image_url, use_container_width=True)
            with c2:
                st.markdown(f"**{idx}. {track.name}**")
                st.caption(f"by {track.artist}")
                if track.preview_url:
                    st.audio(track.preview_url)
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("▶ Play", key=f"play_{idx}", use_container_width=True):
                        ok, msg = spotify.play_track(track.track_uri)
                        (st.success if ok else st.warning)(msg)
                        if not ok:
                            spotify.open_in_browser(track.track_url)
                with b2:
                    st.link_button("Spotify ↗", track.track_url, use_container_width=True)