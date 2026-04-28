"""Spotify integration for the emotion-based music recommender.

Uses Spotipy with the Authorization Code flow so we can:
  * search for mood-matching playlists / tracks
  * start playback on the user's currently active Spotify device
  * fall back to opening the track URL in the browser when no device
    is active or the user is not Premium.
"""
from __future__ import annotations

import os
import webbrowser
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth

load_dotenv()

# Scopes required to read playback state and start/transfer playback.
SCOPE = "user-read-playback-state user-modify-playback-state streaming"

# Curated mood -> Spotify search query map.
EMOTION_QUERY = {
    "happy":    "happy upbeat feel good hits",
    "sad":      "sad songs emotional acoustic",
    "angry":    "angry rock metal rage",
    "fear":     "calm soothing relaxing ambient",
    "disgust":  "chill lo-fi mellow",
    "surprise": "energetic party dance hits",
    "neutral":  "today's top hits",
}


@dataclass
class TrackInfo:
    name: str
    artist: str
    track_url: str
    track_uri: str
    preview_url: Optional[str]
    image_url: Optional[str]


class SpotifyClient:
    """Thin Spotipy wrapper exposing emotion-based recommendation + playback."""

    def __init__(self, cache_path: str = ".spotipy_cache") -> None:
        client_id = os.getenv("SPOTIPY_CLIENT_ID")
        client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")
        redirect_uri = os.getenv(
            "SPOTIPY_REDIRECT_URI", "http://127.0.0.1:8888/callback"
        )
        if not client_id or not client_secret:
            raise RuntimeError(
                "Missing SPOTIPY_CLIENT_ID / SPOTIPY_CLIENT_SECRET. "
                "Copy .env.example to .env and fill in your credentials."
            )

        auth_manager = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=SCOPE,
            cache_path=cache_path,
            open_browser=True,
        )
        self.sp = spotipy.Spotify(auth_manager=auth_manager)

    # ------------------------------------------------------------------ #
    # Recommendation
    # ------------------------------------------------------------------ #
    def recommend_track(self, emotion: str) -> Optional[TrackInfo]:
        """Search Spotify for a track matching the given emotion."""
        query = EMOTION_QUERY.get(emotion, EMOTION_QUERY["neutral"])
        results = self.sp.search(q=query, type="track", limit=20, market="from_token")
        items = (results or {}).get("tracks", {}).get("items", []) or []
        if not items:
            return None

        # Prefer a track that has a preview_url so non-premium users still hear something.
        items.sort(key=lambda t: (t.get("preview_url") is None, -t.get("popularity", 0)))
        track = items[0]
        images = (track.get("album", {}) or {}).get("images", []) or []
        return TrackInfo(
            name=track.get("name", "Unknown"),
            artist=", ".join(a["name"] for a in track.get("artists", [])) or "Unknown",
            track_url=(track.get("external_urls", {}) or {}).get("spotify", ""),
            track_uri=track.get("uri", ""),
            preview_url=track.get("preview_url"),
            image_url=images[0]["url"] if images else None,
        )

    # ------------------------------------------------------------------ #
    # Playback
    # ------------------------------------------------------------------ #
    def get_active_device_id(self) -> Optional[str]:
        devices = (self.sp.devices() or {}).get("devices", []) or []
        for d in devices:
            if d.get("is_active"):
                return d.get("id")
        return devices[0]["id"] if devices else None

    def play_track(self, track_uri: str) -> tuple[bool, str]:
        """Try to start playback on an active device.

        Returns ``(success, message)``. On failure (no device / not premium /
        scope error) the caller can fall back to opening the URL in a browser.
        """
        try:
            device_id = self.get_active_device_id()
            if not device_id:
                return False, (
                    "No active Spotify device found. Open Spotify on your phone "
                    "or desktop and start any song once, then try again."
                )
            self.sp.start_playback(device_id=device_id, uris=[track_uri])
            return True, "Playing on your active Spotify device."
        except spotipy.SpotifyException as e:
            if e.http_status == 403:
                return False, (
                    "Spotify rejected playback (403). Web API playback requires "
                    "a Spotify Premium account."
                )
            return False, f"Spotify error: {e.msg or e}"
        except Exception as e:  # pragma: no cover - defensive
            return False, f"Unexpected error: {e}"

    @staticmethod
    def open_in_browser(track_url: str) -> None:
        if track_url:
            webbrowser.open(track_url, new=2)
