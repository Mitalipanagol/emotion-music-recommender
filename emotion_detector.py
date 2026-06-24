from __future__ import annotations
import os
from typing import Optional
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")
_MODEL_ID = "trpakov/vit-face-expression"

_LABEL_ALIASES = {
    "angry": "angry", "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy", "happiness": "happy",
    "sad": "sad", "sadness": "sad",
    "surprise": "surprise", "surprised": "surprise",
    "neutral": "neutral",
}


class EmotionDetector:

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(_MODEL_ID)
        self.model = AutoModelForImageClassification.from_pretrained(_MODEL_ID)
        self.model.to(self.device).eval()

        cascade_path = os.path.join(
            cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
        )
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def _detect_largest_face(self, image_bgr: np.ndarray):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            return None
        # Pick the largest face (most likely the subject).
        return max(faces, key=lambda r: r[2] * r[3])

    @torch.inference_mode()
    def _classify(self, face_rgb: np.ndarray) -> dict[str, float]:
        pil = Image.fromarray(face_rgb)
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        scores = {emo: 0.0 for emo in EMOTIONS}
        id2label = self.model.config.id2label
        for idx, p in enumerate(probs):
            label = id2label[idx].lower()
            canonical = _LABEL_ALIASES.get(label)
            if canonical:
                scores[canonical] = float(p)
        return scores

    def analyze(self, image_bgr: np.ndarray) -> Optional[dict]:
        
        if image_bgr is None or image_bgr.size == 0:
            return None

        face_box = self._detect_largest_face(image_bgr)
        if face_box is None:
           
            face_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            region = {"x": 0, "y": 0, "w": image_bgr.shape[1], "h": image_bgr.shape[0]}
        else:
            x, y, w, h = (int(v) for v in face_box)
            pad = int(0.15 * max(w, h))
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(image_bgr.shape[1], x + w + pad)
            y1 = min(image_bgr.shape[0], y + h + pad)
            face_bgr = image_bgr[y0:y1, x0:x1]
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            region = {"x": x, "y": y, "w": w, "h": h}

        try:
            scores = self._classify(face_rgb)
        except Exception:
            return None

        dominant_emotion = max(scores, key=scores.get)
        return {
            "dominant_emotion": dominant_emotion,
            "scores": scores,
            "region": region,
        }
