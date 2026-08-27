"""Clarifai logo-recognition model — an alternate/experimental backend for Q3's make
classification, via Clarifai's REST API directly (no SDK dependency, since the exact
client-library class names can't be verified without live credentials — a raw REST call
against Clarifai's stable, documented endpoint pattern is more predictable). NOT wired
into production — for the test/inference interface only (``experiments/``).

No ``CLARIFAI_API_KEY`` (a Personal Access Token) is configured in this environment —
degrades gracefully ("not configured") until one is added. The model reference
(``CLARIFAI_USER_ID``/``CLARIFAI_APP_ID``/``CLARIFAI_MODEL_ID``, see ``config.py``)
defaults to Clarifai's general public logo model — adjust to whatever logo-recognition
model you actually have access to in your account/community models, and verify against
the real sample images once a key is available, before trusting this for Indian truck
manufacturer badges specifically (unverified here).
"""
from __future__ import annotations

import base64
import io
import os

from vfiv import config

_API_URL = "https://api.clarifai.com/v2/users/{user_id}/apps/{app_id}/models/{model_id}/outputs"


def classify_logo(image) -> dict:
    """image: file path or PIL.Image. Returns {"checked", "make", "make_confidence",
    "candidates": [(name, score), ...]} — the same shape any Q3 make-vote source needs."""
    key = os.environ.get("CLARIFAI_API_KEY")
    if not key:
        return {"checked": False, "error": "no CLARIFAI_API_KEY"}

    try:
        import requests
    except Exception as e:
        return {"checked": False, "error": f"requests import: {e}"}

    try:
        from PIL import Image
        img = image if isinstance(image, Image.Image) else Image.open(image)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=92)
        b64 = base64.b64encode(buf.getvalue()).decode()

        url = _API_URL.format(user_id=config.CLARIFAI_USER_ID, app_id=config.CLARIFAI_APP_ID,
                              model_id=config.CLARIFAI_MODEL_ID)
        resp = requests.post(
            url,
            headers={"Authorization": f"Key {key}", "Content-Type": "application/json"},
            json={"inputs": [{"data": {"image": {"base64": b64}}}]},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status", {}).get("code") != 10000:  # Clarifai's "success" code
            return {"checked": False, "error": data.get("status", {}).get("description", "Clarifai error")}

        concepts = data["outputs"][0]["data"].get("concepts", [])
        if not concepts:
            return {"checked": True, "make": "", "make_confidence": 0.0, "candidates": []}
        concepts.sort(key=lambda c: c["value"], reverse=True)
        best = concepts[0]
        return {"checked": True, "make": best["name"],
                "make_confidence": round(best["value"] * 100, 1),
                "candidates": [(c["name"], c["value"]) for c in concepts]}
    except Exception as e:
        return {"checked": False, "error": str(e)}
