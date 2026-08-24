"""Google Cloud Vision Logo Detection — an alternate/experimental backend for Q3's
make classification. This is a dedicated logo-recognition API (narrower and more
purpose-built than SigLIP's zero-shot brand classifier or a general VLM reading a
badge), separate from Vision's generic label detection. NOT wired into production —
for the test/inference interface only (``experiments/``).

No ``GOOGLE_APPLICATION_CREDENTIALS`` is configured in this environment — degrades
gracefully ("not configured") until a service-account key is added.

Known caveat, unverified without credentials: Google's logo-detection training set is
generic/commercial-brand-heavy (consumer electronics, apparel, fast food, etc.) — it's
unconfirmed whether it recognises Indian truck manufacturer badges at all. Test against
the real sample images once credentials are available before trusting this.
"""
from __future__ import annotations

import io


def classify_logo(image) -> dict:
    """image: file path or PIL.Image. Returns {"checked", "make", "make_confidence",
    "candidates": [(name, score), ...]} — the same shape any Q3 make-vote source needs."""
    try:
        from google.cloud import vision
    except Exception as e:
        return {"checked": False, "error": f"google-cloud-vision import: {e}"}

    import os
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return {"checked": False, "error": "no GOOGLE_APPLICATION_CREDENTIALS"}

    try:
        from PIL import Image
        img = image if isinstance(image, Image.Image) else Image.open(image)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=92)

        client = vision.ImageAnnotatorClient()
        gcv_image = vision.Image(content=buf.getvalue())
        resp = client.logo_detection(image=gcv_image)
        if resp.error.message:
            return {"checked": False, "error": resp.error.message}

        logos = sorted(((l.description, float(l.score)) for l in resp.logo_annotations),
                       key=lambda t: t[1], reverse=True)
        if not logos:
            return {"checked": True, "make": "", "make_confidence": 0.0, "candidates": []}
        best_name, best_score = logos[0]
        return {"checked": True, "make": best_name,
                "make_confidence": round(best_score * 100, 1), "candidates": logos}
    except Exception as e:
        return {"checked": False, "error": str(e)}
