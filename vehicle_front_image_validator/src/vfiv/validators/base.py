"""Shared Claude VLM call plumbing, reused by every image-type validator
(front today; side / fastag plug in the same way)."""
import base64
import io
import json
import os

from PIL import Image


def to_jpeg_b64(image) -> str:
    img = Image.open(image).convert("RGB") if isinstance(image, (str, os.PathLike)) else image.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def call_vlm_json(image, prompt: str, model: str, max_tokens: int = 200) -> dict:
    """Send one image + prompt to Claude, parse the STRICT-JSON reply.

    Returns {"checked": True, **parsed_json} on success, or
    {"checked": False, "error": "..."} if the SDK/key is missing or the call fails.
    """
    try:
        import anthropic
    except Exception as e:
        return {"checked": False, "error": f"anthropic import: {e}"}
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return {"checked": False, "error": "no ANTHROPIC_API_KEY"}

    try:
        b64 = to_jpeg_b64(image)
        msg = anthropic.Anthropic(api_key=key).messages.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": prompt},
            ]}],
        )
        txt = msg.content[0].text.strip()
        parsed = json.loads(txt[txt.find("{"): txt.rfind("}") + 1])
        return {"checked": True, **parsed}
    except Exception as e:
        return {"checked": False, "error": str(e)}
