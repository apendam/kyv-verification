"""Offline tests for the new experimentation backends (Gemini, Google Cloud Vision
Logo Detection, Clarifai) — none have credentials configured in this environment, so
these confirm graceful degradation, matching backends/qwen.py's pattern."""
from vfiv.backends import clarifai_backend, google_vision
from vfiv.backends.gemini import call_gemini_json


def test_gemini_not_configured_without_any_credentials(monkeypatch, tmp_path):
    """Neither auth mode configured (API key OR Vertex AI service account) -> not
    configured. Must clear all three env vars — Vertex AI credentials are configured
    in this environment now, so clearing just GEMINI_API_KEY would fall through to
    the (working) Vertex AI path instead of failing."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    monkeypatch.delenv("GEMINI_VERTEX_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    from PIL import Image
    img_path = tmp_path / "blank.jpg"
    Image.new("RGB", (10, 10)).save(img_path)

    result = call_gemini_json(str(img_path), "irrelevant prompt")
    assert result["checked"] is False
    assert "GEMINI_API_KEY" in result["error"]


def test_gemini_falls_back_to_vertex_when_only_api_key_missing(monkeypatch, tmp_path):
    """Confirms the fallback actually engages: with GEMINI_API_KEY absent but Vertex
    AI credentials present (as they are in this environment), the call still
    succeeds rather than reporting unavailable."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    if not __import__("os").environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        import pytest
        pytest.skip("requires GOOGLE_APPLICATION_CREDENTIALS + GEMINI_VERTEX_PROJECT")
    from PIL import Image
    img_path = tmp_path / "blank.jpg"
    Image.new("RGB", (10, 10)).save(img_path)

    result = call_gemini_json(str(img_path), 'Reply with STRICT JSON only: {"ok": true}')
    assert result["checked"] is True


def test_gcv_logo_not_configured_without_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
    from PIL import Image
    img_path = tmp_path / "blank.jpg"
    Image.new("RGB", (10, 10)).save(img_path)

    result = google_vision.classify_logo(str(img_path))
    assert result["checked"] is False
    assert "GOOGLE_APPLICATION_CREDENTIALS" in result["error"]


def test_clarifai_not_configured_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("CLARIFAI_API_KEY", raising=False)
    from PIL import Image
    img_path = tmp_path / "blank.jpg"
    Image.new("RGB", (10, 10)).save(img_path)

    result = clarifai_backend.classify_logo(str(img_path))
    assert result["checked"] is False
    assert "CLARIFAI_API_KEY" in result["error"]
