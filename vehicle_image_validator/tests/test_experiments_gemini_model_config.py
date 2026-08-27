"""Confirms each stage's Gemini call uses its own configurable model
(VFIV_GEMINI_MODEL_Q1/Q2/Q3_MAKE/Q3_MODEL, falling back to VFIV_GEMINI_MODEL) rather
than one hardcoded/global model — verified by capturing the `model` kwarg actually
passed to call_gemini_json, no live API call needed.

NOTE: deliberately does NOT test the env-var fallback via importlib.reload(config) —
reloading a shared module mid-suite leaks into every other test that imports `config`
(monkeypatch's env-var teardown runs AFTER the test body, so a "restore" reload inside
the test still sees the patched env and bakes in the wrong value for the rest of the
process). monkeypatch.setattr on the already-computed config values (below) is safe;
reloading the module that computes them is not.
"""
from vfiv import config
from vfiv.experiments import q1_select, q2_select, q3_select


def _capture_model(monkeypatch, module):
    captured = {}

    def fake_call_gemini_json(image, prompt, model=None):
        captured["model"] = model
        return {"checked": False, "error": "stubbed"}

    monkeypatch.setattr(module, "call_gemini_json", fake_call_gemini_json)
    return captured


def test_q1_gemini_uses_its_own_configured_model(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_MODEL_Q1", "gemini-test-q1")
    captured = _capture_model(monkeypatch, q1_select)
    q1_select.classify_q1("does-not-matter.jpg", backend="gemini")
    assert captured["model"] == "gemini-test-q1"


def test_q2_gemini_uses_its_own_configured_model(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_MODEL_Q2", "gemini-test-q2")
    captured = _capture_model(monkeypatch, q2_select)
    q2_select.classify_q2("does-not-matter.jpg", backend="gemini")
    assert captured["model"] == "gemini-test-q2"


def test_q3_make_gemini_uses_its_own_configured_model(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_MODEL_Q3_MAKE", "gemini-test-q3-make")
    captured = _capture_model(monkeypatch, q3_select)
    q3_select.collect_make_votes("does-not-matter.jpg", backend="gemini")
    assert captured["model"] == "gemini-test-q3-make"


def test_q3_model_gemini_uses_its_own_configured_model(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_MODEL_Q3_MODEL", "gemini-test-q3-model")
    captured = _capture_model(monkeypatch, q3_select)
    q3_select.classify_model("does-not-matter.jpg", backend="gemini")
    assert captured["model"] == "gemini-test-q3-model"
