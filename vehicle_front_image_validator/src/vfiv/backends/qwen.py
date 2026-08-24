"""Qwen2.5-VL wrapper — copied from truck_front_extractor/src/tfe/backends/real.py's
``_QwenVLM`` class. Real code, NOT run live in this environment: the model is ~16GB and
documented as taking minutes/image on CPU (no GPU here) — wiring it now means it's ready
to point at a GPU box in production without further porting, but Q3's model reading
defaults to Claude (see validators/make_model_check.py) until then.

Opt in explicitly once a GPU box is available:
    from vfiv.backends.qwen import get_qwen
    result = get_qwen().make_model(truck_crop)   # {make, make_conf, model, model_conf}
"""
from __future__ import annotations

import json
import re
from typing import Optional

from vfiv import config
from vfiv.backends.device import resolve_device


class QwenVLM:
    """Heavy; lazy-loads on first use."""

    def __init__(self):
        self.model = None

    def warmup(self):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        mid = config.QWEN_MODEL
        dev = resolve_device(config.DEVICE)
        self.proc = AutoProcessor.from_pretrained(mid)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            mid, torch_dtype=torch.float32 if dev == "cpu" else "auto").to(dev).eval()

    def _ask(self, crop, prompt: str, max_new_tokens: int = 128) -> str:
        import torch
        from PIL import Image
        if self.model is None:
            self.warmup()
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        img = crop if isinstance(crop, Image.Image) else Image.fromarray(crop)
        inp = self.proc(text=[text], images=[img], return_tensors="pt").to(resolve_device(config.DEVICE))
        with torch.no_grad():
            out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[:, inp.input_ids.shape[1]:]
        return self.proc.batch_decode(gen, skip_special_tokens=True)[0].strip()

    def make_model(self, truck_crop) -> dict:
        ans = self._ask(truck_crop,
            'Identify this truck. Reply ONLY JSON: '
            '{"make": "...", "model": "...", "confidence": 0-1}')
        d = _parse_json(ans)
        c = float(d.get("confidence", 0.7))
        return {"make": d.get("make"), "make_conf": c * 100.0,
                "model": d.get("model"), "model_conf": c * 100.0}


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else {}
    except Exception:  # noqa: BLE001
        return {}


_qwen: Optional[QwenVLM] = None


def get_qwen() -> QwenVLM:
    global _qwen
    if _qwen is None:
        _qwen = QwenVLM()
    return _qwen
