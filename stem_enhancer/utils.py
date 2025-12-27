from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional


def _extract_path(obj: Any) -> Optional[str]:
    """Extract a file path from Gradio's file payloads."""
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict) and "name" in obj:
        return obj["name"]
    if hasattr(obj, "name"):
        return getattr(obj, "name")
    return None


def normalize_file_list(files: Any) -> List[str]:
    if not files:
        return []
    if isinstance(files, (list, tuple)):
        out: List[str] = []
        for f in files:
            p = _extract_path(f)
            if p:
                out.append(p)
        return out
    p = _extract_path(files)
    return [p] if p else []


def basename(path: str) -> str:
    return os.path.basename(path)


def parse_mapping_text(text: str) -> Dict[str, str]:
    """Parse a simple mapping from UI text.

    Supported:
    - JSON dict: {"file.wav": "drums"}
    - Line format: file.wav: drums
    - Line format: file.wav = drums
    """
    text = (text or "").strip()
    if not text:
        return {}

    if text.startswith("{"):
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return {str(k): str(v) for k, v in obj.items()}
        except Exception:
            pass

    mapping: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sep = ":" if ":" in line else ("=" if "=" in line else None)
        if not sep:
            continue
        left, right = line.split(sep, 1)
        key = left.strip()
        val = right.strip()
        if key and val:
            mapping[key] = val
    return mapping


def infer_instrument_label(filename: str) -> str:
    s = filename.lower()

    if any(k in s for k in ["cymbal", "ride", "crash", "hihat", "hi-hat", "hat", "overhead", "oh"]):
        return "cymbals"
    if any(k in s for k in ["drum", "kick", "snare", "tom", "perc", "percussion"]):
        return "drums"
    if "bass" in s or "sub" in s:
        return "bass"
    if "back" in s and "vocal" in s:
        return "backing_vocals"
    if any(k in s for k in ["vocal", "vox", "leadvox", "lead_vocal", "mainvocal"]):
        return "vocals"
    if "guitar" in s:
        return "guitar"
    if any(k in s for k in ["keys", "key", "keyboard", "piano", "rhodes"]):
        return "keys"
    if any(k in s for k in ["synth", "pad"]):
        return "synth"
    if any(k in s for k in ["fx", "sfx"]):
        return "fx"
    return "other"
