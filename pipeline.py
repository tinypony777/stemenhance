from __future__ import annotations

import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .audio_io import AudioData, apply_global_gain, load_audio, mono_mix, peak_dbfs, save_wav
from .cymbal import CymbalTameOptions, tame_cymbals
from .denoise import reduce_artifacts_and_hiss
from .instrument import InstrumentGuess, guess_instrument
from .phase import PhaseAlignOptions, align_stems
from .transient import TransientOptions, transient_restore


@dataclass
class StemItem:
    name: str
    path: str
    instrument: str = "auto"
    midi_hint: Optional[str] = None


@dataclass
class EnhanceOptions:
    target_sr: int = 48000
    artifact_strength: float = 0.55  # 0..1
    hiss_strength: float = 0.35  # 0..1
    transient_amount: float = 0.45  # 0..1
    cymbal_tame: float = 0.35  # 0..1 (mainly for drums / cymbals)
    do_phase_align: bool = True
    phase_anchor: str = "auto"  # stem name or "auto"
    phase_match: bool = False
    max_delay_ms: float = 80.0
    keep_relative_level: bool = True
    output_subtype: str = "PCM_24"


# Per-instrument multipliers (tunable)
PRESET_MULT = {
    "vocals": {"artifact": 0.45, "hiss": 0.70, "transient": 0.10, "cymbal": 0.0},
    "drums": {"artifact": 0.65, "hiss": 0.30, "transient": 0.85, "cymbal": 1.0},
    "bass": {"artifact": 0.30, "hiss": 0.10, "transient": 0.35, "cymbal": 0.0},
    "guitar": {"artifact": 0.55, "hiss": 0.45, "transient": 0.40, "cymbal": 0.0},
    "keys": {"artifact": 0.60, "hiss": 0.30, "transient": 0.20, "cymbal": 0.0},
    "music": {"artifact": 0.55, "hiss": 0.30, "transient": 0.20, "cymbal": 0.35},
    "fx": {"artifact": 0.75, "hiss": 0.55, "transient": 0.00, "cymbal": 0.0},
    "other": {"artifact": 0.50, "hiss": 0.30, "transient": 0.20, "cymbal": 0.0},
}


def _pad_to_length(y: np.ndarray, length: int) -> np.ndarray:
    if y.shape[0] == length:
        return y
    if y.shape[0] > length:
        return y[:length]
    pad = np.zeros((length - y.shape[0], y.shape[1]), dtype=y.dtype)
    return np.concatenate([y, pad], axis=0)


def _rms(y: np.ndarray) -> float:
    m = mono_mix(y)
    return float(np.sqrt(np.mean(m * m) + 1e-12))


def _choose_anchor(stems_audio: Dict[str, np.ndarray], stems_instruments: Dict[str, str]) -> str:
    # Prefer drums, then bass, else highest RMS
    for preferred in ["drums", "bass", "vocals"]:
        for k, inst in stems_instruments.items():
            if inst == preferred:
                return k

    best_k = None
    best = -1.0
    for k, y in stems_audio.items():
        r = _rms(y)
        if r > best:
            best = r
            best_k = k
    return best_k or list(stems_audio.keys())[0]


def analyze_and_guess_instruments(stems: List[StemItem], loaded: Dict[str, AudioData]) -> Dict[str, InstrumentGuess]:
    out: Dict[str, InstrumentGuess] = {}
    for item in stems:
        ad = loaded[item.name]
        guess = guess_instrument(item.name, mono_mix(ad.y), ad.sr, midi_hint=item.midi_hint)
        out[item.name] = guess
    return out


def enhance_stems(
    stems: List[StemItem],
    opt: EnhanceOptions,
    output_dir: Optional[str] = None,
) -> Tuple[str, Dict]:
    """Enhance stems and return a ZIP path + metadata.

    Returns:
      (zip_path, meta)
    """

    if len(stems) == 0:
        raise ValueError("No stems provided")

    # 1) load + resample
    loaded: Dict[str, AudioData] = {}
    for item in stems:
        ad = load_audio(item.path, target_sr=opt.target_sr)
        loaded[item.name] = ad

    # 2) pad to same length
    max_len = max(ad.y.shape[0] for ad in loaded.values())
    for k in list(loaded.keys()):
        ad = loaded[k]
        loaded[k] = AudioData(y=_pad_to_length(ad.y, max_len), sr=ad.sr)

    # 3) instrument mapping
    guesses = analyze_and_guess_instruments(stems, loaded)
    inst_map: Dict[str, str] = {}
    for item in stems:
        if item.instrument and item.instrument != "auto":
            inst_map[item.name] = item.instrument
        else:
            inst_map[item.name] = guesses[item.name].label

    # 4) per-stem processing
    processed: Dict[str, np.ndarray] = {}
    per_stem_params: Dict[str, Dict] = {}

    for item in stems:
        name = item.name
        y = loaded[name].y
        inst = inst_map[name]
        mult = PRESET_MULT.get(inst, PRESET_MULT["other"])

        a_strength = float(np.clip(opt.artifact_strength * mult["artifact"], 0.0, 1.0))
        h_strength = float(np.clip(opt.hiss_strength * mult["hiss"], 0.0, 1.0))
        t_amount = float(np.clip(opt.transient_amount * mult["transient"], 0.0, 1.0))
        c_amount = float(np.clip(opt.cymbal_tame * mult.get("cymbal", 0.0), 0.0, 1.0))

        y2 = y
        if a_strength > 0 or h_strength > 0:
            # slightly lower cutoff for vocals to avoid harsh air noise
            cut = 6500.0 if inst == "vocals" else 7000.0
            y2 = reduce_artifacts_and_hiss(y2, opt.target_sr, a_strength, h_strength, high_cut_hz=cut)

        # Cymbal-focused cleanup (mainly drums; optionally light on full music bus)
        if c_amount > 1e-6:
            y2 = tame_cymbals(
                y2,
                opt.target_sr,
                CymbalTameOptions(
                    strength=c_amount,
                    # tuned for common AI cymbal artifacts
                    cut1_hz=4500.0,
                    cut2_hz=7000.0,
                    attack_ms=3.0,
                    release_ms=220.0,
                    min_gain=0.28,
                    smooth_strength=0.20,
                    width_tighten=0.22,
                ),
            )
        if t_amount > 0:
            y2 = transient_restore(y2, opt.target_sr, TransientOptions(amount=t_amount))

        processed[name] = y2.astype(np.float32)
        per_stem_params[name] = {
            "instrument": inst,
            "artifact_strength": a_strength,
            "hiss_strength": h_strength,
            "transient_amount": t_amount,
            "cymbal_tame": c_amount,
            "peak_in_dbfs": peak_dbfs(y),
            "peak_out_dbfs": peak_dbfs(y2),
            "guess": {
                "label": guesses[name].label,
                "confidence": guesses[name].confidence,
                "reasons": guesses[name].reasons,
            },
        }

    # 5) phase align (time/polarity + optional phase-match)
    phase_meta = None
    if opt.do_phase_align:
        anchor_key = opt.phase_anchor
        if anchor_key == "auto" or anchor_key not in processed:
            anchor_key = _choose_anchor(processed, inst_map)

        aligned, align_result = align_stems(
            processed,
            sr=opt.target_sr,
            anchor_key=anchor_key,
            opt=PhaseAlignOptions(
                max_delay_ms=opt.max_delay_ms,
                do_polarity=True,
                phase_match=opt.phase_match,
            ),
        )
        processed = aligned
        phase_meta = {
            "anchor": anchor_key,
            "delays_samples": align_result.delays_samples,
            "inverted": align_result.inverted,
        }

    # 6) safety headroom (preserve relative levels)
    if opt.keep_relative_level:
        peak = max(float(np.max(np.abs(v))) for v in processed.values())
        if peak > 0.999:
            g = 0.999 / peak
            for k in list(processed.keys()):
                processed[k] = (processed[k] * g).astype(np.float32)

    # 7) write outputs
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="stem_enhanced_")
    os.makedirs(output_dir, exist_ok=True)

    out_files: List[str] = []
    for name, y in processed.items():
        base = os.path.splitext(os.path.basename(name))[0]
        out_path = os.path.join(output_dir, f"{base}_enhanced.wav")
        save_wav(out_path, y, opt.target_sr, subtype=opt.output_subtype)
        out_files.append(out_path)

    # write meta
    meta = {
        "options": opt.__dict__,
        "instrument_map": inst_map,
        "per_stem": per_stem_params,
        "phase": phase_meta,
    }
    meta_path = os.path.join(output_dir, "enhance_report.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    out_files.append(meta_path)

    # 8) zip
    zip_path = os.path.join(output_dir, "enhanced_stems.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in out_files:
            z.write(p, arcname=os.path.basename(p))

    return zip_path, meta
