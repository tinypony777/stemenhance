from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class InstrumentGuess:
    label: str
    confidence: float
    reasons: List[str]


INSTRUMENT_CHOICES = [
    "auto",
    "vocals",
    "drums",
    "bass",
    "guitar",
    "keys",
    "music",
    "fx",
    "other",
]


_FILENAME_RULES: List[Tuple[str, str]] = [
    (r"(vocal|vox|leadvox|choir|chorus)", "vocals"),
    (r"(drum|kick|snare|hat|hihat|perc|percussion)", "drums"),
    (r"(bass|sub)", "bass"),
    (r"(guitar|gtr)", "guitar"),
    (r"(piano|keys|key|synth|pad|organ)", "keys"),
    (r"(fx|sfx|noise)", "fx"),
]


def _filename_guess(name: str) -> Optional[str]:
    s = os.path.basename(name).lower()
    s = s.replace(" ", "_")
    for pat, label in _FILENAME_RULES:
        if re.search(pat, s):
            return label
    return None


def guess_instrument(
    filename: str,
    y_mono: np.ndarray,
    sr: int,
    midi_hint: Optional[str] = None,
) -> InstrumentGuess:
    """Heuristic instrument guessing.

    This is intentionally lightweight (no large ML model).

    Priority:
    1) MIDI hint (track name / program bucket)
    2) filename keyword
    3) audio feature heuristic
    """

    reasons: List[str] = []

    # 1) MIDI hint (if provided)
    if midi_hint:
        mh = midi_hint.lower()
        for pat, label in _FILENAME_RULES:
            if re.search(pat, mh):
                reasons.append(f"midi_hint matched '{pat}'")
                return InstrumentGuess(label=label, confidence=0.85, reasons=reasons)

    # 2) filename
    fg = _filename_guess(filename)
    if fg is not None:
        reasons.append("filename keyword")
        return InstrumentGuess(label=fg, confidence=0.9, reasons=reasons)

    # 3) audio features
    try:
        from scipy.signal import stft

        y = y_mono.astype(np.float32)
        if y.size < sr:
            # too short
            return InstrumentGuess(label="music", confidence=0.25, reasons=["too short"])

        # Use short excerpt for speed
        max_sec = 30.0
        n = int(min(y.size, sr * max_sec))
        y = y[:n]

        # Frame-based features via STFT
        n_fft = 2048
        hop = 512
        noverlap = n_fft - hop
        freqs, _, Z = stft(
            y,
            fs=sr,
            window="hann",
            nperseg=n_fft,
            noverlap=noverlap,
            nfft=n_fft,
            boundary=None,
            padded=False,
        )
        mag = np.abs(Z) + 1e-12

        # Spectral centroid (Hz)
        centroid = float(np.mean((freqs[:, None] * mag).sum(axis=0) / mag.sum(axis=0)))

        # Spectral rolloff (Hz)
        cum = np.cumsum(mag, axis=0)
        total = cum[-1, :]
        thresh = 0.85 * total
        idx = np.argmax(cum >= thresh[None, :], axis=0)
        rolloff = float(np.mean(freqs[idx]))

        # Spectral flatness (geometric/arithmetic)
        flatness = float(np.mean(np.exp(np.mean(np.log(mag), axis=0)) / (np.mean(mag, axis=0) + 1e-12)))

        # Zero crossing rate
        zc = np.sum(np.abs(np.diff(np.sign(y))) > 0)
        zcr = float(zc / max(1, y.size))

        # Onset rate proxy: frame RMS derivative peaks
        frame_rms = np.sqrt(np.mean(mag * mag, axis=0))
        diff = np.diff(frame_rms)
        if diff.size > 0:
            thr2 = np.percentile(diff, 75)
            onset_peaks = np.sum(diff > thr2)
            seconds = (hop * frame_rms.size) / float(sr)
            onset_rate = float(onset_peaks / max(1e-6, seconds))
        else:
            onset_rate = 0.0

        # Low energy ratio (<200Hz)
        low_band = mag[freqs < 200.0]
        high_band = mag[freqs >= 200.0]
        low_ratio = float(np.sum(low_band) / (np.sum(low_band) + np.sum(high_band) + 1e-9))

        # Heuristics
        if low_ratio > 0.60 and centroid < 800:
            reasons.append(f"low_ratio={low_ratio:.2f}, centroid={centroid:.0f}")
            return InstrumentGuess(label="bass", confidence=0.55, reasons=reasons)

        if onset_rate > 2.0 and centroid > 1200 and flatness > 0.2:
            reasons.append(f"onset_rate={onset_rate:.2f}, centroid={centroid:.0f}, flatness={flatness:.2f}")
            return InstrumentGuess(label="drums", confidence=0.50, reasons=reasons)

        # Vocals often have mid centroid, moderate flatness, moderate zcr
        if 900 < centroid < 3000 and flatness < 0.35 and zcr < 0.12:
            reasons.append(f"centroid={centroid:.0f}, flatness={flatness:.2f}, zcr={zcr:.2f}")
            return InstrumentGuess(label="vocals", confidence=0.40, reasons=reasons)

        if centroid > 2500 and flatness < 0.25:
            reasons.append(f"centroid={centroid:.0f}")
            return InstrumentGuess(label="keys", confidence=0.35, reasons=reasons)

        # Default
        reasons.append(f"centroid={centroid:.0f}, rolloff={rolloff:.0f}, flatness={flatness:.2f}")
        return InstrumentGuess(label="music", confidence=0.25, reasons=reasons)

    except Exception as e:
        return InstrumentGuess(label="music", confidence=0.2, reasons=[f"feature error: {e}"])
