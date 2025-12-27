from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class AudioData:
    """Simple container for multichannel audio."""

    y: np.ndarray  # shape (n_samples, n_channels), float32 in [-1, 1]
    sr: int


def _ensure_2d(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y[:, None]
    if y.ndim == 2:
        return y
    raise ValueError(f"Audio array must be 1D or 2D, got shape={y.shape}")


def _to_float32(y: np.ndarray) -> np.ndarray:
    if y.dtype.kind == "f":
        y = y.astype(np.float32, copy=False)
    else:
        # assume integer PCM
        max_val = np.iinfo(y.dtype).max
        y = (y.astype(np.float32) / float(max_val)).astype(np.float32)
    return y


def load_audio(path: str, target_sr: Optional[int] = None) -> AudioData:
    """Load an audio file as float32.

    - Uses soundfile first (fast, reliable for wav/flac/aiff).
    - Falls back to pydub (requires ffmpeg) for formats like mp3.
    - Optionally resamples to `target_sr`.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()

    y: np.ndarray
    sr: int

    try:
        import soundfile as sf

        y, sr = sf.read(path, always_2d=True)
        y = _to_float32(y)
    except Exception:
        # Fallback: pydub
        try:
            from pydub import AudioSegment

            seg = AudioSegment.from_file(path)
            sr = seg.frame_rate
            samples = np.array(seg.get_array_of_samples())
            if seg.channels > 1:
                samples = samples.reshape((-1, seg.channels))
            else:
                samples = samples[:, None]

            # Convert to float32
            y = samples.astype(np.float32) / float(1 << (8 * seg.sample_width - 1))
        except Exception as e:
            raise RuntimeError(
                f"Failed to load audio: {path}. Install ffmpeg or provide WAV/FLAC/AIFF.\nOriginal error: {e}"
            )

    y = _ensure_2d(y)

    if target_sr is not None and target_sr != sr:
        y = resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return AudioData(y=y, sr=sr)


def resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """High-quality resample using scipy (polyphase).

    We avoid heavy optional deps (and JIT backends) for portability.
    """

    from fractions import Fraction
    from scipy.signal import resample_poly

    y2d = _ensure_2d(y)

    # Rational approximation of the sample-rate ratio
    ratio = Fraction(target_sr, orig_sr).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator

    out = []
    for ch in range(y2d.shape[1]):
        out.append(resample_poly(y2d[:, ch], up, down))
    y_rs = np.stack(out, axis=1)
    return y_rs.astype(np.float32)


def save_wav(path: str, y: np.ndarray, sr: int, subtype: str = "PCM_24") -> None:
    """Save as WAV using soundfile."""

    import soundfile as sf

    y2d = _ensure_2d(y).astype(np.float32)
    # Safety clamp
    y2d = np.clip(y2d, -1.0, 1.0)
    sf.write(path, y2d, sr, subtype=subtype)


def mono_mix(y: np.ndarray) -> np.ndarray:
    y2d = _ensure_2d(y)
    return np.mean(y2d, axis=1)


def peak_dbfs(y: np.ndarray) -> float:
    y2d = _ensure_2d(y)
    peak = float(np.max(np.abs(y2d)))
    if peak <= 0:
        return -np.inf
    return 20.0 * np.log10(peak)


def apply_global_gain(y: np.ndarray, gain_db: float) -> np.ndarray:
    g = 10 ** (gain_db / 20.0)
    return (_ensure_2d(y) * g).astype(np.float32)
