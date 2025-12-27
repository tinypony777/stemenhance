from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass
class Audio:
    sr: int
    data: np.ndarray  # shape (n, ch), float32


def load_audio(path: str) -> Audio:
    """Load audio via soundfile (libsndfile).

    WAV/AIFF/FLAC should work out of the box.
    MP3/M4A are not guaranteed; convert to WAV if needed.
    """
    try:
        import soundfile as sf
    except Exception as e:
        raise RuntimeError(
            "soundfile is required. If installation failed, install libsndfile (brew install libsndfile)."
        ) from e

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    data, sr = sf.read(path, always_2d=True)
    data = np.asarray(data, dtype=np.float32)
    return Audio(sr=int(sr), data=data)


def write_wav(path: str, audio: Audio) -> None:
    import soundfile as sf

    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, audio.data, audio.sr, subtype="PCM_24")


def ensure_stereo(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return np.stack([x, x], axis=1)
    if x.shape[1] == 1:
        return np.concatenate([x, x], axis=1)
    return x


def peak_normalize_safe(x: np.ndarray, peak: float = 0.99) -> np.ndarray:
    m = float(np.max(np.abs(x))) if x.size else 0.0
    if m > peak and m > 0:
        x = x * (peak / m)
    return x


def resample(audio: Audio, sr_out: int) -> Audio:
    if audio.sr == sr_out:
        return audio

    from scipy.signal import resample_poly

    x = audio.data
    sr_in = audio.sr
    g = int(np.gcd(sr_in, sr_out))
    up = sr_out // g
    down = sr_in // g

    ys = []
    for ch in range(x.shape[1]):
        y_ch = resample_poly(x[:, ch], up=up, down=down).astype(np.float32)
        ys.append(y_ch)
    y = np.stack(ys, axis=1)
    return Audio(sr=sr_out, data=y)


def pad_or_trim(audio: Audio, n_samples: int) -> Audio:
    x = audio.data
    if x.shape[0] == n_samples:
        return audio
    if x.shape[0] > n_samples:
        return Audio(sr=audio.sr, data=x[:n_samples])
    pad = n_samples - x.shape[0]
    x2 = np.pad(x, ((0, pad), (0, 0)), mode="constant")
    return Audio(sr=audio.sr, data=x2)
