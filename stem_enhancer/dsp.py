from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import ndimage, signal


DRUM_LIKE = {"drums", "cymbals", "overheads", "percussion"}


def smooth_mask(mask: np.ndarray, strength: float) -> np.ndarray:
    """Reduce musical-noise artifacts by smoothing the time-frequency mask."""
    if strength <= 0:
        return mask

    strength = float(np.clip(strength, 0.0, 1.0))
    kt = 1 + 2 * int(strength * 3)  # 1..7
    kf = 1 + 2 * int(strength * 1)  # 1..3
    m = ndimage.median_filter(mask, size=(kf, kt), mode="nearest")
    sigma_t = 0.2 + 0.8 * strength
    sigma_f = 0.0 + 0.5 * strength
    m = ndimage.gaussian_filter(m, sigma=(sigma_f, sigma_t), mode="nearest")
    return m.astype(np.float32)


def compute_onset_envelope_from_mag(freqs: np.ndarray, mag: np.ndarray, f_lo: float = 2000, f_hi: float = 18000) -> np.ndarray:
    """Compute a simple onset envelope from a magnitude spectrogram.

    Returns shape (t,) normalized to 0..1.
    """
    band = (freqs >= f_lo) & (freqs <= f_hi)
    if not np.any(band):
        band = slice(None)
    m = mag[band]
    if m.shape[1] < 2:
        return np.zeros((mag.shape[1],), dtype=np.float32)
    dm = m[:, 1:] - m[:, :-1]
    flux = np.sum(np.maximum(dm, 0.0), axis=0)
    flux = np.concatenate([[0.0], flux]).astype(np.float32)
    if flux.size >= 5:
        flux = ndimage.uniform_filter1d(flux, size=5, mode="nearest")
    p = float(np.percentile(flux, 95)) if flux.size else 1.0
    if p <= 1e-9:
        return np.zeros_like(flux)
    return np.clip(flux / p, 0.0, 1.0).astype(np.float32)


def transient_boost_mask(freqs: np.ndarray, onset_env: np.ndarray, label: str, strength: float) -> np.ndarray:
    """Create a multiplicative mask (f,t) to boost transients for drum-like stems."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return np.ones((freqs.shape[0], onset_env.shape[0]), dtype=np.float32)

    t = onset_env[None, :]
    f = freqs[:, None]

    if label == "cymbals":
        fw = np.zeros_like(f, dtype=np.float32)
        fw += ((f >= 3000) & (f < 6000)).astype(np.float32) * 0.5
        fw += ((f >= 6000) & (f <= 18000)).astype(np.float32) * 1.0
    else:
        fw = np.zeros_like(f, dtype=np.float32)
        fw += ((f >= 1500) & (f < 8000)).astype(np.float32) * 1.0
        fw += ((f >= 8000) & (f <= 16000)).astype(np.float32) * 0.7
        fw += ((f >= 16000) & (f <= 18000)).astype(np.float32) * 0.4

    return (1.0 + (1.8 * strength) * (fw * t)).astype(np.float32)


def apply_highband_cymbal_tame(
    audio: np.ndarray,
    sr: int,
    strength: float,
    band: Tuple[float, float] = (6000.0, 18000.0),
) -> np.ndarray:
    """Shorten cymbal sustain and reduce wide sizzle (high band only)."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return audio

    x = audio.astype(np.float32)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)

    lo, hi = band
    nyq = 0.5 * sr
    hi = min(hi, nyq * 0.98)
    if lo >= hi:
        return audio

    sos = signal.butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    hb = signal.sosfiltfilt(sos, x, axis=0).astype(np.float32)

    mono = np.mean(hb, axis=1)
    abs_m = np.abs(mono).astype(np.float32)
    win_long = max(1, int(sr * 0.26))
    win_short = max(1, int(sr * 0.08))
    env_long = ndimage.uniform_filter1d(abs_m, size=win_long, mode="nearest")
    env_short = ndimage.uniform_filter1d(abs_m, size=win_short, mode="nearest")

    ratio = env_short / (env_long + 1e-6)
    ratio = np.clip(ratio, 0.2, 1.0)

    gain = (1.0 - (0.85 * strength) * (1.0 - ratio)).astype(np.float32)
    hb2 = hb * gain[:, None]

    # Reduce high-band stereo sizzle width a bit
    if hb2.shape[1] >= 2:
        mid = 0.5 * (hb2[:, 0] + hb2[:, 1])
        side = 0.5 * (hb2[:, 0] - hb2[:, 1])
        side *= (1.0 - 0.55 * strength)
        hb2[:, 0] = mid + side
        hb2[:, 1] = mid - side

    return (x + (hb2 - hb)).astype(np.float32)


def apply_cymbal_attack_restore(
    stem_audio: np.ndarray,
    mix_audio: np.ndarray,
    sr: int,
    strength: float,
    band: Tuple[float, float] = (6000.0, 18000.0),
) -> np.ndarray:
    """Restore cymbal attack by injecting mix highband at missing-onset moments."""
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0:
        return stem_audio

    x = stem_audio.astype(np.float32)
    m = mix_audio.astype(np.float32)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    if m.ndim == 1:
        m = np.stack([m, m], axis=1)

    n = min(x.shape[0], m.shape[0])
    x = x[:n]
    m = m[:n]

    lo, hi = band
    nyq = 0.5 * sr
    hi = min(hi, nyq * 0.98)
    if lo >= hi:
        return stem_audio

    sos = signal.butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    x_hb = signal.sosfiltfilt(sos, x, axis=0).astype(np.float32)
    m_hb = signal.sosfiltfilt(sos, m, axis=0).astype(np.float32)

    def onset_curve(hb: np.ndarray) -> np.ndarray:
        mono = np.mean(hb, axis=1)
        abs_m = np.abs(mono).astype(np.float32)
        win_fast = max(1, int(sr * 0.06))
        win_slow = max(1, int(sr * 0.26))
        env_fast = ndimage.uniform_filter1d(abs_m, size=win_fast, mode="nearest")
        env_slow = ndimage.uniform_filter1d(abs_m, size=win_slow, mode="nearest")
        onset = np.maximum(env_fast - env_slow, 0.0)
        p = float(np.percentile(onset, 99)) if onset.size else 1.0
        if p <= 1e-9:
            return np.zeros_like(onset)
        return np.clip(onset / p, 0.0, 1.0).astype(np.float32)

    o_mix = onset_curve(m_hb)
    o_stem = onset_curve(x_hb)
    missing = np.clip(o_mix - o_stem, 0.0, 1.0)

    inj = m_hb * (missing[:, None] * (0.9 * strength))
    return (x + inj).astype(np.float32)
