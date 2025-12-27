from __future__ import annotations

"""Cymbal / high-frequency repair utilities.

AI-generated 2-mix (and stems derived from it) often exhibits:

- Over-long cymbal sustain (ride / crash tails that never die)
- Digital "fizz" / grain / metallic sizzle
- Unnaturally wide, phasey high band

This module provides an *offline* DSP approach that is intentionally conservative:

- High-band sustain reduction using a two-time-constant envelope.
- Optional high-band median smoothing (no hard gating) to reduce "musical noise".
- Optional high-band width tightening (M/S) to reduce phasey sizzle.

It cannot reconstruct information that never existed, but it can make common
generation artifacts noticeably less objectionable.
"""

from dataclasses import dataclass

import numpy as np

from .audio_io import _ensure_2d


@dataclass
class CymbalTameOptions:
    """Parameters for cymbal taming.

    strength: overall 0..1.
    cut1_hz: start of "cymbal" band.
    cut2_hz: start of "sizzle" band.
    attack_ms / release_ms: envelope follower constants for sustain reduction.
    min_gain: lower bound for sustain gain reduction.
    smooth_strength: 0..1 amount of STFT median smoothing applied only to the sizzle band.
    width_tighten: 0..1 amount of stereo narrowing applied only to the high band.
    """

    strength: float = 0.45
    cut1_hz: float = 4500.0
    cut2_hz: float = 7000.0
    attack_ms: float = 3.0
    release_ms: float = 220.0
    min_gain: float = 0.28
    smooth_strength: float = 0.20
    width_tighten: float = 0.20


def _env_follower(abs_x: np.ndarray, sr: int, tau_ms: float) -> np.ndarray:
    """One-pole IIR smoothing of absolute value."""

    from scipy.signal import lfilter

    tau = max(0.1, float(tau_ms)) / 1000.0
    alpha = np.exp(-1.0 / (sr * tau))
    b = [1.0 - alpha]
    a = [1.0, -alpha]
    return lfilter(b, a, abs_x)


def _sustain_reduce(x: np.ndarray, sr: int, amount: float, attack_ms: float, release_ms: float, min_gain: float) -> np.ndarray:
    """Reduce sustain while preserving attack.

    Uses env_fast (attack) and env_slow (release). When signal is decaying,
    env_fast drops faster than env_slow, and we treat that region as "sustain".

    A soft gate based on a percentile of env_slow prevents over-processing near
    the noise floor.
    """

    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 1e-6:
        return x.astype(np.float32)

    ax = np.abs(x).astype(np.float32)
    env_fast = _env_follower(ax, sr, tau_ms=attack_ms)
    env_slow = _env_follower(ax, sr, tau_ms=release_ms)

    sustain = np.maximum(env_slow - env_fast, 0.0)
    ratio = sustain / (env_slow + 1e-6)

    # Soft gate based on an estimate of the high-band "noise" level.
    nf = float(np.percentile(env_slow, 10.0))
    t0 = nf * 2.0
    t1 = nf * 8.0
    gate = np.clip((env_slow - t0) / (t1 - t0 + 1e-12), 0.0, 1.0)

    gain = 1.0 - gate * (amount * ratio)
    gain = np.clip(gain, float(np.clip(min_gain, 0.05, 0.95)), 1.0)

    return (x * gain).astype(np.float32)


def _spectral_median_smooth(y: np.ndarray, sr: int, strength: float, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    """Median smoothing in STFT magnitude domain.

    This is NOT a hard gate. It keeps energy but removes isolated time/freq grains.
    """

    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 1e-6:
        return y.astype(np.float32)

    from scipy.ndimage import median_filter
    from scipy.signal import stft, istft
    import warnings

    x = y.astype(np.float32)
    nperseg = n_fft
    noverlap = n_fft - hop
    _, _, S = stft(
        x,
        fs=sr,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    mag = np.abs(S)
    phase = np.exp(1j * np.angle(S))

    # Slightly larger median in time than in freq, tuned for cymbal grain.
    mag_med = median_filter(mag, size=(5, 9))
    mag_out = (1.0 - strength) * mag + strength * mag_med
    S_out = mag_out * phase

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        _, x_out = istft(
            S_out,
            fs=sr,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=n_fft,
            input_onesided=True,
            boundary=None,
        )

    # Match original length
    if x_out.size < x.size:
        x_out = np.pad(x_out, (0, x.size - x_out.size))
    elif x_out.size > x.size:
        x_out = x_out[: x.size]

    return x_out.astype(np.float32)


def tame_cymbals(y: np.ndarray, sr: int, opt: CymbalTameOptions) -> np.ndarray:
    """Apply cymbal-focused cleanup.

    Notes:
    - Uses zero-phase filters (sosfiltfilt). Offline-safe.
    - Applies processing only to high bands; the remainder is untouched.
    """

    y2d = _ensure_2d(y).astype(np.float32)
    strength = float(np.clip(opt.strength, 0.0, 1.0))
    if strength <= 1e-6:
        return y2d

    from scipy.signal import butter, sosfiltfilt

    # High split filters
    sos_hp1 = butter(4, float(opt.cut1_hz), btype="highpass", fs=sr, output="sos")
    sos_hp2 = butter(4, float(opt.cut2_hz), btype="highpass", fs=sr, output="sos")

    # Per-channel processing
    rest = np.zeros_like(y2d)
    hi = np.zeros_like(y2d)

    for ch in range(y2d.shape[1]):
        x = y2d[:, ch]
        hi1plus = sosfiltfilt(sos_hp1, x)
        hi2 = sosfiltfilt(sos_hp2, x)
        hi1 = hi1plus - hi2
        rest[:, ch] = (x - hi1plus).astype(np.float32)

        # Sustain reduction: strongest on the top band
        hi1r = _sustain_reduce(
            hi1,
            sr,
            amount=strength * 0.55,
            attack_ms=opt.attack_ms,
            release_ms=opt.release_ms * 0.85,
            min_gain=max(0.45, opt.min_gain + 0.10),
        )
        hi2r = _sustain_reduce(
            hi2,
            sr,
            amount=strength * 1.00,
            attack_ms=opt.attack_ms,
            release_ms=opt.release_ms,
            min_gain=opt.min_gain,
        )

        # Optional "de-fizz" smoothing on the sizzle band.
        sm = float(np.clip(opt.smooth_strength * strength, 0.0, 1.0))
        if sm > 1e-6:
            hi2r = _spectral_median_smooth(hi2r, sr, strength=sm)

        hi[:, ch] = (hi1r + hi2r).astype(np.float32)

    # Optional high-band width tightening (only if stereo)
    wt = float(np.clip(opt.width_tighten * strength, 0.0, 1.0))
    if y2d.shape[1] == 2 and wt > 1e-6:
        hiL = hi[:, 0]
        hiR = hi[:, 1]
        m = 0.5 * (hiL + hiR)
        s = 0.5 * (hiL - hiR)
        s = s * (1.0 - wt)
        hi[:, 0] = (m + s).astype(np.float32)
        hi[:, 1] = (m - s).astype(np.float32)

    return (rest + hi).astype(np.float32)
