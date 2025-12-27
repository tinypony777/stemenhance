from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .audio_io import _ensure_2d


@dataclass
class SpectralGateOptions:
    strength: float = 0.5  # 0..1
    n_fft: int = 2048
    hop_length: int = 512
    noise_percentile: float = 10.0
    smooth_time: int = 5
    smooth_freq: int = 3
    high_only: bool = False
    high_cut_hz: float = 8000.0
    residual_mix: float = 0.05  # keep tiny residual to avoid “underwater”


def _softmask(mag: np.ndarray, thresh: np.ndarray, power: float = 2.0) -> np.ndarray:
    # wiener-like softmask
    mag_p = np.power(mag, power)
    thr_p = np.power(thresh, power)
    return mag_p / (mag_p + thr_p + 1e-12)


def spectral_gate(y: np.ndarray, sr: int, opt: SpectralGateOptions) -> np.ndarray:
    """STFT-domain artifact/noise reduction.

    - Estimates a per-frequency noise floor by percentile.
    - Applies a soft mask + optional smoothing.

    This is designed to be conservative by default.
    """

    strength = float(np.clip(opt.strength, 0.0, 1.0))
    if strength <= 1e-6:
        return _ensure_2d(y).astype(np.float32)

    from scipy.ndimage import median_filter
    from scipy.signal import stft, istft
    import warnings

    y2d = _ensure_2d(y)
    out = np.zeros_like(y2d, dtype=np.float32)

    for ch in range(y2d.shape[1]):
        x = y2d[:, ch].astype(np.float32)
        nperseg = opt.n_fft
        noverlap = opt.n_fft - opt.hop_length
        _, _, S = stft(
            x,
            fs=sr,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=opt.n_fft,
            boundary=None,
            padded=False,
        )
        mag = np.abs(S)
        phase = np.exp(1j * np.angle(S))

        # median smoothing (reduces isolated musical noise)
        mag_med = median_filter(mag, size=(opt.smooth_freq, opt.smooth_time))
        mag_smooth = (1.0 - strength) * mag + strength * mag_med

        # noise floor estimate (per freq)
        noise = np.percentile(mag_smooth, opt.noise_percentile, axis=1, keepdims=True)

        # threshold scaling
        # strength=0.5 -> ~2.5x, strength=1 -> ~5x
        scale = 1.0 + 4.0 * strength
        thresh = noise * scale

        if opt.high_only:
            freqs = np.fft.rfftfreq(opt.n_fft, d=1.0 / sr)
            high_mask = (freqs[:, None] >= opt.high_cut_hz).astype(np.float32)
            # For low frequencies, keep as-is (mask=1)
            mask = _softmask(mag_smooth, thresh)
            mask = high_mask * mask + (1.0 - high_mask) * 1.0
        else:
            mask = _softmask(mag_smooth, thresh)

        # Keep a tiny residual to avoid metallic / over-gated feel
        mag_out = mag_smooth * mask + mag * (1.0 - mask) * float(opt.residual_mix)

        S_out = mag_out * phase

        with warnings.catch_warnings():
            # SciPy may warn about perfect invertibility; we handle length explicitly.
            warnings.simplefilter("ignore", UserWarning)
            _, x_out = istft(
                S_out,
                fs=sr,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=opt.n_fft,
                input_onesided=True,
                boundary=None,
            )
        # match original length
        if x_out.size < x.size:
            x_out = np.pad(x_out, (0, x.size - x_out.size))
        elif x_out.size > x.size:
            x_out = x_out[: x.size]

        out[:, ch] = x_out.astype(np.float32)

    return out


def reduce_artifacts_and_hiss(
    y: np.ndarray,
    sr: int,
    artifact_strength: float,
    hiss_strength: float,
    high_cut_hz: float = 7000.0,
) -> np.ndarray:
    """Combined pass to reduce compute.

    We apply a base gate/smoothing (artifact_strength) and add extra gating
    only in the high band (hiss_strength).
    """

    a = float(np.clip(artifact_strength, 0.0, 1.0))
    h = float(np.clip(hiss_strength, 0.0, 1.0))
    if a <= 1e-6 and h <= 1e-6:
        return _ensure_2d(y).astype(np.float32)

    # Merge into one option pass. We keep median smoothing proportional to artifact.
    opt = SpectralGateOptions(
        strength=max(a, h),
        noise_percentile=10.0,
        smooth_time=7,
        smooth_freq=3,
        high_only=False,
        residual_mix=0.06,
    )

    from scipy.ndimage import median_filter
    from scipy.signal import stft, istft
    import warnings

    y2d = _ensure_2d(y)
    out = np.zeros_like(y2d, dtype=np.float32)

    # Precompute frequency mask
    freqs = np.fft.rfftfreq(opt.n_fft, d=1.0 / sr)
    high_mask = (freqs[:, None] >= high_cut_hz).astype(np.float32)

    for ch in range(y2d.shape[1]):
        x = y2d[:, ch].astype(np.float32)
        nperseg = opt.n_fft
        noverlap = opt.n_fft - opt.hop_length
        _, _, S = stft(
            x,
            fs=sr,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=opt.n_fft,
            boundary=None,
            padded=False,
        )
        mag = np.abs(S)
        phase = np.exp(1j * np.angle(S))

        # smoothing focuses on artifacts (a)
        if a > 1e-6:
            mag_med = median_filter(mag, size=(opt.smooth_freq, opt.smooth_time))
            mag_smooth = (1.0 - a) * mag + a * mag_med
        else:
            mag_smooth = mag

        noise = np.percentile(mag_smooth, 10.0, axis=1, keepdims=True)

        # Threshold scale: low band uses artifact_strength, high band adds hiss_strength
        scale_low = 1.0 + 4.0 * a
        scale_high = 1.0 + 4.0 * min(1.0, a + h)
        scale = scale_low * (1.0 - high_mask) + scale_high * high_mask
        thresh = noise * scale

        mask = _softmask(mag_smooth, thresh)
        mag_out = mag_smooth * mask + mag * (1.0 - mask) * float(opt.residual_mix)
        S_out = mag_out * phase

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _, x_out = istft(
                S_out,
                fs=sr,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=opt.n_fft,
                input_onesided=True,
                boundary=None,
            )
        if x_out.size < x.size:
            x_out = np.pad(x_out, (0, x.size - x_out.size))
        elif x_out.size > x.size:
            x_out = x_out[: x.size]

        out[:, ch] = x_out.astype(np.float32)

    return out


def reduce_artifacts(y: np.ndarray, sr: int, strength: float) -> np.ndarray:
    """Convenience wrapper tuned for stem separation artifacts."""

    opt = SpectralGateOptions(
        strength=strength,
        noise_percentile=10.0,
        smooth_time=7,
        smooth_freq=3,
        high_only=False,
        residual_mix=0.07,
    )
    return spectral_gate(y, sr, opt)


def reduce_hiss(y: np.ndarray, sr: int, strength: float, high_cut_hz: float = 7000.0) -> np.ndarray:
    """Convenience wrapper for hiss / high-frequency junk."""

    opt = SpectralGateOptions(
        strength=strength,
        noise_percentile=15.0,
        smooth_time=5,
        smooth_freq=3,
        high_only=True,
        high_cut_hz=high_cut_hz,
        residual_mix=0.05,
    )
    return spectral_gate(y, sr, opt)
