from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class AlignResult:
    estimated_delay_samples: int
    estimated_delay_seconds: float
    confidence: float
    polarity_inverted: bool


def _to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    return np.mean(x, axis=1)


def gcc_phat(sig: np.ndarray, refsig: np.ndarray, fs: int, max_tau: float = 0.05, interp: int = 16) -> Tuple[float, float]:
    """Return (delay_seconds, confidence) using GCC-PHAT.

    max_tau limits delay search (seconds).
    confidence is a peak-to-average metric (higher is better).
    """
    sig = _to_mono(sig).astype(np.float32)
    refsig = _to_mono(refsig).astype(np.float32)

    n = sig.shape[0] + refsig.shape[0]
    nfft = 1 << (n - 1).bit_length()

    SIG = np.fft.rfft(sig, n=nfft)
    REFSIG = np.fft.rfft(refsig, n=nfft)
    R = SIG * np.conj(REFSIG)
    denom = np.abs(R)
    denom[denom < 1e-12] = 1e-12
    R /= denom
    cc = np.fft.irfft(R, n=interp * nfft)

    max_shift = int(interp * fs * max_tau)
    max_shift = max(max_shift, 1)

    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    shift = int(np.argmax(np.abs(cc)) - max_shift)
    conf = float(np.max(np.abs(cc)) / (np.mean(np.abs(cc)) + 1e-9))
    delay = shift / float(interp * fs)
    return delay, conf


def apply_sample_shift(x: np.ndarray, shift_samples: int) -> np.ndarray:
    """Positive shift delays (pads front). Negative shift advances."""
    if shift_samples == 0:
        return x
    n = x.shape[0]
    if shift_samples > 0:
        pad = np.zeros((shift_samples, x.shape[1]), dtype=x.dtype)
        y = np.concatenate([pad, x], axis=0)
        return y[:n]
    k = -shift_samples
    y = x[k:]
    pad = np.zeros((k, x.shape[1]), dtype=x.dtype)
    y = np.concatenate([y, pad], axis=0)
    return y[:n]


def global_align_sum_to_mix(stems: list[np.ndarray], mix: np.ndarray, sr: int, max_tau: float = 0.05) -> AlignResult:
    """Estimate global delay/polarity between sum(stems) and mix."""
    if not stems:
        return AlignResult(0, 0.0, 0.0, False)
    sum_stems = np.zeros_like(mix)
    for s in stems:
        sum_stems += s

    delay_s, conf = gcc_phat(sum_stems, mix, fs=sr, max_tau=max_tau)
    shift_samples = int(np.round(delay_s * sr))

    a = _to_mono(sum_stems)
    b = _to_mono(mix)
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    corr = float(np.dot(a, b) / denom)
    invert = corr < -0.05

    return AlignResult(
        estimated_delay_samples=shift_samples,
        estimated_delay_seconds=float(shift_samples) / float(sr),
        confidence=float(conf),
        polarity_inverted=bool(invert),
    )
