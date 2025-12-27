from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .audio_io import _ensure_2d, mono_mix


@dataclass
class PhaseAlignResult:
    delays_samples: Dict[str, float]
    inverted: Dict[str, bool]


@dataclass
class PhaseAlignOptions:
    max_delay_ms: float = 80.0
    interp: int = 16  # GCC-PHAT interpolation factor
    do_polarity: bool = True
    phase_match: bool = False  # frequency-constant phase rotation
    stft_n_fft: int = 2048
    stft_hop_length: int = 512


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def gcc_phat_delay(sig: np.ndarray, ref: np.ndarray, max_delay_samples: int, interp: int = 16) -> float:
    """Estimate delay between sig and ref via GCC-PHAT.

    Returns delay in samples. Positive means `sig` lags `ref`.
    """

    # Ensure 1D float64 for precision
    sig = np.asarray(sig, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)

    n = _next_pow2(sig.size + ref.size)
    SIG = np.fft.rfft(sig, n=n)
    REF = np.fft.rfft(ref, n=n)

    R = SIG * np.conj(REF)
    denom = np.abs(R)
    R /= (denom + 1e-12)

    # Interpolated cross-correlation
    cc = np.fft.irfft(R, n=interp * n)

    max_shift = int(interp * max_delay_samples)
    if max_shift <= 0:
        return 0.0

    # Wrap-around: cc[0] is 0 lag, negative lags are at the end
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    shift = int(np.argmax(np.abs(cc)) - max_shift)

    delay = shift / float(interp)
    return float(delay)


def _shift_keep_length(y: np.ndarray, shift: int) -> np.ndarray:
    """Shift audio by integer samples, keeping the original length.

    shift > 0: advance (move left)
    shift < 0: delay (move right)
    """

    y2d = _ensure_2d(y)
    n = y2d.shape[0]

    if shift == 0:
        return y2d

    if shift > 0:
        if shift >= n:
            return np.zeros_like(y2d)
        out = np.zeros_like(y2d)
        out[: n - shift] = y2d[shift:]
        return out
    else:
        k = -shift
        if k >= n:
            return np.zeros_like(y2d)
        out = np.zeros_like(y2d)
        out[k:] = y2d[: n - k]
        return out


def phase_match_constant(y: np.ndarray, ref: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Apply a frequency-dependent constant phase rotation so that y aligns to ref.

    This does NOT solve time-varying phase, but can correct a stable phase rotation.
    """

    from scipy.signal import stft, istft
    import warnings

    y2d = _ensure_2d(y)
    ref2d = _ensure_2d(ref)
    assert y2d.shape == ref2d.shape, "y and ref must have same shape"

    out = np.zeros_like(y2d, dtype=np.float32)

    for ch in range(y2d.shape[1]):
        nperseg = n_fft
        noverlap = n_fft - hop_length
        _, _, Y = stft(
            y2d[:, ch].astype(np.float32),
            fs=sr,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=n_fft,
            boundary=None,
            padded=False,
        )
        _, _, X = stft(
            ref2d[:, ch].astype(np.float32),
            fs=sr,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=n_fft,
            boundary=None,
            padded=False,
        )

        C = X * np.conj(Y)
        phi = np.angle(np.sum(C, axis=1))  # shape (freq,)
        rot = np.exp(1j * phi)[:, None]

        Y2 = Y * rot
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _, y2 = istft(
                Y2,
                fs=sr,
                window="hann",
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=n_fft,
                input_onesided=True,
                boundary=None,
            )
        # match length
        if y2.size < y2d.shape[0]:
            y2 = np.pad(y2, (0, y2d.shape[0] - y2.size))
        elif y2.size > y2d.shape[0]:
            y2 = y2[: y2d.shape[0]]
        out[:, ch] = y2.astype(np.float32)

    return out


def align_stems(
    stems: Dict[str, np.ndarray],
    sr: int,
    anchor_key: str,
    opt: PhaseAlignOptions,
) -> Tuple[Dict[str, np.ndarray], PhaseAlignResult]:
    """Align stems to an anchor by estimated time delay and optional polarity.

    `stems` values must be 2D arrays (n, ch) or 1D (n,).
    """

    if anchor_key not in stems:
        raise ValueError(f"anchor_key '{anchor_key}' not in stems")

    max_delay_samples = int(sr * opt.max_delay_ms / 1000.0)

    anchor = _ensure_2d(stems[anchor_key])
    anchor_mono = mono_mix(anchor)

    aligned: Dict[str, np.ndarray] = {}
    delays: Dict[str, float] = {}
    inverted: Dict[str, bool] = {}

    # Pre-normalize to avoid issues in correlation
    def _norm(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float64)
        m = np.max(np.abs(x)) + 1e-12
        return (x / m).astype(np.float64)

    anchor_n = _norm(anchor_mono)

    for k, y in stems.items():
        y2d = _ensure_2d(y)
        if k == anchor_key:
            aligned[k] = y2d.astype(np.float32)
            delays[k] = 0.0
            inverted[k] = False
            continue

        mono = mono_mix(y2d)
        mono_n = _norm(mono)

        d = gcc_phat_delay(mono_n, anchor_n, max_delay_samples=max_delay_samples, interp=opt.interp)
        delays[k] = d

        # Apply integer shift (fractional is possible but we keep it stable)
        shift = int(round(d))
        y_shifted = _shift_keep_length(y2d, shift)

        inv = False
        if opt.do_polarity:
            # correlation after shift
            a = _norm(mono_mix(y_shifted))
            corr = float(np.sum(a * anchor_n) / (np.sqrt(np.sum(a * a) * np.sum(anchor_n * anchor_n)) + 1e-12))
            if corr < 0:
                y_shifted = (-y_shifted).astype(np.float32)
                inv = True

        if opt.phase_match:
            try:
                y_shifted = phase_match_constant(y_shifted, anchor, sr, n_fft=opt.stft_n_fft, hop_length=opt.stft_hop_length)
            except Exception:
                # If phase-match fails, keep shifted
                pass

        aligned[k] = y_shifted.astype(np.float32)
        inverted[k] = inv

    return aligned, PhaseAlignResult(delays_samples=delays, inverted=inverted)
