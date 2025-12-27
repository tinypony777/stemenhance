from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import signal


def stft_multi(x: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 1024) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """STFT for multi-channel audio.

    Returns (freqs, times, Z) where Z shape = (ch, f, t)
    """
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    ch = x.shape[1]
    Zs = []
    f = t = None
    for c in range(ch):
        f, t, Z = signal.stft(
            x[:, c],
            fs=sr,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop,
            nfft=n_fft,
            boundary=None,
            padded=False,
        )
        Zs.append(Z.astype(np.complex64))
    Zm = np.stack(Zs, axis=0)
    return f, t, Zm


def istft_multi(Z: np.ndarray, sr: int, n_fft: int = 2048, hop: int = 1024, length: int | None = None) -> np.ndarray:
    """Inverse STFT.

    Z shape = (ch, f, t). Returns audio shape (n, ch).
    """
    ch = Z.shape[0]
    ys = []
    for c in range(ch):
        _, y = signal.istft(
            Z[c],
            fs=sr,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop,
            nfft=n_fft,
            boundary=None,
        )
        ys.append(y.astype(np.float32))
    y2 = np.stack(ys, axis=1)
    if length is not None:
        if y2.shape[0] > length:
            y2 = y2[:length]
        elif y2.shape[0] < length:
            pad = length - y2.shape[0]
            y2 = np.pad(y2, ((0, pad), (0, 0)), mode="constant")
    return y2


def mag_from_stft(Z: np.ndarray) -> np.ndarray:
    """Combine stereo magnitudes into a single magnitude matrix (f, t)."""
    if Z.ndim != 3:
        raise ValueError("Z must be (ch, f, t)")
    if Z.shape[0] == 1:
        return np.abs(Z[0]).astype(np.float32)
    mag = np.sqrt(np.abs(Z[0]) ** 2 + np.abs(Z[1]) ** 2)
    return mag.astype(np.float32)
