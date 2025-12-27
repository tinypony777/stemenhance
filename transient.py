from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio_io import _ensure_2d


@dataclass
class TransientOptions:
    amount: float = 0.4  # 0..1
    low_hz: float = 200.0
    high_hz: float = 2000.0
    order: int = 4
    attack_ms: float = 5.0
    release_ms: float = 80.0
    max_gain: float = 2.5


def _env_follower(abs_x: np.ndarray, sr: int, tau_ms: float) -> np.ndarray:
    # One-pole IIR smoothing of absolute value.
    # tau_ms large -> smoother
    from scipy.signal import lfilter

    tau = max(0.1, tau_ms) / 1000.0
    alpha = np.exp(-1.0 / (sr * tau))
    b = [1.0 - alpha]
    a = [1.0, -alpha]
    return lfilter(b, a, abs_x)


def _transient_shaper(x: np.ndarray, sr: int, amount: float, attack_ms: float, release_ms: float, max_gain: float) -> np.ndarray:
    if amount <= 1e-6:
        return x

    ax = np.abs(x).astype(np.float32)
    env_fast = _env_follower(ax, sr, tau_ms=attack_ms)
    env_slow = _env_follower(ax, sr, tau_ms=release_ms)

    trans = np.maximum(env_fast - env_slow, 0.0)
    ratio = trans / (env_slow + 1e-6)

    gain = 1.0 + amount * ratio
    gain = np.clip(gain, 1.0, max_gain)

    return (x * gain).astype(np.float32)


def transient_restore(y: np.ndarray, sr: int, opt: TransientOptions) -> np.ndarray:
    """Multiband transient restoration (zero-phase crossover).

    The goal is to compensate for transient smearing / loss commonly introduced
    by some stem separation methods.

    Notes:
    - Uses sosfiltfilt (zero-phase). Not real-time, but safe for offline.
    - Applies stronger shaping in high band by default.
    """

    amount = float(np.clip(opt.amount, 0.0, 1.0))
    if amount <= 1e-6:
        return _ensure_2d(y).astype(np.float32)

    from scipy.signal import butter, sosfiltfilt

    y2d = _ensure_2d(y).astype(np.float32)
    out = np.zeros_like(y2d, dtype=np.float32)

    # Crossovers
    sos_lp = butter(opt.order, opt.low_hz, btype="lowpass", fs=sr, output="sos")
    sos_hp = butter(opt.order, opt.high_hz, btype="highpass", fs=sr, output="sos")

    for ch in range(y2d.shape[1]):
        x = y2d[:, ch]

        low = sosfiltfilt(sos_lp, x)
        high = sosfiltfilt(sos_hp, x)
        mid = x - low - high

        # Apply shaping: low is subtle, high is strongest
        low2 = _transient_shaper(low, sr, amount * 0.15, opt.attack_ms, opt.release_ms, opt.max_gain)
        mid2 = _transient_shaper(mid, sr, amount * 0.45, opt.attack_ms, opt.release_ms, opt.max_gain)
        high2 = _transient_shaper(high, sr, amount * 1.00, opt.attack_ms, opt.release_ms, opt.max_gain)

        out[:, ch] = (low2 + mid2 + high2).astype(np.float32)

    return out
