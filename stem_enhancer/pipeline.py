from __future__ import annotations

import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import __version__
from .align import AlignResult, apply_sample_shift, global_align_sum_to_mix
from .audio_io import Audio, load_audio, pad_or_trim, peak_normalize_safe, resample, write_wav
from .dsp import DRUM_LIKE, apply_cymbal_attack_restore, apply_highband_cymbal_tame, compute_onset_envelope_from_mag, smooth_mask, transient_boost_mask
from .tf import istft_multi, mag_from_stft, stft_multi
from .utils import basename, infer_instrument_label


@dataclass
class Options:
    artifact: float = 0.5
    hiss: float = 0.3
    transient: float = 0.5
    cymbal: float = 0.5
    cymbal_attack: float = 0.3
    phase_align: bool = True


@dataclass
class Stem:
    path: str
    name: str
    label: str
    audio: Audio


def _stable_temp_root() -> str:
    # Respect caller's env, else fallback.
    return os.environ.get("GRADIO_TEMP_DIR") or os.environ.get("TMPDIR") or tempfile.gettempdir()


def _make_workdir() -> str:
    root = _stable_temp_root()
    os.makedirs(root, exist_ok=True)
    return tempfile.mkdtemp(prefix="bakuage-stem-enhancer-", dir=root)


def _attenuate_hiss_in_mask(mask: np.ndarray, freqs: np.ndarray, hiss: float) -> np.ndarray:
    """Reduce very-high frequency leakage in non-drum stems.

    This is intentionally conservative: it mainly reduces 'air hiss' leakage.
    """
    hiss = float(np.clip(hiss, 0.0, 1.0))
    if hiss <= 0:
        return mask
    nyq = float(freqs.max()) if freqs.size else 24000.0
    f0 = 12000.0
    if nyq <= f0:
        return mask
    # Frequency ramp 0..1 above f0
    ramp = np.clip((freqs - f0) / (nyq - f0), 0.0, 1.0).astype(np.float32)
    att = 1.0 - (0.75 * hiss) * ramp
    return (mask * att[:, None]).astype(np.float32)


def process_stems(
    stem_paths: List[str],
    mix_path: Optional[str],
    mapping: Dict[str, str],
    opts: Options,
    progress: Optional[Callable[[str, float], None]] = None,
) -> Tuple[str, Dict]:
    """Main processing pipeline.

    Returns (zip_path, report_dict).
    """
    if not stem_paths:
        raise ValueError("No stems provided")

    if progress:
        progress("Loading audio", 0.02)

    stems: List[Stem] = []
    for p in stem_paths:
        name = basename(p)
        label = mapping.get(name) or mapping.get(p) or infer_instrument_label(name)
        a = load_audio(p)
        stems.append(Stem(path=p, name=name, label=label, audio=a))

    mix_audio: Optional[Audio] = None
    if mix_path:
        mix_audio = load_audio(mix_path)

    # Target SR/length
    sr_target = int(mix_audio.sr) if mix_audio else int(stems[0].audio.sr)
    if progress:
        progress("Resampling", 0.05)

    for s in stems:
        if s.audio.sr != sr_target:
            s.audio = resample(s.audio, sr_target)
    if mix_audio and mix_audio.sr != sr_target:
        mix_audio = resample(mix_audio, sr_target)

    n_target = int(mix_audio.data.shape[0]) if mix_audio else int(max(st.audio.data.shape[0] for st in stems))
    for s in stems:
        s.audio = pad_or_trim(s.audio, n_target)
    if mix_audio:
        mix_audio = pad_or_trim(mix_audio, n_target)

    # Optional global align
    align_result: Optional[AlignResult] = None
    if opts.phase_align and mix_audio:
        if progress:
            progress("Phase/timing align (global)", 0.10)
        stem_arrays = [st.audio.data for st in stems]
        align_result = global_align_sum_to_mix(stem_arrays, mix_audio.data, sr_target, max_tau=0.05)
        shift = align_result.estimated_delay_samples
        if abs(shift) > 0:
            for st in stems:
                st.audio = Audio(sr_target, apply_sample_shift(st.audio.data, shift))
        if align_result.polarity_inverted:
            for st in stems:
                st.audio = Audio(sr_target, -st.audio.data)

    # Mix-guided mask reprojection (core)
    if not mix_audio:
        raise ValueError("2mix is required for v0.4.0 (mix-guided mode)")

    if progress:
        progress("STFT (2mix)", 0.15)
    freqs, times, X = stft_multi(mix_audio.data, sr_target, n_fft=2048, hop=1024)
    mix_mag = mag_from_stft(X)
    onset_env = compute_onset_envelope_from_mag(freqs, mix_mag)

    # First pass: magnitudes + sum
    if progress:
        progress("STFT (stems) + masks", 0.22)
    mags: List[np.ndarray] = []
    for idx, st in enumerate(stems):
        _, _, Z = stft_multi(st.audio.data, sr_target, n_fft=2048, hop=1024)
        mags.append(mag_from_stft(Z))
        if progress:
            progress(f"Analyzing stem {idx+1}/{len(stems)}", 0.22 + 0.18 * (idx + 1) / max(1, len(stems)))

    A_sum = np.zeros_like(mags[0], dtype=np.float32)
    for A in mags:
        A_sum += A
    A_sum += 1e-8

    # Second pass: build masks, apply smoothing and transient re-allocation
    masks: List[np.ndarray] = []
    for idx, (st, A) in enumerate(zip(stems, mags)):
        M = (A / A_sum).astype(np.float32)

        # Artifact smoothing
        M = smooth_mask(M, opts.artifact)

        # Reduce very-high leakage hiss for non-drum stems
        if st.label not in DRUM_LIKE:
            M = _attenuate_hiss_in_mask(M, freqs, opts.hiss)

        # Transient restore: allocate more mix energy to drum-like stems at onset frames
        if st.label in DRUM_LIKE:
            boost = transient_boost_mask(freqs, onset_env, st.label, opts.transient)
            M = (M * boost).astype(np.float32)

        masks.append(M)

    # Renormalize so that sum masks == 1 per TF bin
    sumM = np.zeros_like(masks[0], dtype=np.float32)
    for M in masks:
        sumM += M
    sumM += 1e-8
    for i in range(len(masks)):
        masks[i] = (masks[i] / sumM).astype(np.float32)

    if progress:
        progress("iSTFT + post", 0.45)

    # Reproject + inverse STFT
    out_stems: List[Stem] = []
    for idx, (st, M) in enumerate(zip(stems, masks)):
        # Apply same mask to both channels of mix STFT (preserves stereo)
        U = (X * M[None, :, :]).astype(np.complex64)
        y = istft_multi(U, sr_target, n_fft=2048, hop=1024, length=n_target)

        # Post-processing for cymbal wash / missing attack
        if st.label in DRUM_LIKE:
            y = apply_highband_cymbal_tame(y, sr_target, opts.cymbal)
            y = apply_cymbal_attack_restore(y, mix_audio.data, sr_target, opts.cymbal_attack)

        y = peak_normalize_safe(y, peak=0.99)
        out_stems.append(Stem(path=st.path, name=st.name, label=st.label, audio=Audio(sr_target, y)))
        if progress:
            progress(f"Rendering stem {idx+1}/{len(stems)}", 0.45 + 0.40 * (idx + 1) / max(1, len(stems)))

    workdir = _make_workdir()
    out_dir = Path(workdir) / "output"
    stems_dir = out_dir / "stems"
    stems_dir.mkdir(parents=True, exist_ok=True)

    if progress:
        progress("Writing WAV + report", 0.90)

    out_files: List[str] = []
    for st in out_stems:
        stem_out_name = st.name
        if stem_out_name.lower().endswith(".wav"):
            stem_out_name = stem_out_name[:-4] + "_enhanced.wav"
        else:
            stem_out_name = stem_out_name + "_enhanced.wav"
        out_path = str(stems_dir / stem_out_name)
        write_wav(out_path, st.audio)
        out_files.append(out_path)

    report = {
        "version": __version__,
        "mode": "mix-guided", 
        "inputs": [st.name for st in stems],
        "mix": basename(mix_path) if mix_path else None,
        "labels": {st.name: st.label for st in stems},
        "options": {
            "artifact": opts.artifact,
            "hiss": opts.hiss,
            "transient": opts.transient,
            "cymbal": opts.cymbal,
            "cymbal_attack": opts.cymbal_attack,
            "phase_align": opts.phase_align,
        },
        "stft": {"n_fft": 2048, "hop": 1024},
        "sample_rate": sr_target,
        "samples": n_target,
        "align": (align_result.__dict__ if align_result else None),
    }

    report_path = str(out_dir / "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Zip
    zip_path = str(Path(workdir) / f"bakuage-stem-enhancer-{__version__}.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(report_path, arcname="report.json")
        for fp in out_files:
            z.write(fp, arcname=f"stems/{Path(fp).name}")

    if progress:
        progress("Done", 1.0)
    return zip_path, report
