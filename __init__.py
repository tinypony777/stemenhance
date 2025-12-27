"""Bakuage Stem Enhancer.

Local stem enhancement utilities:
- artifact reduction (spectral gating)
- hiss reduction
- transient restoration
- phase/timing alignment

This package is intentionally self-contained.
"""

from .pipeline import enhance_stems, EnhanceOptions, StemItem
