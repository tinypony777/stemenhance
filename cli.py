from __future__ import annotations

import argparse
import glob
import os
import sys

from .pipeline import EnhanceOptions, StemItem, enhance_stems


def _parse_args(argv):
    p = argparse.ArgumentParser(description="Bakuage Stem Enhancer (CLI)")
    p.add_argument("--stems", nargs="+", required=True, help="Stem audio paths (supports glob)")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--sr", type=int, default=48000, help="Target sample rate")
    p.add_argument("--artifact", type=float, default=0.55, help="Artifact reduction strength 0..1")
    p.add_argument("--hiss", type=float, default=0.35, help="Hiss reduction strength 0..1")
    p.add_argument("--transient", type=float, default=0.45, help="Transient restore amount 0..1")
    p.add_argument(
        "--cymbal",
        type=float,
        default=0.35,
        help="Cymbal tame strength 0..1 (mainly drums; reduces long sustain / fizz)",
    )
    p.add_argument("--phase_align", type=str, default="auto", help="Anchor stem name or 'off' or 'auto'")
    p.add_argument("--phase_match", action="store_true", help="Enable frequency-constant phase matching")
    p.add_argument("--max_delay_ms", type=float, default=80.0, help="Max delay for alignment")
    p.add_argument("--no_keep_relative", action="store_true", help="Do not preserve relative level when applying safety scaling")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    paths = []
    for pat in args.stems:
        expanded = glob.glob(pat)
        if expanded:
            paths.extend(expanded)
        else:
            paths.append(pat)

    stems = [StemItem(name=os.path.basename(p), path=p, instrument="auto") for p in sorted(paths)]

    opt = EnhanceOptions(
        target_sr=args.sr,
        artifact_strength=args.artifact,
        hiss_strength=args.hiss,
        transient_amount=args.transient,
        cymbal_tame=args.cymbal,
        do_phase_align=(args.phase_align != "off"),
        phase_anchor=("auto" if args.phase_align in ["auto", "off"] else args.phase_align),
        phase_match=args.phase_match,
        max_delay_ms=args.max_delay_ms,
        keep_relative_level=(not args.no_keep_relative),
    )

    zip_path, meta = enhance_stems(stems, opt=opt, output_dir=args.out)
    print(f"Wrote: {zip_path}")


if __name__ == "__main__":
    main()
