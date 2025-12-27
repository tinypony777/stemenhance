from __future__ import annotations

import json
import os
import re
import tempfile
from typing import Dict, List, Optional, Tuple

import gradio as gr

from stem_enhance.pipeline import EnhanceOptions, StemItem, enhance_stems


def _parse_mapping(text: str) -> Dict[str, str]:
    """Parse user mapping.

    Accepts:
    - JSON dict: {"filename.wav": "vocals", ...}
    - YAML-ish lines: filename: instrument
    """

    if not text or not text.strip():
        return {}

    t = text.strip()

    # JSON first
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return {str(k): str(v) for k, v in obj.items()}
    except Exception:
        pass

    # YAML-ish
    mapping: Dict[str, str] = {}
    for line in t.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        mapping[k.strip()] = v.strip()
    return mapping


def _match_midi_hints(stem_paths: List[str], midi_paths: List[str]) -> Dict[str, str]:
    """Return stem filename -> midi filename mapping by basename similarity."""

    midi_by_base = {}
    for mp in midi_paths:
        base = os.path.splitext(os.path.basename(mp))[0].lower()
        midi_by_base[base] = mp

    out: Dict[str, str] = {}
    for sp in stem_paths:
        sbase = os.path.splitext(os.path.basename(sp))[0].lower()
        # direct match
        if sbase in midi_by_base:
            out[os.path.basename(sp)] = os.path.basename(midi_by_base[sbase])
            continue
        # partial match
        for mb, mp in midi_by_base.items():
            if mb in sbase or sbase in mb:
                out[os.path.basename(sp)] = os.path.basename(mp)
                break
    return out


def run(
    stems_files,
    midi_files,
    instrument_mapping,
    artifact_strength,
    hiss_strength,
    transient_amount,
    cymbal_tame,
    do_phase_align,
    phase_match,
    max_delay_ms,
    anchor_name,
    output_subtype,
):
    # gradio 2.x File input returns file objects or paths depending on `type`
    def _to_paths(file_in):
        if file_in is None:
            return []
        if isinstance(file_in, list):
            return [getattr(f, "name", f) for f in file_in]
        return [getattr(file_in, "name", file_in)]

    stem_paths = _to_paths(stems_files)
    midi_paths = _to_paths(midi_files)

    if not stem_paths:
        return None, "No stems uploaded."

    user_map = _parse_mapping(instrument_mapping)
    midi_hint_map = _match_midi_hints(stem_paths, midi_paths)

    stems: List[StemItem] = []
    for p in stem_paths:
        name = os.path.basename(p)
        inst = user_map.get(name) or user_map.get(os.path.splitext(name)[0]) or "auto"
        midi_hint = midi_hint_map.get(name)
        stems.append(StemItem(name=name, path=p, instrument=inst, midi_hint=midi_hint))

    opt = EnhanceOptions(
        target_sr=48000,
        artifact_strength=float(artifact_strength),
        hiss_strength=float(hiss_strength),
        transient_amount=float(transient_amount),
        cymbal_tame=float(cymbal_tame),
        do_phase_align=bool(do_phase_align),
        phase_anchor=(anchor_name.strip() if anchor_name and anchor_name.strip() else "auto"),
        phase_match=bool(phase_match),
        max_delay_ms=float(max_delay_ms),
        keep_relative_level=True,
        output_subtype=str(output_subtype),
    )

    out_dir = tempfile.mkdtemp(prefix="bakuage_stem_enhance_")
    zip_path, meta = enhance_stems(stems, opt=opt, output_dir=out_dir)

    # Human-readable report
    lines = []
    lines.append("Bakuage Stem Enhancer report")
    lines.append(f"Output: {zip_path}")
    lines.append("")
    lines.append("Instrument map:")
    for k, v in meta.get("instrument_map", {}).items():
        lines.append(f"  - {k}: {v}")

    name_map = meta.get("name_map")
    if name_map:
        lines.append("")
        lines.append("Renamed duplicate stems:")
        for original, uniques in name_map.items():
            for unique in uniques:
                lines.append(f"  - {original} -> {unique}")

    phase = meta.get("phase")
    if phase:
        lines.append("")
        lines.append(f"Phase align anchor: {phase.get('anchor')}")
        delays = phase.get("delays_samples", {})
        inv = phase.get("inverted", {})
        for k in sorted(delays.keys()):
            lines.append(f"  - {k}: delay={delays[k]:.2f} samples, inverted={inv.get(k, False)}")

    lines.append("")
    lines.append("Per-stem applied params:")
    for k, d in meta.get("per_stem", {}).items():
        lines.append(
            f"  - {k} ({d.get('instrument')}): artifact={d.get('artifact_strength'):.2f}, hiss={d.get('hiss_strength'):.2f}, transient={d.get('transient_amount'):.2f}, cymbal={d.get('cymbal_tame'):.2f}, peak_in={d.get('peak_in_dbfs'):.2f} dBFS, peak_out={d.get('peak_out_dbfs'):.2f} dBFS"
        )
        g = d.get("guess", {})
        if g:
            lines.append(f"      guess={g.get('label')} conf={g.get('confidence'):.2f} reasons={g.get('reasons')}")

    return zip_path, "\n".join(lines)


def main():
    title = "Bakuage Stem Enhancer (local)"
    description = "AI生成音源のステムに対して、アーティファクト低減 / ヒス低減 / トランジェント復元 / 簡易フェーズアラインをまとめて実行します。"

    with gr.Blocks(title=title, theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"# {title}\n{description}\n\n**推奨:** 48kHz WAVステム")

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## 1. 入力")
                stems_files = gr.File(
                    label="Stems (multiple)",
                    file_count="multiple",
                    type="filepath",
                )
                midi_files = gr.File(
                    label="MIDI (optional, multiple)",
                    file_count="multiple",
                    type="filepath",
                )
                instrument_mapping = gr.Textbox(
                    label="Instrument mapping (optional)",
                    value="# JSON 例\n# {\"vocals.wav\": \"vocals\", \"drums.wav\": \"drums\"}\n\n# YAML風 例\n# vocals.wav: vocals\n# drums.wav: drums\n",
                    lines=6,
                )
                output_subtype = gr.Dropdown(
                    choices=["PCM_16", "PCM_24", "PCM_32"],
                    value="PCM_24",
                    label="Output WAV bit depth",
                )
            with gr.Column(scale=3):
                gr.Markdown("## 2. 調整")
                with gr.Tabs():
                    with gr.TabItem("Noise & Detail"):
                        artifact_strength = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.55,
                            label="Artifact reduction strength",
                        )
                        hiss_strength = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.35,
                            label="Hiss reduction strength",
                        )
                        cymbal_tame = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.35,
                            label="Cymbal tame (drums) – reduce long sustain / fizz / wide sizzle",
                        )
                    with gr.TabItem("Transient"):
                        transient_amount = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0.45,
                            label="Transient restore amount",
                        )
                    with gr.TabItem("Phase"):
                        do_phase_align = gr.Checkbox(value=True, label="Phase/timing align")
                        phase_match = gr.Checkbox(value=False, label="Phase match (freq-constant)")
                        max_delay_ms = gr.Slider(
                            minimum=0,
                            maximum=150,
                            value=80,
                            step=1,
                            label="Max delay (ms)",
                        )
                        anchor_name = gr.Textbox(label="Anchor stem name (optional, default=auto)", value="auto")

        with gr.Row():
            run_button = gr.Button("Run enhancement", variant="primary")
            clear_button = gr.Button("Clear")

        gr.Markdown("## 3. 出力")
        output_zip = gr.File(label="Download enhanced stems (zip)")
        report = gr.Textbox(label="Report", lines=12)

        run_button.click(
            fn=run,
            inputs=[
                stems_files,
                midi_files,
                instrument_mapping,
                artifact_strength,
                hiss_strength,
                transient_amount,
                cymbal_tame,
                do_phase_align,
                phase_match,
                max_delay_ms,
                anchor_name,
                output_subtype,
            ],
            outputs=[output_zip, report],
        )
        clear_button.click(
            fn=lambda: (
                [],
                [],
                "# JSON 例\n# {\"vocals.wav\": \"vocals\", \"drums.wav\": \"drums\"}\n\n# YAML風 例\n# vocals.wav: vocals\n# drums.wav: drums\n",
                0.55,
                0.35,
                0.45,
                0.35,
                True,
                False,
                80,
                "auto",
                "PCM_24",
                None,
                "",
            ),
            inputs=[],
            outputs=[
                stems_files,
                midi_files,
                instrument_mapping,
                artifact_strength,
                hiss_strength,
                transient_amount,
                cymbal_tame,
                do_phase_align,
                phase_match,
                max_delay_ms,
                anchor_name,
                output_subtype,
                output_zip,
                report,
            ],
        )

    demo.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
