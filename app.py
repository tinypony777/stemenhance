from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr

from stem_enhancer.pipeline import Options, process_stems
from stem_enhancer.utils import normalize_file_list, parse_mapping_text


def _ensure_stable_tmp() -> str:
    tmp_base = Path.home() / "Library" / "Caches" / "bakuage-stem-enhancer" / "tmp"
    tmp_base.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(tmp_base))
    os.environ.setdefault("GRADIO_TEMP_DIR", str(tmp_base))
    return str(tmp_base)


_ensure_stable_tmp()


def run(
    stems,
    mix,
    mapping_text,
    artifact,
    hiss,
    transient,
    cymbal,
    cymbal_attack,
    phase_align,
    progress=gr.Progress(),
):
    stem_paths = normalize_file_list(stems)
    mix_paths = normalize_file_list(mix)
    mix_path = mix_paths[0] if mix_paths else None
    mapping = parse_mapping_text(mapping_text)

    opts = Options(
        artifact=float(artifact),
        hiss=float(hiss),
        transient=float(transient),
        cymbal=float(cymbal),
        cymbal_attack=float(cymbal_attack),
        phase_align=bool(phase_align),
    )

    def cb(msg: str, p: float):
        try:
            progress(p, desc=msg)
        except Exception:
            pass

    zip_path, report = process_stems(stem_paths, mix_path, mapping, opts, progress=cb)
    report_text = json.dumps(report, ensure_ascii=False, indent=2)
    return zip_path, report_text


with gr.Blocks(title="Bakuage Stem Enhancer") as demo:
    gr.Markdown(
        """# Bakuage Stem Enhancer (v0.4.0)

2mixをガイドに、ステム分離で欠損したトランジェントを再配分し、シンバルの不快な洗い感を抑えます。

推奨: **2mix を必ず入力**。
"""
    )

    with gr.Row():
        stems_in = gr.File(
            label="Stems (multiple WAV)",
            file_count="multiple",
            file_types=[".wav", ".aif", ".aiff", ".flac"],
        )
        mix_in = gr.File(
            label="2mix (WAV) - required in v0.4.0",
            file_count="single",
            file_types=[".wav", ".aif", ".aiff", ".flac"],
        )

    mapping = gr.Textbox(
        label="Instrument mapping (optional)",
        lines=4,
        placeholder="example:\nDrums.wav: drums\nOverheads.wav: cymbals\n",
    )

    with gr.Row():
        artifact = gr.Slider(0, 1, value=0.6, step=0.05, label="Artifact reduction (mask smoothing)")
        hiss = gr.Slider(0, 1, value=0.3, step=0.05, label="Hiss/leak reduction (non-drums high band)")
    with gr.Row():
        transient = gr.Slider(0, 1, value=0.6, step=0.05, label="Transient restore (mix-guided reallocation)")
        phase_align = gr.Checkbox(value=True, label="Global align (delay/polarity) to 2mix")
    with gr.Row():
        cymbal = gr.Slider(0, 1, value=0.55, step=0.05, label="Cymbal tame (drums high band)")
        cymbal_attack = gr.Slider(0, 1, value=0.35, step=0.05, label="Cymbal attack restore (inject from 2mix)")

    run_btn = gr.Button("Process")
    clear_btn = gr.Button("Clear")

    out_zip = gr.File(label="Output ZIP")
    out_report = gr.Textbox(label="Report", lines=14)

    run_btn.click(
        fn=run,
        inputs=[stems_in, mix_in, mapping, artifact, hiss, transient, cymbal, cymbal_attack, phase_align],
        outputs=[out_zip, out_report],
    )

    def _clear():
        return None, None, "", 0.6, 0.3, 0.6, 0.55, 0.35, True, None, ""

    clear_btn.click(
        fn=_clear,
        inputs=[],
        outputs=[stems_in, mix_in, mapping, artifact, hiss, transient, cymbal, cymbal_attack, phase_align, out_zip, out_report],
    )


def main():
    demo.queue(concurrency_count=1).launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
