#!/usr/bin/env python3
"""Render one compact, Git-friendly Markdown log for a scale experiment."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re


ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
EPOCH = re.compile(
    r"\[(\d+)\]\s+time\s+([0-9.]+)\s+lr\s+([0-9.]+)\s+"
    r"3d_train\s+([0-9.]+)\s+e1\s+([0-9.]+)\s+e2\s+([0-9.]+)"
)
RUNTIME = re.compile(
    r"RUNTIME epoch=(\d+) train_it_per_sec=([0-9.]+) "
    r"peak_allocated_mib=([0-9.]+) peak_reserved_mib=([0-9.]+)"
)
PROGRESS = re.compile(r"(\d+)it \[([^\]]+)\]")

REGISTRY = {
    "S1": {
        "title": "S1 W128/D20",
        "config": "configs/pose3d/graph_posemamba_h36m_w128_d20_scale_80e.yaml",
        "width": 128,
        "depth": 20,
        "parameters": 6_836_355,
        "preflight_key": "S1",
    },
    "S2": {
        "title": "S2 W256/D10",
        "config": "configs/pose3d/graph_posemamba_h36m_w256_d10_scale_80e.yaml",
        "width": 256,
        "depth": 10,
        "parameters": 12_646_107,
        "preflight_key": "S2",
    },
}


def clean(path: Path) -> str:
    return ANSI.sub("", path.read_text(encoding="utf-8", errors="replace")).replace(
        "\r", "\n"
    )


def monitor_summary(path: Path, phase: str) -> dict:
    if not path.is_file():
        return {}
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("phase") != phase:
                continue
            try:
                rows.append(
                    {
                        key: float(row[key])
                        for key in (
                            "gpu_used_mib",
                            "gpu_util_percent",
                            "temp_c",
                            "power_w",
                        )
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    if not rows:
        return {}
    return {
        "samples": len(rows),
        "latest": rows[-1],
        "max": {key: max(row[key] for row in rows) for key in rows[0]},
        "mean": {
            key: sum(row[key] for row in rows) / len(rows) for key in rows[0]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=sorted(REGISTRY), required=True)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--status", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--preflight",
        default=".experiments/model_scaling_80e_20260903/PRELAUNCH_PASS.json",
    )
    parser.add_argument(
        "--monitor",
        default=".experiments/model_scaling_80e_20260903/runtime/gpu_monitor.csv",
    )
    parser.add_argument("--source-commit", default="")
    parser.add_argument("--queue-pid", default="")
    parser.add_argument("--train-pid", default="")
    args = parser.parse_args()

    spec = REGISTRY[args.experiment]
    run = Path(args.run_dir) if args.run_dir else None
    epochs = []
    runtimes = []
    progress = None
    errors = []
    if run is not None and (run / "log.txt").is_file():
        text = clean(run / "log.txt")
        epochs = [
            {
                "epoch": int(a),
                "minutes": float(b),
                "lr": float(c),
                "loss": float(d),
                "p1": float(e),
                "p2": float(f),
            }
            for a, b, c, d, e, f in EPOCH.findall(text)
        ]
        runtimes = [
            {
                "epoch": int(a),
                "it_per_sec": float(b),
                "peak_allocated_mib": float(c),
                "peak_reserved_mib": float(d),
            }
            for a, b, c, d in RUNTIME.findall(text)
        ]
        progress_matches = PROGRESS.findall(text)
        progress = progress_matches[-1] if progress_matches else None
        errors = re.findall(
            r".*(?:Traceback|CUDA out of memory|\bnan\b|\binf\b).*$",
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )

    preflight = {}
    preflight_path = Path(args.preflight)
    if preflight_path.is_file():
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    stages = preflight.get("smokes", {}).get(spec["preflight_key"], {})
    monitor = monitor_summary(Path(args.monitor), args.experiment)
    best = min(epochs, key=lambda item: item["p1"]) if epochs else None
    latest = epochs[-1] if epochs else None
    status = args.status or (
        "COMPLETED" if len(epochs) == 80 else ("RUNNING" if run else "READY")
    )
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    lines = [
        f"# {spec['title']} training log",
        "",
        f"- Last synchronized snapshot (UTC): `{now}`.",
        f"- Status: `{status}`.",
        f"- Config: `{spec['config']}`.",
        f"- Source commit: `{args.source_commit or preflight.get('source_commit', 'unknown')}`.",
        f"- Width/depth: `{spec['width']}/{spec['depth']}`.",
        f"- Trainable parameters: `{spec['parameters']:,}`.",
        "- Protocol: H36M-SH, T243/S81, batch 4, FP32 compiled, seed 0, 80 epochs.",
        f"- Run directory: `{str(run) if run else 'not created'}`.",
        f"- Queue/train PID at snapshot: `{args.queue_pid or 'n/a'}` / `{args.train_pid or 'n/a'}`.",
        "",
        "## Current summary",
        "",
        f"- Completed epochs: `{len(epochs)}/80`.",
        f"- Latest EMA P1/P2: `{latest['p1']:.4f}/{latest['p2']:.4f} mm` at epoch {latest['epoch']}."
        if latest
        else "- Latest EMA P1/P2: pending first completed epoch.",
        f"- Best EMA P1 and paired P2: `{best['p1']:.4f}/{best['p2']:.4f} mm` at epoch {best['epoch']}."
        if best
        else "- Best EMA result: pending first completed epoch.",
        f"- Current iteration trace: `{progress[0]}it [{progress[1]}]`."
        if progress
        else "- Current iteration trace: unavailable or waiting to start.",
        f"- Error matches: `{len(errors)}`.",
    ]
    if runtimes:
        stable = [item["it_per_sec"] for item in runtimes if item["epoch"] > 1]
        lines.extend(
            [
                f"- Latest train throughput: `{runtimes[-1]['it_per_sec']:.3f} it/s`.",
                f"- Stable mean throughput: `{sum(stable) / len(stable):.3f} it/s`."
                if stable
                else "- Stable mean throughput: pending epoch 2.",
                f"- Trainer peak reserved VRAM: `{max(item['peak_reserved_mib'] for item in runtimes):.0f} MiB`.",
            ]
        )
    if monitor:
        lines.extend(
            [
                f"- External monitor latest total GPU memory/utilization: `{monitor['latest']['gpu_used_mib']:.0f} MiB / {monitor['latest']['gpu_util_percent']:.0f}%`.",
                f"- External monitor max temperature/power: `{monitor['max']['temp_c']:.0f} C / {monitor['max']['power_w']:.1f} W`.",
                "- External memory includes the retained `wh` process and is not a per-model peak.",
            ]
        )

    lines.extend(["", "## Registered preflight", ""])
    for batch in ("B1", "B2", "B4"):
        stage = stages.get(batch)
        if not stage:
            lines.append(f"- {batch}: pending.")
            continue
        lines.append(
            f"- {batch}: {'compiled' if stage.get('compiled') else 'eager'}, "
            f"peak reserved `{stage.get('measured_peak_reserved_mib', 0):.0f} MiB`, "
            f"loss `{stage.get('loss', 0):.6f}`, "
            f"throughput `{stage.get('iterations_per_second', 0):.3f} it/s`."
        )

    lines.extend(
        [
            "",
            "## Completed-epoch history",
            "",
            "| Epoch | Train min | LR | Train loss | EMA P1 | Paired P2 | it/s | Reserved MiB |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    runtime_by_epoch = {item["epoch"]: item for item in runtimes}
    for item in epochs:
        runtime = runtime_by_epoch.get(item["epoch"], {})
        lines.append(
            f"| {item['epoch']} | {item['minutes']:.2f} | {item['lr']:.6f} | "
            f"{item['loss']:.6f} | {item['p1']:.4f} | {item['p2']:.4f} | "
            f"{runtime.get('it_per_sec', 0):.3f} | {runtime.get('peak_reserved_mib', 0):.0f} |"
        )
    if not epochs:
        lines.append("| — | — | — | — | — | — | — | — |")

    lines.extend(
        [
            "",
            "## Interpretation guard",
            "",
            "Training-time results are provisional. The paper result is the minimum EMA P1 within epochs 1-80 and the P2 from that same checkpoint; raw/EMA checkpoints are strictly replayed after completion.",
        ]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
