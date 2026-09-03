#!/usr/bin/env python3
"""Build the registered minimal ablation CSV and Markdown table."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
EPOCH = re.compile(
    r"\[(\d+)\]\s+time\s+([0-9.]+)\s+lr\s+([0-9.]+)\s+"
    r"3d_train\s+([0-9.]+)\s+e1\s+([0-9.]+)\s+e2\s+([0-9.]+)"
)
RUNTIME = re.compile(
    r"RUNTIME epoch=(\d+) train_it_per_sec=([0-9.]+) "
    r"peak_allocated_mib=([0-9.]+) peak_reserved_mib=([0-9.]+)"
)
PARAMETERS = re.compile(r"Trainable parameter count:(\d+)")
P1 = re.compile(r"Protocol #1 Error \(MPJPE\):([0-9.]+)mm")
P2 = re.compile(r"Protocol #2 Error \(P-MPJPE\):([0-9.]+)mm")


def clean_text(path: Path) -> str:
    return ANSI.sub("", path.read_text(encoding="utf-8", errors="replace")).replace(
        "\r", "\n"
    )


def verified_metrics(path: Path) -> tuple[float | None, float | None]:
    if not path.is_file():
        return None, None
    text = clean_text(path)
    p1 = P1.findall(text)
    p2 = P2.findall(text)
    return (
        float(p1[-1]) if p1 else None,
        float(p2[-1]) if p2 else None,
    )


def run_row(run: Path | None, base: dict) -> dict:
    row = dict(base)
    if run is None:
        row.update(
            {
                "params": "",
                "best_epoch": "",
                "P1": "",
                "P2": "",
                "raw_P1": "",
                "raw_P2": "",
                "peak_vram": "",
                "it_per_sec": "",
                "status": "PLANNED",
                "run_dir": "",
            }
        )
        return row

    log_path = run / "log.txt"
    if not log_path.is_file():
        row.update({"status": "FAILED_NO_LOG", "run_dir": str(run)})
        return row
    text = clean_text(log_path)
    epochs = [
        {
            "epoch": int(match.group(1)),
            "minutes": float(match.group(2)),
            "lr": float(match.group(3)),
            "loss": float(match.group(4)),
            "p1": float(match.group(5)),
            "p2": float(match.group(6)),
        }
        for match in EPOCH.finditer(text)
        if int(match.group(1)) <= 80
    ]
    runtimes = [
        {
            "epoch": int(match.group(1)),
            "it_per_sec": float(match.group(2)),
            "peak_allocated": float(match.group(3)),
            "peak_reserved": float(match.group(4)),
        }
        for match in RUNTIME.finditer(text)
    ]
    parameters = PARAMETERS.findall(text)
    if not epochs:
        row.update({"status": "RUNNING_NO_COMPLETE_EPOCH", "run_dir": str(run)})
        return row

    best = min(epochs, key=lambda item: item["p1"])
    ema_p1, ema_p2 = verified_metrics(run / "verification_best_ema_epoch.stdout")
    raw_p1, raw_p2 = verified_metrics(run / "verification_best_epoch.stdout")
    stable_runtime = [item["it_per_sec"] for item in runtimes if item["epoch"] > 1]
    complete = len(epochs) == 80 and (run / "latest_epoch.bin").is_file()
    row.update(
        {
            "params": int(parameters[-1]) if parameters else "",
            "best_epoch": best["epoch"],
            "P1": ema_p1 if ema_p1 is not None else best["p1"],
            "P2": ema_p2 if ema_p2 is not None else best["p2"],
            "raw_P1": raw_p1 if raw_p1 is not None else "",
            "raw_P2": raw_p2 if raw_p2 is not None else "",
            "peak_vram": max(
                (item["peak_reserved"] for item in runtimes), default=""
            ),
            "it_per_sec": (
                sum(stable_runtime) / len(stable_runtime)
                if stable_runtime
                else ""
            ),
            "status": "COMPLETED" if complete else f"RUNNING_EPOCH_{len(epochs)}",
            "run_dir": str(run),
        }
    )
    return row


def display(value, digits=4):
    if value == "" or value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def monitored_gpu_peaks(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    peaks: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            phase = str(row.get("phase", ""))
            try:
                used = float(row["gpu_used_mib"])
            except (KeyError, TypeError, ValueError):
                continue
            peaks[phase] = max(peaks.get(phase, 0.0), used)
    return peaks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study-dir",
        default=".experiments/minimal_ablation_80e_20260903",
    )
    parser.add_argument("--a1-run", default="")
    parser.add_argument("--a2-run", default="")
    args = parser.parse_args()

    study = Path(args.study_dir)
    monitor_peaks = monitored_gpu_peaks(study / "runtime/gpu_monitor.csv")
    a1 = run_row(
        Path(args.a1_run).resolve() if args.a1_run else None,
        {
            "experiment": "A1 Factorized Only",
            "factorized": 1,
            "graph": 0,
            "feature_fusion": 0,
            "topology_conditioned": 0,
        },
    )
    a2 = run_row(
        Path(args.a2_run).resolve() if args.a2_run else None,
        {
            "experiment": "A2 Graph Feature Fusion",
            "factorized": 1,
            "graph": 1,
            "feature_fusion": 1,
            "topology_conditioned": 0,
        },
    )
    if "A1" in monitor_peaks:
        a1["peak_vram"] = monitor_peaks["A1"]
    if "A2" in monitor_peaks:
        a2["peak_vram"] = monitor_peaks["A2"]

    variants = [
        {
            "experiment": "A0 PoseMamba",
            "factorized": 0,
            "graph": 0,
            "feature_fusion": 0,
            "topology_conditioned": 0,
            "params": 790083,
            "best_epoch": 60,
            "P1": 40.22602202713938,
            "P2": 33.51762634743647,
            "raw_P1": "",
            "raw_P2": "",
            "peak_vram": "",
            "it_per_sec": "",
            "status": "COMPLETED_EXISTING",
            "run_dir": "existing PoseMamba W64/D6/M1 seed0",
        },
        a1,
        a2,
        {
            "experiment": "A3 Full",
            "factorized": 1,
            "graph": 1,
            "feature_fusion": 0,
            "topology_conditioned": 1,
            "params": 800083,
            "best_epoch": 53,
            "P1": 39.84516172662857,
            "P2": 33.23224035987441,
            "raw_P1": 41.61853340867378,
            "raw_P2": 34.26450820132349,
            "peak_vram": 3734,
            "it_per_sec": "",
            "status": "COMPLETED_EXISTING",
            "run_dir": "existing GraphConditionedPoseMamba seed0",
        },
    ]

    study.mkdir(parents=True, exist_ok=True)
    fields = (
        "experiment",
        "factorized",
        "graph",
        "feature_fusion",
        "topology_conditioned",
        "params",
        "best_epoch",
        "P1",
        "P2",
        "raw_P1",
        "raw_P2",
        "peak_vram",
        "it_per_sec",
        "status",
        "run_dir",
    )
    with (study / "ablation_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(variants)

    lines = [
        "# Minimal ablation table",
        "",
        "All rows use the best EMA P1 within the registered budget and the P2 from the same checkpoint.",
        "",
        "| Variant | Factorized | Graph | Feature Fusion | Topology-conditioned Dynamics | Params | P1 | P2 | Status |",
        "|---|---|---|---|---|---:|---:|---:|---|",
    ]
    mark = {0: "—", 1: "✓"}
    for row in variants:
        lines.append(
            "| {experiment} | {factorized} | {graph} | {feature_fusion} | "
            "{topology_conditioned} | {params} | {p1} | {p2} | {status} |".format(
                experiment=row["experiment"],
                factorized=mark[row["factorized"]],
                graph=mark[row["graph"]],
                feature_fusion=mark[row["feature_fusion"]],
                topology_conditioned=mark[row["topology_conditioned"]],
                params=display(row.get("params"), 0),
                p1=display(row.get("P1")),
                p2=display(row.get("P2")),
                status=row["status"],
            )
        )
    (study / "ablation_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
