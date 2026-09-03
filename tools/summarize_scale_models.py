#!/usr/bin/env python3
"""Build registered CSV and Markdown results for the model-scaling study."""

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


def clean(path: Path) -> str:
    return ANSI.sub("", path.read_text(encoding="utf-8", errors="replace")).replace(
        "\r", "\n"
    )


def verified(path: Path) -> tuple[float | None, float | None]:
    if not path.is_file():
        return None, None
    text = clean(path)
    p1 = P1.findall(text)
    p2 = P2.findall(text)
    return (float(p1[-1]) if p1 else None, float(p2[-1]) if p2 else None)


def run_row(run: Path | None, base: dict) -> dict:
    row = dict(base)
    if run is None:
        row.update(
            {
                "parameters": row.get("parameters", ""),
                "best_epoch": "",
                "ema_p1": "",
                "ema_p2": "",
                "raw_p1": "",
                "raw_p2": "",
                "peak_vram_mib": "",
                "it_per_sec": "",
                "status": "PLANNED",
                "run_dir": "",
            }
        )
        return row
    log = run / "log.txt"
    if not log.is_file():
        row.update({"status": "FAILED_NO_LOG", "run_dir": str(run)})
        return row
    text = clean(log)
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
        if int(a) <= 80
    ]
    runtimes = [
        {
            "epoch": int(a),
            "it_per_sec": float(b),
            "peak_reserved": float(d),
        }
        for a, b, c, d in RUNTIME.findall(text)
    ]
    parameters = PARAMETERS.findall(text)
    if not epochs:
        row.update({"status": "RUNNING_NO_COMPLETE_EPOCH", "run_dir": str(run)})
        return row
    best = min(epochs, key=lambda item: item["p1"])
    ema_p1, ema_p2 = verified(run / "verification_best_ema_epoch.stdout")
    raw_p1, raw_p2 = verified(run / "verification_best_epoch.stdout")
    stable = [item["it_per_sec"] for item in runtimes if item["epoch"] > 1]
    complete = len(epochs) == 80 and (run / "latest_epoch.bin").is_file()
    row.update(
        {
            "parameters": int(parameters[-1]) if parameters else row.get("parameters", ""),
            "best_epoch": best["epoch"],
            "ema_p1": ema_p1 if ema_p1 is not None else best["p1"],
            "ema_p2": ema_p2 if ema_p2 is not None else best["p2"],
            "raw_p1": raw_p1 if raw_p1 is not None else "",
            "raw_p2": raw_p2 if raw_p2 is not None else "",
            "peak_vram_mib": max(
                (item["peak_reserved"] for item in runtimes), default=""
            ),
            "it_per_sec": sum(stable) / len(stable) if stable else "",
            "status": "COMPLETED" if complete else f"RUNNING_EPOCH_{len(epochs)}",
            "run_dir": str(run),
        }
    )
    return row


def display(value, digits=4):
    if value == "" or value is None:
        return "—"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study-dir", default=".experiments/model_scaling_80e_20260903"
    )
    parser.add_argument("--s1-run", default="")
    parser.add_argument("--s2-run", default="")
    args = parser.parse_args()
    study = Path(args.study_dir)
    rows = [
        {
            "experiment": "Full W64/D8",
            "width": 64,
            "depth": 8,
            "parameters": 800083,
            "best_epoch": 53,
            "ema_p1": 39.84516172662857,
            "ema_p2": 33.23224035987441,
            "raw_p1": 41.61853340867378,
            "raw_p2": 34.26450820132349,
            "peak_vram_mib": 3734,
            "it_per_sec": "",
            "status": "COMPLETED_EXISTING",
            "run_dir": "existing GraphConditionedPoseMamba seed0",
        },
        run_row(
            Path(args.s1_run).resolve() if args.s1_run else None,
            {
                "experiment": "S1 W128/D20",
                "width": 128,
                "depth": 20,
                "parameters": 6836355,
            },
        ),
        run_row(
            Path(args.s2_run).resolve() if args.s2_run else None,
            {
                "experiment": "S2 W256/D10",
                "width": 256,
                "depth": 10,
                "parameters": 12646107,
            },
        ),
    ]
    study.mkdir(parents=True, exist_ok=True)
    fields = (
        "experiment",
        "width",
        "depth",
        "parameters",
        "best_epoch",
        "ema_p1",
        "ema_p2",
        "raw_p1",
        "raw_p2",
        "peak_vram_mib",
        "it_per_sec",
        "status",
        "run_dir",
    )
    with (study / "scaling_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Model-scaling results",
        "",
        "P2 is always paired with the EMA checkpoint selected by minimum P1 within 80 epochs.",
        "",
        "| Variant | Width | Depth | Params | Best epoch | EMA P1 | Paired P2 | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {experiment} | {width} | {depth} | {parameters} | {best_epoch} | "
            "{p1} | {p2} | {status} |".format(
                experiment=row["experiment"],
                width=row["width"],
                depth=row["depth"],
                parameters=display(row.get("parameters"), 0),
                best_epoch=display(row.get("best_epoch"), 0),
                p1=display(row.get("ema_p1")),
                p2=display(row.get("ema_p2")),
                status=row["status"],
            )
        )
    (study / "scaling_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
