#!/usr/bin/env python3
"""Stop the named MPI Full process after complete epoch-80 checkpoints exist."""

import json
import os
import shutil
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


PID = 1136724
ROOT = Path("/scratch/home/caiwei/GraphConditionedPoseMamba_MPI_FULL_20260906")
RUN = ROOT / "runs/mpi_full_testbest_seed0"
MARKER = ROOT / "verification/user_stop_at_epoch80.json"


def process_identity():
    proc = Path(f"/proc/{PID}")
    if not proc.exists():
        return None
    command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(
        "utf-8", "replace"
    )
    cwd = (proc / "cwd").resolve()
    expected = (
        "train_3dhp.py --config "
        "configs/pose3d_3dhp/graph_posemamba_3dhp_w64_d8_testbest.yaml"
    )
    if cwd != ROOT or expected not in command:
        raise RuntimeError(f"refusing unexpected PID identity: cwd={cwd} cmd={command}")
    return {"cwd": str(cwd), "command": command}


def load_epoch(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("epoch", -1)) != 80:
        raise RuntimeError(f"{path}: expected epoch80, got {payload.get('epoch')}")
    state = payload.get("model_pos")
    if not state:
        raise RuntimeError(f"{path}: missing model_pos")
    finite = all(
        torch.isfinite(value).all().item()
        for value in state.values()
        if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex())
    )
    if not finite:
        raise RuntimeError(f"{path}: non-finite model state")
    return payload


def latest_metric_epoch():
    history = json.loads((RUN / "metrics.json").read_text())
    return max((int(row["epoch"]) for row in history), default=0)


MARKER.parent.mkdir(parents=True, exist_ok=True)
while True:
    identity = process_identity()
    if identity is None:
        raise RuntimeError("target exited before a complete epoch80 stop point")
    if latest_metric_epoch() >= 80:
        raw = RUN / "latest_epoch.bin"
        ema = RUN / "latest_ema_epoch.bin"
        raw_payload = load_epoch(raw)
        ema_payload = load_epoch(ema)
        break
    time.sleep(5)

os.kill(PID, signal.SIGTERM)
deadline = time.time() + 120
while Path(f"/proc/{PID}").exists() and time.time() < deadline:
    time.sleep(1)
if Path(f"/proc/{PID}").exists():
    raise RuntimeError("SIGTERM did not stop the target within 120 seconds")

fixed_raw = RUN / "raw_fixed_epoch80_user_stop.bin"
fixed_ema = RUN / "ema_fixed_epoch80_user_stop.bin"
shutil.copy2(RUN / "latest_epoch.bin", fixed_raw)
shutil.copy2(RUN / "latest_ema_epoch.bin", fixed_ema)
load_epoch(fixed_raw)
load_epoch(fixed_ema)

record = {
    "status": "STOPPED_BY_USER_AT_ALIGNED_EPOCH80",
    "time": datetime.now(timezone.utc).isoformat(),
    "pid": PID,
    "signal": "SIGTERM",
    "identity": identity,
    "metric_epoch": latest_metric_epoch(),
    "raw_checkpoint": str(fixed_raw),
    "ema_checkpoint": str(fixed_ema),
    "raw_checkpoint_type": raw_payload.get("checkpoint_type"),
    "ema_checkpoint_type": ema_payload.get("checkpoint_type"),
}
MARKER.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
