"""MPI-INF-3DHP adapter repaired against the official T=81 protocol.

Training follows MotionAGFormer's published T=81 sampling contract: stride-9
starts and deterministic resampling of the final short tail.  Test windows are
centred on every *valid* evaluation frame exactly once and edge padded.  The
adapter never clips or otherwise rewrites 3D labels.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from lib.utils.utils_data import flip_data


# Project/H36M order -> MPI-INF-3DHP order.
H36M_TO_MPI = {
    0: 14, 1: 11, 2: 12, 3: 13, 4: 8, 5: 9, 6: 10,
    7: 15, 8: 1, 9: 16, 10: 0, 11: 2, 12: 3, 13: 4,
    14: 5, 15: 6, 16: 7,
}
MPI_INDEX_FOR_H36M = [H36M_TO_MPI[i] for i in range(17)]


def _sequence_token(name: str) -> str:
    """Return TS1..TS6 even when a train key is ``S1 TS1``."""

    return str(name).strip().split()[-1]


class MotionDatasetMPI3DHPProtocolV2(Dataset):
    """Windowed 3DHP data with explicit centre-frame evaluation semantics."""

    protocol_name = "mpi-inf-3dhp-gt2d-f81-official-repaired-v1"

    def __init__(self, args: Any, data_split: str):
        if data_split not in {"train", "test"}:
            raise ValueError(f"unknown split: {data_split}")
        self.data_root = os.fspath(args.data_root)
        self.data_split = data_split
        self.clip_len = int(args.clip_len)
        if self.clip_len % 2 != 1:
            raise ValueError("protocol-v2 requires an odd clip_len for a centre frame")
        self.center_offset = self.clip_len // 2
        self.stride = int(
            getattr(args, "data_stride", 9)
            if data_split == "train"
            else getattr(args, "test_data_stride", 1)
        )
        if self.stride < 1:
            raise ValueError("data stride must be positive")
        if data_split == "test" and self.stride != 1:
            raise ValueError(
                "protocol-v2 evaluates every valid test centre exactly once; "
                "test_data_stride must be 1"
            )
        self.train_tail_policy = str(
            getattr(args, "mpi3dhp_train_tail_policy", "resample")
        )
        if data_split == "train" and self.train_tail_policy != "resample":
            raise ValueError(
                "official-repaired 3DHP training requires "
                "mpi3dhp_train_tail_policy=resample"
            )
        self.flip = bool(getattr(args, "flip", False)) and data_split == "train"
        self.sequences: list[dict[str, Any]] = []
        self.indices: list[tuple[int, int]] = []
        self._load_sequences()
        self._build_indices()

    @staticmethod
    def _sequence_hw(seq_name: str) -> tuple[int, int]:
        # TS5/TS6 use 1920x1080; all other sequences use 2048x2048.
        if _sequence_token(seq_name) in {"TS5", "TS6"}:
            return 1920, 1080
        return 2048, 2048

    @staticmethod
    def _normalize_2d(data_2d: np.ndarray, width: int, height: int) -> np.ndarray:
        x = np.asarray(data_2d, dtype=np.float32).copy()
        if x.ndim != 3 or x.shape[1] != 17 or x.shape[2] < 2:
            raise ValueError(f"expected 2D data [T,17,2(+)], got {x.shape}")
        out = np.zeros((x.shape[0], 17, 3), dtype=np.float32)
        out[..., :2] = x[..., :2]
        out[..., 0] = out[..., 0] / float(width) * 2.0 - 1.0
        out[..., 1] = out[..., 1] / float(width) * 2.0 - float(height) / float(width)
        # The released 3DHP package has no detector confidence channel.  Keep
        # this explicit and auditable rather than silently treating it as real.
        out[..., 2] = 1.0
        return out

    @staticmethod
    def _normalize_3d(data_3d: np.ndarray, width: int, height: int) -> np.ndarray:
        x = np.asarray(data_3d, dtype=np.float32).copy()
        if x.ndim != 3 or x.shape[1] != 17 or x.shape[2] < 3:
            raise ValueError(f"expected 3D data [T,17,3(+)], got {x.shape}")
        x = x[..., :3]
        x[..., 0] = x[..., 0] / float(width) * 2.0 - 1.0
        x[..., 1] = x[..., 1] / float(width) * 2.0 - float(height) / float(width)
        x[..., 2] = x[..., 2] / float(width) * 2.0
        return x

    def _append_sequence(
        self,
        seq_name: str,
        source_name: str,
        data_2d_raw: np.ndarray,
        data_3d_raw: np.ndarray,
        valid_mask: np.ndarray | None,
    ) -> None:
        width, height = self._sequence_hw(seq_name)
        data_2d_raw = np.asarray(data_2d_raw)[:, MPI_INDEX_FOR_H36M, :]
        data_3d_raw = np.asarray(data_3d_raw)[:, MPI_INDEX_FOR_H36M, :]
        data_2d = self._normalize_2d(data_2d_raw, width, height)
        data_3d = self._normalize_3d(data_3d_raw, width, height)
        if valid_mask is None:
            valid = np.ones((data_2d.shape[0],), dtype=np.float32)
        else:
            valid = np.asarray(valid_mask, dtype=np.float32).reshape(-1)
            if valid.shape[0] != data_2d.shape[0]:
                raise ValueError(
                    f"valid mask length {valid.shape[0]} != frames {data_2d.shape[0]}"
                )
        self.sequences.append({
            "source": source_name,
            "sequence": _sequence_token(seq_name),
            "width": float(width),
            "height": float(height),
            "input_2d": data_2d,
            "label_3d": data_3d,
            "valid": valid,
        })

    def _load_sequences(self) -> None:
        filename = "data_train_3dhp.npz" if self.data_split == "train" else "data_test_3dhp.npz"
        path = os.path.join(self.data_root, filename)
        data = np.load(path, allow_pickle=True)["data"].item()
        if self.data_split == "train":
            for seq_key in sorted(data):
                cam_dict = data[seq_key][0]
                for cam_key in sorted(cam_dict):
                    anim = cam_dict[cam_key]
                    self._append_sequence(
                        seq_key,
                        f"{seq_key}_cam{cam_key}",
                        anim["data_2d"],
                        anim["data_3d"],
                        None,
                    )
        else:
            for seq_name in sorted(data):
                anim = data[seq_name]
                self._append_sequence(
                    seq_name,
                    str(seq_name),
                    anim["data_2d"],
                    anim["data_3d"],
                    anim.get("valid"),
                )

    def _build_indices(self) -> None:
        for seq_idx, seq in enumerate(self.sequences):
            n_frames = int(seq["input_2d"].shape[0])
            if self.data_split == "test":
                # Use the official valid mask as the evaluation population.
                # The tuple stores a centre index, not a window start.
                self.indices.extend(
                    (seq_idx, center)
                    for center in np.flatnonzero(seq["valid"] > 0.5).tolist()
                )
                continue

            # Match MotionAGFormer's official MPI3DHP.partition(): every
            # stride-aligned start is retained, including the final short tail.
            self.indices.extend(
                (seq_idx, start) for start in range(0, n_frames, self.stride)
            )

    @staticmethod
    def _resample_indices(original_length: int, target_length: int) -> np.ndarray:
        """Return MotionAGFormer-compatible deterministic tail indices."""

        if original_length < 1:
            raise ValueError("cannot resample an empty 3DHP tail")
        even = np.linspace(0, original_length, num=target_length, endpoint=False)
        return np.clip(
            np.floor(even), 0, original_length - 1
        ).astype(np.int64)

    def protocol_summary(self) -> dict[str, Any]:
        valid = [float(seq["valid"].sum()) for seq in self.sequences]
        return {
            "protocol": self.protocol_name,
            "split": self.data_split,
            "clip_len": self.clip_len,
            "center_offset": self.center_offset,
            "stride": self.stride,
            "sequence_count": len(self.sequences),
            "window_count": len(self.indices),
            "evaluation_population": (
                "all_valid_centres_once_with_edge_padded_context"
                if self.data_split == "test"
                else "stride_aligned_windows_with_resampled_short_tail"
            ),
            "train_tail_policy": (
                self.train_tail_policy if self.data_split == "train" else None
            ),
            "valid_frame_counts": {
                seq["source"]: int(seq["valid"].sum()) for seq in self.sequences
            },
            "confidence_definition": "constant_1.0_for_3dhp_release_npz",
            "valid_frames_total": int(sum(valid)),
        }

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        seq_idx, anchor = self.indices[index]
        seq = self.sequences[seq_idx]
        if self.data_split == "test":
            center = int(anchor)
            offsets = np.arange(-self.center_offset, self.center_offset + 1)
            frame_indices = np.clip(
                center + offsets,
                0,
                int(seq["input_2d"].shape[0]) - 1,
            )
            motion_2d = seq["input_2d"][frame_indices].copy()
            motion_3d = seq["label_3d"][frame_indices].copy()
        else:
            start = int(anchor)
            stop = start + self.clip_len
            motion_2d = seq["input_2d"][start:stop]
            motion_3d = seq["label_3d"][start:stop]
            if motion_2d.shape[0] != self.clip_len:
                indices = self._resample_indices(
                    int(motion_2d.shape[0]), self.clip_len
                )
                motion_2d = motion_2d[indices]
                motion_3d = motion_3d[indices]
            motion_2d = motion_2d.copy()
            motion_3d = motion_3d.copy()
        if self.flip and random.random() > 0.5:
            motion_2d = flip_data(motion_2d)
            motion_3d = flip_data(motion_3d)
        if self.data_split == "train":
            return torch.from_numpy(motion_2d).float(), torch.from_numpy(motion_3d).float()
        return (
            torch.from_numpy(motion_2d).float(),
            torch.from_numpy(motion_3d).float(),
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(seq["width"], dtype=torch.float32),
            torch.tensor(seq["height"], dtype=torch.float32),
            seq["source"],
            torch.tensor(center, dtype=torch.long),
        )
