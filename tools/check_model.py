"""Print shape, graph, protocol and parameter checks for the new backbone."""

import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lib.model.PoseMamba import GraphConditionedPoseMamba, PoseMamba
from lib.model.graph_mixer import h36m_neighbor_names
from lib.utils.learning import load_backbone
from lib.utils.tools import get_config


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/pose3d/graph_posemamba_h36m_w64_d8_0p8m.yaml",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    config = get_config(str(config_path))

    print("Graph neighborhoods")
    for joint, relations in h36m_neighbor_names(config.use_symmetry_edges).items():
        print(
            f"{joint:10s} bone={','.join(relations['bone']) or '-':35s} "
            f"symmetry={','.join(relations['symmetry']) or '-'}"
        )

    candidate = load_backbone(config)
    original = PoseMamba(
        num_frame=config.maxlen,
        num_joints=config.num_joints,
        in_chans=2 if config.no_conf else 3,
        embed_dim_ratio=config.dim_feat,
        depth=config.depth,
        mlp_ratio=config.mlp_ratio,
        drop_rate=config.dropout,
        drop_path_rate=config.drop_path_rate,
    )
    original_params = parameter_count(original)
    candidate_params = parameter_count(candidate)
    added = candidate_params - original_params
    print("\nParameter count")
    print(f"Original PoseMamba parameters:         {original_params:,}")
    print(f"GraphConditionedPoseMamba parameters: {candidate_params:,}")
    print(f"Net parameter change:                 {added:+,} ({added / original_params:+.2%})")

    if not torch.cuda.is_available():
        raise RuntimeError("the real selective-scan shape check requires CUDA")
    device = torch.device("cuda")
    candidate = candidate.to(device).eval()
    in_chans = 2 if config.no_conf else 3
    input_tensor = torch.randn(
        args.batch_size,
        config.maxlen,
        config.num_joints,
        in_chans,
        device=device,
    )
    with torch.no_grad():
        prediction, trace = candidate(input_tensor, return_shape_trace=True)

    print("\nTensor shapes")
    ordered = (
        "input",
        "embedding",
        "graph_feature",
        "spatial_ssm_input",
        "spatial_ssm_output",
        "temporal_ssm_input",
        "temporal_ssm_output",
        "final_prediction",
    )
    for name in ordered:
        print(f"{name:24s} {trace[name]}")
    expected = (args.batch_size, config.maxlen, config.num_joints, 3)
    if tuple(prediction.shape) != expected:
        raise AssertionError(f"prediction shape {tuple(prediction.shape)} != {expected}")
    if not bool(torch.isfinite(prediction).all()):
        raise AssertionError("prediction contains NaN or Inf")
    print("\nAll graph, parameter and real CUDA shape checks passed.")


if __name__ == "__main__":
    main()
