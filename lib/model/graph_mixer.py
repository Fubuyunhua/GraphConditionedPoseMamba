"""Fixed-topology relation mixing for Human3.6M 17-joint poses.

The mixer deliberately keeps graph construction static.  Bone and bilateral
symmetry relations are represented by separate directed edge lists, while the
same lightweight relation/neighbor projections are shared by both edge types.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


H36M_JOINT_NAMES: Tuple[str, ...] = (
    "root",
    "rhip",
    "rknee",
    "rankle",
    "lhip",
    "lknee",
    "lankle",
    "belly",
    "neck",
    "nose",
    "head",
    "lshoulder",
    "lelbow",
    "lwrist",
    "rshoulder",
    "relbow",
    "rwrist",
)

H36M_BONE_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 4),
    (4, 5),
    (5, 6),
    (0, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (8, 11),
    (11, 12),
    (12, 13),
    (8, 14),
    (14, 15),
    (15, 16),
)

H36M_SYMMETRY_EDGES: Tuple[Tuple[int, int], ...] = (
    (1, 4),
    (2, 5),
    (3, 6),
    (11, 14),
    (12, 15),
    (13, 16),
)


def _make_directed_edges(
    edges: Sequence[Tuple[int, int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    targets: List[int] = []
    sources: List[int] = []
    for left, right in edges:
        targets.extend((left, right))
        sources.extend((right, left))
    return torch.tensor(targets, dtype=torch.long), torch.tensor(sources, dtype=torch.long)


def h36m_neighbor_names(use_symmetry_edges: bool = True) -> Dict[str, Dict[str, List[str]]]:
    """Return human-readable fixed graph neighborhoods for diagnostics."""

    bone: List[List[str]] = [[] for _ in H36M_JOINT_NAMES]
    symmetry: List[List[str]] = [[] for _ in H36M_JOINT_NAMES]
    for left, right in H36M_BONE_EDGES:
        bone[left].append(H36M_JOINT_NAMES[right])
        bone[right].append(H36M_JOINT_NAMES[left])
    if use_symmetry_edges:
        for left, right in H36M_SYMMETRY_EDGES:
            symmetry[left].append(H36M_JOINT_NAMES[right])
            symmetry[right].append(H36M_JOINT_NAMES[left])
    return {
        name: {"bone": bone[index], "symmetry": symmetry[index]}
        for index, name in enumerate(H36M_JOINT_NAMES)
    }


class SkeletonGraphMixer(nn.Module):
    """Mix fixed bone and bilateral-symmetry relations without replacing input.

    For every directed neighbor relation ``j -> i`` the message is

    ``m_ij = W_relation(x_j - x_i) + W_neighbor(x_j)``.

    Bone and symmetry messages are summed independently, scaled by learned
    scalars initialized to one, combined, activated, and projected back to the
    model width.  The return value is a residual/context feature ``G``; callers
    decide whether and where to combine it with ``X``.
    """

    def __init__(
        self,
        dim: int,
        hidden_ratio: float = 0.5,
        use_symmetry_edges: bool = True,
        num_joints: int = 17,
    ) -> None:
        super().__init__()
        if int(num_joints) != len(H36M_JOINT_NAMES):
            raise ValueError("SkeletonGraphMixer requires the Human3.6M 17-joint topology")
        hidden_dim = max(1, int(dim * float(hidden_ratio)))
        self.dim = int(dim)
        self.hidden_dim = hidden_dim
        self.num_joints = int(num_joints)
        self.use_symmetry_edges = bool(use_symmetry_edges)

        # Both transforms consume the same joint feature in the dense path.
        # A single wider GEMM is algebraically identical to two small Linear
        # calls and considerably easier for Inductor/cuBLAS to schedule.
        self.message_proj = nn.Linear(self.dim, 2 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, self.dim)
        self.activation = nn.GELU()

        bone_targets, bone_sources = _make_directed_edges(H36M_BONE_EDGES)
        sym_targets, sym_sources = _make_directed_edges(H36M_SYMMETRY_EDGES)
        self.register_buffer("bone_targets", bone_targets, persistent=True)
        self.register_buffer("bone_sources", bone_sources, persistent=True)
        self.register_buffer("symmetry_targets", sym_targets, persistent=True)
        self.register_buffer("symmetry_sources", sym_sources, persistent=True)

        bone_adjacency = torch.zeros(self.num_joints, self.num_joints)
        bone_adjacency[bone_targets, bone_sources] = 1.0
        symmetry_adjacency = torch.zeros(self.num_joints, self.num_joints)
        symmetry_adjacency[sym_targets, sym_sources] = 1.0
        self.register_buffer("bone_adjacency", bone_adjacency, persistent=False)
        self.register_buffer(
            "symmetry_adjacency", symmetry_adjacency, persistent=False
        )
        self.register_buffer(
            "bone_degree", bone_adjacency.sum(dim=-1), persistent=False
        )
        self.register_buffer(
            "symmetry_degree", symmetry_adjacency.sum(dim=-1), persistent=False
        )
        self.use_dense_aggregation = True

        self.alpha_bone = nn.Parameter(torch.ones(1))
        self.alpha_sym = nn.Parameter(torch.ones(1))

    def _aggregate(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        sources: torch.Tensor,
    ) -> torch.Tensor:
        source_features = x.index_select(1, sources)
        target_features = x.index_select(1, targets)
        relation_weight, neighbor_weight = self.message_proj.weight.split(
            self.hidden_dim, dim=0
        )
        relation_bias, neighbor_bias = self.message_proj.bias.split(
            self.hidden_dim, dim=0
        )
        messages = F.linear(
            source_features - target_features,
            relation_weight,
            relation_bias,
        )
        messages = messages + F.linear(
            source_features,
            neighbor_weight,
            neighbor_bias,
        )
        output = x.new_zeros(x.shape[0], self.num_joints, self.hidden_dim)
        return output.index_add(1, targets, messages)

    def _aggregate_dense(self, x: torch.Tensor) -> torch.Tensor:
        """Algebraically fuse all fixed-edge messages into one 17x17 matmul."""
        relation, neighbor = self.message_proj(x).split(self.hidden_dim, dim=-1)
        adjacency = self.alpha_bone * self.bone_adjacency
        degree = self.alpha_bone * self.bone_degree
        if self.use_symmetry_edges:
            adjacency = adjacency + self.alpha_sym * self.symmetry_adjacency
            degree = degree + self.alpha_sym * self.symmetry_degree

        mixed = torch.matmul(adjacency.to(dtype=x.dtype), relation + neighbor)
        mixed = mixed - degree.to(dtype=x.dtype)[None, :, None] * relation
        relation_bias = self.message_proj.bias[: self.hidden_dim]
        mixed = mixed + (
            degree.to(dtype=x.dtype)[None, :, None]
            * relation_bias[None, None, :]
        )
        return mixed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected [B,T,J,C], received shape {tuple(x.shape)}")
        batch, frames, joints, channels = x.shape
        if joints != self.num_joints or channels != self.dim:
            raise ValueError(
                f"expected J={self.num_joints}, C={self.dim}; received J={joints}, C={channels}"
            )

        flat = x.reshape(batch * frames, joints, channels)
        if self.use_dense_aggregation:
            mixed = self._aggregate_dense(flat)
        else:
            bone = self._aggregate(flat, self.bone_targets, self.bone_sources)
            mixed = self.alpha_bone * bone
            if self.use_symmetry_edges:
                symmetry = self._aggregate(
                    flat, self.symmetry_targets, self.symmetry_sources
                )
                mixed = mixed + self.alpha_sym * symmetry
        output = self.out_proj(self.activation(mixed))
        return output.reshape(batch, frames, joints, channels)

    def neighbor_names(self) -> Dict[str, Dict[str, List[str]]]:
        return h36m_neighbor_names(self.use_symmetry_edges)


__all__ = [
    "H36M_JOINT_NAMES",
    "H36M_BONE_EDGES",
    "H36M_SYMMETRY_EDGES",
    "SkeletonGraphMixer",
    "h36m_neighbor_names",
]
