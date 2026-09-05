"""Fixed-topology relation mixing for Human3.6M 17-joint poses.

The mixer deliberately keeps graph construction static.  Bone and bilateral
symmetry relations are represented by separate directed edge lists, while the
same lightweight relation/neighbor projections are shared by both edge types.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict, List, Sequence, Tuple

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


def _canonical_edge(edge: Sequence[int]) -> Tuple[int, int]:
    left, right = (int(edge[0]), int(edge[1]))
    if left == right:
        raise ValueError(f"self-loop is not allowed: {(left, right)}")
    return (left, right) if left < right else (right, left)


def _normalized_edges(
    edges: Sequence[Sequence[int]],
    *,
    num_nodes: int,
) -> Tuple[Tuple[int, int], ...]:
    normalized = tuple(_canonical_edge(edge) for edge in edges)
    if any(left < 0 or right >= num_nodes for left, right in normalized):
        raise ValueError(f"edge outside [0,{num_nodes}): {normalized}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"duplicate undirected edge: {normalized}")
    return tuple(sorted(normalized))


def _degrees(edges: Sequence[Sequence[int]], num_nodes: int) -> Tuple[int, ...]:
    degree = [0] * int(num_nodes)
    for left, right in edges:
        degree[int(left)] += 1
        degree[int(right)] += 1
    return tuple(degree)


def _is_connected(edges: Sequence[Sequence[int]], num_nodes: int) -> bool:
    adjacency = [[] for _ in range(num_nodes)]
    for left, right in edges:
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == num_nodes


def _degree_preserving_rewire(
    edges: Sequence[Sequence[int]],
    *,
    num_nodes: int,
    seed: int,
    require_connected: bool,
    successful_swaps: int,
) -> Tuple[Tuple[int, int], ...]:
    """Deterministic undirected double-edge swaps with a private RNG."""

    original = _normalized_edges(edges, num_nodes=num_nodes)
    current = set(original)
    rng = random.Random(int(seed))
    accepted = 0
    attempts = 0
    max_attempts = max(10_000, int(successful_swaps) * 2_000)
    while accepted < successful_swaps and attempts < max_attempts:
        attempts += 1
        first, second = rng.sample(sorted(current), 2)
        a, b = first
        c, d = second
        if len({a, b, c, d}) != 4:
            continue
        if rng.randrange(2):
            proposed = {_canonical_edge((a, c)), _canonical_edge((b, d))}
        else:
            proposed = {_canonical_edge((a, d)), _canonical_edge((b, c))}
        if len(proposed) != 2:
            continue
        remaining = current - {first, second}
        if proposed & remaining:
            continue
        candidate = remaining | proposed
        if require_connected and not _is_connected(candidate, num_nodes):
            continue
        current = candidate
        accepted += 1
    if accepted != successful_swaps:
        raise RuntimeError(
            "unable to satisfy degree-preserving rewiring constraints: "
            f"accepted={accepted} requested={successful_swaps} attempts={attempts}"
        )
    result = tuple(sorted(current))
    if result == original:
        raise RuntimeError("degree-preserving rewiring returned the anatomical graph")
    if _degrees(result, num_nodes) != _degrees(original, num_nodes):
        raise RuntimeError("degree sequence changed during rewiring")
    if require_connected and not _is_connected(result, num_nodes):
        raise RuntimeError("rewired bone graph is disconnected")
    return result


def build_h36m_graph_spec(
    mode: str = "anatomical",
    *,
    seed: int = 3407,
) -> Dict[str, Any]:
    """Build one immutable, JSON-serializable topology specification."""

    mode = str(mode).lower()
    if mode not in {"anatomical", "degree_preserving_rewired"}:
        raise ValueError(
            "graph_topology_mode must be anatomical or "
            f"degree_preserving_rewired, received {mode!r}"
        )
    anatomical_bone = tuple(H36M_BONE_EDGES)
    anatomical_symmetry = tuple(H36M_SYMMETRY_EDGES)
    if mode == "anatomical":
        bone_edges = anatomical_bone
        symmetry_edges = anatomical_symmetry
        generator = "fixed_h36m_anatomical_v1"
    else:
        bone_edges = _degree_preserving_rewire(
            anatomical_bone,
            num_nodes=len(H36M_JOINT_NAMES),
            seed=int(seed) * 2 + 1,
            require_connected=True,
            successful_swaps=len(anatomical_bone) * 8,
        )
        symmetry_edges = _degree_preserving_rewire(
            anatomical_symmetry,
            num_nodes=len(H36M_JOINT_NAMES),
            seed=int(seed) * 2 + 2,
            require_connected=False,
            successful_swaps=len(anatomical_symmetry) * 8,
        )
        generator = "private_python_rng_double_edge_swap_v1"

    bone_set = set(_normalized_edges(bone_edges, num_nodes=17))
    symmetry_set = set(_normalized_edges(symmetry_edges, num_nodes=17))
    anatomical_bone_set = set(_normalized_edges(anatomical_bone, num_nodes=17))
    anatomical_symmetry_set = set(
        _normalized_edges(anatomical_symmetry, num_nodes=17)
    )
    payload: Dict[str, Any] = {
        "mode": mode,
        "graph_rewire_seed": int(seed),
        "generator": generator,
        "joint_names": list(H36M_JOINT_NAMES),
        "bone_edges": [list(edge) for edge in bone_edges],
        "symmetry_edges": [list(edge) for edge in symmetry_edges],
        "bone_degrees": list(_degrees(bone_edges, 17)),
        "symmetry_degrees": list(_degrees(symmetry_edges, 17)),
        "bone_connected": _is_connected(bone_edges, 17),
        "bone_edge_overlap_with_anatomical": len(bone_set & anatomical_bone_set)
        / len(anatomical_bone_set),
        "symmetry_edge_overlap_with_anatomical": len(
            symmetry_set & anatomical_symmetry_set
        )
        / len(anatomical_symmetry_set),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _make_directed_edges(
    edges: Sequence[Tuple[int, int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    targets: List[int] = []
    sources: List[int] = []
    for left, right in edges:
        targets.extend((left, right))
        sources.extend((right, left))
    return torch.tensor(targets, dtype=torch.long), torch.tensor(sources, dtype=torch.long)


def h36m_neighbor_names(
    use_symmetry_edges: bool = True,
    *,
    bone_edges: Sequence[Sequence[int]] = H36M_BONE_EDGES,
    symmetry_edges: Sequence[Sequence[int]] = H36M_SYMMETRY_EDGES,
) -> Dict[str, Dict[str, List[str]]]:
    """Return human-readable fixed graph neighborhoods for diagnostics."""

    bone: List[List[str]] = [[] for _ in H36M_JOINT_NAMES]
    symmetry: List[List[str]] = [[] for _ in H36M_JOINT_NAMES]
    for left, right in bone_edges:
        bone[left].append(H36M_JOINT_NAMES[right])
        bone[right].append(H36M_JOINT_NAMES[left])
    if use_symmetry_edges:
        for left, right in symmetry_edges:
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
        graph_topology_mode: str = "anatomical",
        graph_rewire_seed: int = 3407,
        topology_spec: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if int(num_joints) != len(H36M_JOINT_NAMES):
            raise ValueError("SkeletonGraphMixer requires the Human3.6M 17-joint topology")
        hidden_dim = max(1, int(dim * float(hidden_ratio)))
        self.dim = int(dim)
        self.hidden_dim = hidden_dim
        self.num_joints = int(num_joints)
        self.use_symmetry_edges = bool(use_symmetry_edges)
        topology_spec = topology_spec or build_h36m_graph_spec(
            graph_topology_mode,
            seed=graph_rewire_seed,
        )
        if topology_spec["mode"] != str(graph_topology_mode).lower():
            raise ValueError("topology_spec mode does not match graph_topology_mode")
        self.graph_topology_mode = str(topology_spec["mode"])
        self.graph_rewire_seed = int(topology_spec["graph_rewire_seed"])
        self.graph_topology_hash = str(topology_spec["sha256"])
        self._topology_spec = json.loads(json.dumps(topology_spec))

        # Both transforms consume the same joint feature in the dense path.
        # A single wider GEMM is algebraically identical to two small Linear
        # calls and considerably easier for Inductor/cuBLAS to schedule.
        self.message_proj = nn.Linear(self.dim, 2 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, self.dim)
        self.activation = nn.GELU()

        bone_edges = tuple(tuple(edge) for edge in topology_spec["bone_edges"])
        symmetry_edges = tuple(
            tuple(edge) for edge in topology_spec["symmetry_edges"]
        )
        bone_targets, bone_sources = _make_directed_edges(bone_edges)
        sym_targets, sym_sources = _make_directed_edges(symmetry_edges)
        persistent_topology = self.graph_topology_mode == "anatomical"
        self.register_buffer(
            "bone_targets", bone_targets, persistent=persistent_topology
        )
        self.register_buffer(
            "bone_sources", bone_sources, persistent=persistent_topology
        )
        self.register_buffer(
            "symmetry_targets", sym_targets, persistent=persistent_topology
        )
        self.register_buffer(
            "symmetry_sources", sym_sources, persistent=persistent_topology
        )

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
        bone_edges = tuple(
            (int(left), int(right))
            for left, right in zip(
                self.bone_targets[::2].tolist(), self.bone_sources[::2].tolist()
            )
        )
        symmetry_edges = tuple(
            (int(left), int(right))
            for left, right in zip(
                self.symmetry_targets[::2].tolist(),
                self.symmetry_sources[::2].tolist(),
            )
        )
        return h36m_neighbor_names(
            self.use_symmetry_edges,
            bone_edges=bone_edges,
            symmetry_edges=symmetry_edges,
        )

    def topology_metadata(self) -> Dict[str, Any]:
        return json.loads(json.dumps(self._topology_spec))


__all__ = [
    "H36M_JOINT_NAMES",
    "H36M_BONE_EDGES",
    "H36M_SYMMETRY_EDGES",
    "SkeletonGraphMixer",
    "build_h36m_graph_spec",
    "h36m_neighbor_names",
]
