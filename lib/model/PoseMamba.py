## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

import math
import logging
from functools import partial
from collections import OrderedDict
from einops import rearrange, repeat
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as activation_checkpoint

import time

from math import sqrt
import os
import sys
# 获取当前工作目录
current_directory = os.path.dirname(__file__) + '/../' + '../'
sys.path.append(current_directory)
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import torch.nn.functional as F
from functools import partial
import torch.fft

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
import numpy as np

from lib.model.graph_mixer import SkeletonGraphMixer, build_h36m_graph_spec
from lib.model.mambablocks import BiSTSSM, BiSTSSMBlock, FactorizedBiSSM, Mlp
class  PoseMamba(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=256, depth=6, mlp_ratio=2., drop_rate=0., drop_path_rate=0.2,  norm_layer=None, posemamba_backward_mode="legacy"):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        posemamba_backward_mode = str(posemamba_backward_mode).lower()
        if posemamba_backward_mode not in {"legacy", "exact"}:
            raise ValueError(
                "posemamba_backward_mode must be legacy or exact, received "
                f"{posemamba_backward_mode!r}"
            )
        self.posemamba_backward_mode = posemamba_backward_mode
        scan_forward_type = (
            "v2_plus_poselimbs"
            if posemamba_backward_mode == "legacy"
            else "v2_plus_poselimbs_exact_backward"
        )

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = embed_dim_ratio   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        out_dim = 3     #### output dimension is num_joints * 3
        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.block_depth = depth
        self.STEblocks = nn.ModuleList([
           BiSTSSMBlock(
                hidden_dim = embed_dim_ratio, 
                mlp_ratio = mlp_ratio, 
                drop_path=dpr[i], 
                norm_layer=norm_layer,
                forward_type=scan_forward_type
                )
            for i in range(depth)])

        self.TTEblocks = nn.ModuleList([
           BiSTSSMBlock(
                hidden_dim = embed_dim, 
                mlp_ratio = mlp_ratio, 
                drop_path=dpr[i], 
                norm_layer=norm_layer,
                forward_type=scan_forward_type
                )
            for i in range(depth)])

        self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim , out_dim),
        )


    def STE_forward(self, x):
        b, f, n, c = x.shape  ##### b is batch size, f is number of frames, n is number of joints, c is channel size?
        x = rearrange(x, 'b f n c  -> (b f) n c', )
        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        x = rearrange(x, '(b f) n c  -> b f n c', f=f)
        blk = self.STEblocks[0]
        x = blk(x)

        x = self.Spatial_norm(x)
        return x

    def TTE_foward(self, x):
        # assert len(x.shape) == 3, "shape is equal to 3"
        b, f, n, c  = x.shape
        x = rearrange(x, 'b f n cw -> (b n) f cw', f=f)
        x += self.Temporal_pos_embed[:,:f,:]
        x = self.pos_drop(x)
        x = rearrange(x, '(b n) f cw -> b f n cw', n=n)
        blk = self.TTEblocks[0]
        x = blk(x)

        x = self.Temporal_norm(x)
        return x

    def ST_foward(self, x):
        assert len(x.shape)==4, "shape is equal to 4"
        b, f, n, cw = x.shape
        for i in range(1, self.block_depth):
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]
            x = steblock(x)
            x = self.Spatial_norm(x)
            x = tteblock(x)
            x = self.Temporal_norm(x)
        return x

    def forward(self, x):
        b, f, n, c = x.shape
        x = self.STE_forward(x)
        x = self.TTE_foward(x)
        x = self.ST_foward(x)
        x = self.head(x)
        x = x.view(b, f, n, -1)
        return x

    def execution_spec(self):
        return {
            "model": "PoseMamba",
            "posemamba_backward_mode": self.posemamba_backward_mode,
            "scan_forward_type": (
                "v2_plus_poselimbs"
                if self.posemamba_backward_mode == "legacy"
                else "v2_plus_poselimbs_exact_backward"
            ),
            "k_group": 4,
        }


class GraphConditionedPoseBlock(nn.Module):
    """Graph-local mixing followed by factorized or coupled BiSSMs."""

    def __init__(
        self,
        hidden_dim,
        num_joints=17,
        graph_hidden_ratio=0.5,
        use_graph_mixer=True,
        use_symmetry_edges=True,
        graph_conditioned_ssm=True,
        graph_injection_mode=None,
        reuse_graph_context=True,
        factorized_spatial_temporal=True,
        coupled_ssm_forward_type="v2_plus_poselimbs",
        graph_scale=1.0,
        spatial_res_scale=1.0,
        temporal_res_scale=1.0,
        graph_topology_mode="anatomical",
        graph_rewire_seed=3407,
        graph_topology_spec=None,
        recurrence_scope="independent",
        spatial_ssm_conv=1,
        temporal_ssm_conv=3,
        compile_compatible_scan=False,
        ssm_d_state=16,
        ssm_ratio=2.0,
        mlp_ratio=2.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        if graph_injection_mode is None:
            graph_injection_mode = (
                "control" if graph_conditioned_ssm else "none"
            )
        graph_injection_mode = str(graph_injection_mode).lower()
        if graph_injection_mode not in {"none", "feature", "control"}:
            raise ValueError(
                "graph_injection_mode must be one of none, feature, control; "
                f"received {graph_injection_mode!r}"
            )
        if graph_injection_mode in {"feature", "control"} and not use_graph_mixer:
            raise ValueError(
                f"graph_injection_mode={graph_injection_mode!r} requires "
                "use_graph_mixer=True"
            )
        expected_conditioned = graph_injection_mode == "control"
        if bool(graph_conditioned_ssm) != expected_conditioned:
            raise ValueError(
                "graph_conditioned_ssm must agree with graph_injection_mode: "
                f"mode={graph_injection_mode!r} requires "
                f"graph_conditioned_ssm={expected_conditioned}"
            )
        self.hidden_dim = int(hidden_dim)
        self.num_joints = int(num_joints)
        self.use_graph_mixer = bool(use_graph_mixer)
        self.graph_injection_mode = graph_injection_mode
        self.graph_conditioned_ssm = expected_conditioned
        self.reuse_graph_context = bool(reuse_graph_context)
        self.factorized_spatial_temporal = bool(factorized_spatial_temporal)
        self.coupled_ssm_forward_type = str(coupled_ssm_forward_type)
        self.compile_compatible_scan = bool(compile_compatible_scan)
        self.graph_scale = float(graph_scale)
        self.graph_topology_mode = str(graph_topology_mode).lower()
        self.graph_rewire_seed = int(graph_rewire_seed)
        self.recurrence_scope = str(recurrence_scope).lower()
        if self.recurrence_scope not in {"independent", "joined"}:
            raise ValueError(
                "recurrence_scope must be independent or joined, received "
                f"{self.recurrence_scope!r}"
            )
        if self.recurrence_scope == "joined" and not self.factorized_spatial_temporal:
            raise ValueError(
                "joined recurrence requires factorized_spatial_temporal=True"
            )

        self.norm_spatial = norm_layer(hidden_dim)
        self.norm_temporal = norm_layer(hidden_dim)
        self.norm_mlp = norm_layer(hidden_dim)
        self.graph_mixer = (
            SkeletonGraphMixer(
                dim=hidden_dim,
                hidden_ratio=graph_hidden_ratio,
                use_symmetry_edges=use_symmetry_edges,
                num_joints=num_joints,
                graph_topology_mode=self.graph_topology_mode,
                graph_rewire_seed=self.graph_rewire_seed,
                topology_spec=graph_topology_spec,
            )
            if self.use_graph_mixer
            else None
        )
        if self.factorized_spatial_temporal:
            self.spatial_ssm = FactorizedBiSSM(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                d_conv=spatial_ssm_conv,
                axis="spatial",
                compile_compatible_scan=compile_compatible_scan,
                recurrence_scope=self.recurrence_scope,
            )
            self.temporal_ssm = FactorizedBiSSM(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                d_conv=temporal_ssm_conv,
                axis="temporal",
                compile_compatible_scan=compile_compatible_scan,
                recurrence_scope=self.recurrence_scope,
            )
        else:
            self.spatial_ssm = BiSTSSM(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                d_conv=spatial_ssm_conv,
                conv_mode="2d",
                forward_type=self.coupled_ssm_forward_type,
                k_group=4,
            )
            self.temporal_ssm = BiSTSSM(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                d_conv=temporal_ssm_conv,
                conv_mode="2d",
                forward_type=self.coupled_ssm_forward_type,
                k_group=4,
            )
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
            channels_first=False,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.gamma_s = nn.Parameter(torch.ones(1) * float(spatial_res_scale))
        self.gamma_t = nn.Parameter(torch.ones(1) * float(temporal_res_scale))

    def _route_graph_injection(self, x, graph_feature=None):
        """Return graph feature, recurrent content, and selective context.

        ``none`` leaves both recurrent content and Delta/B/C input untouched.
        ``feature`` feeds ``x + graph`` as recurrent content with no independent
        context.  ``control`` preserves the released Full definition: recurrent
        content comes from ``x`` while ``x + graph`` controls Delta/B/C.
        """

        if self.graph_injection_mode == "none":
            return None, x, None
        if graph_feature is None:
            graph_feature = self.graph_mixer(x)
        graph_enhanced = x + self.graph_scale * graph_feature
        if self.graph_injection_mode == "feature":
            return graph_feature, graph_enhanced, None
        return graph_feature, x, graph_enhanced

    @staticmethod
    def factorize_spatial(x):
        """Map each frame to an independent joint sequence."""
        return rearrange(x, "b t j c -> (b t) 1 j c")

    @staticmethod
    def restore_spatial(x, batch, frames):
        return rearrange(x, "(b t) 1 j c -> b t j c", b=batch, t=frames)

    @staticmethod
    def factorize_temporal(x):
        """Map each joint to an independent temporal sequence."""
        return rearrange(x, "b t j c -> (b j) 1 t c")

    @staticmethod
    def restore_temporal(x, batch, joints):
        return rearrange(x, "(b j) 1 t c -> b t j c", b=batch, j=joints)

    def forward(self, x, joint_pos, temporal_pos, return_shape_trace=False):
        b, t, j, c = x.shape
        if j != self.num_joints or c != self.hidden_dim:
            raise ValueError(
                f"expected [B,T,{self.num_joints},{self.hidden_dim}], received {tuple(x.shape)}"
            )

        spatial_feature = self.norm_spatial(x) + joint_pos
        graph_feature, spatial_ssm_feature, spatial_context = (
            self._route_graph_injection(spatial_feature)
        )
        spatial_input = (
            self.factorize_spatial(spatial_ssm_feature)
            if self.factorized_spatial_temporal
            else spatial_ssm_feature
        )
        spatial_context_1d = (
            None
            if spatial_context is None
            else (
                self.factorize_spatial(spatial_context)
                if self.factorized_spatial_temporal
                else spatial_context
            )
        )
        spatial_kwargs = {"context": spatial_context_1d}
        if isinstance(self.spatial_ssm, FactorizedBiSSM):
            spatial_kwargs["segments_per_sample"] = t
        spatial_output = self.spatial_ssm(spatial_input, **spatial_kwargs)
        if self.factorized_spatial_temporal:
            spatial_output = self.restore_spatial(spatial_output, b, t)
        x = x + self.gamma_s * self.drop_path(spatial_output)

        temporal_feature = self.norm_temporal(x) + temporal_pos
        if self.reuse_graph_context and graph_feature is not None:
            temporal_graph_feature, temporal_ssm_feature, temporal_context = (
                self._route_graph_injection(
                    temporal_feature,
                    graph_feature=graph_feature,
                )
            )
        else:
            temporal_graph_feature, temporal_ssm_feature, temporal_context = (
                self._route_graph_injection(temporal_feature)
            )
        temporal_input = (
            self.factorize_temporal(temporal_ssm_feature)
            if self.factorized_spatial_temporal
            else temporal_ssm_feature
        )
        temporal_context_1d = (
            None
            if temporal_context is None
            else (
                self.factorize_temporal(temporal_context)
                if self.factorized_spatial_temporal
                else temporal_context
            )
        )
        temporal_kwargs = {"context": temporal_context_1d}
        if isinstance(self.temporal_ssm, FactorizedBiSSM):
            temporal_kwargs["segments_per_sample"] = j
        temporal_output = self.temporal_ssm(temporal_input, **temporal_kwargs)
        if self.factorized_spatial_temporal:
            temporal_output = self.restore_temporal(temporal_output, b, j)
        x = x + self.gamma_t * self.drop_path(temporal_output)
        x = x + self.drop_path(self.mlp(self.norm_mlp(x)))

        if not return_shape_trace:
            return x
        trace = {
            "graph_injection_mode": self.graph_injection_mode,
            "graph_topology_mode": self.graph_topology_mode,
            "recurrence_scope": self.recurrence_scope,
            "factorized_spatial_temporal": self.factorized_spatial_temporal,
            "ssm_forward_type": (
                "v2_1d_bidir_k2_compile"
                if self.factorized_spatial_temporal and self.compile_compatible_scan
                else (
                    "v2_1d_bidir_k2"
                    if self.factorized_spatial_temporal
                    else self.coupled_ssm_forward_type
                )
            ),
            "graph_feature": (
                None if graph_feature is None else (b, t, j, self.hidden_dim)
            ),
            "spatial_context": (
                None if spatial_context is None else tuple(spatial_context.shape)
            ),
            "spatial_ssm_input": tuple(spatial_input.shape),
            "spatial_ssm_output": tuple(spatial_output.shape),
            "temporal_graph_feature": (
                None
                if temporal_graph_feature is None
                else (b, t, j, self.hidden_dim)
            ),
            "temporal_context": (
                None if temporal_context is None else tuple(temporal_context.shape)
            ),
            "temporal_ssm_input": tuple(temporal_input.shape),
            "temporal_ssm_output": tuple(temporal_output.shape),
        }
        return x, trace


class GraphConditionedPoseMamba(nn.Module):
    """Graph-conditioned PoseMamba with selectable SSM boundary handling."""

    def __init__(
        self,
        num_frame=243,
        num_joints=17,
        in_chans=3,
        embed_dim_ratio=57,
        depth=10,
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=None,
        use_graph_mixer=True,
        use_symmetry_edges=True,
        graph_hidden_ratio=0.5,
        graph_conditioned_ssm=True,
        graph_injection_mode=None,
        reuse_graph_context=True,
        factorized_spatial_temporal=True,
        coupled_ssm_forward_type="v2_plus_poselimbs",
        spatial_ssm_conv=1,
        temporal_ssm_conv=3,
        compile_compatible_scan=False,
        graph_scale=1.0,
        spatial_res_scale=1.0,
        temporal_res_scale=1.0,
        ssm_d_state=16,
        ssm_ratio=2.0,
        activation_checkpoint_blocks=False,
        graph_topology_mode="anatomical",
        graph_rewire_seed=3407,
        recurrence_scope="independent",
    ):
        super().__init__()
        if int(num_joints) != 17:
            raise ValueError("GraphConditionedPoseMamba currently supports Human3.6M 17 joints")
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim = int(embed_dim_ratio)
        self.num_frame = int(num_frame)
        self.num_joints = int(num_joints)
        self.embed_dim = embed_dim
        self.block_depth = int(depth)
        self.factorized_spatial_temporal = bool(factorized_spatial_temporal)
        self.coupled_ssm_forward_type = str(coupled_ssm_forward_type)
        self.activation_checkpoint_blocks = bool(activation_checkpoint_blocks)
        self.graph_topology_mode = str(graph_topology_mode).lower()
        self.graph_rewire_seed = int(graph_rewire_seed)
        self.recurrence_scope = str(recurrence_scope).lower()
        self.graph_topology_spec = build_h36m_graph_spec(
            self.graph_topology_mode,
            seed=self.graph_rewire_seed,
        )
        self.graph_topology_hash = self.graph_topology_spec["sha256"]

        self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                GraphConditionedPoseBlock(
                    hidden_dim=embed_dim,
                    num_joints=num_joints,
                    graph_hidden_ratio=graph_hidden_ratio,
                    use_graph_mixer=use_graph_mixer,
                    use_symmetry_edges=use_symmetry_edges,
                    graph_conditioned_ssm=graph_conditioned_ssm,
                    graph_injection_mode=graph_injection_mode,
                    reuse_graph_context=reuse_graph_context,
                    factorized_spatial_temporal=factorized_spatial_temporal,
                    coupled_ssm_forward_type=coupled_ssm_forward_type,
                    graph_scale=graph_scale,
                    spatial_res_scale=spatial_res_scale,
                    temporal_res_scale=temporal_res_scale,
                    graph_topology_mode=self.graph_topology_mode,
                    graph_rewire_seed=self.graph_rewire_seed,
                    graph_topology_spec=self.graph_topology_spec,
                    recurrence_scope=self.recurrence_scope,
                    spatial_ssm_conv=spatial_ssm_conv,
                    temporal_ssm_conv=temporal_ssm_conv,
                    compile_compatible_scan=compile_compatible_scan,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[index],
                    norm_layer=norm_layer,
                )
                for index in range(depth)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3),
        )

    def forward(self, x, return_shape_trace=False):
        if x.ndim != 4:
            raise ValueError(f"expected [B,T,J,Cin], received {tuple(x.shape)}")
        b, t, j, _ = x.shape
        if j != self.num_joints:
            raise ValueError(f"expected {self.num_joints} joints, received {j}")
        if t > self.num_frame:
            raise ValueError(f"input has {t} frames but model maximum is {self.num_frame}")

        trace = {"input": tuple(x.shape)} if return_shape_trace else None
        x = self.pos_drop(self.Spatial_patch_to_embedding(x))
        if return_shape_trace:
            trace["embedding"] = tuple(x.shape)
        joint_pos = self.Spatial_pos_embed[:, None, :, :]
        temporal_pos = self.Temporal_pos_embed[:, :t, :][:, :, None, :]
        for index, block in enumerate(self.blocks):
            if return_shape_trace and index == 0:
                x, block_trace = block(
                    x, joint_pos, temporal_pos, return_shape_trace=True
                )
                trace.update(block_trace)
            elif (
                self.activation_checkpoint_blocks
                and self.training
                and torch.is_grad_enabled()
            ):
                x = activation_checkpoint.checkpoint(
                    block,
                    x,
                    joint_pos,
                    temporal_pos,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                x = block(x, joint_pos, temporal_pos)
        prediction = self.head(x).reshape(b, t, j, 3)
        if return_shape_trace:
            trace["final_prediction"] = tuple(prediction.shape)
            return prediction, trace
        return prediction

    def execution_spec(self):
        return {
            "model": "GraphConditionedPoseMamba",
            "graph_injection_mode": self.blocks[0].graph_injection_mode,
            "graph_topology_mode": self.graph_topology_mode,
            "graph_rewire_seed": self.graph_rewire_seed,
            "graph_topology_hash": self.graph_topology_hash,
            "recurrence_scope": self.recurrence_scope,
            "factorized_spatial_temporal": self.factorized_spatial_temporal,
            "scan_forward_type": (
                "v2_1d_bidir_k2_compile"
                if self.blocks[0].compile_compatible_scan
                else "v2_1d_bidir_k2"
            ),
            "k_group": 2,
        }


if __name__ == "__main__":
    torch.cuda.set_device(3)
    model = PoseMamba(num_frame=243, embed_dim_ratio=128,mlp_ratio = 2, depth = 10).cuda()
    from thop import profile, clever_format
    input_shape = (1, 243, 17, 2)
    x = torch.randn(input_shape).cuda()
    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
    print("FLOPs: %s" %(flops))
    print("params: %s" %(params))
