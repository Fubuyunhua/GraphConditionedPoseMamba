"""Minimal model factory for the GraphConditionedPoseMamba release."""

from lib.model.PoseMamba import GraphConditionedPoseMamba, PoseMamba


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def partial_train_layers(model, partial_list):
    for name, parameter in model.named_parameters():
        parameter.requires_grad = any(token in name for token in partial_list)
    return model


def load_backbone(args):
    in_chans = 2 if getattr(args, "no_conf", False) else 3
    common = dict(
        num_frame=args.maxlen,
        num_joints=args.num_joints,
        in_chans=in_chans,
        embed_dim_ratio=args.dim_feat,
        depth=args.depth,
        mlp_ratio=args.mlp_ratio,
        drop_rate=getattr(args, "dropout", 0.0),
        drop_path_rate=getattr(args, "drop_path_rate", 0.2),
    )
    if args.backbone == "PoseMamba":
        return PoseMamba(**common)
    if args.backbone != "GraphConditionedPoseMamba":
        raise ValueError(
            "This focused release supports PoseMamba and "
            f"GraphConditionedPoseMamba, received {args.backbone!r}"
        )
    return GraphConditionedPoseMamba(
        **common,
        use_graph_mixer=getattr(args, "use_graph_mixer", True),
        use_symmetry_edges=getattr(args, "use_symmetry_edges", True),
        graph_hidden_ratio=getattr(args, "graph_hidden_ratio", 0.5),
        graph_conditioned_ssm=getattr(args, "graph_conditioned_ssm", True),
        reuse_graph_context=getattr(args, "reuse_graph_context", True),
        factorized_spatial_temporal=getattr(
            args, "factorized_spatial_temporal", True
        ),
        spatial_ssm_conv=getattr(args, "spatial_ssm_conv", 1),
        temporal_ssm_conv=getattr(args, "temporal_ssm_conv", 3),
        compile_compatible_scan=getattr(args, "compile_compatible_scan", False),
        graph_scale=getattr(args, "graph_scale", 1.0),
        spatial_res_scale=getattr(args, "spatial_res_scale", 1.0),
        temporal_res_scale=getattr(args, "temporal_res_scale", 1.0),
        ssm_d_state=getattr(args, "ssm_d_state", 16),
        ssm_ratio=getattr(args, "ssm_ratio", 2.0),
        activation_checkpoint_blocks=getattr(
            args, "activation_checkpoint_blocks", False
        ),
    )


__all__ = [
    "AverageMeter",
    "GraphConditionedPoseMamba",
    "PoseMamba",
    "load_backbone",
    "partial_train_layers",
]
