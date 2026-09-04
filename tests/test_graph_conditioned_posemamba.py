import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lib.model.PoseMamba import (
    GraphConditionedPoseBlock,
    GraphConditionedPoseMamba,
    PoseMamba,
)
from lib.model.graph_mixer import SkeletonGraphMixer, h36m_neighbor_names
from lib.model.mambablocks import (
    BiSTSSM,
    CrossMerge1DBidirectional,
    CrossScan1DBidirectional,
    FactorizedBiSSM,
)
from lib.utils.tools import get_config


def parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def cuda_selective_scan_available():
    if not torch.cuda.is_available():
        return False
    try:
        import selective_scan_cuda_core  # noqa: F401
    except Exception:
        return False
    return True


class RecordingSSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.context_shape = None
        self.input_value = None
        self.context_value = None

    def forward(self, x, context=None):
        self.input_shape = tuple(x.shape)
        self.context_shape = None if context is None else tuple(context.shape)
        self.input_value = x.detach().clone()
        self.context_value = None if context is None else context.detach().clone()
        return x


class ContextAwareSSM(nn.Module):
    """Small differentiable stand-in for checkpoint equivalence tests."""

    def forward(self, x, context=None):
        return x if context is None else x + 0.125 * context


class FixedGraphMixer(nn.Module):
    def forward(self, x):
        return torch.full_like(x, 0.25)


class GraphMixerTests(unittest.TestCase):
    def test_required_h36m_neighbors(self):
        neighbors = h36m_neighbor_names(use_symmetry_edges=True)
        self.assertEqual(set(neighbors["root"]["bone"]), {"rhip", "lhip", "belly"})
        self.assertEqual(
            set(neighbors["neck"]["bone"]),
            {"belly", "nose", "lshoulder", "rshoulder"},
        )
        self.assertEqual(neighbors["lwrist"]["bone"], ["lelbow"])
        self.assertEqual(neighbors["lwrist"]["symmetry"], ["rwrist"])
        self.assertEqual(neighbors["rwrist"]["bone"], ["relbow"])
        self.assertEqual(neighbors["rwrist"]["symmetry"], ["lwrist"])

    def test_graph_mixer_preserves_shape_and_has_unit_edge_scales(self):
        mixer = SkeletonGraphMixer(dim=57, hidden_ratio=0.5)
        x = torch.randn(2, 5, 17, 57, requires_grad=True)
        output = mixer(x)
        self.assertEqual(output.shape, x.shape)
        self.assertEqual(mixer.hidden_dim, 28)
        self.assertEqual(mixer.alpha_bone.item(), 1.0)
        self.assertEqual(mixer.alpha_sym.item(), 1.0)
        output.square().mean().backward()
        self.assertIsNotNone(mixer.message_proj.weight.grad)

    def test_dense_graph_aggregation_matches_edge_reference(self):
        torch.manual_seed(7)
        mixer = SkeletonGraphMixer(12).double()
        x = torch.randn(2, 5, 17, 12, dtype=torch.double, requires_grad=True)

        mixer.use_dense_aggregation = False
        edge_output = mixer(x)
        edge_output.square().mean().backward()
        edge_input_grad = x.grad.detach().clone()
        edge_parameter_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in mixer.named_parameters()
        }

        mixer.zero_grad(set_to_none=True)
        x.grad = None
        mixer.use_dense_aggregation = True
        dense_output = mixer(x)
        dense_output.square().mean().backward()

        torch.testing.assert_close(dense_output, edge_output, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(x.grad, edge_input_grad, rtol=1e-12, atol=1e-12)
        for name, parameter in mixer.named_parameters():
            torch.testing.assert_close(
                parameter.grad,
                edge_parameter_grads[name],
                rtol=1e-11,
                atol=1e-12,
            )


class FactorizedShapeTests(unittest.TestCase):
    def test_factorization_is_exact_and_never_concatenates_samples(self):
        x = torch.arange(2 * 3 * 17 * 2).reshape(2, 3, 17, 2)
        spatial = GraphConditionedPoseBlock.factorize_spatial(x)
        temporal = GraphConditionedPoseBlock.factorize_temporal(x)
        self.assertEqual(spatial.shape, (6, 1, 17, 2))
        self.assertEqual(temporal.shape, (34, 1, 3, 2))
        for batch in range(2):
            for frame in range(3):
                self.assertTrue(torch.equal(spatial[batch * 3 + frame, 0], x[batch, frame]))
            for joint in range(17):
                self.assertTrue(
                    torch.equal(temporal[batch * 17 + joint, 0], x[batch, :, joint])
                )
        self.assertTrue(
            torch.equal(GraphConditionedPoseBlock.restore_spatial(spatial, 2, 3), x)
        )
        self.assertTrue(
            torch.equal(GraphConditionedPoseBlock.restore_temporal(temporal, 2, 17), x)
        )

    def test_bidirectional_scan_and_merge_alignment(self):
        x = torch.randn(2, 8, 1, 17, requires_grad=True)
        scans = CrossScan1DBidirectional.apply(x)
        self.assertEqual(scans.shape, (2, 2, 8, 17))
        merged = CrossMerge1DBidirectional.apply(scans.reshape(2, 2, 8, 1, 17))
        self.assertTrue(torch.allclose(merged, 2.0 * x.squeeze(2)))
        merged.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.full_like(x, 2.0)))

    def test_full_shape_contract_without_cuda_kernel(self):
        model = GraphConditionedPoseMamba(
            num_frame=243,
            in_chans=3,
            embed_dim_ratio=57,
            depth=1,
            mlp_ratio=2.0,
        ).eval()
        spatial_recorder = RecordingSSM()
        temporal_recorder = RecordingSSM()
        model.blocks[0].spatial_ssm = spatial_recorder
        model.blocks[0].temporal_ssm = temporal_recorder

        x = torch.randn(2, 243, 17, 3)
        with torch.no_grad():
            prediction, trace = model(x, return_shape_trace=True)
        self.assertEqual(prediction.shape, (2, 243, 17, 3))
        self.assertEqual(trace["embedding"], (2, 243, 17, 57))
        self.assertEqual(trace["graph_feature"], (2, 243, 17, 57))
        self.assertEqual(spatial_recorder.input_shape, (486, 1, 17, 57))
        self.assertEqual(spatial_recorder.context_shape, (486, 1, 17, 57))
        self.assertEqual(temporal_recorder.input_shape, (34, 1, 243, 57))
        self.assertEqual(temporal_recorder.context_shape, (34, 1, 243, 57))


class GraphInjectionModeTests(unittest.TestCase):
    def _block(self, mode, *, use_graph=True, conditioned=False):
        block = GraphConditionedPoseBlock(
            hidden_dim=8,
            num_joints=17,
            use_graph_mixer=use_graph,
            graph_conditioned_ssm=conditioned,
            graph_injection_mode=mode,
            reuse_graph_context=True,
            drop_path=0.0,
        )
        if use_graph:
            block.graph_mixer = FixedGraphMixer()
        return block

    def test_none_leaves_content_unchanged_and_has_no_context(self):
        block = self._block("none", use_graph=False, conditioned=False)
        x = torch.randn(2, 3, 17, 8)
        graph, content, context = block._route_graph_injection(x)
        self.assertIsNone(graph)
        self.assertIs(content, x)
        self.assertIsNone(context)

    def test_feature_fuses_graph_into_content_only(self):
        block = self._block("feature", conditioned=False)
        x = torch.randn(2, 3, 17, 8)
        graph, content, context = block._route_graph_injection(x)
        torch.testing.assert_close(graph, torch.full_like(x, 0.25))
        torch.testing.assert_close(content, x + 0.25)
        self.assertIsNone(context)

    def test_control_preserves_content_and_routes_graph_to_context(self):
        block = self._block("control", conditioned=True)
        x = torch.randn(2, 3, 17, 8)
        graph, content, context = block._route_graph_injection(x)
        torch.testing.assert_close(graph, torch.full_like(x, 0.25))
        self.assertIs(content, x)
        torch.testing.assert_close(context, x + 0.25)

    def test_invalid_or_contradictory_modes_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "one of none, feature, control"):
            self._block("unknown", conditioned=False)
        with self.assertRaisesRegex(ValueError, "requires use_graph_mixer=True"):
            self._block("feature", use_graph=False, conditioned=False)
        with self.assertRaisesRegex(ValueError, "must agree"):
            self._block("control", use_graph=True, conditioned=False)

    def test_legacy_full_inference_matches_explicit_control(self):
        torch.manual_seed(20260903)
        legacy = GraphConditionedPoseMamba(
            num_frame=9,
            in_chans=3,
            embed_dim_ratio=12,
            depth=1,
            mlp_ratio=1.5,
            drop_path_rate=0.0,
            graph_conditioned_ssm=True,
            graph_injection_mode=None,
        ).eval()
        explicit = GraphConditionedPoseMamba(
            num_frame=9,
            in_chans=3,
            embed_dim_ratio=12,
            depth=1,
            mlp_ratio=1.5,
            drop_path_rate=0.0,
            graph_conditioned_ssm=True,
            graph_injection_mode="control",
        ).eval()
        explicit.load_state_dict(legacy.state_dict(), strict=True)
        for model in (legacy, explicit):
            model.blocks[0].spatial_ssm = ContextAwareSSM()
            model.blocks[0].temporal_ssm = ContextAwareSSM()
        x = torch.randn(2, 9, 17, 3)
        with torch.no_grad():
            old_style = legacy(x)
            control = explicit(x)
        torch.testing.assert_close(control, old_style, rtol=0, atol=0)

    def test_nonfactorized_control_keeps_full_tensor_and_decouples_content(self):
        model = GraphConditionedPoseMamba(
            num_frame=9,
            in_chans=3,
            embed_dim_ratio=8,
            depth=1,
            mlp_ratio=1.5,
            drop_path_rate=0.0,
            graph_conditioned_ssm=True,
            graph_injection_mode="control",
            factorized_spatial_temporal=False,
        ).eval()
        block = model.blocks[0]
        block.graph_mixer = FixedGraphMixer()
        spatial = RecordingSSM()
        temporal = RecordingSSM()
        block.spatial_ssm = spatial
        block.temporal_ssm = temporal
        x = torch.randn(2, 9, 17, 3)
        with torch.no_grad():
            prediction, trace = model(x, return_shape_trace=True)
        self.assertEqual(prediction.shape, (2, 9, 17, 3))
        self.assertEqual(trace["spatial_ssm_input"], (2, 9, 17, 8))
        self.assertEqual(trace["temporal_ssm_input"], (2, 9, 17, 8))
        self.assertEqual(trace["graph_feature"], (2, 9, 17, 8))
        self.assertEqual(trace["spatial_context"], (2, 9, 17, 8))
        self.assertEqual(trace["temporal_context"], (2, 9, 17, 8))
        self.assertEqual(spatial.context_shape, spatial.input_shape)
        self.assertEqual(temporal.context_shape, temporal.input_shape)
        torch.testing.assert_close(
            spatial.context_value - spatial.input_value,
            torch.full_like(spatial.input_value, 0.25),
        )
        torch.testing.assert_close(
            temporal.context_value - temporal.input_value,
            torch.full_like(temporal.input_value, 0.25),
        )
        self.assertEqual(trace["graph_injection_mode"], "control")
        self.assertFalse(trace["factorized_spatial_temporal"])
        self.assertEqual(trace["ssm_forward_type"], "v2_plus_poselimbs")


class ConfigurationAndCapacityTests(unittest.TestCase):
    def test_w64_0p8m_config_capacity_and_batch(self):
        config_path = (
            REPO_ROOT
            / "configs/pose3d/"
            / "graph_posemamba_h36m_w64_d8_0p8m.yaml"
        )
        config = get_config(str(config_path))
        from lib.utils.learning import load_backbone

        model = load_backbone(config)
        self.assertEqual(config.dim_feat, 64)
        self.assertEqual(config.depth, 8)
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.test_batch_size, 4)
        self.assertEqual(parameter_count(model), 800_083)

    def test_release_config_freezes_training_protocol(self):
        config_path = (
            REPO_ROOT
            / "configs/pose3d/"
            / "graph_posemamba_h36m_w64_d8_0p8m.yaml"
        )
        config = get_config(str(config_path))
        expected = {
            "epochs": 120,
            "warmup_epochs": 8,
            "batch_size": 4,
            "test_batch_size": 4,
            "learning_rate": 5e-4,
            "weight_decay": 0.012,
            "lr_decay": 0.99,
            "use_ema": True,
            "ema_decay": 0.9998,
            "clip_len": 243,
            "data_stride": 81,
            "sample_stride": 1,
            "no_conf": False,
            "lambda_3d_velocity": 20.0,
            "lambda_scale": 0.5,
            "lambda_3d": 1.0,
            "flip": True,
            "mask_ratio": 0.0,
            "mask_T_ratio": 0.0,
            "noise": False,
        }
        for field, value in expected.items():
            self.assertEqual(getattr(config, field), value, field)

    def test_capacity_and_unit_residual_initialization(self):
        original = PoseMamba(
            num_frame=243,
            in_chans=3,
            embed_dim_ratio=57,
            depth=10,
            mlp_ratio=2.0,
        )
        candidate = GraphConditionedPoseMamba(
            num_frame=243,
            in_chans=3,
            embed_dim_ratio=57,
            depth=10,
            mlp_ratio=2.0,
        )
        ratio = parameter_count(candidate) / parameter_count(original)
        # The factorized backbone uses exactly two directional SSM parameter
        # groups.  Original PoseMamba's nominal "2direction" wrapper allocates
        # four groups by duplicating both forward and backward sequences.
        self.assertGreater(ratio, 0.65)
        self.assertLess(ratio, 0.8)
        for block in candidate.blocks:
            self.assertEqual(block.spatial_ssm.x_proj_weight.shape[0], 2)
            self.assertEqual(block.temporal_ssm.x_proj_weight.shape[0], 2)
        for block in candidate.blocks:
            self.assertEqual(block.gamma_s.item(), 1.0)
            self.assertEqual(block.gamma_t.item(), 1.0)

    def test_memory_optimized_configs_preserve_protocol_and_capacity(self):
        from lib.utils.learning import load_backbone

        baseline = get_config(
            str(
                REPO_ROOT
                / "configs/pose3d/graph_posemamba_h36m_w64_d8_0p8m.yaml"
            )
        )
        for name, checkpointed, compile_mode in (
            ("graph_posemamba_h36m_w64_d8_0p8m_memopt_default.yaml", False, "default"),
            ("graph_posemamba_h36m_w64_d8_0p8m_memopt_checkpoint.yaml", True, "default"),
            ("graph_posemamba_h36m_w64_d8_0p8m_memopt_speed.yaml", False, "reduce-overhead"),
        ):
            candidate = get_config(str(REPO_ROOT / "configs/pose3d" / name))
            for field in (
                "epochs",
                "warmup_epochs",
                "batch_size",
                "test_batch_size",
                "learning_rate",
                "weight_decay",
                "lr_decay",
                "use_ema",
                "ema_decay",
                "clip_len",
                "data_stride",
                "sample_stride",
                "lambda_3d",
                "lambda_scale",
                "lambda_3d_velocity",
                "lambda_diff",
            ):
                self.assertEqual(getattr(candidate, field), getattr(baseline, field))
            self.assertEqual(candidate.compile_mode, compile_mode)
            self.assertTrue(candidate.eager_eval_when_compiled)
            self.assertEqual(candidate.activation_checkpoint_blocks, checkpointed)
            self.assertEqual(parameter_count(load_backbone(candidate)), 800_083)

    def test_activation_checkpoint_preserves_outputs_and_gradients(self):
        torch.manual_seed(20260902)
        reference = GraphConditionedPoseMamba(
            num_frame=9,
            in_chans=3,
            embed_dim_ratio=12,
            depth=2,
            mlp_ratio=1.5,
            drop_path_rate=0.0,
            activation_checkpoint_blocks=False,
        ).train()
        checkpointed = GraphConditionedPoseMamba(
            num_frame=9,
            in_chans=3,
            embed_dim_ratio=12,
            depth=2,
            mlp_ratio=1.5,
            drop_path_rate=0.0,
            activation_checkpoint_blocks=True,
        ).train()
        for model in (reference, checkpointed):
            for block in model.blocks:
                block.spatial_ssm = ContextAwareSSM()
                block.temporal_ssm = ContextAwareSSM()
        checkpointed.load_state_dict(reference.state_dict(), strict=True)

        x_reference = torch.randn(2, 9, 17, 3, requires_grad=True)
        x_checkpointed = x_reference.detach().clone().requires_grad_(True)
        output_reference = reference(x_reference)
        output_reference.square().mean().backward()
        output_checkpointed = checkpointed(x_checkpointed)
        output_checkpointed.square().mean().backward()

        torch.testing.assert_close(output_checkpointed, output_reference, rtol=0, atol=0)
        torch.testing.assert_close(x_checkpointed.grad, x_reference.grad, rtol=0, atol=0)
        for (_, parameter_reference), (_, parameter_checkpointed) in zip(
            reference.named_parameters(), checkpointed.named_parameters()
        ):
            torch.testing.assert_close(
                parameter_checkpointed.grad,
                parameter_reference.grad,
                rtol=1e-6,
                atol=1e-8,
            )

    def test_minimal_ablation_configs_use_80_epoch_frozen_protocol(self):
        from lib.utils.learning import load_backbone

        full = get_config(
            str(
                REPO_ROOT
                / "configs/pose3d/graph_posemamba_h36m_w64_d8_0p8m_memopt_speed.yaml"
            )
        )
        registered = (
            ("ablation_factorized_only.yaml", "none", False, False),
            ("ablation_graph_feature.yaml", "feature", True, False),
        )
        frozen_fields = (
            "warmup_epochs",
            "batch_size",
            "test_batch_size",
            "learning_rate",
            "weight_decay",
            "lr_decay",
            "use_ema",
            "ema_decay",
            "maxlen",
            "dim_feat",
            "depth",
            "mlp_ratio",
            "ssm_d_state",
            "ssm_ratio",
            "dropout",
            "drop_path_rate",
            "factorized_spatial_temporal",
            "spatial_ssm_conv",
            "temporal_ssm_conv",
            "graph_scale",
            "spatial_res_scale",
            "temporal_res_scale",
            "compile_model",
            "compile_mode",
            "compile_compatible_scan",
            "eager_eval_when_compiled",
            "data_root",
            "subset_list",
            "dt_file",
            "clip_len",
            "data_stride",
            "sample_stride",
            "num_joints",
            "rootrel",
            "no_conf",
            "lambda_3d",
            "lambda_scale",
            "lambda_3d_velocity",
            "lambda_diff",
            "flip",
            "mask_ratio",
            "mask_T_ratio",
            "noise",
        )
        for filename, mode, use_graph, conditioned in registered:
            config = get_config(str(REPO_ROOT / "configs/pose3d" / filename))
            self.assertEqual(config.epochs, 80)
            self.assertEqual(config.graph_injection_mode, mode)
            self.assertEqual(config.use_graph_mixer, use_graph)
            self.assertEqual(config.graph_conditioned_ssm, conditioned)
            for field in frozen_fields:
                self.assertEqual(getattr(config, field), getattr(full, field), field)
            model = load_backbone(config)
            self.assertTrue(all(block.graph_injection_mode == mode for block in model.blocks))

    def test_no_factorization_config_uses_80_epoch_frozen_protocol(self):
        from lib.utils.learning import load_backbone

        full = get_config(
            str(
                REPO_ROOT
                / "configs/pose3d/graph_posemamba_h36m_w64_d8_0p8m_memopt_speed.yaml"
            )
        )
        candidate = get_config(
            str(
                REPO_ROOT
                / "configs/pose3d/ablation_graph_conditioned_no_factorization.yaml"
            )
        )
        frozen_fields = (
            "warmup_epochs",
            "batch_size",
            "test_batch_size",
            "learning_rate",
            "weight_decay",
            "lr_decay",
            "use_ema",
            "ema_decay",
            "maxlen",
            "dim_feat",
            "depth",
            "mlp_ratio",
            "ssm_d_state",
            "ssm_ratio",
            "dropout",
            "drop_path_rate",
            "use_graph_mixer",
            "use_symmetry_edges",
            "graph_hidden_ratio",
            "graph_conditioned_ssm",
            "reuse_graph_context",
            "spatial_ssm_conv",
            "temporal_ssm_conv",
            "graph_scale",
            "spatial_res_scale",
            "temporal_res_scale",
            "compile_model",
            "compile_mode",
            "compile_compatible_scan",
            "cuda_graph_model",
            "eager_eval_when_compiled",
            "activation_checkpoint_blocks",
            "data_root",
            "subset_list",
            "dt_file",
            "clip_len",
            "data_stride",
            "sample_stride",
            "num_joints",
            "rootrel",
            "no_conf",
            "gt_2d",
            "train_2d",
            "pretrain_3d_curriculum",
            "no_eval",
            "finetune",
            "partial_train",
            "lambda_3d",
            "lambda_scale",
            "lambda_3d_velocity",
            "lambda_diff",
            "lambda_lv",
            "lambda_lg",
            "lambda_a",
            "lambda_av",
            "lambda_3dw",
            "lambda_attn_diag",
            "lambda_attn_entropy",
            "lambda_tail_aux",
            "lambda_gate_sparsity",
            "synthetic",
            "flip",
            "mask_ratio",
            "mask_T_ratio",
            "noise",
        )
        self.assertEqual(candidate.epochs, 80)
        self.assertEqual(candidate.graph_injection_mode, "control")
        self.assertFalse(candidate.factorized_spatial_temporal)
        self.assertEqual(candidate.coupled_ssm_forward_type, "v2_plus_poselimbs")
        for field in frozen_fields:
            self.assertEqual(getattr(candidate, field), getattr(full, field), field)
        model = load_backbone(candidate)
        self.assertTrue(
            all(not block.factorized_spatial_temporal for block in model.blocks)
        )
        self.assertTrue(all(isinstance(block.spatial_ssm, BiSTSSM) for block in model.blocks))
        self.assertTrue(all(isinstance(block.temporal_ssm, BiSTSSM) for block in model.blocks))
        self.assertTrue(all(block.graph_injection_mode == "control" for block in model.blocks))

    def test_scale_configs_preserve_full_protocol(self):
        from lib.utils.learning import load_backbone

        full = get_config(
            str(
                REPO_ROOT
                / "configs/pose3d/graph_posemamba_h36m_w64_d8_0p8m_memopt_speed.yaml"
            )
        )
        registered = (
            ("graph_posemamba_h36m_w128_d20_scale_80e.yaml", 128, 20, 6_836_355),
            ("graph_posemamba_h36m_w256_d10_scale_80e.yaml", 256, 10, 12_646_107),
        )
        frozen_fields = (
            "warmup_epochs",
            "batch_size",
            "test_batch_size",
            "learning_rate",
            "weight_decay",
            "lr_decay",
            "checkpoint_frequency",
            "use_ema",
            "ema_decay",
            "backbone",
            "model_type",
            "maxlen",
            "mlp_ratio",
            "ssm_d_state",
            "ssm_ratio",
            "dropout",
            "drop_path_rate",
            "use_graph_mixer",
            "use_symmetry_edges",
            "graph_hidden_ratio",
            "graph_conditioned_ssm",
            "reuse_graph_context",
            "factorized_spatial_temporal",
            "spatial_ssm_conv",
            "temporal_ssm_conv",
            "graph_scale",
            "spatial_res_scale",
            "temporal_res_scale",
            "compile_model",
            "compile_mode",
            "compile_compatible_scan",
            "cuda_graph_model",
            "eager_eval_when_compiled",
            "activation_checkpoint_blocks",
            "data_root",
            "subset_list",
            "dt_file",
            "clip_len",
            "data_stride",
            "sample_stride",
            "num_joints",
            "rootrel",
            "no_conf",
            "gt_2d",
            "train_2d",
            "pretrain_3d_curriculum",
            "no_eval",
            "finetune",
            "partial_train",
            "lambda_3d",
            "lambda_scale",
            "lambda_3d_velocity",
            "lambda_diff",
            "lambda_lv",
            "lambda_lg",
            "lambda_a",
            "lambda_av",
            "lambda_3dw",
            "lambda_attn_diag",
            "lambda_attn_entropy",
            "lambda_tail_aux",
            "lambda_gate_sparsity",
            "synthetic",
            "flip",
            "mask_ratio",
            "mask_T_ratio",
            "noise",
        )
        for filename, width, depth, expected_parameters in registered:
            config = get_config(str(REPO_ROOT / "configs/pose3d" / filename))
            self.assertEqual(config.epochs, 80)
            self.assertEqual(config.dim_feat, width)
            self.assertEqual(config.depth, depth)
            self.assertEqual(config.graph_injection_mode, "control")
            for field in frozen_fields:
                self.assertEqual(getattr(config, field), getattr(full, field), field)
            model = load_backbone(config)
            self.assertEqual(parameter_count(model), expected_parameters)
            self.assertEqual(len(model.blocks), depth)
            self.assertEqual(model.embed_dim, width)
            self.assertTrue(
                all(block.graph_injection_mode == "control" for block in model.blocks)
            )
            self.assertTrue(
                all(block.factorized_spatial_temporal for block in model.blocks)
            )

    def test_w256_d16_60e_config_is_a_controlled_depth_scale(self):
        from lib.utils.learning import load_backbone

        reference = get_config(
            str(
                REPO_ROOT
                / "configs/pose3d/graph_posemamba_h36m_w256_d10_scale_80e.yaml"
            )
        )
        candidate = get_config(
            str(
                REPO_ROOT
                / "configs/pose3d/graph_posemamba_h36m_w256_d16_scale_60e.yaml"
            )
        )
        self.assertEqual(candidate.epochs, 60)
        self.assertTrue(candidate.enable_linear_warmup)
        self.assertEqual(candidate.warmup_start_factor, 0.1)
        self.assertEqual(candidate.max_grad_norm, 1.0)
        self.assertTrue(candidate.grad_clip_error_if_nonfinite)
        self.assertEqual(candidate.dim_feat, 256)
        self.assertEqual(candidate.depth, 16)
        for field in (
            "warmup_epochs",
            "batch_size",
            "test_batch_size",
            "learning_rate",
            "weight_decay",
            "lr_decay",
            "use_ema",
            "ema_decay",
            "mlp_ratio",
            "ssm_d_state",
            "ssm_ratio",
            "dropout",
            "drop_path_rate",
            "use_graph_mixer",
            "use_symmetry_edges",
            "graph_hidden_ratio",
            "graph_conditioned_ssm",
            "graph_injection_mode",
            "reuse_graph_context",
            "factorized_spatial_temporal",
            "spatial_ssm_conv",
            "temporal_ssm_conv",
            "graph_scale",
            "compile_model",
            "compile_mode",
            "compile_compatible_scan",
            "eager_eval_when_compiled",
            "activation_checkpoint_blocks",
            "dt_file",
            "clip_len",
            "data_stride",
            "rootrel",
            "no_conf",
            "lambda_3d",
            "lambda_scale",
            "lambda_3d_velocity",
            "lambda_diff",
            "flip",
        ):
            self.assertEqual(getattr(candidate, field), getattr(reference, field), field)
        model = load_backbone(candidate)
        self.assertEqual(parameter_count(model), 20_192_451)
        self.assertEqual(len(model.blocks), 16)

    def test_linear_warmup_then_epoch_decay_schedule(self):
        from train import LinearWarmupEpochDecay

        parameter = nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        schedule = LinearWarmupEpochDecay(
            optimizer,
            steps_per_epoch=2,
            warmup_epochs=2,
            start_factor=0.1,
            lr_decay=0.5,
        )
        observed = []
        for _ in range(7):
            observed.append(schedule.prepare_step()[0])
            schedule.complete_step()
        expected = [1e-4, 4e-4, 7e-4, 1e-3, 1e-3, 1e-3, 5e-4]
        for actual, target in zip(observed, expected):
            self.assertAlmostEqual(actual, target, places=12)
        self.assertEqual(schedule.state_dict()["global_step"], 7)

    def test_linear_warmup_then_cosine_schedule(self):
        from train import LinearWarmupEpochDecay

        parameter = nn.Parameter(torch.ones(()))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        schedule = LinearWarmupEpochDecay(
            optimizer,
            steps_per_epoch=2,
            warmup_epochs=2,
            start_factor=0.1,
            lr_decay=0.5,
            decay_mode="cosine",
            total_epochs=4,
            min_lr_ratio=0.1,
        )
        observed = []
        for _ in range(8):
            observed.append(schedule.prepare_step()[0])
            schedule.complete_step()
        expected = [
            1e-4,
            4e-4,
            7e-4,
            1e-3,
            1e-3,
            7.75e-4,
            3.25e-4,
            1e-4,
        ]
        for actual, target in zip(observed, expected):
            self.assertAlmostEqual(actual, target, places=12)
        state = schedule.state_dict()
        self.assertEqual(state["decay_mode"], "cosine")
        self.assertEqual(state["total_epochs"], 4)
        self.assertEqual(state["min_lr_ratio"], 0.1)

    def test_adamw_groups_honor_only_explicit_no_decay_markers(self):
        from train import build_adamw_parameter_groups

        module = FactorizedBiSSM(
            d_model=16,
            d_state=4,
            ssm_ratio=2.0,
            d_conv=1,
            axis="spatial",
        )
        legacy = build_adamw_parameter_groups(
            module,
            weight_decay=0.012,
            honor_no_weight_decay=False,
        )
        self.assertEqual(len(legacy), 1)
        self.assertEqual(legacy[0]["weight_decay"], 0.012)

        groups = build_adamw_parameter_groups(
            module,
            weight_decay=0.012,
            honor_no_weight_decay=True,
        )
        self.assertEqual([group["group_name"] for group in groups], ["decay", "no_decay"])
        self.assertEqual([group["weight_decay"] for group in groups], [0.012, 0.0])
        expected_no_decay = {
            id(parameter)
            for parameter in module.parameters()
            if getattr(parameter, "_no_weight_decay", False)
        }
        actual_no_decay = {id(parameter) for parameter in groups[1]["params"]}
        self.assertEqual(actual_no_decay, expected_no_decay)
        self.assertGreater(len(actual_no_decay), 0)

    def test_w256_d16_r3_is_optimizer_only_delta(self):
        from lib.utils.learning import load_backbone

        r2 = get_config(
            str(
                REPO_ROOT
                / "configs/pose3d/graph_posemamba_h36m_w256_d16_scale_60e.yaml"
            )
        )
        r3 = get_config(
            str(
                REPO_ROOT
                / "configs/pose3d/graph_posemamba_h36m_w256_d16_stable_r3_60e.yaml"
            )
        )
        self.assertEqual(r3.learning_rate, 3e-4)
        self.assertEqual(r3.lr_schedule_mode, "cosine")
        self.assertEqual(r3.min_lr_ratio, 0.1)
        self.assertTrue(r3.honor_no_weight_decay)
        self.assertTrue(r3.track_parameter_update_norm)
        for field in (
            "epochs",
            "warmup_epochs",
            "warmup_start_factor",
            "batch_size",
            "test_batch_size",
            "weight_decay",
            "use_ema",
            "ema_decay",
            "max_grad_norm",
            "dim_feat",
            "depth",
            "mlp_ratio",
            "ssm_d_state",
            "ssm_ratio",
            "drop_path_rate",
            "graph_injection_mode",
            "reuse_graph_context",
            "factorized_spatial_temporal",
            "lambda_3d",
            "lambda_scale",
            "lambda_3d_velocity",
            "lambda_diff",
            "dt_file",
        ):
            self.assertEqual(getattr(r3, field), getattr(r2, field), field)
        model = load_backbone(r3)
        self.assertEqual(parameter_count(model), 20_192_451)


@unittest.skipUnless(cuda_selective_scan_available(), "CUDA selective scan is unavailable")
class CudaIntegrationTests(unittest.TestCase):
    def test_actual_b2_t243_forward(self):
        model = GraphConditionedPoseMamba(
            num_frame=243,
            in_chans=3,
            embed_dim_ratio=57,
            depth=1,
            mlp_ratio=2.0,
        ).cuda().eval()
        x = torch.randn(2, 243, 17, 3, device="cuda")
        with torch.no_grad():
            prediction, trace = model(x, return_shape_trace=True)
        self.assertEqual(prediction.shape, (2, 243, 17, 3))
        self.assertEqual(trace["spatial_ssm_input"], (486, 1, 17, 57))
        self.assertEqual(trace["temporal_ssm_input"], (34, 1, 243, 57))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_graph_context_changes_parameters_but_not_u(self):
        module = FactorizedBiSSM(
            d_model=16,
            d_state=4,
            ssm_ratio=2.0,
            d_conv=1,
            axis="spatial",
        ).cuda().eval()
        setattr(module, "__DEBUG__", True)
        x = torch.randn(2, 1, 17, 16, device="cuda")
        with torch.no_grad():
            module(x, context=x)
            first = {key: value.clone() for key, value in module.__data__.items()}
            module(x, context=x + torch.randn_like(x))
            second = module.__data__
        self.assertTrue(torch.equal(first["us"], second["us"]))
        self.assertFalse(torch.equal(first["dts"], second["dts"]))
        self.assertFalse(torch.equal(first["Bs"], second["Bs"]))
        self.assertFalse(torch.equal(first["Cs"], second["Cs"]))

    def test_coupled_graph_context_changes_parameters_but_not_u(self):
        module = BiSTSSM(
            d_model=16,
            d_state=4,
            ssm_ratio=2.0,
            d_conv=3,
            conv_mode="2d",
            forward_type="v2_plus_poselimbs",
            k_group=4,
        ).cuda().eval()
        setattr(module, "__DEBUG__", True)
        x = torch.randn(2, 9, 17, 16, device="cuda")
        with torch.no_grad():
            module(x, context=x)
            first = {key: value.clone() for key, value in module.__data__.items()}
            module(x, context=x + torch.randn_like(x))
            second = module.__data__
        self.assertTrue(torch.equal(first["us"], second["us"]))
        self.assertFalse(torch.equal(first["dts"], second["dts"]))
        self.assertFalse(torch.equal(first["Bs"], second["Bs"]))
        self.assertFalse(torch.equal(first["Cs"], second["Cs"]))

    def test_context_content_projection_matches_full_projection(self):
        torch.manual_seed(17)
        module = FactorizedBiSSM(
            d_model=16,
            axis="temporal",
            d_conv=3,
        ).cuda()
        x = torch.randn(3, 1, 11, 16, device="cuda", requires_grad=True)
        context = torch.randn_like(x, requires_grad=True)

        module.context_content_only = False
        full_output = module(x, context=context)
        full_output.square().mean().backward()
        full_x_grad = x.grad.detach().clone()
        full_context_grad = context.grad.detach().clone()
        full_parameter_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in module.named_parameters()
            if parameter.grad is not None
        }

        module.zero_grad(set_to_none=True)
        x.grad = None
        context.grad = None
        module.context_content_only = True
        optimized_output = module(x, context=context)
        optimized_output.square().mean().backward()

        torch.testing.assert_close(optimized_output, full_output, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(x.grad, full_x_grad, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(
            context.grad,
            full_context_grad,
            rtol=1e-5,
            atol=1e-7,
        )
        for name, parameter in module.named_parameters():
            if parameter.grad is not None:
                torch.testing.assert_close(
                    parameter.grad,
                    full_parameter_grads[name],
                    rtol=1e-5,
                    atol=1e-7,
                )

    def test_compile_compatible_scan_matches_original_kernel(self):
        torch.manual_seed(29)
        original = FactorizedBiSSM(
            d_model=16,
            d_state=4,
            ssm_ratio=2.0,
            d_conv=3,
            axis="temporal",
        ).cuda()
        compatible = FactorizedBiSSM(
            d_model=16,
            d_state=4,
            ssm_ratio=2.0,
            d_conv=3,
            axis="temporal",
            compile_compatible_scan=True,
        ).cuda()
        compatible.load_state_dict(original.state_dict(), strict=True)
        x_original = torch.randn(
            3, 1, 19, 16, device="cuda", requires_grad=True
        )
        context_original = torch.randn_like(
            x_original, requires_grad=True
        )
        x_compatible = x_original.detach().clone().requires_grad_(True)
        context_compatible = (
            context_original.detach().clone().requires_grad_(True)
        )

        output_original = original(x_original, context=context_original)
        output_original.square().mean().backward()
        output_compatible = compatible(
            x_compatible,
            context=context_compatible,
        )
        output_compatible.square().mean().backward()

        torch.testing.assert_close(
            output_compatible,
            output_original,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            x_compatible.grad,
            x_original.grad,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            context_compatible.grad,
            context_original.grad,
            rtol=1e-5,
            atol=2e-11,
        )
        for (_, parameter_original), (_, parameter_compatible) in zip(
            original.named_parameters(),
            compatible.named_parameters(),
        ):
            torch.testing.assert_close(
                parameter_compatible.grad,
                parameter_original.grad,
                rtol=1e-5,
                atol=2e-10,
            )

    def test_original_posemamba_still_runs(self):
        model = PoseMamba(
            num_frame=9,
            in_chans=2,
            embed_dim_ratio=16,
            depth=1,
            mlp_ratio=2.0,
        ).cuda().eval()
        with torch.no_grad():
            output = model(torch.randn(1, 9, 17, 2, device="cuda"))
        self.assertEqual(output.shape, (1, 9, 17, 3))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
