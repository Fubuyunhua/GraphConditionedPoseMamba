# GraphConditionedPoseMamba 研究交接

## 1. 快照范围

- 快照日期：2026-09-02（UTC）
- 目标：只研究 PoseMamba backbone 如何通过更合理的 skeleton × time inductive bias
  降低 Human3.6M MPJPE。
- 最新模型：`GraphConditionedPoseMamba` W64/D8。
- 精确参数量：800,083。
- batch：4；clip：243；joints：17；输入通道：xy+confidence。
- 本包不包含 ReliPose 的 observation-risk、dual-frequency、coarse branch、dynamic
  reliability gate、MoE、新 prediction head 或新 loss。

## 2. 研究假设

原 PoseMamba 把 `T×J` 当二维规则网格并通过 flatten scan 传播状态，可能产生不属于人体
时空图的连续转移。新模型把职责拆开：

1. SkeletonGraphMixer：固定 bone + bilateral symmetry 的局部结构。
2. Spatial BiSSM：每个 frame 是独立 sample，只沿17 joints 双向传播。
3. Temporal BiSSM：每个 joint 是独立 sample，只沿243 frames 双向传播。
4. Graph-conditioned dynamics：recurrent content `u` 来自原 feature；`Δ/B/C` 来自
   `x + graph_context`。

详细公式和 shape 见 `docs/POSEMAMBA_CHANGES.md`。

## 3. 相对原 PoseMamba 的代码变化

### 新增

- `lib/model/graph_mixer.py`
  - Human3.6M 17-joint bone edges。
  - 6 对 bilateral symmetry edges。
  - 固定图 relation/neighbor message；无 dynamic fully-connected graph。
- `GraphConditionedPoseBlock` / `GraphConditionedPoseMamba`
  - PreNorm spatial residual → temporal residual → MLP residual。
  - `gamma_s/gamma_t` 从1初始化，无zero-init/sigmoid gate。
- `FactorizedBiSSM`
  - 真正 K=2：forward + backward 各一次。
  - spatial `[B,T,J,C] -> [B*T,1,J,C]`。
  - temporal `[B,T,J,C] -> [B*J,1,T,C]`。
- `selective_scan_compile.py`
  - 用 `torch.library.custom_op` 暴露原CUDA core forward/backward。
  - 不改变 selective-scan 数值实现，只让 Dynamo/AOTAutograd 看见 shape/autograd contract。

### 修改

- `mambablocks.py`
  - `forward_corev2(x, context=None)`：`u` scan自`x`，`Δ/B/C` projection自context。
  - spatial `d_conv=1`；temporal 用真正 Conv1D `d_conv=3`。
  - factorized方向数由旧K4重复路径改为严格K2。
- `PoseMamba.py`
  - 原 `PoseMamba` 类保留，用于基线回归。
  - 新主干使用独立 joint/time positional embeddings。
- `train.py`
  - 保留AdamW、EMA、loss、augmentation和evaluation。
  - 兼容 `torch.compile` checkpoint unwrap。
  - 日志标量在GPU累计后一次同步；这不参与反向。
  - 首次compile增加明确等待提示。

## 4. 当前配置

`configs/pose3d/graph_posemamba_h36m_w64_d8_0p8m.yaml`

| 项 | 值 |
|---|---:|
| dim | 64 |
| depth | 8 |
| graph hidden | 32 |
| SSM inner | 120 (`ratio=1.875`) |
| MLP inner | 126 (`ratio=1.96875`) |
| d_state | 16 |
| spatial conv | 1 |
| temporal conv | 3 |
| parameters | 800,083 |
| batch/test batch | 4/4 |

为什么不是默认ratio2：W64/D8/ratio2为842,083参数；W64/D7仅约739K。保留8层并轻微
缩小inner width可落在0.800M档。

## 5. 已验证状态

- graph neighbor correctness：通过。
- B4/T243真实CUDA shape：通过。
- prediction：`[4,243,17,3]`，finite。
- 完整loss/backward/AdamW/EMA：通过。
- opaque custom-op vs原CUDA Autograd.Function：输出逐位一致；梯度仅有归约级浮点差。
- 原PoseMamba CUDA forward回归：通过。
- 源工作区测试：52/52通过。
- 本机短编译基准：16 steps，约195.6ms/step、5.11it/s。
- 同正式入口诊断：连续55 batches通过后人工SIGINT。

### 尚未验证

- 120 epoch未完成；发布快照时正式run完成64次评估并进入下一轮。
- 最新中途P1/P2为`40.0475/33.1396 mm`，不是最终或多seed结果。
- 无可发布checkpoint。
- 未做multi-seed统计。
- 未证明0.8M配置精度优于原PoseMamba。

### 显存状态

- 正式FP32/compile训练短采样约`15,844--16,005 / 16,311 MiB`，余量偏紧。
- 参数本身不是主因；factorized short scan、graph context激活和reduce-overhead CUDA Graph
  私有池共同决定高水位。
- 当前优化重点曾是吞吐而非峰值显存。训练结束后应先做eager/default/reduce-overhead同配置
  A/B，再验证train-only compile、eager eval和activation checkpoint。
- 训练权重可在全新FP32 eager推理进程中低显存运行；不能把243帧Temporal BiSSM切块。

完整证据、计算和验收标准见`docs/VRAM_AND_INFERENCE.md`。

## 6. 已停止/否决的优化

- GraphMixer out projection与SSM context projection代数合并：数学等价但编译后更慢。
- 外层CUDA Graph叠加torch.compile：capture失败。
- max-autotune-no-cudagraphs：首次编译过慢，不适合作为正式入口。
- DataLoader 2 workers + pin memory：本机无收益。
- AMP、TF32、改batch：为保持协议未启用。
- 新L17定制kernel：可能继续提速，但违反当前“不改底层scan数值kernel”的研究边界。

## 7. CUDA编译交接

模型实际需要`selective_scan_cuda_core`，不是只编译oflex。发布包`setup.py`默认：

```bash
SELECTIVE_SCAN_MODES=core pip install -v --no-build-isolation ./kernels/selective_scan
```

opaque custom-op仍调用这个同一个core extension：

```text
GraphConditionedPoseMamba
  -> FactorizedBiSSM
  -> SelectiveScanCoreCompile
  -> posemamba::selective_scan_core_fwd/bwd
  -> selective_scan_cuda_core.fwd/bwd
```

构建必须发生在最终训练Python环境中。升级PyTorch、CUDA或编译器后应重新构建，不要复制
旧`.so`。详细排查见`docs/BUILD.md`。

## 8. 下一位维护者直接执行

```bash
python -m pip install -r requirements.txt
bash scripts/build_selective_scan.sh
python scripts/verify_install.py
python tools/check_model.py
python -m unittest discover -s tests -v
```

准备数据后：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/pose3d/graph_posemamba_h36m_w64_d8_0p8m.yaml \
  --checkpoint runs/graph_posemamba/h36m/w64_d8_0p8m_seed0 \
  --seed 0
```

首轮必须记录：稳定吞吐、epoch耗时、raw/EMA P1/P2、5类checkpoint是否完整。最终报告时
不得把短基准或早期loss写成精度结论。

## 9. 不在包内的资产

- Human3.6M/MPI-INF-3DHP数据。
- 所有checkpoint、runs、TensorBoard日志。
- 旧ReliPose/ReliPoseMU模型和实验。
- 论文、投稿材料、个人笔记和凭据。

## 10. GitHub建议

- 仓库必须设为private。
- 默认分支建议`main`。
- 首次提交建议：`feat: package graph-conditioned factorized PoseMamba`。
- push前运行`python tools/audit_release.py`，并人工确认无`.so`、权重、数据和token。
