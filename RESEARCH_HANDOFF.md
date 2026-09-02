# GraphConditionedPoseMamba 研究接手文档

> 快照时间：2026-09-02 13:13 UTC<br>
> 仓库：`Fubuyunhua/GraphConditionedPoseMamba`（private）<br>
> 研究状态：实现与工程验收完成；Human3.6M seed0 正式训练进行中；最终精度结论未形成。

本文供下一位研究者直接接手。它记录当前研究问题、模型实现、训练协议、实验进度、工程证据、
已知风险和下一步优先级。不要把本文中的中途指标写成论文最终结果。

## 1. 一句话状态

当前模型是一个 **Graph-Conditioned Factorized BiSSM PoseMamba**：固定人体骨架图建模局部
拓扑，逐帧 Spatial BiSSM 建模单帧全局关节依赖，逐关节 Temporal BiSSM 建模长时轨迹，
并只用 graph context 生成 selective SSM 的 `Delta/B/C`。主配置为 W64/D8、800,083 参数、
batch 4、243 帧。正式 seed0 已完成65次评估，尚未完成120 epoch。

## 2. 研究目标与硬约束

唯一目标：在保持已经验证的 PoseMamba 训练策略、数据处理、loss、augmentation、EMA、
optimizer 和 Human3.6M evaluation protocol 不变的条件下，只修改 PoseMamba backbone，争取
降低 MPJPE。

明确不研究、也不应重新加入：

- ReliPose observation-risk；
- dual-frequency 或 low/high-pass branch；
- coarse temporal branch、stride-3 branch、temporal downsampling；
- dynamic reliability/confidence/risk/frequency gate；
- MoE、多尺度动态routing、新prediction head或新loss。

第一版固定使用真实 Human3.6M skeleton bone edges和6组左右对称edges，不使用fully-connected
dynamic graph或复杂attention。`gamma_s/gamma_t`均从1初始化，不使用zero-init或sigmoid gate。

## 3. 核心研究假设

原PoseMamba将`X [B,T,J,C]`作为一个二维规则网格并flatten scan，可能形成不属于真实人体
时空图的状态转移：

```text
joint16(frame t) -> joint0(frame t+1)
last frame(joint j) -> first frame(joint j+1)
```

其3x3 DWConv还把joint index邻近误当成骨骼空间邻近。新模型的职责拆分是：

```text
真实局部骨架拓扑           -> SkeletonGraphMixer
单帧内全局joint dependency -> Spatial BiSSM
单joint长时trajectory       -> Temporal BiSSM
状态写入/遗忘/读取参数       -> graph-conditioned Delta/B/C
```

该假设是否降低最终MPJPE仍必须由完成后的正式配对实验回答。

## 4. 模型数据流

输入与输出：

```text
input      [B,T,17,3]   # x, y, confidence
embedding  [B,T,17,64]
prediction [B,T,17,3]
```

每个`GraphConditionedPoseBlock`执行：

```text
z_s = LN_s(x) + E_joint
G_s = GraphMixer(z_s)
s   = SpatialBiSSM(z_s, context=z_s + graph_scale * G_s)
x   = x + gamma_s * s

z_t = LN_t(x) + E_time
t   = TemporalBiSSM(z_t, context=z_t + graph_scale * reused_G_s)
x   = x + gamma_t * t

x   = x + MLP(LN_mlp(x))
```

Spatial factorization：

```text
[B,T,J,C] -> [B*T,1,J,C]
batch4时  -> [972,1,17,64]
```

每个frame是独立selective-scan sample，正向`j0...j16`，反向`j16...j0`，state不会跨frame。
Spatial local topology已交给GraphMixer，所以`d_conv=1`。

Temporal factorization：

```text
[B,T,J,C] -> [B*J,1,T,C]
batch4时  -> [68,1,243,64]
```

每个joint是独立trajectory，正向`t0...t242`，反向`t242...t0`，state不会跨joint。时间局部邻域
真实存在，因此使用Conv1D `d_conv=3`。

Graph-conditioned selective dynamics保持：

```text
u = encode(x)
context = x + graph_scale * GraphMixer(x)
(Delta, B, C) = projection(encode(context))
y = selective_scan(u, Delta, A, B, C, D)
```

状态内容`u`始终来自原feature；graph context只控制状态如何写入、遗忘和读取。输入与context
共享input projection权重，CUDA selective-scan数值kernel未修改。

## 5. GraphMixer定义

Human3.6M 17 joints：root、双腿、belly/neck/nose/head、双肩肘腕。固定bone edges为：

```text
(0,1)(1,2)(2,3)  (0,4)(4,5)(5,6)
(0,7)(7,8)(8,9)(9,10)
(8,11)(11,12)(12,13)  (8,14)(14,15)(15,16)
```

symmetry edges为：

```text
(1,4)(2,5)(3,6)(11,14)(12,15)(13,16)
```

消息形式：

```text
r_ij = x_j - x_i
m_ij = W_relation(r_ij) + W_neighbor(x_j)
g_i  = alpha_bone * sum_bone(m_ij) + alpha_sym * sum_sym(m_ij)
G_i  = W_out(GELU(g_i))
```

hidden dimension为32，`alpha_bone/alpha_sym`从1初始化。

## 6. 当前主配置

文件：`configs/pose3d/graph_posemamba_h36m_w64_d8_0p8m.yaml`

| 项目 | 当前值 |
|---|---:|
| width | 64 |
| depth | 8 |
| parameters | 800,083 |
| graph hidden | 32 |
| SSM inner | 120 (`ratio=1.875`) |
| MLP inner | 126 (`ratio=1.96875`) |
| d_state | 16 |
| directional groups | 2（forward/backward） |
| spatial conv | 1 |
| temporal conv | 3 |
| clip/joints/input | 243 / 17 / xy+confidence |
| train/test batch | 4 / 4 |

W64/D8若直接使用两个ratio=2会达到842,083参数；当前轻微缩小inner width以保持8层并落在
0.800M档，不是通过明显扩容获取精度。

## 7. 冻结的训练协议

以下配置来自当前已验证PoseMamba配方，不能为了新模型随意调整：

```text
optimizer           AdamW
learning rate       5e-4
weight decay        0.012
epochs              120
warmup              8 epochs
lr decay            0.99
EMA                  enabled, decay 0.9998
batch                4
clip/data stride     243 / 81
root-relative        enabled
flip                 enabled
confidence input     enabled
lambda_3d            1.0
lambda_scale         0.5
lambda_3d_velocity   20.0
lambda_diff          0.5
```

loss、augmentation、数据预处理、clip、stride和Protocol #1/#2均不因新backbone修改。

## 8. 工程优化历史

早期正确性优先版本约`1.8--1.9 it/s`。之后依次完成：

1. 固定图dense等价聚合、context有效投影、零权loss shortcut和foreach EMA；
2. 修复伪双向K4重复为真正K2 forward/backward；
3. 为原`selective_scan_cuda_core`注册opaque compile接口，消除主要Dynamo graph break；
4. temporal复用单次graph context；
5. 合并GraphMixer relation/neighbor projection、延迟日志同步并标记compile step边界；
6. W64/D8配置的真实H36M短基准达到约`195.6 ms/step`、`5.11 it/s`。

同环境曾测默认PoseMamba约`4.210 it/s`，同等编译PoseMamba约`5.174 it/s`。因此当前GCF
已接近同等编译PoseMamba速度；剩余问题主要是显存，而不是明显吞吐不足。

## 9. 正式Human3.6M实验状态

Docker源工作区（不在GitHub包中）的活跃run：

```text
/workspace/ReliPose_release/runs/graph_conditioned_posemamba/h36m/
W64_D8_0p8M_seed0_restart_2026_09_02_T_04_36_22
```

发布快照时：

```text
完成评估次数       65
正在训练           epoch 65
最新P1/P2          39.9882 / 33.1112 mm
P1最低             39.9882 mm（第65次评估）
P2独立最低         33.0836 mm（第62次评估）
最终120 epoch       未完成
multi-seed          未运行
```

配置启用EMA，`train.py`训练期评估在`ema_helper.average_parameters(...)`上下文内完成，所以日志
中的`e1/e2`是EMA参数评估。当前代码用该EMA P1判断是否更新best，同时保存raw和EMA两个best
文件；因此`best_epoch.bin`是“由EMA指标选择时刻保存的raw权重”，不能把它误称为raw自身最优。
训练完成后必须分别加载raw/EMA checkpoint做明确评估。

以上均是中途状态，只支持“训练正常收敛”，不支持“已经优于PoseMamba”的最终claim。

## 10. 显存状态与原因

正式FP32、batch4、`torch.compile(mode="reduce-overhead")`训练短采样：

```text
GPU total     16,311 MiB
used          15,844--16,005 MiB
utilization   96--98%
```

短采样没有单调增长，进程也已跨越数十轮，当前更像CUDA Graph/allocator稳定高水位而不是明显
泄漏。但余量仅约306--467 MiB，不适合直接扩大模型。

主要因素：

- 全FP32训练激活；
- Spatial将batch4变成972条独立L=17 scan；
- graph content/context双路中间量；
- selective-scan backward state；
- `reduce-overhead`隐式启用CUDA Graph私有池；
- 训练与评估compiled graph及Inductor/caching allocator保留。

完整公式、显存测量要求、保持精度的推理方式见`docs/VRAM_AND_INFERENCE.md`。

## 11. 已验证与未验证

已验证：

- graph neighbors和symmetry edges正确；
- Spatial/Temporal reshape与restore精确；
- B4/T243输出`[4,243,17,3]`；
- 800,083参数；
- loss/backward/AdamW/EMA完整step；
- opaque custom-op与原CUDA kernel输出/梯度一致；
- 原PoseMamba仍可运行；
- 从本包CUDA源码全新sm120编译并forward/backward通过；
- GitHub包CPU聚焦测试、安全审计和SHA256校验通过。

尚未验证：

- 120 epoch最终raw/EMA P1/P2；
- 与完全同配方PoseMamba的最终配对精度；
- multi-seed均值/方差；
- 当前W64/D8在eager/default/reduce-overhead下的严格同commit显存A/B；
- 更大参数模型是否提高精度；
- BF16或量化部署精度。

## 12. 下一步优先级

### Priority 0：完成当前实验

不修改、不停止当前run。完成120 epoch后：

1. 记录日志最终值和最低值；
2. 分别评估`best_epoch.bin`、`best_ema_epoch.bin`、`latest_epoch.bin`和`latest_ema_epoch.bin`；
3. 明确P1-selected checkpoint对应的P1/P2；
4. 与冻结协议PoseMamba配对比较；
5. 只有完整结果后才能形成精度结论。

### Priority 1：显存专项优化

用同一checkpoint/config在全新进程中测：

```text
eager
torch.compile(mode="default")
torch.compile(mode="reduce-overhead")
compiled training + eager evaluation
```

分别记录`max_memory_allocated/reserved`。优先尝试train-only compile和非CUDA-Graph default
compile；仍不足时增加非重入activation checkpoint并保持DropPath RNG。不能用`empty_cache()`
代替测量，也不能通过减小batch/T或删除graph conditioning伪装显存优化。

### Priority 2：更大容量

只有显存优化通过输出、loss、梯度和checkpoint一致性验收后，才依次测试约1.0M、1.2M、1.5M，
每次只改变容量。更大模型不保证更低MPJPE，不进行无依据的大规模超参数搜索。

### Priority 3：可选数值优化

BF16/FP16/INT8、剪枝、量化和新short-scan kernel均可能改变数值或研究边界，必须作为独立实验，
不能覆盖FP32基线。

## 13. 直接运行命令

安装并编译：

```bash
python -m pip install -r requirements.txt
bash scripts/build_selective_scan.sh
python scripts/verify_install.py
python tools/check_model.py
python -m unittest discover -s tests -v
```

训练：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/pose3d/graph_posemamba_h36m_w64_d8_0p8m.yaml \
  --checkpoint runs/graph_posemamba/h36m/w64_d8_0p8m_seed0 \
  --seed 0
```

评估：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/pose3d/graph_posemamba_h36m_w64_d8_0p8m.yaml \
  --evaluate path/to/checkpoint.bin \
  --checkpoint runs/eval/graph_posemamba_w64_d8 \
  --seed 0
```

首次compile可能在`0it`停留10--60秒或更久；TF32 deprecation、TF32未启用和SM不足以
max-autotune是warning。没有最终`RuntimeError`/`CUDA error`时不要在首次backward中断。

## 14. 数据、权重与安全边界

仓库不包含：

- Human3.6M或MPI-INF-3DHP数据；
- runs、TensorBoard日志或checkpoint；
- 编译后的`.so/.o`；
- GitHub token、Docker credential或个人笔记；
- ReliPose/ReliPoseMU和无关旧实验。

数据路径和格式见`DATA.md`。训练完成后若要发布权重，应单独确认具体raw/EMA文件，并使用
GitHub Release或Git LFS，不要直接把大型checkpoint写入普通Git历史。

## 15. 接手检查清单

接手者应按顺序确认：

1. 阅读本文件、`HANDOFF.md`、`docs/POSEMAMBA_CHANGES.md`和
   `docs/VRAM_AND_INFERENCE.md`；
2. 核对Python/PyTorch/CUDA/GPU架构，重新编译`selective_scan_cuda_core`；
3. 运行安装验证、参数/shape检查和单元测试；
4. 准备外部数据，但不要提交数据或本地绝对路径；
5. 若原Docker仍存在，先完成当前seed0，不要重复启动同一正式实验；
6. 将最终实验数值、checkpoint类型、命令和run路径写入正式实验记录；
7. 未完成配对baseline和multi-seed前，保持研究结论审慎。
