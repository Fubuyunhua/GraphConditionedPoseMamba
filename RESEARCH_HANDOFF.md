> **2026-09-06 update:** R3 stopped by user at 54/60; E-RWG-0 is RUNNING. The serial experiment queue and GitHub sync are authorized. See [.experiments/paper_remaining_evidence/CURRENT_HANDOFF_20260906.md](.experiments/paper_remaining_evidence/CURRENT_HANDOFF_20260906.md). Earlier running-status and confirmation statements below are historical.

# GraphConditionedPoseMamba 研究接手文档

> 快照时间：2026-09-05 12:05（Asia/Shanghai）<br>
> GitHub：`Fubuyunhua/GraphConditionedPoseMamba`（private）<br>
> 当前远程实验分支：`codex/minimal-ablation-80e-20260903`<br>
> 论文证据本地分支：`codex/paper-remaining-evidence-20260905`（未push）

本文记录当前论文论点、实现身份、历史结果、运行中实验、剩余证据和接手顺序。完整论文结构
见[论文大纲](docs/PAPER_OUTLINE_AND_CONTRIBUTIONS.md)，工程安装与CUDA构建仍参考
[HANDOFF.md](HANDOFF.md)。

## 1. 一句话状态

W64/D8 GraphConditionedPoseMamba的seed0、A1和A2已完成，当前核心观察为Full
`39.8452/33.2322 mm`。论文尚缺真实拓扑、匹配递推边界、多seed和第二数据集结果。
对应代码与配置已在隔离worktree完成预检，但没有push，也没有启动长训练。并行的W256/D16
R3属于尺度/优化诊断，仍在5090运行，不是论文核心消融。

## 2. 仓库与分支身份

远程GitHub目前是线性继承的三个分支：

```text
main (14216cc)
  -> codex/memory-opt-5090-20260902 (6e60b78)
     -> codex/minimal-ablation-80e-20260903 (a931525 at paper-worktree creation)
```

论文证据代码在独立本地worktree：

```text
D:\gpu5090\GraphConditionedPoseMamba-paper-evidence
branch: codex/paper-remaining-evidence-20260905
implementation: ad4e2fa
records: 4d736c6
```

提示词依据提交`3fac1a4`，实际创建worktree时HEAD为`a931525`；差异只有R3日志同步，没有
模型源码或冻结配置变化。不得把本地论文分支误称为已经发布到GitHub。

## 3. 论文中心问题

GraphConditionedPoseMamba研究结构信息进入选择性状态空间计算的位置，而不是简单增加图特征：

```text
content U, output gate Z <- current pose feature H
Delta/B/C                <- H + GraphMixer(H)
```

三个问题分别由以下证据回答：

1. 注入位置：Graph Feature Fusion A2 vs Full A3。
2. 图内容：Anatomical Full vs Full with Rewired Graph。
3. 递推边界：Full vs参数匹配的Full w/o Recurrence Reset。

`SkeletonGraphMixer`是上下文编码器，不单独宣传成全新GNN。时空factorization是互补设计，
不宣称首次提出。ReliPose风险、频率分解、MoE、额外loss和D16优化器不是本文主贡献。

## 4. Full模型数据流

输入输出：

```text
input      [B,T,17,3]  # x,y,confidence
embedding  [B,T,17,64]
prediction [B,T,17,3]
```

每个block：

```text
H_s   = LN_s(H) + E_joint
G     = GraphMixer(H_s)
H_mid = H + gamma_s * SpatialBiSSM(H_s ; H_s + graph_scale*G)
H_t   = LN_t(H_mid) + E_time
H_tmp = H_mid + gamma_t * TemporalBiSSM(H_t ; H_t + graph_scale*G)
H_out = H_tmp + MLP(LN_m(H_tmp))
```

空间`[B,T,J,C]->[B*T,1,J,C]`，每帧独立；时间
`[B,T,J,C]->[B*J,1,T,C]`，每条关节轨迹独立。两者均为真正K=2正反向scan。空间
`d_conv=1`，时间为逐关节Conv1D `d_conv=3`。时间阶段复用同一block空间阶段得到的`G`。

## 5. 已完成H36M结果

以下均为单seed、逐轮监测H36M测试集得到的最佳EMA，P2来自同一checkpoint：

| Variant | Params | Best epoch | EMA P1 | Paired P2 | 状态 |
|---|---:|---:|---:|---:|---|
| A0 PoseMamba released | 790,083 | 60 | 40.2260 | 33.5176 | 完成120轮 |
| A1 Factorized Only | 749,891 | 47 | 40.0605 | 33.3565 | 完成80轮 |
| A2 Graph Feature Fusion | 800,083 | 52 | 40.0588 | 33.2873 | 完成80轮 |
| A3 Full | 800,083 | 53 | 39.8452 | 33.2322 | 完成120轮 |

当前观察：Full比A0低0.3809 mm，比A2低0.2137 mm。差异很小，未做完整multi-seed前不能称为
稳定或统计显著。

固定epoch120 EMA：A0为`40.4098/33.3151`，Full为`40.9894/33.3666`，Full没有优势。
因此必须分开报告“测试监测best”和“固定终点”。A0/Full没有保存epoch80权重，不能补造
`ema_fixed_epoch80`；A1/A2的epoch80 EMA文件存在，但尚未重放固定终点指标。

## 6. 实际80轮协议

```text
dataset              H36M-SH xy+confidence, T243/S81
dataset sha256        73b642f2567a8d0b194f88c54a3182c7b635c003c832b48ae6ee559f10232175
train samples         17,748
batch / steps         4 / 4,437 per epoch
optimizer             AdamW
LR / WD / decay       5e-4 / 0.012 / 0.99 per epoch
effective warmup      disabled
epochs / steps        80 / 354,960
EMA decay / updates   0.9998 / 354,960
loss                  position 1 + scale 0.5 + velocity 20 + difference 0.5
best rule             best_ema_test_monitored_first80
fixed rule            ema_fixed_epoch80
```

历史YAML含`warmup_epochs: 8`，但训练入口没有启用warmup。新增匹配配置明确写
`enable_linear_warmup: false`，以保持实际协议，而不是采用D16 R3的新优化器。

## 7. 论文证据实现

### 7.1 Full with Rewired Graph

配置：`configs/pose3d/ablation_full_rewired_graph.yaml`。

- seed3407独立RNG固定生成，不消耗模型/数据全局RNG。
- 骨骼16边、对称6边，节点度数分别保持，骨骼图连通。
- 无自环、重复边，所有层共享同一图。
- 图SHA256：`f9037c7265d94ba73c5941fc3070dec76cd022e8c302d141543e94c85627efad`。
- 参数量与Full相同：800,083。

### 7.2 Full w/o Recurrence Reset — matched

配置：`configs/pose3d/ablation_full_no_recurrence_reset_matched.yaml`。

局部Conv1D、输入/context投影、u、Delta/B/C和z先按Full计算，再连接scan：

```text
spatial [B*T,2,D,J] -> [B,2,D,T*J]
temporal [B*J,2,D,T] -> [B,2,D,J*T]
```

T243/J17时scan长度为4,131。K=2、参数名/形状/数量、图、loss和训练协议均不变。旧1.028M
K4 coupled版本保留为旧预检，不是这一消融。

### 7.3 PoseMamba corrected backward

历史A0源码已确认：前向含`x+x[...,indices]`，backward遗漏索引路径的`P^Tg`。新增
`posemamba_backward_mode: legacy|exact`，默认legacy；exact只补齐导数，不改变前向、K4、
CrossMerge和CUDA scan。两者均为790,083参数，历史权重可严格加载；exact重放仍为
`40.226022/33.517626 mm`。corrected结果必须作为独立诊断，不能与released A0混成多seed。

### 7.4 重复性与MPI

- A0、A2、Full的seed1/2配置已生成。
- MPI预注册baseline为released PoseMamba W64/D6/M1，不在看到结果后切换corrected身份。
- MPI使用已审计T81、stride9、2,875个有效测试中心、固定epoch120协议。
- 两个MPI模型的协议和B4 CUDA forward/backward已通过。

## 8. 预检状态

- 47项CPU/CUDA单元测试通过。
- Rewired、matched no-reset、corrected A0真实H36M单步通过。
- Rewired/No-reset均为800,083参数；corrected A0为790,083。
- 默认Full相对未修改源码：预测/loss逐位一致；梯度差在CUDA重复运行噪声内。
- 历史Full与A0 checkpoint严格加载；完整评估口径保持。
- 新H36M训练会显式保存`raw_fixed_epoch80.bin`和`ema_fixed_epoch80.bin`。
- 独占GPU正式速度/FLOPs尚未测；并发smoke计时不能用于论文。
- 所有新长训练均为`NOT_RUN`，等待用户逐个确认。

详细记录：`.experiments/paper_remaining_evidence/check_report.md`、`ledger.json`、
`experiment_registry.csv`和`results.csv`。

## 9. 当前尺度实验状态

远程唯一GraphConditionedPoseMamba长训练是W256/D16 R3：

```text
remote root   /scratch/home/caiwei/GraphConditionedPoseMamba_W256_D16_R3_60e_20260905
source        0e23c5d
params        20,192,451
budget        60 epochs
snapshot      26/60 completed, epoch27 running
best/latest   37.8653 / 31.8041 mm at epoch26
loss          0.010022
LR            2.28e-4
reserved VRAM 21,140 MiB trainer
```

R3曾在epoch7—8有限值震荡后自行恢复。用户明确要求不得自主中断：监控只能记录、同步和告警，
禁止发送任何停止、暂停、恢复或改配置命令。

尺度历史：

- W128/D20：用户在65/80取消，部分最佳37.6593/31.8717，不是完成结果。
- W256/D10：首轮前取消。
- D16 R1：3轮后因无效warmup与过冲判INVALID。
- D16 R2：16轮后出现严重有限值震荡并已停止，判INVALID。
- D16 R3：当前运行中，不能写入论文最终精度表。

## 10. 下一步执行顺序

新长训练没有获得授权。用户确认后一次只启动一个：

1. Rewired Graph seed0，80轮。
2. Matched no-reset seed0，80轮。
3. Corrected PoseMamba seed0，80轮。
4. A0/A2/Full seed1。
5. MPI released A0/Full。
6. A0/A2/Full seed2。

不得自动push、自动排队全部实验、搜索重连图/seed/超参数，或删除任何不支持Full的有效结果。

## 11. 写作口径

- A2 vs Full只能称“图信息注入策略对照”。A2中内容、Delta/B/C和输出门控都来自图增强输入。
- 真实图只有优于固定重连图时才能支持“解剖拓扑有价值”。
- reset只有优于参数匹配joined版本时才能作为精度设计主张。
- MPI训练并测试是第二数据集验证，不是零样本泛化。
- best与fixed endpoint分列；P1/P2必须来自同一checkpoint。
- 公开方法数值须核对原论文，并注明输入、帧长、规模和额外数据。

## 12. 接手检查清单

1. 阅读本文、论文大纲、`HANDOFF.md`和paper evidence检查报告。
2. 区分远程实验分支、运行中R3源码和未push的论文证据分支。
3. 不修改或停止R3；先核对PID、日志和checkpoint。
4. 新实验必须从`ad4e2fa`或其已审计后继提交建立独立目录。
5. 长训练前重新做独占GPU显存门禁，并等待用户指定单个run。
6. 每个run记录git/config/data哈希、seed、graph seed、实际执行路径、raw/EMA身份和两种统计口径。
7. 完成后如实更新证据—主张强度；不通过调参或选择结果迎合论文叙事。
