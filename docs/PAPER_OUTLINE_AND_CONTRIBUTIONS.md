> **2026-09-06 update:** R3 stopped by user at 54/60. E-RWG-0 completed with best EMA `40.416912/33.822350 mm`, `+0.571750 mm` P1 worse than anatomical Full for seed0. E-NR-0 passed its real-data B4 gate and is next in the authorized serial queue. See [.experiments/paper_remaining_evidence/CURRENT_HANDOFF_20260906.md](../.experiments/paper_remaining_evidence/CURRENT_HANDOFF_20260906.md). Earlier status statements below are historical.

# 论文大纲与主要创新点

## 论文题目

**Graph-Conditioned Factorized State Space Modeling for 3D Human Pose Estimation**

中文：**面向三维人体姿态估计的图条件化解耦状态空间建模**

模型名沿用 `GraphConditionedPoseMamba`。

本文档整合截至2026-09-05的实现、历史实验和剩余证据计划。核心历史结果来自
`codex/minimal-ablation-80e-20260903`；新增论文证据代码位于本地隔离分支
`codex/paper-remaining-evidence-20260905`，科学实现提交为`ad4e2fa`。尚未完成的实验只保留
为计划或占位，不提前写成结论。

## 1. 中心论点

在空间—时间分解的状态空间模型中，姿态内容和骨架关系不必以相同方式参与计算。本文保留
当前姿态特征作为递推内容和输出门控的输入，用骨架增强上下文生成选择性状态参数，并在
帧内关节序列与逐关节时间轨迹上执行作用域明确的双向递推。

论文回答三个问题：

1. 骨架信息直接混入内容路径，还是用于条件化选择参数更有效？
2. 真实人体拓扑是否优于同规模、同度数的非解剖关系？
3. 在参数和局部预处理不变时，递推状态是否应跨帧/关节轨迹边界连接？

ReliPose可靠性、频率分解、检测器切换、D16优化器修复和编译技巧均不作为本文核心算法贡献。

## 2. 主要创新及边界

### 2.1 核心机制：内容路径与骨架上下文的角色分离

设当前姿态隐特征为`H`，图上下文为`G=GraphMixer(H)`，图增强上下文为`Q=H+λG`：

```text
U = content_projection(H)
Z = output_gate(H)
(Delta, B, C) = selective_parameter_projection(Q)
```

普通Graph Feature Fusion把`Q`作为完整SSM输入，因此内容、选择参数和输出门控都受图增强输入
影响。Full保持`U/Z`来自当前姿态特征，只让`Q`控制Delta/B/C。

建议表述：本文提出骨架上下文条件化的选择性状态建模，区分当前姿态内容路径与选择参数
生成上下文，在不额外改变当前层内容和门控输入的情况下，用骨架增强上下文控制状态写入、
保留与读取。

边界：这是作用路径分离，不是统计独立或正交解耦。后续层的`H`已经包含前层图信息，不能
宣称整个网络的内容路径完全不受图影响。

### 2.2 互补设计：作用域明确的空间—时间双向递推

```text
spatial: [N,T,J,d] -> [N*T,J,d]
temporal: [N,T,J,d] -> [N*J,T,d]
```

每帧空间序列和每条关节时间轨迹拥有独立初始状态。空间局部卷积为`d_conv=1`，时间分支使用
逐关节Conv1D `d_conv=3`。正反方向严格使用K=2。

建议表述：本文在帧内关节序列和逐关节时间轨迹上组织独立双向递推，使递推作用域与两种
建模任务一致，并以骨架上下文补充跨关节关系。

边界：空间扫描仍依赖关节排列顺序，不等价于逐步沿骨骼边传播；不宣称首次进行时空分解，
也不把A0到A1的全部差异归因于状态重置。

### 2.3 支撑模块与实证贡献

`SkeletonGraphMixer`使用固定骨骼边和左右对应边编码上下文。它是支撑模块，不单独宣传为
全新GNN。第三条贡献可写为紧凑模型与受控证据体系，包括注入位置、图内容、递推边界、
实现诊断、多seed、第二数据集和同条件效率。

## 3. 方法章节

### 3.1 问题定义与总体结构

输入`X∈R^{N×T×J×3}`，末维为`x/y/confidence`；输出根相对3D姿态
`Y_hat∈R^{N×T×J×3}`。整体由输入嵌入、多个图条件化时空块和
`LayerNorm+Linear(d,3)`组成。

### 3.2 Anatomical Context Encoding

对有向邻居`j→i`：

```text
m_ij = Linear_r(H_j-H_i) + Linear_n(H_j)
G_i = Linear_o(GELU(alpha_b * sum_bone(m_ij)
                    + alpha_s * sum_sym(m_ij)))
```

说明固定关系、共享投影、偏置、图隐藏维数和可学习关系尺度。该模块不强制骨长、对称或
关节角；图固定也不意味着输出上下文固定。

### 3.3 Content-Preserving Context Conditioning

局部输入与context共享内容投影权重。内容`U`和普通门控`Z`来自`H`，Delta/B/C来自`H+λG`
的编码。双向scan输出对齐合并、归一化，与`Z`相乘后进行输出投影。离散化公式按实际
selective-scan实现描述，不另造与代码不一致的公式。

### 3.4 Factorized Bidirectional Recurrence

分别定义空间/时间序列、正反方向、状态初始化和逆映射。时间阶段复用同一block空间阶段
得到的图上下文，不画成第二次图网络计算。

### 3.5 Full Block

```text
H_s   = LN_s(H_l) + E_joint
G_l   = GraphMixer(H_s)
H_mid = H_l + gamma_s * DropPath(BiSSM_s(H_s ; H_s + lambda*G_l))
H_t   = LN_t(H_mid) + E_time
H_tmp = H_mid + gamma_t * DropPath(BiSSM_t(H_t ; H_t + lambda*G_l))
H_l+1 = H_tmp + DropPath(FFN(LN_m(H_tmp)))
```

分号左侧为内容输入，右侧为选择参数上下文。

### 3.6 目标函数与实现成本

```text
L = L_pos + 0.5 L_scale + 20 L_velocity + 0.5 L_diff
```

该损失沿用基线，不是本文创新。参数量、完整MACs/FLOPs、推理延迟、训练吞吐、allocated和
reserved显存分别报告；编译和activation checkpoint属于工程实现。

## 4. 论文目录

### Abstract

依次写任务、注入位置问题、内容/上下文分工、独立时空递推和最终证据。所有结果保留占位，
不得提前使用“显著优于”或“SOTA”。

### 1. Introduction

1. 2D-to-3D姿态恢复及结构/时间信息的重要性。
2. 图增强、SSM、双向扫描和时空分解的已有进展。
3. 结构信息进入内容路径与选择参数路径并不等价。
4. 方法概述：骨架上下文、选择参数条件化、独立时空递推。
5. 贡献：核心机制、互补递推设计、受控实证。

### 2. Related Work

- 2D-to-3D Pose Lifting：MixSTE、MotionBERT、MotionAGFormer等。
- State-Space Models for Pose Estimation：PoseMamba、SAMA、DBMambaPose。
- Graph-aware State-Space Modeling：Hamba、PS-Mamba等。

落点是本文具体的内容/上下文注入方式及其受控验证，不宣称“首次图+SSM”。正式写作前须以
原论文核对方法与数值。

### 3. Method

按3.1—3.6展开，并明确继承PoseMamba和selective-scan实现。

### 4. Experiments

#### 4.1 数据、协议与实现

披露H36M划分、检测器、confidence、T243/S81、根相对坐标、flip TTA、P1/P2聚合和EMA。
历史80轮协议实际未启用warmup，尽管配置含`warmup_epochs: 8`；新匹配实验明确保持这一实际
行为。测试集监测best与固定epoch80 EMA分列。

#### 4.2 Human3.6M比较

公开论文结果、当前协议复现与本文结果分区，注明输入、额外数据、帧长和参数量。官方A0与
corrected-backward诊断分行，后者不替代前者。

#### 4.3 注入策略消融

A1、A2、Full回答无图、普通融合和条件化。A2与Full参数匹配，是Q1核心证据；变化包含内容
与门控输入来源，不能缩写为“只改变U”。

#### 4.4 解剖拓扑消融

Full对比固定保度重连图。重连图seed为3407，边数和每类节点度数不变，骨骼图保持连通，
不按结果换图。

#### 4.5 递推边界消融

Full对比参数匹配no-reset。保持K=2、投影和局部Conv1D，只在scan边界连接段；不使用旧
1,028,563参数K4 coupled配置。

#### 4.6 重复性与统计

A0、A2、Full补seed1/2。报告逐seed、配对差值、mean/std；逐源序列bootstrap作为补充，
不能替代训练seed。

#### 4.7 MPI-INF-3DHP

预注册released A0与Full，在同一T81修复协议上训练和测试。称为第二数据集验证，不称为
H36M到MPI零样本泛化。

#### 4.8 效率与定性分析

同设备、batch、长度、dtype、执行模式、预热和同步条件下测效率。展示逐动作、逐关节、
成功/失败案例；案例不替代结构消融。

### 5. Discussion and Limitations

- 不把学习到的Delta直接解释为关节可信度。
- 固定H36M图限制跨关节定义迁移。
- 空间扫描存在顺序偏置。
- 双向时序使用未来帧，不是因果在线模型。
- 小幅增益需要多seed，best结果受测试集监测影响。
- 参数效率和显存效率分开。
- released/corrected PoseMamba身份分开。

### 6. Conclusion

只总结最终证据支持的主张。若重连图或no-reset不支持假设，必须降低摘要、引言和结论中的
表述强度，而不是修改实验迎合论点。

## 5. 当前证据

以下均为单seed、H36M测试集逐轮监测选出的最佳EMA，P2来自同一checkpoint：

| 模型 | Params | Best epoch | P1 | P2 |
|---|---:|---:|---:|---:|
| PoseMamba A0 | 790,083 | 60 | 40.2260 | 33.5176 |
| Factorized Only A1 | 749,891 | 47 | 40.0605 | 33.3565 |
| Graph Feature Fusion A2 | 800,083 | 52 | 40.0588 | 33.2873 |
| Full A3 | 800,083 | 53 | 39.8452 | 33.2322 |
| Full with Rewired Graph | 800,083 | 41 | 40.4169 | 33.8223 |

Full相对A0的观察差值为0.3809 mm，相对A2为0.2137 mm。这些差异尚不能称为稳定或统计
显著。固定epoch120下Full为40.9894/33.3666，A0为40.4098/33.3151，Full没有优势，必须
与best口径同时披露。

尺度探索不是核心消融：W128/D20在65/80按用户要求取消，部分最佳为37.6593/31.8717；
W256/D16 R1/R2因优化震荡无效；R3按用户要求在54/60停止，部分最佳为
37.4302/31.5206，不能作为完整60轮结果。

## 6. 剩余证据

| 主张 | 必需证据 | 当前状态 | 不支持时处理 |
|---|---|---|---|
| 条件化注入优于普通融合 | A2 vs Full，seed0/1/2 | seed0已有 | 降低核心机制精度主张 |
| 真实拓扑有价值 | Full vs Rewired | seed0支持：Rewired P1差0.5718 mm；仍是单seed | 改称一般关系上下文 |
| reset边界有价值 | Full vs matched no-reset | B4门禁通过，等待/进入正式训练 | 降为组织方式 |
| 整体优于PoseMamba | released A0、corrected诊断、多seed | seed0及诊断预检已有 | 限定适用设置 |
| 不限于H36M | MPI released A0 vs Full | 协议/CUDA预检通过 | 不写第二数据集提升 |
| 更高运行效率 | 独占GPU同条件测量 | 未完成 | 只说紧凑/参数效率 |

详细配置、固定图、实现差异和PASS/BLOCKED状态见
`.experiments/paper_remaining_evidence/`。长训练必须逐个获得确认，不自动排队或push。

## 7. 图表安排

| 图表 | 内容 |
|---|---|
| Figure 1 | 总体结构及空间图上下文向时间阶段的复用 |
| Figure 2 | feature/control路径与independent/joined递推 |
| Figure 3 | 逐动作/逐关节误差和成功失败案例 |
| Table 1 | H36M公开比较与受控比较分区 |
| Table 2 | A0/A1/A2/Full、Rewired、matched no-reset |
| Table 3 | A0/A2/Full多seed与配对差值 |
| Table 4 | MPI released A0与Full |
| Table 5 | 同条件效率 |
| Supplement | 配置、图规格、导数诊断、曲线和完整误差 |

## 8. 参考入口

- SAMA, ICCV 2025: https://openaccess.thecvf.com/content/ICCV2025/html/Lu_A_Structure-aware_and_Motion-adaptive_Framework_for_3D_Human_Pose_Estimation_ICCV_2025_paper.html
- DBMambaPose, Pattern Recognition: https://www.sciencedirect.com/science/article/abs/pii/S0031320325015857
- PS-Mamba, ICCV 2025: https://openaccess.thecvf.com/content/ICCV2025/html/Dong_PS-Mamba_Spatial-Temporal_Graph_Mamba_for_Pose_Sequence_Refinement_ICCV_2025_paper.html

这些文献是最接近工作的入口，不代表已完成绝对首创性检索。
