# 显存、CUDA Graph 与推理交接

本文记录 2026-09-02 在 RTX 5060 Ti 16GB 环境中的实测现象、原因分析和后续优化边界。
它不是最终性能结论；所有显存优化必须与同配置、同进程阶段的基线配对测量。

## 1. 已观察状态

W64/D8/800,083 参数、batch 4、T=243 的正式 FP32 训练启用了：

```python
torch.compile(model, mode="reduce-overhead", fullgraph=False)
```

短时采样为 `15,844--16,005 MiB / 16,311 MiB`，GPU 利用率约 `96--98%`。该进程已跨越
数十次训练/评估，采样没有显示逐 step 单调增长，因此当前证据更符合稳定高水位，而不是明显
显存泄漏。物理余量仍只有约 306--467 MiB，不能直接用于更大模型。

800,083 个 FP32 权重本身约 3.05 MiB；即使加入梯度、AdamW 两份状态和 EMA，也只是十几
MiB 量级。训练显存主要由激活、selective-scan backward state、编译 workspace 和缓存池决定，
不能由参数量推断。

## 2. 为什么原 PoseMamba 显存更低

原 PoseMamba 对完整 `T x J` 平面扫描，近似让 CUDA kernel 处理少量长序列：

```text
B_scan = B = 4
L = T * J = 4,131
```

factorized Spatial SSM 为保证 frame 间 state reset，改成：

```text
B_scan = B * T = 972
L = J = 17
```

CUDA core 为 backward 保存的状态近似为：

```text
[B_scan, K * d_inner, ceil(L / 2048), 2 * d_state]
```

当前 spatial 配置 `K=2, d_inner=120, d_state=16`，单层核心 state 约 28.5 MiB；8 层
spatial 约 228 MiB。temporal 的 `[B*J=68, L=243]` 单层约 2.0 MiB。该状态只是总显存
的一部分，但说明大量独立 L=17 scan 的 workspace 特性明显不同于原版少量 L=4131 scan。

新模型还需要同时保留 recurrent content 和 graph parameter context。Temporal factorization 的
`T/J` 交换、正反向 CrossScan、GraphMixer 和 context projection 也可能产生额外连续副本或
中间激活。原版 flatten scan 的人体边界不合理，但对现有 CUDA kernel 更友好。

## 3. CUDA Graph 显存池

本机 PyTorch 的 `reduce-overhead` 模式启用 `triton.cudagraphs=True`。CUDA Graph 录制一串
kernel 和固定显存地址，后续用 graph replay 减少 Python、allocator 和 kernel launch 开销。
代价是静态输入/输出、workspace 和私有池在 graph 存活期间不能像 eager tensor 一样自由释放。

训练 forward/backward、eval graph、Inductor workspace 和 caching allocator 都可能体现在
`nvidia-smi` 的进程占用中。`nvidia-smi` 不区分活跃 tensor 与 reserved/private pool。
`torch.cuda.empty_cache()`也不能释放仍被活跃 CUDA Graph 引用的地址。

历史 eager 版本曾观察约 9.3--9.7 GiB，而当前 compiled W64/D8 约 15.47--15.63 GiB；因为
模型深度和实现版本不同，这个约 6 GiB 表面差值只能作为线索，不能写成 CUDA Graph 的精确
增量。精确值必须用同一 commit、同一配置、全新独立进程做 A/B。

## 4. 当前已经做过什么

当前实现主要完成的是吞吐优化：

- 伪 K4 重复方向改为真正 K2 forward/backward；
- temporal 复用单次 GraphMixer context；
- 合并 GraphMixer relation/neighbor projection；
- content/context 共享 input projection 权重；
- selective-scan 注册 compile-compatible opaque custom op；
- 标记 compile step 边界、减少无效 loss 和逐 batch 同步。

这些工作把早期约 1.8--1.9 it/s 提升到 W64/D8 短基准约 5.11 it/s。当前尚未完成系统性的
峰值显存优化：没有 activation checkpoint、train-only compile、eager evaluation、分阶段显存
测量或 mixed precision。`reduce-overhead`本身还是一项以显存换速度的选择。

## 5. 训练结束后的优化顺序

第一步必须在新进程中对同一个 W64/D8 checkpoint/config 测量：

```text
A. eager
B. torch.compile(mode="default")
C. torch.compile(mode="reduce-overhead")
D. compiled training + eager evaluation
```

每组记录模型初始化、forward、loss、backward、optimizer step 和 evaluation 后的：

```python
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
```

然后按风险从低到高处理：

1. 训练使用普通 Inductor compile，评估调用原始 eager model，避免额外 eval graph 池。
2. 将 compile mode 配置化，优先验证 `default`（无 reduce-overhead CUDA Graph）。
3. 对 `GraphConditionedPoseBlock`使用非重入 activation checkpoint，并保持 DropPath RNG。
4. 缩短 graph/context 生命周期，减少重复 `permute/contiguous/CrossScan`副本。
5. 只有 FP32 仍不足时，单独建立 BF16 baseline；不能与当前 FP32 结果混为同一协议。

activation checkpoint 不改变模型公式，但会在 backward 重算 forward，预期牺牲吞吐。BF16、
FP16、INT8、剪枝、缩短 T、降低 d_state 或 temporal chunking 都可能改变数值或模型能力，不能
作为严格等价优化。

专门支持每 17 个 joint reset 的 segmented short-scan kernel 可能同时改善速度和workspace，
但会修改底层 CUDA kernel，超出当前“不改 selective-scan 数值实现”的研究边界。

## 6. 低显存推理且保持精度

训练得到的普通 state dict 不依赖 CUDA Graph。推荐在训练进程退出后启动全新推理进程：

```python
model.load_state_dict(checkpoint["model_pos"])
model.cuda().eval()
with torch.inference_mode():
    prediction = model(batch_input.cuda())
    prediction_cpu = prediction.cpu()
```

使用 FP32 eager、batch 1 或 2、只加载模型权重并及时把输出转到 CPU，可以去掉训练梯度、
optimizer、EMA helper、backward state 和训练 CUDA Graph 池。模型只有 LayerNorm、没有依赖
batch 统计的 BatchNorm，因此减小推理 batch 不应系统性改变输出。

必须保持同一 raw/EMA checkpoint、243 帧输入、confidence/root-relative/flip TTA、输出聚合及
MPJPE/P-MPJPE 协议。不能沿时间把 243 帧切块，因为双向 Temporal BiSSM 需要完整前后上下文，
chunk boundary 会改变 state reset 和预测。

正式验收应比较标准 FP32 evaluation 与低显存 FP32 eager evaluation，建议要求输出最大绝对
误差不超过 `1e-5`，完整 MPJPE/P-MPJPE 差值不超过 `0.001 mm`。BF16/FP16/量化必须作为
可能改变精度的独立部署实验。

