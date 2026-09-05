# Graph-Conditioned Factorized PoseMamba

> Private research snapshot, 2026-09-02. Keep this repository private until
> the associated research has been reviewed for release.

这是一个从 PoseMamba 派生的 2D-to-3D Human Pose Estimation 研究包。当前版本只研究
PoseMamba backbone：固定人体骨架图负责局部关系，逐帧 Spatial BiSSM 负责同一姿态内的
全局关节依赖，逐关节 Temporal BiSSM 负责长时轨迹，并用骨架图上下文生成 selective SSM
的 `Δ/B/C`。包内不包含 ReliPose 的 risk/frequency/routing 分支。

English summary: this repository contains a focused, private research snapshot
of a graph-conditioned, spatial/temporal-factorized PoseMamba backbone for
Human3.6M 2D-to-3D pose lifting.

## 当前状态

- 主配置：W64 / D8 / 800,083 parameters / batch 4 / T=243 / J=17。
- 输入：`[B,T,J,3]`，最后一维是 `(x,y,confidence)`。
- 输出：`[B,T,J,3]`。
- 已验证环境包括RTX 5060 Ti上的Python 3.10.18/PyTorch
  `2.10.0.dev+cu128`，以及RTX 5090上的Python 3.10.20/PyTorch
  `2.11.0+cu128`。
- 真实 Human3.6M 短基准完成 forward/loss/backward/AdamW/EMA；16-step 编译基准约
  `195.6 ms/step`、`5.11 it/s`。吞吐仅是本机工程数据，不是精度结果。
- 52 项源工作区回归测试通过；本包另含聚焦测试。
- RTX 5090 seed0已完成120轮：最佳EMA为第53轮
  `39.8452 / 33.2322 mm`，第120轮EMA为`40.9894 / 33.3666 mm`。
- 同数据、配方与当前评价器下，790,083参数PoseMamba W64/D6/M1最佳EMA为
  `40.2260 / 33.5176 mm`；Graph模型最佳EMA改善`0.3809 / 0.2854 mm`，
  但第120轮固定终点没有改善。
- 当前结果是单seed且逐轮监测H36M测试集；不得描述成multi-seed或无偏测试集优势。
- 完整结果、显存优化、checkpoint复评和公平性限制见
  [RTX5090实验报告](docs/RTX5090_EXPERIMENT_RESULTS_20260903.md)。

面向研究接手的完整状态见[RESEARCH_HANDOFF.md](RESEARCH_HANDOFF.md)，工程交接见
[HANDOFF.md](HANDOFF.md)，结构差异见
[docs/POSEMAMBA_CHANGES.md](docs/POSEMAMBA_CHANGES.md)，显存、CUDA Graph和低显存推理见
[docs/VRAM_AND_INFERENCE.md](docs/VRAM_AND_INFERENCE.md)。论文中心论点、章节结构、证据边界
和图表安排见[论文大纲](docs/PAPER_OUTLINE_AND_CONTRIBUTIONS.md)。

## 仓库内容

```text
configs/pose3d/                 独立、无隐藏 base 依赖的 W64/D8 配置
lib/model/PoseMamba.py          原 PoseMamba + 新 GraphConditionedPoseMamba
lib/model/graph_mixer.py        固定骨架与左右对称关系 mixer
lib/model/mambablocks.py        K=2 factorized BiSSM 与 context-conditioned Δ/B/C
lib/model/selective_scan_compile.py
                                torch.compile 可见的 opaque custom-op 包装
kernels/selective_scan/         selective_scan_cuda_core CUDA 源码和构建脚本
train.py                        Human3.6M 训练、EMA、评估与 checkpoint
tools/check_model.py            图、参数量和真实 CUDA shape 检查
tools/benchmark_training.py     完整训练步短基准
scripts/build_selective_scan.sh CUDA extension 构建入口
scripts/verify_install.py       安装后 forward/backward 验证
```

数据、checkpoint、runs、TensorBoard 日志和论文材料均未打包。

## 1. 环境

要求：

- Linux
- Python 3.10
- NVIDIA GPU
- 与所装 PyTorch 兼容的 CUDA toolkit（必须有 `nvcc`）
- CUDA-enabled PyTorch；必须提供`torch.library.custom_op/register_autograd`。本快照只在
  PyTorch `2.10.0.dev+cu128`和`2.11.0+cu128`完成验证；其他版本需先运行安装验证。

先根据你的 GPU/CUDA 安装 PyTorch 和 torchvision（以及该PyTorch发行版匹配的Triton）。
不要让`pip`单独升级到与PyTorch不兼容的Triton，也不要直接照搬另一台机器的wheel。
例如应从 [PyTorch 官方安装器](https://pytorch.org/get-started/locally/) 选择匹配命令，然后：

```bash
python -m pip install -r requirements.txt
```

检查环境：

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
PY
nvcc --version
```

## 2. 编译 selective-scan CUDA extension

当前模型依赖的模块名是 **`selective_scan_cuda_core`**。必须在安装 PyTorch 后、同一个
Python 环境内编译：

```bash
bash scripts/build_selective_scan.sh
```

等价的手工命令：

```bash
export SELECTIVE_SCAN_MODES=core
export MAX_JOBS=4
python -m pip install -v --no-build-isolation ./kernels/selective_scan
python -c "import torch, selective_scan_cuda_core; print('CUDA extension OK')"
```

如果构建机没有可见 GPU，需要显式设置架构。例如 Ampere 8.6：

```bash
export TORCH_CUDA_ARCH_LIST="8.6"
bash scripts/build_selective_scan.sh
```

RTX 5060 Ti 是 compute capability 12.0；必须使用能识别该架构的 PyTorch/CUDA 版本。
更多 ABI、架构和错误排查见 [docs/BUILD.md](docs/BUILD.md)。

## 3. 安装验证

```bash
python scripts/verify_install.py
python tools/check_model.py
python -m unittest discover -s tests -p 'test_*.py' -v
```

预期关键输出：

```text
parameters: 800,083
prediction: (1, 243, 17, 3)
forward/backward: OK
```

`tools/check_model.py` 默认使用 batch 4，并打印：

```text
[4,243,17,3] -> embedding [4,243,17,64]
spatial scan input  [972,1,17,64]
temporal scan input [68,1,243,64]
prediction          [4,243,17,3]
```

## 4. 数据

数据集不在仓库内。默认配置要求：

```text
data/motion3d/MB3D_f243s81/h36m_sh_conf_cam_source_final.pkl
```

详见 [DATA.md](DATA.md)。不要提交数据、个人路径或派生缓存。

## 5. 训练

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/pose3d/graph_posemamba_h36m_w64_d8_0p8m_memopt_speed.yaml \
  --checkpoint runs/graph_posemamba/h36m/w64_d8_0p8m_speed_seed0 \
  --seed 0
```

配置冻结：AdamW、LR `5e-4`、weight decay `0.012`、120 epochs、8 warmup epochs、
EMA `0.9998`、batch 4、velocity loss、augmentation、clip/stride 和评估协议均沿用已验证
PoseMamba 配方。

### 首次 `0it` 是编译，不是卡死

`compile_model: true` 会在第一次 forward/backward 编译图，进度条可能在 `0it` 保持
10–60 秒甚至更久。TF32 deprecation、TF32 未开启和 “Not enough SMs to use
max_autotune_gemm” 是 warning。没有最终 `RuntimeError:`/`CUDA error:` 时不要按
`Ctrl+C`。一次诊断曾在同一入口连续完成 55 个 batch。

如需诊断而非正式训练：

```bash
python tools/benchmark_training.py --warmup-steps 4 --steps 16
```

加 `--real-data` 才会读取 Human3.6M；默认使用合成张量。

## 6. 评估

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/pose3d/graph_posemamba_h36m_w64_d8_0p8m.yaml \
  --evaluate path/to/checkpoint.bin \
  --checkpoint runs/eval/graph_posemamba_w64_d8 \
  --seed 0
```

报告时必须区分 raw 与 EMA checkpoint，并同时给出 MPJPE（Protocol #1）和
P-MPJPE（Protocol #2）。当前仓库没有随附权重。若需要在低显存环境保持FP32精度，请使用
独立eager推理进程并遵守[显存与推理交接](docs/VRAM_AND_INFERENCE.md)中的边界。

## 7. GitHub / 安全策略

`.gitignore` 已排除：

- `data/`
- `runs/`
- `checkpoint*/`
- `*.bin`, `*.pth`, `*.pt`, `*.ckpt`
- 编译产物、TensorBoard 日志和 IDE 文件

上传前运行：

```bash
python tools/audit_release.py
git status --short
```

## License and provenance

项目沿用根目录 [LICENSE](LICENSE)。PoseMamba/Mamba/selective-scan 派生代码的来源和
声明见 [NOTICE](NOTICE)，相关源文件中的版权头应保留。
