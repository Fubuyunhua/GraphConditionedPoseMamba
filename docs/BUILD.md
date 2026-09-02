# CUDA selective-scan 编译说明

## 必需产物

当前GraphConditionedPoseMamba要求Python能导入：

```python
import selective_scan_cuda_core
```

`selective_scan_cuda_oflex`不是当前主路径的替代品。默认构建只编译core；如需额外实验可：

```bash
SELECTIVE_SCAN_MODES=core,oflex bash scripts/build_selective_scan.sh
```

## 推荐顺序

1. 创建Python 3.10环境。
2. 安装与GPU/driver/toolkit匹配的CUDA PyTorch与torchvision。
3. 安装`requirements.txt`。
4. 在同一环境运行`build_selective_scan.sh`。
5. 运行`verify_install.py`。

```bash
which python
python -m pip --version
which nvcc
python -c "import torch; print(torch.__version__, torch.version.cuda)"
bash scripts/build_selective_scan.sh
python scripts/verify_install.py
```

## 架构

有可见GPU时PyTorch extension通常自动检测架构。无GPU构建节点必须设置：

```bash
export TORCH_CUDA_ARCH_LIST="8.6"   # 示例，不要盲目复制
```

常见架构必须根据实际设备和PyTorch支持核对。当前验证机RTX 5060 Ti报告`(12,0)`，使用CUDA
12.8和能识别sm_120的PyTorch nightly。

## 版本/ABI

- 不要从另一套PyTorch环境复制`.so`。
- PyTorch升级后重新构建。
- `_GLIBCXX_USE_CXX11_ABI`不匹配会导致undefined symbol。
- `nvcc` toolkit与`torch.version.cuda`最好同主版本；明显不匹配时先修环境。
- 使用`--no-build-isolation`，因为构建脚本必须导入当前环境的torch。

## 常见错误

### `ModuleNotFoundError: selective_scan_cuda_core`

没有编译core，或装到了不同Python：

```bash
SELECTIVE_SCAN_MODES=core python -m pip install -v --no-build-isolation ./kernels/selective_scan
python -c "import torch, selective_scan_cuda_core"
```

### `nvcc: command not found`

PyTorch wheel自带CUDA runtime不等于系统有CUDA compiler。安装兼容toolkit并确保`nvcc`在PATH。

### `no kernel image is available`

扩展没包含当前GPU架构。清理构建缓存并用正确`TORCH_CUDA_ARCH_LIST`重建。不要提交`.so`。

### `undefined symbol`

通常是PyTorch/CUDA/C++ ABI不匹配。确认`which python`和`python -m pip`一致，然后重新构建。

### 首步`0it`很久

这是`torch.compile(mode="reduce-overhead")`编译forward/backward，与CUDA extension编译是两件事。
首次训练可能在0it等待10–60秒或更久。没有最终RuntimeError时不要中断。

### TF32 / max-autotune warning

TF32 API deprecation、TF32未开启、SM不足以运行max-autotune GEMM都是非致命warning。本实验为保持
协议不主动开启TF32。

## 数值边界

`selective_scan_compile.py`只注册opaque custom operator；实际计算仍调用：

```text
selective_scan_cuda_core.fwd
selective_scan_cuda_core.bwd
```

不要在未建立独立数值对照的情况下替换CUDA kernel、启用AMP或修改force-fp32行为。
