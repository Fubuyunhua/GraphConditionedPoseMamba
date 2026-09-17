# GCS-Pose

**Graph-Conditioned Selective State Space Modeling for 3D Human Pose Estimation**

视频二维关键点到三维人体姿态的训练与推理实现。姿态特征生成 U/Z，骨架增强上下文
生成 Δ/B/C，空间与时间分别递推。代码类名保留 `GraphConditionedPoseMamba`。

**与 PoseMamba 的关系：**本实现建立在
[PoseMamba](https://github.com/nankingjing/PoseMamba) 的开源代码基础上，并非全部从零编写。
GCS-Pose 是独立的研究扩展，不是 PoseMamba 官方版本。来源与贡献边界详见
[第三方说明](docs/THIRD_PARTY_NOTICES.md)。

公开仓库只保留复现所需代码、配置、模型说明和权重发布信息；完整实验日志、内部
工作流和交接材料不作为公开内容。详细命令见 [英文 README](README.md)。

## 复现步骤

1. Linux 环境安装兼容的 CUDA PyTorch/torchvision，参考环境为 Python3.10、PyTorch2.11.0+cu128。
2. 安装 `requirements.txt`，运行 `bash scripts/build_selective_scan.sh` 编译扩展。
3. 运行 `python scripts/verify_install.py` 检查 CUDA 前向、反向及参数量。
4. 根据 [数据说明](docs/DATA.md) 准备合法取得的 Human3.6M 预处理数据。
5. 用 `scripts/reproduce_gcspose.py` 生成配置；加 `--run` 才开始从头训练。
6. 下载并核验公开权重后，用 `train.py --evaluate` 评估，配置必须与权重一致。

模型选项为 `w64d8/w128d10/w128d20`，输入协议为 `detector/gt2d`，公开训练配置
统一80轮、batch4。历史权重的实际训练预算和最佳轮次见模型卡，不把新配方当作新结果。

| 模型 | 参数量 | Detector P1 / P2（mm） |
|---|---:|---:|
| W64D8 | 800,083 | 39.845 / 33.232 |
| W128D10 | 3,435,395 | 38.484 / 32.383 |
| W128D20 | 6,836,355 | 37.659 / 31.872 |

GT2D 与 Detector 不能混用。当前 GT 评价保留输入 GT XY，属于已知 XY 的深度提升
协议；详细边界见 [模型卡](docs/MODEL_CARD.md)。[权重发布状态](docs/WEIGHTS.md)。

## 致谢与引用

感谢 [PoseMamba](https://github.com/nankingjing/PoseMamba) 和
[Mamba](https://github.com/state-spaces/mamba) 作者开放源码。
本项目继承姿态提升代码基础及 selective-scan 实现；GCS-Pose 实现的扩展包括分解的
空间/时间递推，以及由骨架增强上下文生成 Δ/B/C、由姿态特征生成 U/Z 的图条件化设计。
基础 Mamba 公式、双向扫描及继承的数据和训练工具不作为本项目原创贡献。
致谢不表示上游作者认可或背书本项目。

使用本项目时，请同时引用相关基础工作，特别是
[PoseMamba（AAAI 2025）](https://ojs.aaai.org/index.php/AAAI/article/view/32401)，
其正式引用信息见 [原仓库 Citation](https://github.com/nankingjing/PoseMamba#citation)。
请保留 [LICENSE](LICENSE)、[NOTICE](NOTICE) 和源文件版权声明。
GCS-Pose 的作者列表、DOI 和正式出版信息待提供，不虚构发表信息。
