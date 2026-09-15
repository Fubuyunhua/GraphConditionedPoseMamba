# GCS-Pose

**Graph-Conditioned Selective State Space Modeling for 3D Human Pose Estimation**

姿态特征生成递推内容 U 与输出门控 Z，骨架增强上下文生成选择参数 Δ/B/C，空间与时间分别递推。
代码类名保留 `GraphConditionedPoseMamba`，不破坏检查点兼容性。

完整安装、训练、评估、计算量命令见 [English README](README.md)，协议边界见
[复现说明](docs/GCS_POSE_REPRODUCTION.md)。

## 主模型与历史成绩

| 模型 | 参数量 | 检测 P1 / P2（mm） | 最佳 EMA 轮次 | 实际训练 |
|---|---:|---:|---:|---|
| W64D8 | 800,083 | 39.845 / 33.232 | 53 | 120 轮 |
| W128D10 | 3,435,395 | 38.484 / 32.383 | 54 | batch 4，80 轮 |
| W128D20 | 6,836,355 | 37.659 / 31.872 | 45 | 原始 S1，停止于 65/80 |

GT2D 最佳 P1 分别为 16.816、12.653、11.310 mm，均来自 80 轮从头训练。
当前 GT 评价器用输入 GT XY 替换预测 XY，属于已知 XY 的深度提升协议，不能与
检测结果或不同 GT 协议直接比较。以上为历史单种子监控测试集选出的最佳 EMA，
不是本次独立复评；W128D10 不混用旧 batch 8 版本。

## 复现

六份独立配置位于 `configs/gcs_pose/detector` 与 `configs/gcs_pose/gt2d`。
统一 80 轮、batch 4，保留各模型科学设置。80 轮是新复现预算，不冒充已完成结果，
也不保证未来最优值一定出现在预算内。

使用 Linux、NVIDIA GPU、匹配的 CUDA toolkit/nvcc 和 PyTorch/torchvision。
历史环境是 Python 3.10、PyTorch 2.11.0+cu128，不是完整依赖锁定。
先安装匹配的 CUDA PyTorch，再执行：

```bash
python -m pip install -r requirements.txt
bash scripts/build_selective_scan.sh
python scripts/verify_install.py

# 仅检查布局、生成配置和哈希，不训练；output 必须不存在。
python scripts/reproduce_gcspose.py --model w128d20 --protocol detector \
  --data-root /path/to/DATA_ROOT --output runs/check_w128d20 --seed 0

# 使用另一个新目录，显式开始训练。
CUDA_VISIBLE_DEVICES=0 python scripts/reproduce_gcspose.py \
  --model w128d20 --protocol detector --data-root /path/to/DATA_ROOT \
  --output runs/w128d20_detector_seed0 --seed 0 --run

python -X utf8 train.py --config runs/w128d20_detector_seed0/config.yaml \
  --evaluate /path/to/best_epoch_ema.bin --checkpoint runs/eval_w128d20 --seed 0
```

可选模型 `w64d8/w128d10/w128d20`；GT 改为 `--protocol gt2d`。
启动器不提供续训/预训练入口，但布局检查不替代数值预检。权重必须可信且配置匹配，
记录 SHA256、raw/EMA、真实 epoch。训练器会在 checkpoint 前缀后加时间戳。

Human3.6M 须依法取得。数据目录包含 `h36m_sh_conf_cam_source_final.pkl` 与
`H36M-SH/{train,test}/*.pkl`，详见 [DATA.md](DATA.md)。原始视频不能直接代替预处理文件。

## 计算量与未完成材料

三个模型每个 243 帧 clip 的修正 MAC-equivalent 分别为 4.602/18.739/37.474 G。
采用密集乘加加解析 scan FLOPs/2，不是精确硬件指令数，不能与漏算 functional/scan
的 THOP 直接比较。见 [全部模型效率](docs/ALL_MODELS_EFFICIENCY_20260915.md)。

尚缺公开检查点、完整环境锁定、从原始视频生成相同预处理的完整流程，以及干净环境
全程复现。Q/H 尚未通过原生 CUDA 预检，重复实验不因整理而宣称完成，MPI 指标需
独立协议审计。本轮不启动或停止训练，历史队列不是执行指令。

保留 LICENSE/NOTICE 及第三方署名。数据和第三方权重许可证另行适用。
作者、DOI 和正式发表信息待提供，不虚构引用。
