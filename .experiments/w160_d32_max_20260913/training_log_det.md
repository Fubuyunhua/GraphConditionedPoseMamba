# W160/D32 H36M检测器输入训练

2026-09-13 05:02:03北京时间正式启动，PID2193893，第1/80轮。
16,534,531参数，batch4，4437batches/epoch，总354960更新，seed0从头训练。
随机初始hash54e3a6df9785667cf559baba3b103d2ab927e9d7d253f110c7ddb2247fcc8683通过，未载入checkpoint。
LR3e-5真实warmup8轮至3e-4，随后cosine至3e-5；EMA.9998，DP.20，clip1，
WD.012且652800个SSM参数免衰减。FP32，逐块非重入activation checkpoint。

训练DropPath下开启/关闭checkpointing前向和梯度一致性PASS。
真实B1/B2/B4有限梯度、graph路径可达、EMA保存重载一致性PASS。
B4预检峰值11612MiB，第一步LR符合3e-5起步。编译后的实际整轮速度待观察。
数据哈希73b642f...，完整配置/源/初始hash见PREFLIGHT_PASS。

root /scratch/home/caiwei/GraphConditionedPoseMamba_W160_D32_DET_20260913，
log launch_logs/train.log。源88d6f59。SAMA保持暂停，GT最大版尚未启动。
固定80轮，除失败或明确用户指令不提前停止，不自动扩展预算。
