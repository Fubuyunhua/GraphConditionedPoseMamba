# W128D10检测器最佳轮次微调

2026-09-11 09:08:02北京时间启动，PID2058759，第1/8轮。
源是完整80e训练的第63轮最佳EMA；严格复评39.3835665/32.7786591mm。
源hash cb5a9153f473e9280ba95b78b5276f4d0645336ef9287eed6d8b0b424425ee59。

仅末2块和头训练，共680835参数；冻结前8块、输入/位置，冻结EMA不更新。
batch8，blockLR2e-6/head1e-5，一轮warmup然后余弦至.1倍，8轮。
EMA.99960004，WD.012，clip1，DP.20，原损失/检测器输入不变。
新优化器/EMA计数0，载入独立weights-only export，不resume旧优化器。

B1/B8及同编译模型尾批4有限检查通过；冻结参数逐项相等，可训练块/头
均确认更新，学习率比例5，EMA保存重载预测一致。预检峰值4090MiB。
本次必须比较39.3835665的初始基线；不能保证提升。
若8轮无收益，保留原始权重，不自动延长。

root /scratch/home/caiwei/GraphConditionedPoseMamba_W128_D10_BEST_FT_20260911，
log launch_logs/finetune.log。源192a5de。SAMA继续暂停，消融/D10GT仍待执行。
