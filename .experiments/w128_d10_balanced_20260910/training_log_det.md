# W128/D10 H36M检测器输入实验

2026-09-10 10:12:16北京时间正式启动，PID2027023，第1/80轮。
3,435,395参数，batch8，2219批次/轮，随机seed0；正式初始化指纹
7506c395336ddd719843a4709c1b6b1ec29dfbaab31b493dbbc70d7c18f8e49b校验通过。
gt_2d=false，没有预训练或resume。LR5e-4逐轮.99，无warmup，DP.20，
WD.012，clip1，EMA.99960004。

B1/B2/B4/B8与同一编译模型末批4检查均PASS，预检峰值13760MiB。
初始正式快照：D10进程13540MiB，D20 GT进程15374MiB，整卡28931MiB/32607，
95%利用率；D10约2.33batch/s，只有短窗口，完整轮时长待观察。
共享GPU吞吐不能作论文效率证据。D20未停止，SAMA继续等待。

用户纠正前D10 GT曾短暂进入第1轮，PID2025903已停止，0完整轮次；
不计作有效结果，不再启动GT版本。检测器实验在新独立目录。
root /scratch/home/caiwei/GraphConditionedPoseMamba_W128_D10_DET_PARALLEL_20260910，
log launch_logs/det_parallel.log。源9bc4642；定时任务仍关闭。
