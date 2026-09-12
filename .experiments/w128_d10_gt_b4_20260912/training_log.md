# W128/D10 GT2D batch4 从头训练

2026-09-12 13:58:31北京时间启动，PID2118071，80轮，batch4，3,435,395参数。
随机seed0，未加载任何checkpoint；正式初始hash7506c395336ddd719843a4709c1b6b1ec29dfbaab31b493dbbc70d7c18f8e49b匹配。
LR5e-4逐轮.99，WD.012，EMA.9998，DP.20，无warmup/裁剪，匹配最新D10检测配方。
GTxy+c1、有限梯度、图梯度可达、B1/B2/B4、EMA保存重载均PASS，B4预检峰值7448MiB。

|Epoch|EMA P1 mm|Paired P2 mm|Train minutes|
|---|---:|---:|---:|
|1|147.418176|103.697579|8.04|
|2|117.156634|89.767215|7.94|

第3轮运行中。初期EMA尚未收敛，不能判定最终性能。进程显存8152MiB，
另一个进程3998MiB；总体利用率快照98%。完整一轮约8.6分钟，共享条件可变化。
源a94da51，日志launch_logs/gt_scratch.log，根目录
/scratch/home/caiwei/GraphConditionedPoseMamba_W128_D10_GT_B4_20260912。
SAMA仍暂停，未重启定时监控。GT评价使用已知xy，成绩单独报告。
