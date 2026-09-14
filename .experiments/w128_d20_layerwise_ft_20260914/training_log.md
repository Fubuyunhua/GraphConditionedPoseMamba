# W128/D20全参数分层微调

2026-09-14 11:23:42北京时间正式启动，PID2339948，第1/8轮。
旧最佳EMA第45轮独立复评37.659287/31.871699mm，原文件保留。
全部6,836,355参数参与训练：input33792个参数LR1e-6；前10块3400960个LR1e-6；
后10块3400960个LR3e-6；head643个LR1e-5。一轮warmup后余弦至各自峰值.1倍。
batch4、EMA.9998、新AdamW、WD.012、clip1、DP.20、FP32、原损失；无锚定项。

单元测试和全20块/位置/输入/头更新检查通过；B1/B2/B4有限梯度、重置及
保存重载PASS。B4预检峰值14668MiB；运行快照进程14936MiB，GPU86%。
每轮4437batch，当前短窗口4.92batch/s；完整轮耗时需首轮完成后确认。
根目录 /scratch/home/caiwei/GraphConditionedPoseMamba_W128_D20_LAYERWISE_FT_20260914，
log launch_logs/finetune.log。源3256b63。比较保留第0轮权重，不保证精度提升。
SAMA暂停，不自动扩展微调或恢复已取消实验。
