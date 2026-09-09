# W128/D20 GT2D从头训练

2026-09-09 21:31:59北京时间正式启动，PID2004698，第1/80轮。
正式日志确认Fresh random initialization VERIFIED; no checkpoint loaded。
初始指纹cfe4a27267c7ea07ca2b3564a695d13fdd8fdbf6cd36a0bcc79131355fcb145a
与独立seed0构造器一致。未传--pretrained/--resume；finetune=false。

6,836,355参数全训练，batch4，4437批次/轮，80轮。初始LR5e-4，逐轮.99，
无warmup，DP.20，WD.012，clip1，EMA.9998，FP32。GTxy+c1输入检查PASS。
原数据集和已知xy评价协议保持，指标单独报告为GT；不承诺精度优于其它模型。
B1/B2/B4有限梯度、随机起点、优化器/EMA新状态及保存重载检查均PASS。

W256 GT微调已由用户停止9/30，不作正式完成结果。旧权重保留不覆盖。
GT微调计划已被用户否决；后续不得依据旧文档自动做GT迁移学习。
SAMA在独立目录恢复等待，只有所有GPU任务退出后可进入GPU预检。
日志launch_logs/gt_scratch.log，根目录
/scratch/home/caiwei/GraphConditionedPoseMamba_W128_GT2D_SCRATCH_20260909。
当前source220a479，定时监控仍关闭。
