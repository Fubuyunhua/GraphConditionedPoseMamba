# W256最佳检测器权重对应的GT2D适配

已按2026-09-09最新用户指令停止，完成9/30，打断第10轮，exit143。
最后完整EMA19.968368/16.242261mm。用户拒绝GT预训练微调作为所需实验，
改为W128随机初始化80轮。以下为历史记录，不能标为完成GT实验。

2026-09-09 16:53:32北京时间正式启动，PID1991232，第1/30轮，4437批次/轮。

源：同宽stage1最佳EMA，原文件保留。GT微调前完整评估为
P1=31.847862889436183mm，P2=26.12329633203479mm。该GT已知xy协议与
检测器输入37.416mm不直接比较；不是微调取得的提升。

W256/D16，20,192,451参数全部训练，batch4，30轮，seed0，FP32。
峰值LR3e-5，3轮warmup从3e-6起，余弦至3e-6；DropPath.20；
AdamW重置，WD.012/SSM免衰减，clip1，EMA.9998从源权重开始、计数0。
原损失不变，gt_2d=true，训练/测试xy均来自真实标签，置信度1。

严格源/导出等值、GT坐标、全参数覆盖、重置、B1/B2/B4有限梯度及
EMA保存重载预测检查PASS。B4预检峰值21164MiB。权重/源/配置/数据
hash在PREFLIGHT_PASS.json中。只上传精简证据，不上传checkpoint或数据。

远程root /scratch/home/caiwei/GraphConditionedPoseMamba_W256_BEST_GT2D_20260909，
日志launch_logs/gt2d.log。源8be2ac0；定时监控仍关闭。
