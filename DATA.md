# Human3.6M data

数据不包含在仓库中。默认配置读取：

```text
data/motion3d/MB3D_f243s81/h36m_sh_conf_cam_source_final.pkl
```

该pickle需与原PoseMamba/ReliPose Human3.6M-SH预处理协议一致，提供243帧clip、17 joints、
xy+confidence输入和3D ground truth。默认配置：

```yaml
clip_len: 243
data_stride: 81
sample_stride: 1
subset_list: [H36M-SH]
no_conf: false
```

仓库不提供Human3.6M原始数据或派生文件。使用者必须自行获得合法授权并按项目预处理格式准备。

不要提交以下内容：

```text
data/
runs/
checkpoint*/
*.bin
*.pth
*.pt
*.ckpt
```
