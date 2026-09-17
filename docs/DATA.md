# Human3.6M data

Obtain Human3.6M under its own license. No dataset or raw video is redistributed.
Use H36M-SH detector xy/confidence, 17 joints, clips243, train stride81, sample stride1.
Training subjects: S1/S5/S6/S7/S8. Test subjects: S9/S11.

Expected layout under the root passed to `--data-root`:

```
h36m_sh_conf_cam_source_final.pkl
H36M-SH/train/*.pkl
H36M-SH/test/*.pkl
```

Reference metadata SHA256:
`73b642f2567a8d0b194f88c54a3182c7b635c003c832b48ae6ee559f10232175`.
Reference clip counts:17748 train,2228 test. A matching hash verifies file identity,
not legal access. Do not substitute another detector, joint order, clip boundary,
camera normalization or 2.5D factor without reporting a protocol change.

Raw videos alone are not sufficient. A complete raw-video-to-identical-preprocessing
pipeline is not currently provided. The data reader and dataset classes specify
the consumed schema; exact-score reproduction requires compatible prepared data.
GT2D uses the same metadata through its separate configuration.
