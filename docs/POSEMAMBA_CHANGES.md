# 相对 PoseMamba 的结构修改

## 1. 原问题

输入为 `X ∈ R^{B×T×J×C}`。原PoseMamba的spatial/temporal block都使用完整`T×J`
CrossScan，并用joint reorder后相加、二维DWConv和flatten sequence。这可能产生：

- `joint16(frame t) -> joint0(frame t+1)`；
- `last frame(joint j) -> first frame(joint j+1)`；
- tensor joint index邻近被误当作真实骨骼邻近。

## 2. Graph mixer

Human3.6M bone edges：

```text
(0,1)(1,2)(2,3)  (0,4)(4,5)(5,6)
(0,7)(7,8)(8,9)(9,10)
(8,11)(11,12)(12,13)  (8,14)(14,15)(15,16)
```

Symmetry edges：

```text
(1,4)(2,5)(3,6)(11,14)(12,15)(13,16)
```

对`j -> i`：

```text
r_ij = x_j - x_i
m_ij = W_relation(r_ij) + W_neighbor(x_j)
g_i = alpha_bone * sum_bone(m_ij) + alpha_sym * sum_sym(m_ij)
G_i = W_out(GELU(g_i))
```

`alpha_bone/alpha_sym`从1初始化。图固定，不使用dynamic fully-connected attention。

## 3. Spatial state reset

```text
[B,T,J,C] -> [B*T,1,J,C]
```

每个frame成为selective-scan独立batch item：

```text
j0 -> ... -> j16
j16 -> ... -> j0
```

因此frame间不会共享state。GraphMixer负责局部骨骼，Spatial BiSSM负责单帧全局joint
context，`d_conv=1`。

## 4. Temporal state reset

```text
[B,T,J,C] -> [B*J,1,T,C]
```

每个joint trajectory成为独立batch item：

```text
t0 -> ... -> t242
t242 -> ... -> t0
```

因此joint间不会共享temporal state。时间局部邻域真实存在，使用Conv1D `d_conv=3`。

## 5. Graph-conditioned selective dynamics

```text
u = encode(x)
context = x + graph_scale * G(x)
(Delta, B, C) = projection(encode(context))
y = selective_scan(u, Delta, A, B, C, D)
```

内容`u`仍来自原feature；graph context只决定状态如何写入、遗忘和读取。输入和context共享
同一input projection权重，没有第二套完整embedding参数。CUDA kernel接口和数值实现未改。

## 6. Block

```text
z_s = LN_s(x) + E_joint
G_s = GraphMixer(z_s)
x   = x + gamma_s * SpatialBiSSM(z_s, z_s + G_s)

z_t = LN_t(x) + E_time
x   = x + gamma_t * TemporalBiSSM(z_t, z_t + reused_G_s)
x   = x + MLP(LN_mlp(x))
```

`gamma_s=gamma_t=1`。无zero-init、sigmoid gate、frequency/coarse branch。

## 7. K=2而不是伪K=4

factorized scan的参数group为2：一个forward、一个backward。早期wrapper为适配SS2D分配4组，
实际重复了两个方向。严格K2同时更符合定义并显著减少参数/计算。

## 8. 位置编码

- joint embedding：`[1,J,C]`，只在spatial阶段使用；
- time embedding：`[1,T,C]`，只在temporal阶段使用。

不再混成单一`T×J`位置表示。
