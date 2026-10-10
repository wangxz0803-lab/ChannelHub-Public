# 原始观测数据接口 v1

`observations.npz` 是无 pickle 的 NumPy 压缩归档，所有复数数组均为 `complex128`。标识为 `channelhub-observation-v1`。

| 字段 | 维度 | 单位 / 解释 |
|---|---|---|
| `desired_h` | `[T, R, P, K]` | 无量纲频域信道 |
| `desired_cir` | `[T, R, P, L]` | 离散路径系数，无量纲 |
| `desired_delays_s` | `[R, P, L]` | 路径时延，秒 |
| `desired_x` | `[T, P, K]` | 发射幅度，sqrt(W) / RE |
| `desired_active_re` | `[T, K]` | 至少一个端口非零的发射掩码 |
| `interferer_j_h / cir / delays_s / x / active_re` | 同上，端口数可不同 | 第 j 个干扰源，j 从 0 开始 |
| `signal` | `[T, R, K]` | 有用接收分量，sqrt(W) / RE |
| `interference` | `[T, R, K]` | 所有干扰源接收分量之和 |
| `noise` | `[T, R, K]` | 独立复高斯噪声 |
| `y` | `[T, R, K]` | 未经处理的总接收观测 |
| `frequency_offset_hz` | `[K]` | 相对于载频的基带频偏 |
| `time_s` | `[T]` | 时间坐标，起点 0 |
| `receiver_position_m` | `[T, 3]` | 配置的匀速接收轨迹坐标 |

`T=14*num_slots`，`K=12*num_rb`；`R` 为接收端物理天线端口数，`P` 为对应发射端口数，`L` 为路径数。不存在的 RT 路径以负时延和零有效贡献表示，不能把负时延当作物理路径。

## 方程

```text
H[t,r,p,k] = sum_l a[t,r,p,l] * exp(-j*2*pi*f[k]*tau[r,p,l])
signal[t,r,k] = sum_p H[t,r,p,k] * X[t,p,k]
interference[t,r,k] = sum_j sum_p H_j[t,r,p,k] * X_j[t,p,k]
Y = signal + interference + noise
```

矩阵乘法没有隐含共轭。所有链路在同一资源网格上计算。`frequency_offset_hz` 从 `-K/2*SCS` 到 `(K/2-1)*SCS`，中心包含一个 0 Hz 频点；不另行插入保护子载波或 DC 空洞。

生成式波形先将每个活跃 RE 上的跨端口总功率归一为 1，再乘以 `sqrt(10**((power_dbm-30)/10))`。自定义 X 不进行此步骤，`power_dbm` 和 `activity` 对自定义 X 不生效；其占用与功率完全由矩阵给定。

复噪声满足 `E[|N|²] = 10**((noise_psd_dbm_hz + noise_figure_db - 30)/10) * SCS_Hz`，实部和虚部各占一半方差。随机数流在期望链路、各干扰源、发射信号和噪声之间隔离；增加干扰源不会重抽期望信道。

## 物理语义

CDL/TDL 使用 Sionna 的单位平均功率小尺度信道，再施加显式路径损耗。不按每个随机样本的瞬时能量重新归一化。统计模型的时延是 excess delay，不包含端点间飞行时间；不要用它当作绝对测距真值。

RT 保留实际路径增益和绝对时延。路径系数已经是载频处的复基带系数，转换到子载波时只乘基带频偏项，避免重复施加载频相位。

SSB 的 240 子载波参考块放在载波中心，PSS 位于相对符号 0、SSS 位于相对符号 2、PBCH-DMRS 位于相对符号 1/2/3。PBCH 载荷位置为零。CSI-RS 单逻辑端口经单位范数 DFT 权值映射到物理发射端口。端口顺序与上游 Sionna 阵列一致；TDL 仅把端口当作独立通道。

结果中的 `signal`、`interference`、`noise`、真实 H 和路径是**仿真真值**。算法是否可以读取这些字段应在实验协议中声明；常见接收算法只读取 Y、已知导频 X 和资源位置，以真值作离线对照。

## 可复现性

`metadata.json` 保存完整解析配置、配置哈希、ChannelHub 与引擎版本、数组维度、单位及 `observations.npz` 的 SHA-256。自定义 X 记录连续复数数组字节的哈希。路径、日志、机器用户名和凭据不是导出协议字段。

相同依赖、设备和配置下可重复生成；不承诺不同 GPU、驱动、浮点后端或上游版本之间逐比特相同。RT 自定义场景应连同引用的网格自行版本化。
