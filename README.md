# ChannelHub · University Research Edition

面向高校教学和非商业科研的信道与干扰实验平台。配置场景、收发阵列、标准导频和干扰源，生成原始信道 **H / CIR**、发射网格 **X** 和未处理接收观测 **Y**，在自己的算法环境中完成验证。

```text
场景 / 阵列 / 波形 / 干扰配置
               ↓
       CDL / TDL 或 Sionna RT
               ↓
      Y = H X + Σ Hᵢ Xᵢ + N
               ↓
    原始数组 + 完整配置 + 版本信息
```

**使用许可：仅高校教学和非商业科研。** 可内部修改；公开修改版或其他再分发须另行取得书面许可。详细范围与 GitHub 平台权利例外见 [LICENSE](LICENSE)。这是公开可读的科研源码许可，不是 MIT / Apache 或 OSI 开源许可。

## 功能

- **统计信道**：官方 Sionna PHY 2.2.0 的 CDL-A～E、TDL-A～E，支持多天线、时延扩展与多普勒。CPU 可运行。
- **射线追踪**：官方 Sionna RT 2.2.0，自由空间、合成地面反射场景，以及命令行载入自定义 XML 场景。
- **发射侧**：SRS Comb-2/4、周期/跳频/多端口配置；单端口 CSI-RS row 2；SSB 的 PSS、SSS 和 PBCH-DMRS；满网格 QPSK；自定义复数 X。
- **通用阵列**：行列数、单/双极化、间距、方位、全向或 3GPP 方向图；公开公式生成的 DFT 波束。
- **干扰与噪声**：最多 8 个独立干扰源，可配置位置、功率、活跃概率和资源；复高斯白噪声。
- **网页平台**：实验配置、二维拓扑预览、任务队列、取消、结果预览、配置导入导出和实验包下载。
- **接口边界**：止于前向传播与接收观测。高校可自行实现信道估计、测量提取、特征工程、训练和评估。

## 安装和启动

需要 Python 3.11 或更新版本。推荐先在虚拟环境中安装。

```bash
git clone https://github.com/wangxz0803-lab/ChannelHub-Public.git
cd ChannelHub-Public
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[statistical,rt]"
.\.venv\Scripts\channelhub.exe serve
```

Linux / macOS：

```bash
source .venv/bin/activate
python -m pip install -e ".[statistical,rt]"
channelhub serve
```

打开 <http://127.0.0.1:5088>。网页静态资源随 Python 包分发，无需 Node.js、前端构建、数据库或 Redis。

仅统计信道使用 `.[statistical]`；仅射线追踪使用 `.[rt]`。Sionna PHY 的上游依赖也可能安装 RT 包，但统计信道不需要可用的 RT 原生后端。RT 需要 Mitsuba/Dr.Jit 支持的 CUDA 或 LLVM 运行环境；原生后端不可用时任务会明确失败，不会替换成另一种模型。环境要求见 [Sionna 官方安装说明](https://nvlabs.github.io/sionna/installation.html)。

服务默认仅监听本机。它是本地实验工具，没有多用户身份认证；部署到共享服务器时应在前方配置认证和访问控制。

## 最小命令行实验

```bash
channelhub generate --config examples/statistical-srs.json --output outputs/first-run
channelhub generate --config examples/rt-reflector.json --output outputs/rt-run
channelhub generate --config examples/typical-mimo-srs.json --output outputs/typical-run
```

结果目录包含 `observations.npz` 和 `metadata.json`；重复写入同一结果目录会报错。随机种子、完整配置及依赖版本记录在元数据中。

```python
import numpy as np

with np.load("outputs/first-run/observations.npz", allow_pickle=False) as data:
    h = data["desired_h"]  # [time, rx_port, tx_port, subcarrier]
    x = data["desired_x"]  # [time, tx_port, subcarrier]
    y = data["y"]  # [time, rx_port, subcarrier]
    reference = np.einsum("trpk,tpk->trk", h, x)
    np.testing.assert_allclose(y, reference + data["interference"] + data["noise"])
    # 将 x、y 及资源位置送入自己的接收算法。
    # h 和分量真值用于评价，是否作为算法输入由实验协议决定。
```

完整字段、功率定义和模型边界见 [数据接口](docs/DATA_CONTRACT.md)。

`typical-mimo-srs.json` 是已实采验证的 2 发 4 收、移动接收端、双干扰源示例。场景参数、独立数值核验、发现的问题与修复见 [v0.1.1 验收记录](docs/VALIDATION.md)。

## 自定义发射信号

将 `tx.waveform.kind` 设为 `custom`，提供 `[14 × num_slots, tx_ports, 12 × num_rb]` 的复数 `.npy` 文件：

```bash
python examples/make_custom_x.py
channelhub generate --config examples/custom-x.json --x outputs/custom-x.npy --output outputs/custom-run
```

X 的单位是每个 RE 的 `sqrt(W)`，会原样使用，不再按 `power_dbm` 归一化或缩放。网页也支持上传同样的 `.npy` 文件，最大 32 MiB。自定义干扰发射矩阵可通过 Python 的 `channelhub.signals.apply_channel()` 直接组合。

自定义 RT 场景：

```bash
channelhub generate --config examples/rt-reflector.json --scene /path/to/scene.xml --output outputs/my-scene
```

场景文件及其引用的网格由使用者提供；包中只带一个自行构造的平面反射场景。每份结果记录用户 XML 文件的哈希；复现实验时还需保留 XML 引用的外部资源。

## 首版明确边界

- 频域逐 RE 前向模型；符号时间采用 normal-CP 时隙平均间隔。不模拟时域 OFDM、CP 失配、ICI、定时同步、解码或接收机处理。
- 统计链路独立生成，没有多用户空间一致性。初始距离用于自由空间路径损耗，也可显式设置 `pathloss_db`；这不是 UMa/UMi 系统级路径损耗模型。运动用于多普勒，不逐时刻更新大尺度路径损耗。
- TDL 默认天线间独立，只使用天线端口数量；方向图、阵列间距与方位仅用于 CDL 和 RT。
- RT 在短观测窗口内固定几何路径，用路径多普勒演进相位，不在每个符号重新追踪移动几何。所有 RT 发射源共用同一种阵列配置。
- CSI-RS 仅实现标准单端口 row 2、density=1、noCDM；SSB 仅包含 PSS/SSS/PBCH-DMRS，按每个配置时隙重复参考块，不实现完整 SSB burst 调度或 PBCH 编码。
- 生成式波形的功率是**每个活跃 RE 上跨发射端口的总功率**，不是全载波发射总功率。
- 默认最大 12,000,000 个频域信道系数；任务串行运行，最多排队 8 个。更大实验可在 Python 层按批次组织。

## 开发与验证

```bash
python -m pip install -e ".[statistical,rt,dev]"
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

有可用 RT 环境时运行原生物理验证：

```powershell
$env:CHANNELHUB_TEST_RT="1"
python -m pytest -q
```

Linux 对应 `CHANNELHUB_TEST_RT=1 python -m pytest -q`。RT 测试以自由空间 Friis 幅度、传播时间与反射路径为独立参照。其他测试覆盖全部 10 个 CDL/TDL profile、随机种子复现、功率缩放、导频映射、叠加恒等式和网页任务下载。

依赖来源见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。使用范围及额外许可请求见 [LICENSE](LICENSE)。
