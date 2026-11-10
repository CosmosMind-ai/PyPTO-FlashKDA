# PyPTO-FlashKDA

[English](README.md) | **中文**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Platform](https://img.shields.io/badge/Platform-Ascend%20910C-blue)
![CANN](https://img.shields.io/badge/CANN-%E2%89%A5%209.0-orange)

> 本仓库属于 **CosmosMind PyPTO 算子系列**（面向昇腾），全部内核由 [MetaRSI](https://www.cosmosmind.ai/research/metarsi-v1) 驱动。

面向 **昇腾 910C** 的 Kimi Delta Attention（KDA）内核库，基于
[PyPTO](https://github.com/hw-native-sys/pypto) 编写——是
[MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA)（CUDA/CUTLASS，SM90+）
在昇腾生态的对应实现。

KDA 是 [Kimi Linear](https://arxiv.org/abs/2510.26692)（式 1）的核心递推算子。
本库提供昇腾 910C 上的前向与反向实现，每次发布均通过密封正确性 holdout 门禁。

## 状态

v0.1.0 —— 递推（recurrent）前向与完整反向已通过全部算子契约
（20/20 计分用例及边界语义套件，见 `tests/`）。面向长序列 prefill 的
独立分块并行（chunked）前向路径已在路线图中。

## 亮点

- **快过官方 PyPTO 实现。** 与 pypto-lib 仓库内置的 GLM KDA 内核
  （`models/glm5_3_flash/kda.py`，上游 main `a18654d`）同卡同协议对拍：
  **decode 4.16×、prefill 1.16×**——且 prefill 数字是我们的 *recurrent* 路径
  对官方 *chunked* 路径。
- **逼近手写 Ascend C。** 在 20 用例 KDA 算子契约套件上，相对官方 Ascend C 实现的
  几何平均延迟比为 **0.878×**，仍在持续优化。
- **前向 + 反向。** 完整梯度（`dq, dk, dv, dg, dbeta, dinitial_state`），
  经 FP64 gradcheck 验证的 autograd oracle 校验——这是昇腾 PyPTO 生态的首个
  KDA 反向实现。
- **实测密封证据。** 下列每个数字都来自通过密封 holdout（精度 + 延迟双门禁）的
  构建版本；原始测量记录随库发布于 `benchmarks/evidence/`。

## 性能

昇腾 910C（`a2a3`），驱动 26.0.rc1，median effective latency，100 轮 / 5 warmup。
完整协议与逐用例表格见 `benchmarks/BENCHMARKS.md`。

| 组件 | 延迟 | 参照 |
|---|---|---|
| 前向（20 用例契约套件） | 52.21 µs | 相对同 DSL 未融合分解 +67.8%；对 Ascend C 几何平均 0.878× |
| 前向（GLM decode 形状） | 23.35 µs vs 官方 97.2 µs | **4.16×** |
| 前向（GLM prefill，211 tokens） | 292.0 µs vs 官方 337.6 µs | **1.16×** |
| 反向 | 85.12 µs | 相对同 DSL 未融合分解 +90.4% |

## 安装

```bash
pip install -e .
```

环境要求：昇腾 910C（`a2a3`），CANN ≥ 9.0，`pypto`、`pypto-lib`（其 `golden`
包支撑经验证的启动路径）、`torch`。

## 快速上手

```python
import torch
import flash_kda

B, T, H, Dk, Dv = 1, 64, 8, 128, 128
q = torch.randn(B, T, H, Dk, dtype=torch.bfloat16)
k = torch.randn(B, T, H, Dk, dtype=torch.bfloat16)
v = torch.randn(B, T, H, Dv, dtype=torch.bfloat16) * 0.5
g = -torch.rand(B, T, H, Dk) * 5.0      # 逐 token 对数衰减，fp32
beta = torch.rand(B, T, H)               # 已激活的写入强度，fp32
s0 = torch.zeros(B, H, Dk, Dv)           # 零张量 = 无历史状态

y, final_state = flash_kda.fwd(q, k, v, g, beta, s0, mode="auto")
# y: [B,T,H,Dv]（dtype 同 q）；final_state: [B,H,Dk,Dv] fp32

dy = torch.randn_like(y.float())
dfs = torch.randn(B, H, Dk, Dv)
dq, dk, dv, dg, dbeta, ds0 = flash_kda.bwd(dy, dfs, q, k, v, g, beta, s0)
```

## 测试

```bash
python tests/test_fwd_contract.py --device 0   # 20 用例契约 vs torch oracle
python tests/test_bwd.py --device 0            # FP64 gradcheck oracle + 上卡比对
```

## 路线图

- [ ] 独立 chunked 前向路径（长序列 prefill 的主要杠杆），
      含 chunk/recurrent 数值等价与分段连续性测试套件
- [ ] 变长（`cu_seqlens`）支持——接入 fla 生态调度的前提
- [ ] 零开销直接启动器（当前 API 经带校验的 `golden` harness 启动，见 `flash_kda/api.py`）
- [ ] 向 `fla.ops.kda` 提调度 PR（fla-org/flash-linear-attention 自动分发）
- [ ] 对 Ascend C 几何平均 ≥ 1.0×（当前 0.878×，优化进行中）

## 致谢

算子契约与 20 用例评测套件与官方 Ascend C 版 KimiDeltaAttention 实现对齐；
数学定义来自 Kimi Linear（arXiv 2510.26692）式 1。接口语义与 MoonshotAI/FlashKDA
对齐，fla 生态调度可无缝路由到任一后端。

## 许可证

MIT，详见 `LICENSE`。
