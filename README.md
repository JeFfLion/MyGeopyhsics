# Kirchhoff PSTM 2D

> 2D Pre-Stack Time Migration using Kirchhoff Integral Method &middot; Multi-GPU Accelerated  
> 基于克希霍夫积分的二维叠前时间偏移 &middot; 多 GPU 加速

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-GPL--3.0-green)](./LICENSE)
[![Author](https://img.shields.io/badge/Author-LJF-lightgrey)](./LICENSE)

---

## Overview / 概述

**Kirchhoff PSTM 2D** is a production-grade 2D pre-stack time migration system that implements the Kirchhoff integral wavefield extrapolation method on multi-GPU using PyTorch. It reconstructs subsurface reflectivity by summing seismic traces along diffraction time curves with amplitude weighting and anti-aliasing filtering.

> **Author: LJF** — This software is released under GPL-3.0. Unauthorized commercial use is strictly prohibited.

**Kirchhoff PSTM 2D** 是一套工业级的二维炮域叠前时间偏移系统，基于克希霍夫积分波场延拓理论，利用 PyTorch 框架的 GPU 张量加速能力与 NCCL 多卡集合通信机制实现。

> **作者: LJF** — 本软件基于 GPL-3.0 协议发布，严禁未经授权的商业使用。

### Key Features / 核心特性

- **DSR traveltime equation** / 双平方根走时方程
- **Shot-domain migration** with geometric ray-tracing / 炮域偏移，几何射线追踪
- **Multi-GPU** via NCCL `all_reduce` / 多 GPU 并行归约
- **Per-shot memory release** protocol / 逐炮内存释放机制
- **OOM auto-degradation** (batch size halving) / 显存溢出自动降级
- **Watchdog timeout** & emergency cleanup chain / 看门狗超时与逃生链
- **SEGY I/O** via `segyio` / SEGY 格式读写

---

## Algorithm / 算法原理

### DSR Traveltime Equation / 双平方根走时方程

```
T = sqrt(t₀²/4 + (x_img − xs)² / Vrms²)  +  sqrt(t₀²/4 + (x_img − xr)² / Vrms²)
```

Where / 其中:
- `t₀` : two-way vertical time / 双程垂向时间
- `x_img` : imaging CMP position / 成像点 CMP 位置
- `xs`, `xr` : source & receiver positions / 炮点、检波点位置
- `Vrms` : RMS velocity / 均方根速度

### Amplitude Weighting / 振幅加权

```
W = depth / (Vrms · dx_src · dx_rec² + ε)

mig += W · (data[n] − data[n−1]) / dt / (2π)
```

### Anti-aliasing / 抗假频

Central-difference filter with configurable max frequency (`f_max`, default 125 Hz).

---

## Quick Demo / 快速演示

A fully self-contained demo is provided — no external data required. It generates synthetic shot gathers (3 flat reflectors + noise) and runs the full PSTM pipeline.

提供完全自包含的演示脚本，无需外部数据。生成合成炮集（3 个水平反射层 + 噪声）并运行完整 PSTM 流程。

```bash
# Single GPU demo / 单 GPU 演示
python demo.py

# Multi-GPU demo / 多 GPU 演示
python demo.py --n-gpus 2

# Large synthetic test / 大规模合成测试
python demo.py --n-shots 100 --n-gpus 4
```

Output: `demo_pstm_result.sgy` — a SEGY migrated section viewable in any seismic interpretation software.

输出：`demo_pstm_result.sgy` — 可在任何地震解释软件中查看的 SEGY 偏移剖面。

---

## Installation / 安装

### Prerequisites / 环境要求

- Python ≥ 3.10
- NVIDIA GPU with CUDA support (tested on RTX 4090)
- PyTorch ≥ 2.0 with CUDA

### Setup / 安装步骤

```bash
# 1. Create conda environment / 创建 conda 环境
conda create -n pstm python=3.10
conda activate pstm

# 2. Install PyTorch with CUDA / 安装 PyTorch（CUDA 版）
pip install torch

# 3. Install other dependencies / 安装其他依赖
pip install -r requirements.txt
```

---

## Usage with Real Data / 真实数据使用方法

```bash
# Quick validation / 快速验证
python run_pstm.py --shot-path ./your_shot.sgy --vel-path ./your_vel.sgy --test-shots 4

# Full run / 全量运行
python run_pstm.py \
    --shot-path ./shot_data.sgy \
    --vel-path ./vel_data.sgy \
    --output ./pstm_result.sgy \
    --n-gpus 4
```

### CLI Arguments / 命令行参数

| Argument | Default | Description |
|----------|---------|-------------|
| `--shot-path` | `shot_data.sgy` | Input shot gathers / 输入炮集 |
| `--vel-path` | `vel_data.sgy` | Input RMS velocity / 输入速度场 |
| `--output` | `pstm_result.sgy` | Output migrated section / 偏移输出 |
| `--n-gpus` | `4` | Number of GPUs / GPU 数量 |
| `--max-aperture` | `3000` | Migration aperture (m) / 偏移孔径 |
| `--rec-batch-size` | `64` | Receiver batch size (GPU memory) / 接收道批大小 |
| `--src-depth` | `9.0` | Source depth (m) / 炮点深度 |
| `--rec-depth` | `10.0` | Receiver depth (m) / 检波点深度 |
| `--cmp-spacing` | `25.0` | CMP output spacing (m) / CMP 间距 |
| `--f-max` | `125.0` | Anti-aliasing max frequency (Hz) / 抗假频最大频率 |
| `--test-shots` | — | Process first N shots only / 仅处理前 N 炮 |
| `--log-level` | `INFO` | `DEBUG` `INFO` `WARNING` `ERROR` |

---

## Architecture / 架构

```
kirchhoff-pstm-2d/
├── pstm_2d/
│   ├── __init__.py        Package entry / 包入口
│   ├── engine.py          Top-level PSTM2DEngine / 顶层调度引擎
│   ├── geometry.py        SEGY header parsing & CMP grid / 几何解析
│   ├── traveltime.py      DSR kernel & amplitude weighting / 走时核
│   ├── worker.py          Single-GPU migration worker / 单 GPU Worker
│   ├── scheduler.py       Multi-GPU scheduler + watchdog / 多 GPU 调度器
│   ├── aggregator.py      Result accumulation & SEGY output / 结果归约
│   └── safety.py          Circuit breaker & emergency cleanup / 安全熔断
├── run_pstm.py            CLI entry for real data / 真实数据 CLI 入口
├── demo.py                Self-contained synthetic demo / 合成数据演示
├── requirements.txt       Python dependencies / 依赖清单
├── LICENSE                GPL-3.0 License / 许可协议
└── docs/
    └── 实验报告_PSTM.md     Full experimental report / 完整实验报告
```

### Pipeline / 流水线

```
Phase 1: SEGY header parsing → Geometry, statics, velocity loading
Phase 2: Shot chunks distributed across N GPUs → NCCL all_reduce
Phase 3: Fold-map normalization → SEGY output
```

### GPU Chunk Strategy / GPU 块分配策略

Shots are partitioned with configurable overlap halos to avoid edge artifacts at chunk boundaries. Each GPU independently reads and migrates its assigned chunk, then results are merged via NCCL `all_reduce`.

炮集按可配置的重叠区（halo）划分给各 GPU，避免块边界处的边缘效应。各 GPU 独立读取并偏移其分配的炮块，结果通过 NCCL `all_reduce` 归约合并。

---

## License / 许可协议

**Copyright (c) LJF. All Rights Reserved.**

This project is licensed under the **GNU General Public License v3.0** — see [LICENSE](./LICENSE) for details.

**商业使用警告 / Commercial Use Warning:**

This software is provided for research and educational purposes. Unauthorized commercial deployment without explicit written permission from the author (LJF) is a violation of both the license terms and the author's intellectual property rights. The LJF signature is embedded throughout the codebase as a safeguard.

本软件仅供研究和教育用途。未经作者 (LJF) 明确书面授权的商业部署，既违反许可协议条款，也侵犯作者知识产权。代码中已嵌入 LJF 签名作为防护标识。

---

## References / 参考文献

- Schneider, W. A. (1978). Integral formulation for migration in two and three dimensions. *Geophysics*, 43(1), 49–76.
- French, W. S. (1975). Computer migration of oblique seismic reflection profiles. *Geophysics*, 40(6), 961–980.
- [docs/实验报告_PSTM.md](./docs/实验报告_PSTM.md) — Full experimental report (Chinese) / 完整实验报告
