# Kirchhoff PSTM 2D

> 2D Pre-Stack Time Migration using Kirchhoff Integral Method &middot; Multi-GPU Accelerated  
> 基于克希霍夫积分的二维叠前时间偏移 &middot; 多 GPU 加速

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-GPL--3.0-green)](./LICENSE)

---

## Overview / 概述

**Kirchhoff PSTM 2D** is a production-grade 2D pre-stack time migration system that implements the Kirchhoff integral wavefield extrapolation method on multi-GPU using PyTorch. It reconstructs subsurface reflectivity by summing seismic traces along diffraction time curves with amplitude weighting and anti-aliasing filtering.

**Kirchhoff PSTM 2D** 是一套工业级的二维炮域叠前时间偏移系统，基于克希霍夫积分波场延拓理论，利用 PyTorch 框架的 GPU 张量加速能力与 NCCL 多卡集合通信机制实现。

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
T = sqrt(t0²/4 + (x-img − xs)² / Vrms²)  +  sqrt(t0²/4 + (x-img − xr)² / Vrms²)
```

Where / 其中:
- `t0` : two-way vertical time / 双程垂向时间
- `x-img` : imaging CMP position / 成像点 CMP 位置
- `xs`, `xr` : source & receiver positions / 炮点、检波点位置
- `Vrms` : RMS velocity / 均方根速度

### Amplitude Weighting / 振幅加权

```
W = depth / (Vrms · dx_src · dx_rec² + ε)

mig += W · (data[n] − data[n-1]) / dt / (2π)
```

### Anti-aliasing / 抗假频

Central-difference filter with configurable max frequency (`f_max`, default 125 Hz).

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

## Usage / 使用方法

```bash
# Quick validation (4 shots only) / 快速验证（仅4炮）
python run_pstm.py --test-shots 4 --log-level DEBUG

# Full run (601 shots, 4 GPUs) / 全量运行（601炮，4 GPU）
python run_pstm.py --log-level INFO

# Custom parameters / 自定义参数
python run_pstm.py \
    --shot-path ./NH1_Shot.sgy \
    --vel-path ./NH1_Vel.sgy \
    --output ./pstm_result.sgy \
    --n-gpus 4 \
    --max-aperture 3000 \
    --rec-batch-size 64
```

### CLI Arguments / 命令行参数

| Argument | Default | Description |
|----------|---------|-------------|
| `--shot-path` | `NH1_Shot.sgy` | Input shot gathers / 输入炮集 |
| `--vel-path` | `NH1_Vel.sgy` | Input RMS velocity / 输入速度场 |
| `--output` | `pstm_result.sgy` | Output migrated section / 偏移输出 |
| `--n-gpus` | `4` | Number of GPUs / GPU 数量 |
| `--max-aperture` | `3000` | Migration aperture (m) / 偏移孔径 |
| `--rec-batch-size` | `64` | Receiver batch size (GPU memory) / 接收道批大小 |
| `--test-shots` | — | Process first N shots only / 仅处理前 N 炮 |
| `--log-level` | `INFO` | `DEBUG` `INFO` `WARNING` `ERROR` |

---

## Architecture / 架构

```
pstm_2d/
├── __init__.py        Package entry
├── engine.py          Top-level PSTM2DEngine orchestrator / 顶层调度
├── geometry.py        SEGY header parsing & CMP grid / 几何解析
├── traveltime.py      DSR kernel & amplitude weighting / 走时核
├── worker.py          Single-GPU migration worker / 单 GPU worker
├── scheduler.py       Multi-GPU scheduler with watchdog / 多 GPU 调度
├── aggregator.py      Result accumulation & SEGY output / 结果归约
└── safety.py          Circuit breaker & emergency cleanup / 安全熔断
```

### Pipeline / 流水线

```
Phase 1: SEGY header parsing → Geometry, statics, velocity loading
Phase 2: Shot chunks distributed across 4 GPUs → NCCL all_reduce
Phase 3: Fold-map normalization → SEGY output
```

### GPU Chunk Assignment / GPU 块分配

| GPU | Shot Range | Halo |
|-----|-----------|------|
| GPU 0 | 1 → 161 | R=10 |
| GPU 1 | 142 → 311 | L=10, R=10 |
| GPU 2 | 292 → 461 | L=10, R=10 |
| GPU 3 | 442 → 601 | L=10 |

---

## Performance / 性能

| Metric | Value |
|--------|-------|
| Hardware | 4× NVIDIA RTX 4090 (24 GB) |
| Input | 601 shots × 282 traces × 2750 samples |
| Runtime | ~51 s (full 601 shots) |
| Per-shot | ~0.22 s/shot |
| Peak GPU memory | ~400 MB / GPU |
| Output | 1341 CMP × 2750 samples, 15 MB |

---

## License / 许可协议

This project is licensed under the **GPL-3.0** License — see [LICENSE](./LICENSE) for details.

本项目采用 **GPL-3.0** 开源许可协议。

---

## References / 参考文献

- Schneider, W. A. (1978). Integral formulation for migration in two and three dimensions. *Geophysics*, 43(1), 49–76.
- French, W. S. (1975). Computer migration of oblique seismic reflection profiles. *Geophysics*, 40(6), 961–980.
- [docs/实验报告_PSTM.md](./docs/实验报告_PSTM.md) — Full experimental report (Chinese) / 完整实验报告
