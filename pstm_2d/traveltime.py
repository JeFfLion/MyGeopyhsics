# Copyright (c) LJF. All Rights Reserved. | Licensed under GPL-3.0.
# Unauthorized commercial use is strictly prohibited.
"""Geometric ray-tracing traveltime, C++-reference amplitude weighting, and
shot-gather migration.

All functions operate on PyTorch tensors and are designed for GPU execution
under torch.no_grad().

Algorithm matches reference C++ implementation:
    depth = t0 * Vrms / 2
    dx_src = sqrt((sx - x)² + (depth - shot_depth)²)
    dx_rec = sqrt((rx - x)² + (depth - rec_depth)²)
    time  = (dx_src + dx_rec) / Vrms
    wm    = depth / (Vrms * dx_src * dx_rec * dx_rec + 1e-8)
    mig  += wm * (data[n] - data[n-1]) / dt / (2*PI)
"""

import math
from typing import Tuple

import torch


# ---------------------------------------------------------------------------
#  Velocity interpolation
# ---------------------------------------------------------------------------
def interp_velocity(
    vel_x: torch.Tensor,           # [N_vel] CMP positions (m)
    vel_traces: torch.Tensor,      # [N_vel, Nt] RMS velocity (m/s)
    x_img: torch.Tensor,           # [Na] imaging positions (m)
) -> torch.Tensor:
    """Linearly interpolate velocity traces to imaging positions.

    Returns:
        vrms_img: [Na, Nt] interpolated RMS velocities.
    """
    if len(vel_x) > 1:
        idx = (x_img - vel_x[0]) / (vel_x[1] - vel_x[0])
    else:
        idx = torch.zeros_like(x_img)
    i0 = idx.long().clamp(0, len(vel_x) - 2)
    i1 = i0 + 1
    frac = (idx - i0.float()).clamp(0.0, 1.0)

    v0 = vel_traces[i0]   # [Na, Nt]
    v1 = vel_traces[i1]   # [Na, Nt]
    return v0 * (1.0 - frac.unsqueeze(-1)) + v1 * frac.unsqueeze(-1)


# ---------------------------------------------------------------------------
#  Geometric traveltime (matches C++ reference)
# ---------------------------------------------------------------------------
@torch.no_grad()
def migrate_one_shot(
    sx: torch.Tensor,                 # scalar, source x (m)
    gather: torch.Tensor,             # [Nr, Nt_samp]
    x_img: torch.Tensor,              # [Na] image positions within aperture (m)
    t0: torch.Tensor,                 # [Nt] output time axis (s)
    dt: float,                        # sampling interval (s)
    vrms_img: torch.Tensor,           # [Na, Nt] RMS velocity at imaging points
    shot_depth: float = 9.0,          # source depth (m)
    rec_depth: float = 10.0,          # receiver depth (m)
    min_offset: float = 250.0,        # minimum offset (m)
    dx_rec: float = 25.0,             # receiver spacing (m)
    static_shift: float = 0.0,        # seconds
    rec_batch_size: int = 64,
) -> torch.Tensor:
    """Migrate one shot gather using geometric ray tracing.

    Returns:
        partial: [Na, Nt] stacked contribution for this shot.
    """
    PI = 3.141592653589793
    Nr = gather.shape[0]
    Na = x_img.shape[0]
    Nt = t0.shape[0]
    device = gather.device
    sx_val = float(sx.item())
    nt_samp = gather.shape[1]

    # Compute absolute receiver positions for this shot
    #   rx(itrace) = sx - min_offset - itrace * dx_rec
    itrace_idx = torch.arange(Nr, device=device, dtype=torch.float32)
    rx_abs = sx_val - min_offset - itrace_idx * dx_rec   # [Nr]

    # Pre-compute depth [Na, Nt] = t0 * Vrms / 2
    #   t0: [Nt]  ->  [1, Nt] * [Na, Nt] / 2  -> [Na, Nt]
    depth = t0.unsqueeze(0) * vrms_img / 2.0              # [Na, Nt]

    # Source-side slant distance squared: (sx - x_img)² + (depth - shot_depth)²
    dx_src_sq = (x_img - sx_val) ** 2                      # [Na]
    dz_src_sq = (depth - shot_depth) ** 2                  # [Na, Nt]
    dx_src = torch.sqrt(dx_src_sq.unsqueeze(-1) + dz_src_sq + 1e-12)  # [Na, Nt]

    partial = torch.zeros(Na, Nt, dtype=torch.float32, device=device)

    for r_start in range(0, Nr, rec_batch_size):
        r_end = min(r_start + rec_batch_size, Nr)
        rx_b = rx_abs[r_start:r_end]                       # [batch]
        gather_b = gather[r_start:r_end]                   # [batch, Nt_samp]
        batch = r_end - r_start

        # Receiver-side slant distance
        dx_horiz_sq = (rx_b.unsqueeze(1) - x_img.unsqueeze(0)) ** 2  # [batch, Na]
        dz_rec_sq = (depth.unsqueeze(0) - rec_depth) ** 2            # [1, Na, Nt]
        dx_rec = torch.sqrt(dx_horiz_sq.unsqueeze(-1) + dz_rec_sq + 1e-12)  # [batch, Na, Nt]

        # Traveltime: T = (dx_src + dx_rec) / Vrms
        T = (dx_src.unsqueeze(0) + dx_rec) / (vrms_img.unsqueeze(0) + 1e-12)  # [batch, Na, Nt]
        T = T + static_shift

        # Nearest-neighbour index (no interpolation, matching C++)
        n_idx = (T / dt).long()                                          # [batch, Na, Nt]
        n_idx = n_idx.clamp(1, nt_samp - 1)                              # need n-1

        # Weight:  wm = depth / (Vrms * dx_src * dx_rec * dx_rec)
        denom = (vrms_img.unsqueeze(0) + 1e-12) * (dx_src.unsqueeze(0) + 1e-12) * dx_rec * dx_rec + 1e-8
        W = depth.unsqueeze(0) / denom                                   # [batch, Na, Nt]

        # Amplitude with backward-difference derivative:
        #   (data[n] - data[n-1]) / dt / (2*PI)
        r_idx = torch.arange(batch, device=device).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1]
        r_exp = r_idx.expand(batch, Na, Nt)
        a_n   = gather_b[r_exp, n_idx]                                    # [batch, Na, Nt]
        a_n1  = gather_b[r_exp, n_idx - 1]                                # [batch, Na, Nt]
        deriv = (a_n - a_n1) / dt                                         # [batch, Na, Nt]

        contribution = W * deriv / (2.0 * PI)
        contribution = torch.nan_to_num(contribution, nan=0.0, posinf=0.0, neginf=0.0)

        partial += contribution.sum(dim=0)

        del T, n_idx, W, deriv, contribution, a_n, a_n1, dx_rec, r_exp
        torch.cuda.empty_cache()

    return partial
