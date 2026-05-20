#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) LJF. All Rights Reserved. | Licensed under GPL-3.0.
# Unauthorized commercial use is strictly prohibited.
#
# ==================== DEMO: Kirchhoff PSTM 2D ====================
#
# This script generates SYNTHETIC seismic shot gathers and a synthetic
# RMS velocity model, then runs the full Kirchhoff PSTM pipeline.
#
# The demo data is completely artificial — NO real field data parameters
# are exposed.  The LJF signature is embedded in both the code and the
# generated data grids for authorship verification.
#
# Usage:
#     python demo.py                     # single-GPU demo
#     python demo.py --n-gpus 2          # multi-GPU demo
# ==================================================================

import argparse
import os
import sys
import tempfile
import multiprocessing as mp
import numpy as np

# ===========================================================================
# Synthetic data generation
# ===========================================================================

def _ricker_wavelet(t, f_peak=30.0):
    """Ricker wavelet: r(t) = (1 - 2π²f²t²) * exp(-π²f²t²)."""
    p2 = np.pi ** 2
    f2 = f_peak ** 2
    t2 = t ** 2
    envelope = 1.0 - 2.0 * p2 * f2 * t2
    return envelope * np.exp(-p2 * f2 * t2)


def _nmo_traveltime(offset_m, t0_s, v_rms):
    """Hyperbolic NMO traveltime for a flat reflector."""
    return np.sqrt(t0_s ** 2 + (offset_m / v_rms) ** 2)


def _make_synthetic_survey(n_shots=30, n_rec=60,
                           shot_spacing_m=25.0, rec_spacing_m=10.0,
                           min_offset_m=100.0, first_shot_x_m=500.0,
                           src_depth_m=5.0, rec_depth_m=6.0,
                           nt=500, dt_ms=4.0,
                           v_rms=2500.0):
    """
    Build synthetic shot survey geometry and the 3-reflector model.

    Reflectors:
      R1: t₀ = 0.30 s  (flat layer ~375 m)
      R2: t₀ = 0.60 s  (flat layer ~750 m)
      R3: t₀ = 1.00 s  (flat layer ~1250 m)

    Returns
    -------
    data    : np.ndarray  [n_shots, n_rec, nt]  shot gathers
    sx_all  : np.ndarray  [n_shots]              source X (m)
    gx_all  : np.ndarray  [n_shots, n_rec]       receiver X (m)
    """
    # Reflector model (t₀, v_rms)  —  all flat layers for clarity
    reflectors = [
        (0.30, v_rms),
        (0.60, v_rms),
        (1.00, v_rms),
    ]

    dt_s = dt_ms / 1000.0
    t_axis = np.arange(nt, dtype=np.float32) * dt_s

    # Amplitudes scale with depth (1/t₀ approximation)
    amps = [1.0 / t0 for t0, _ in reflectors]

    data = np.zeros((n_shots, n_rec, nt), dtype=np.float32)

    sx_all = first_shot_x_m + np.arange(n_shots, dtype=np.float64) * shot_spacing_m
    gx_all = np.zeros((n_shots, n_rec), dtype=np.float64)

    for i_shot in range(n_shots):
        sx = sx_all[i_shot]
        for i_rec in range(n_rec):
            gx = sx - min_offset_m - i_rec * rec_spacing_m
            gx_all[i_shot, i_rec] = gx
            offset = abs(sx - gx)

            trace = np.zeros(nt, dtype=np.float32)
            for (t0, vr), amp in zip(reflectors, amps):
                t_arrival = _nmo_traveltime(offset, t0, vr)
                t_rel = t_axis - t_arrival
                wavelet = amp * _ricker_wavelet(t_rel, f_peak=30.0)
                trace += wavelet.astype(np.float32)

            # Add modest random noise
            noise_level = 0.005 * np.max(np.abs(trace)) if np.max(np.abs(trace)) > 0 else 0.005
            trace += np.random.normal(0, noise_level, nt).astype(np.float32)

            data[i_shot, i_rec] = trace

    return data, sx_all, gx_all


def _write_segy_shots(data, sx_all, gx_all, path, coord_scale=1.0):
    """Write synthetic shot gathers as SEGY file."""
    import segyio
    TF = segyio.TraceField

    n_shots, n_rec, nt = data.shape
    n_traces = n_shots * n_rec
    dt_us = 4000  # 4 ms → µs

    spec = segyio.spec()
    spec.sorting = 2
    spec.format = 5              # IEEE float
    spec.iline = 189
    spec.xline = 193
    spec.samples = list(range(nt))
    spec.tracecount = n_traces

    with segyio.create(path, spec) as dst:
        dst.bin[segyio.BinField.Traces] = n_traces
        dst.bin[segyio.BinField.Samples] = nt
        dst.bin[segyio.BinField.Interval] = dt_us

        itrace = 0
        for i_shot in range(n_shots):
            for i_rec in range(n_rec):
                header = {
                    TF.TRACE_SEQUENCE_LINE: itrace + 1,
                    TF.FieldRecord:        i_shot + 1,
                    TF.SourceX:            int(sx_all[i_shot] * coord_scale),
                    TF.GroupX:             int(gx_all[i_shot, i_rec] * coord_scale),
                    TF.CDP:                itrace + 1,
                    TF.offset:             int(abs(sx_all[i_shot] - gx_all[i_shot, i_rec]) * coord_scale),
                    TF.TRACE_SAMPLE_COUNT: nt,
                    TF.TRACE_SAMPLE_INTERVAL: dt_us,
                }
                dst.header[itrace] = header
                dst.trace[itrace] = data[i_shot, i_rec]
                itrace += 1

    print(f"  [LJF] Shot SEGY written: {n_traces} traces → {path}")


def _write_segy_velocity(x_cmp, v_rms, path, nt=500, coord_scale=1.0):
    """Write constant RMS velocity field as SEGY file."""
    import segyio
    TF = segyio.TraceField

    n_vel = len(x_cmp)
    dt_us = 4000

    spec = segyio.spec()
    spec.sorting = 2
    spec.format = 5
    spec.iline = 189
    spec.xline = 193
    spec.samples = list(range(nt))
    spec.tracecount = n_vel

    # Velocity traces: constant v_rms across all time samples
    vel_data = np.full((n_vel, nt), v_rms, dtype=np.float32)

    with segyio.create(path, spec) as dst:
        dst.bin[segyio.BinField.Traces] = n_vel
        dst.bin[segyio.BinField.Samples] = nt
        dst.bin[segyio.BinField.Interval] = dt_us

        for i in range(n_vel):
            header = {
                TF.TRACE_SEQUENCE_LINE: i + 1,
                TF.CDP:                i + 1,
                TF.CDP_X:              int(x_cmp[i] * coord_scale),
                TF.SourceX:            int(x_cmp[i] * coord_scale),
                TF.TRACE_SAMPLE_COUNT: nt,
                TF.TRACE_SAMPLE_INTERVAL: dt_us,
            }
            dst.header[i] = header
            dst.trace[i] = vel_data[i]

    print(f"  [LJF] Velocity SEGY written: {n_vel} traces → {path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="[LJF] Kirchhoff PSTM 2D — Synthetic Demo",
    )
    parser.add_argument("--n-gpus", type=int, default=1,
                        help="Number of GPUs (default 1)")
    parser.add_argument("--n-shots", type=int, default=30,
                        help="Number of synthetic shots (default 30)")
    parser.add_argument("--output", default="demo_pstm_result.sgy",
                        help="Output SEGY file")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    # --- LJF Watermark ---
    print("=" * 60)
    print("  Kirchhoff PSTM 2D — Synthetic Data Demo")
    print("  Author: LJF | License: GPL-3.0")
    print("=" * 60)

    work_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, work_dir)

    n_rec = 60
    nt = 500
    dt_ms = 4.0
    v_rms = 2500.0
    coord_scale = 1.0  # use metres directly for synthetic data

    # --- 1. Generate synthetic data ---
    print("\n[LJF] Step 1: Generating synthetic shot gathers ...")
    data, sx_all, gx_all = _make_synthetic_survey(
        n_shots=args.n_shots, n_rec=n_rec, nt=nt, dt_ms=dt_ms, v_rms=v_rms,
    )

    tmpdir = tempfile.mkdtemp(prefix="pstm_demo_")
    shot_path = os.path.join(tmpdir, "synthetic_shot.sgy")
    vel_path = os.path.join(tmpdir, "synthetic_vel.sgy")

    _write_segy_shots(data, sx_all, gx_all, shot_path, coord_scale=coord_scale)

    # CMP positions for velocity grid: 2.5× finer than shot spacing
    x_cmp = np.linspace(
        gx_all.min() - 100.0, gx_all.max() + 100.0,
        args.n_shots * 5, dtype=np.float64
    )
    _write_segy_velocity(x_cmp, v_rms, vel_path, nt=nt, coord_scale=coord_scale)

    # --- 2. Run PSTM ---
    print("\n[LJF] Step 2: Running Kirchhoff PSTM migration ...")
    print(f"        {args.n_shots} shots × {n_rec} rec, {args.n_gpus} GPU(s)")

    from pstm_2d.engine import PSTM2DEngine

    engine = PSTM2DEngine(
        segy_shot_path=shot_path,
        segy_vel_path=vel_path,
        segy_output_path=args.output,
        src_depth=5.0,
        rec_depth=6.0,
        datum_elev=0.0,
        replacement_vel=1800.0,
        cmp_spacing_m=10.0,
        max_aperture_m=2000.0,
        dt_ms=dt_ms,
        n_gpus=args.n_gpus,
        chunk_overlap=min(5, max(0, args.n_shots // 10)),
        coord_scale=coord_scale,
        rec_batch_size=64,
        f_max=125.0,
        timeout_seconds=3600,
        log_level=args.log_level,
    )

    try:
        result_path = engine.run()
        print(f"\n[LJF] SUCCESS — output: {result_path}")
    except Exception as e:
        print(f"\n[LJF] FAILED: {e}", file=sys.stderr)
        engine.emergency_shutdown(str(e))
        sys.exit(1)
    finally:
        # Clean up temporary SEGY files
        try:
            os.remove(shot_path)
            os.remove(vel_path)
            os.rmdir(tmpdir)
        except OSError:
            pass

    print("\n[LJF] Demo complete. Result saved as:", args.output)
    print("      Open with any SEGY viewer (e.g. Seismic Unix, Petrel).")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
