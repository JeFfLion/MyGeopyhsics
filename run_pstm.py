#!/usr/bin/env python3
"""CLI entry point for 2D PSTM migration.

Usage:
    python run_pstm.py                        # full 601 shots, 4 GPUs
    python run_pstm.py --test-shots 4         # quick 4-shot validation
    python run_pstm.py --n-gpus 2 --help      # custom config
"""

import argparse
import os
import sys
import torch
import multiprocessing as mp


def main():
    parser = argparse.ArgumentParser(
        description="2D Pre-Stack Time Migration (Kirchhoff PSTM) on Multi-GPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--shot-path", default="NH1_Shot.sgy",
                        help="Input SEGY shot gathers")
    parser.add_argument("--vel-path", default="NH1_Vel.sgy",
                        help="Input SEGY RMS velocity")
    parser.add_argument("--output", default="pstm_result.sgy",
                        help="Output SEGY migrated section")
    parser.add_argument("--work-dir", default=None,
                        help="Working directory (default: script directory)")
    parser.add_argument("--src-depth", type=float, default=9.0,
                        help="Source depth (m)")
    parser.add_argument("--rec-depth", type=float, default=10.0,
                        help="Receiver depth (m)")
    parser.add_argument("--datum-elev", type=float, default=0.0,
                        help="Datum elevation (m)")
    parser.add_argument("--replacement-vel", type=float, default=1800.0,
                        help="Replacement velocity (m/s)")
    parser.add_argument("--cmp-spacing", type=float, default=25.0,
                        help="CMP output spacing (m)")
    parser.add_argument("--max-aperture", type=float, default=3000.0,
                        help="Max migration aperture (m)")
    parser.add_argument("--dt-ms", type=float, default=4.0,
                        help="Sample interval (ms)")
    parser.add_argument("--n-gpus", type=int, default=4,
                        help="Number of GPUs")
    parser.add_argument("--chunk-overlap", type=int, default=10,
                        help="Shot overlap between workers")
    parser.add_argument("--rec-batch-size", type=int, default=64,
                        help="Receiver batch size (GPU memory control)")
    parser.add_argument("--f-max", type=float, default=125.0,
                        help="Anti-aliasing max frequency (Hz)")
    parser.add_argument("--timeout", type=int, default=7200,
                        help="Watchdog timeout (seconds)")
    parser.add_argument("--test-shots", type=int, default=None,
                        help="Process only first N shots (for validation)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    # Resolve working directory
    if args.work_dir:
        work_dir = args.work_dir
    else:
        work_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(work_dir)
    sys.path.insert(0, work_dir)

    shot_path = os.path.join(work_dir, args.shot_path)
    vel_path = os.path.join(work_dir, args.vel_path)
    output_path = os.path.join(work_dir, args.output)

    for p, label in [(shot_path, "shot"), (vel_path, "velocity")]:
        if not os.path.exists(p):
            print(f"ERROR: {label} file not found: {p}")
            sys.exit(1)

    from pstm_2d.engine import PSTM2DEngine

    engine = PSTM2DEngine(
        segy_shot_path=shot_path,
        segy_vel_path=vel_path,
        segy_output_path=output_path,
        src_depth=args.src_depth,
        rec_depth=args.rec_depth,
        datum_elev=args.datum_elev,
        replacement_vel=args.replacement_vel,
        cmp_spacing_m=args.cmp_spacing,
        max_aperture_m=args.max_aperture,
        dt_ms=args.dt_ms,
        n_gpus=args.n_gpus,
        chunk_overlap=args.chunk_overlap,
        rec_batch_size=args.rec_batch_size,
        f_max=args.f_max,
        timeout_seconds=args.timeout,
        log_level=args.log_level,
    )

    if args.test_shots is not None:
        print(f"\n  TEST MODE: processing first {args.test_shots} shots only\n")
        _patch_for_test(engine, args.test_shots)

    try:
        result_path = engine.run()
        print(f"\n  SUCCESS — output written to: {result_path}")
    except Exception as e:
        print(f"\n  FAILED: {e}")
        engine.emergency_shutdown(str(e))
        sys.exit(1)


def _patch_for_test(engine: "PSTM2DEngine", n_shots: int):
    """Monkey-patch to limit processing to first n_shots."""
    orig_run = engine.preprocessor.load_geometry

    def patched_load():
        geom = orig_run()
        geom.n_shots = min(geom.n_shots, n_shots)
        return geom

    engine.preprocessor.load_geometry = patched_load

    if hasattr(engine, "n_gpus"):
        engine.n_gpus = min(engine.n_gpus, n_shots)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
