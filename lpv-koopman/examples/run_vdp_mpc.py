"""Koopman-LPV MPC demos on the true Van der Pol plant.

Three closed-loop tasks, all using the N = 40 autonomous model and the
polynomial LPV field B(x) (built from autonomous data only):

  1. stabilization of the *unstable* origin from a point at 92% of the
     limit-cycle radius (the open-loop plant diverges to the cycle);
  2. tracking of a moving reference x1 = 0.3 sin(1.2 t);
  3. piecewise-constant setpoints x1 = +/-0.25 (documented limitation:
     residual bias from the model's stationarity error at off-origin
     equilibria - see the validation report).

Run:  python examples/run_vdp_mpc.py   (~4 min including model fit)
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lpvkoopman import (
    KoopmanMPC,
    MPCConfig,
    fit_b_field,
    run_closed_loop,
    vdp_scaled,
)
from lpvkoopman.vdp_benchmark import limit_cycle, sample_interior

from run_vdp_control_benchmark import get_model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parents[1] / "reports"))
    args = ap.parse_args()
    out = Path(args.outdir)
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    system = vdp_scaled()
    model = get_model(out / "model_n40.pkl")
    lc = limit_cycle(system)
    bfield = fit_b_field(
        model, system.g, sample_interior(lc, 400, np.random.default_rng(2)),
        degree=4, f=system.f,
    )
    results = {}

    # ---- task 1: stabilize the unstable origin ---------------------------
    x0 = lc[800] * 0.92
    mpc = KoopmanMPC(model, bfield,
                     MPCConfig(horizon=80, Q=(1.0, 1.0), R=1e-3, S=1e-2))
    t0 = time.perf_counter()
    stab = run_closed_loop(mpc, system, x0, reference=lambda t: np.zeros(2),
                           n_steps=600)
    free = system.simulate(x0, stab.t)  # uncontrolled comparison
    nx = np.linalg.norm(stab.X, axis=1)
    results["stabilization"] = {
        "x0": x0.tolist(),
        "norm_at_2s": float(nx[200]), "norm_at_4s": float(nx[400]),
        "norm_final": float(nx[-1]), "u_max": float(np.abs(stab.U).max()),
        "mean_solve_ms": float(stab.solve_ms.mean()),
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    print(f"stabilization: |x| 2s={nx[200]:.2e} 4s={nx[400]:.2e} "
          f"final={nx[-1]:.2e}  solve {stab.solve_ms.mean():.0f} ms")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    ax = axes[0]
    ax.plot(lc[:, 0], lc[:, 1], "k-", lw=1.6, label="limit cycle")
    ax.plot(free[:, 0], free[:, 1], color="0.6", lw=1.2, label="open loop (u=0)")
    ax.plot(stab.X[:, 0], stab.X[:, 1], "tab:green", lw=1.8, label="closed loop")
    ax.plot(*x0, "bo", ms=6, label="$x_0$")
    ax.plot(0, 0, "r*", ms=12, label="target (unstable focus)")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_aspect("equal")
    ax.legend(fontsize=8); ax.set_title("Stabilization of the unstable origin")
    ax = axes[1]
    ax.semilogy(stab.t, np.maximum(nx, 1e-8), "tab:green", label="$\\|x\\|$")
    ax2 = ax.twinx()
    ax2.step(stab.t[:-1], stab.U[:, 0], color="tab:orange", lw=0.9, alpha=0.8)
    ax2.set_ylabel("u", color="tab:orange"); ax2.set_ylim(-1.1, 1.1)
    ax.set_xlabel("t [s]"); ax.set_ylabel("$\\|x\\|$"); ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Convergence and input")
    fig.tight_layout(); fig.savefig(figdir / "fig9_mpc_stabilization.png", dpi=150)
    plt.close(fig)

    # ---- task 2: sinusoidal reference tracking ---------------------------
    ref_sin = lambda t: np.array([0.3 * np.sin(1.2 * t), 0.0])
    mpc = KoopmanMPC(model, bfield,
                     MPCConfig(horizon=40, Q=(10.0, 0.5), R=1e-3, S=1e-2))
    track = run_closed_loop(mpc, system, np.zeros(2), reference=ref_sin,
                            n_steps=1000)
    e = track.X[1:, 0] - track.R[:, 0]
    results["sine_tracking"] = {
        "rms_after_1s": float(np.sqrt(np.mean(e[100:] ** 2))),
        "max_abs_err_after_1s": float(np.abs(e[100:]).max()),
        "u_max": float(np.abs(track.U).max()),
    }
    print(f"sine tracking: RMS(>1s) {results['sine_tracking']['rms_after_1s']:.4f} "
          f"(amplitude 0.3)")

    # ---- task 3: piecewise-constant setpoints (documented limitation) ----
    ref_step = lambda t: np.array([0.25 if (t % 5.0) < 2.5 else -0.25, 0.0])
    mpc = KoopmanMPC(model, bfield,
                     MPCConfig(horizon=25, Q=(10.0, 0.5), R=1e-3, S=1e-2))
    step = run_closed_loop(mpc, system, np.zeros(2), reference=ref_step,
                           n_steps=1000, offset_free=True)
    es = step.X[1:, 0] - step.R[:, 0]
    msk = np.zeros(1000, bool)
    for s in range(4):
        msk[s * 250 + 150 : (s + 1) * 250] = True
    results["step_tracking"] = {
        "settled_rms": float(np.sqrt(np.mean(es[msk] ** 2))),
        "settled_std_segment1": float(step.X[150:250, 0].std()),
        "u_max": float(np.abs(step.U).max()),
        "note": "residual bias from model stationarity error at off-origin "
                "equilibria; see report",
    }
    print(f"step tracking: settled RMS {results['step_tracking']['settled_rms']:.4f} "
          f"(reference +/-0.25)")

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.2), sharex="col")
    for col, (res, ttl) in enumerate(
        [(track, "sine reference"), (step, "step reference (offset-free)")]
    ):
        axes[0, col].plot(res.t[1:], res.R[:, 0], "k--", lw=1.2, label="reference")
        axes[0, col].plot(res.t, res.X[:, 0], "tab:green", lw=1.4, label="$x_1$")
        axes[0, col].set_title(f"Tracking: {ttl}")
        axes[0, col].legend(fontsize=8)
        axes[0, col].set_ylabel("$x_1$")
        axes[1, col].step(res.t[:-1], res.U[:, 0], "tab:orange", lw=0.9)
        axes[1, col].set_ylabel("u"); axes[1, col].set_xlabel("t [s]")
        axes[1, col].set_ylim(-1.1, 1.1)
    fig.tight_layout(); fig.savefig(figdir / "fig10_mpc_tracking.png", dpi=150)
    plt.close(fig)

    (out / "results_mpc.json").write_text(json.dumps(results, indent=1))
    print(f"results -> {out/'results_mpc.json'}")
    print(f"figures -> {figdir}/fig9, fig10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
