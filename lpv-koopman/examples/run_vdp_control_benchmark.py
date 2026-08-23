"""Forced-prediction validation of the LPV input matrix B(x).

B(x) = (dPhi/dx) g_c(x) is built from the autonomous N = 40 model only -
no forced/input-perturbed experiments anywhere (the thesis's central
claim). This script reproduces the controlled-prediction comparison
against Korda & Mezic (2020) Table I (which used a constant B *fitted to
forced data*):

  * gradient-quality diagnostics (eigenfunction-PDE residual, MLS vs FD);
  * forced rollouts under the paper's two test inputs (square wave 300 ms,
    sine 60 ms, unit amplitude, 1 s horizon) from interior ICs, with three
    input models: smooth LPV field B(x), its constant reduction, and raw
    pointwise MLS B(x) (the cautionary tale - it explodes);
  * figures + reports/results_control.json.

Run:  python examples/run_vdp_control_benchmark.py   (~3 min)
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
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lpvkoopman import (
    BMap,
    FitConfig,
    fit_b_field,
    fit_koopman_model,
    pde_residual,
    predict_forced,
    vdp_scaled,
)
from lpvkoopman.metrics import relative_rmse_percent
from lpvkoopman.systems import ContinuousSystem
from lpvkoopman.vdp_benchmark import limit_cycle, make_training_data, sample_interior

PAPER_CONTROLLED = {"square": 6.0, "sine": 2.9}  # Table I, N=20, optimized


def get_model(cache: Path):
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    data = make_training_data(n_traj=100, horizon=7.0)
    model, _ = fit_koopman_model(
        data, FitConfig(n_eig=40, optimize=True, alpha_bounds=(-20.0, 1.0))
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(model, open(cache, "wb"))
    return model


def truth_forced(system, x0, u_seq, ts):
    x = np.asarray(x0, dtype=float).copy()
    out = [x.copy()]
    for uk in u_seq:
        forced = ContinuousSystem(
            f=lambda xx, _u=uk: system.f(xx) + np.array([0.0, 1.0]) * _u, n=2
        )
        x = forced.simulate(x, np.array([0.0, ts]), rtol=1e-9, atol=1e-11)[-1]
        out.append(x.copy())
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parents[1] / "reports"))
    args = ap.parse_args()
    out = Path(args.outdir)
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    system = vdp_scaled()
    t0 = time.perf_counter()
    model = get_model(out / "model_n40.pkl")
    print(f"model ready ({time.perf_counter()-t0:.0f}s)")
    lc = limit_cycle(system)
    tree = cKDTree(lc)

    # ---------------- gradient diagnostics --------------------------------
    rng = np.random.default_rng(11)
    xq = sample_interior(lc, 150 if args.quick else 200, rng)
    dq = tree.query(xq)[0]
    grad_table = {}
    print("== eigenfunction-PDE residual ||J f - A z|| / ||A z|| ==")
    for tag, kw in [
        ("mls_k40_deg2", dict(method="mls")),
        ("fd_h1e-3_o4", dict(method="fd", h=1e-3, order=4)),
        ("fd_h0.01_o4", dict(method="fd", h=0.01, order=4)),
        ("fd_h0.1_o2_thesis_default", dict(method="fd", h=0.1, order=2)),
    ]:
        r = pde_residual(model, system.f, xq, **kw)
        grad_table[tag] = {
            "median": float(np.median(r)),
            "p90": float(np.percentile(r, 90)),
            "near_lc_median": float(np.median(r[dq < 0.1])),
        }
        print(f"   {tag:28s} median {np.median(r):9.3e}  p90 {np.percentile(r,90):9.3e}")

    # ---------------- B(x) models ------------------------------------------
    fit_pts = sample_interior(lc, 400, np.random.default_rng(2))
    bfield = fit_b_field(model, system.g, fit_pts, degree=4, f=system.f)
    bconst = fit_b_field(model, system.g, fit_pts, degree=0, f=system.f)
    braw = BMap(model, system.g)
    nrm = np.linalg.norm(braw.batch(fit_pts)[:, :, 0], axis=1)
    dfit = tree.query(fit_pts)[0]
    b_mag = {
        "raw_interior_median": float(np.median(nrm[dfit > 0.15])),
        "raw_near_lc_median": float(np.median(nrm[dfit < 0.05])),
        "raw_max": float(nrm.max()),
    }
    print(f"== raw ||B(x)||: interior median {b_mag['raw_interior_median']:.0f}, "
          f"near-LC median {b_mag['raw_near_lc_median']:.0f}, max {b_mag['raw_max']:.0f} ==")

    # ---------------- forced benchmark -------------------------------------
    ts, K = 0.01, 100
    tg = np.arange(K) * ts
    signals = {
        "square": np.where((tg % 0.3) < 0.15, 1.0, -1.0),  # paper: 300 ms period
        "sine": np.sin(2 * np.pi * tg / 0.06),  # paper: 60 ms period
    }
    n_ic = 100 if args.quick else 250
    x0s = sample_interior(lc, n_ic, np.random.default_rng(7))
    d0 = tree.query(x0s)[0]

    results = {"gradient_diagnostics": grad_table, "raw_B_magnitude": b_mag,
               "paper_controlled_N20": PAPER_CONTROLLED, "n_ic": n_ic}
    examples = {}
    for nm, useq in signals.items():
        t1 = time.perf_counter()
        truths = [truth_forced(system, x0, useq, ts) for x0 in x0s]
        rows = {}
        for bnm, bb, sel in [
            ("lpv_poly_deg4", bfield, slice(None)),
            ("const_deg0", bconst, slice(None)),
            ("raw_mls", braw, slice(0, 30)),  # explodes; 30 ICs suffice to show it
        ]:
            idx = range(len(x0s))[sel]
            e = np.array([
                relative_rmse_percent(predict_forced(model, bb, x0s[j], useq, ts), truths[j])
                for j in idx
            ])
            dd = d0[list(idx)]
            rows[bnm] = {
                "mean": float(e.mean()), "median": float(np.median(e)),
                "near_lc_mean": float(e[dd < 0.05].mean()) if (dd < 0.05).any() else None,
                "interior_mean": float(e[dd >= 0.05].mean()),
            }
            print(f"   {nm:6s} {bnm:14s} mean {e.mean():12.2f}%  median {np.median(e):8.2f}%"
                  f"  (paper const-B from forced data: {PAPER_CONTROLLED[nm]}%)")
        results[nm] = rows
        # keep two showcase ICs for the figure: mid-region and near-LC
        j_mid = int(np.argmin(np.abs(d0 - np.percentile(d0, 60))))
        j_lc = int(np.argmin(d0))
        examples[nm] = [
            (x0s[j], truths[j], predict_forced(model, bfield, x0s[j], useq, ts), d0[j])
            for j in (j_mid, j_lc)
        ]
        print(f"   [{nm}: {time.perf_counter()-t1:.0f}s]")

    # ---------------- figures ----------------------------------------------
    t_plot = np.arange(K + 1) * ts
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharex=True)
    for col_pair, (nm, exs) in enumerate(examples.items()):
        for col_in, (x0, xt, xp, dd) in enumerate(exs):
            c = 2 * col_pair + col_in
            for row in range(2):
                ax = axes[row, c]
                ax.plot(t_plot, xt[:, row], "b-", lw=1.7, label="true")
                ax.plot(t_plot, xp[:, row], "r--", lw=1.5, label="LPV model")
                if row == 0:
                    err = relative_rmse_percent(xp, xt)
                    ax.set_title(f"{nm}, d(x0,LC)={dd:.2f}\nerr={err:.1f}%", fontsize=9)
                if row == 1:
                    ax.set_xlabel("t [s]")
                if c == 0:
                    ax.set_ylabel(f"$x_{row+1}$")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Forced prediction with LPV B(x) from autonomous data only")
    fig.tight_layout()
    fig.savefig(figdir / "fig6_forced_predictions.png", dpi=150)
    plt.close(fig)

    # spatial error map for the square wave, LPV B
    e_sq = np.array([
        relative_rmse_percent(predict_forced(model, bfield, x0s[j], signals["square"], ts),
                              truth_forced(system, x0s[j], signals["square"], ts))
        for j in range(len(x0s))
    ])
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.plot(lc[:, 0], lc[:, 1], "k-", lw=2, label="limit cycle")
    sc = ax.scatter(x0s[:, 0], x0s[:, 1], c=np.log10(e_sq), s=18, cmap="viridis")
    fig.colorbar(sc, ax=ax, label=r"$\log_{10}$ forced prediction error [%]")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_aspect("equal")
    ax.set_title("Square-wave forced prediction error (LPV B)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "fig7_forced_error_map.png", dpi=150)
    plt.close(fig)

    # B-field sanity map: output-projected input gain C B(x) (exact: [0, 1])
    gx = np.linspace(-0.85, 0.85, 60)
    gy = np.linspace(-0.9, 0.9, 60)
    GX, GY = np.meshgrid(gx, gy)
    CB = np.full(GX.shape, np.nan)
    from matplotlib.path import Path as MplPath
    poly = MplPath(lc)
    pts = np.column_stack([GX.ravel(), GY.ravel()])
    inside = poly.contains_points(pts)
    for i, p in enumerate(pts):
        if inside[i]:
            CB.ravel()[i] = (model.C @ bfield(p))[1, 0]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    pc = ax.pcolormesh(GX, GY, CB, shading="auto", cmap="RdBu_r", vmin=0.7, vmax=1.3)
    fig.colorbar(pc, ax=ax, label=r"$(C\,B(x))_2$   (exact value: 1)")
    ax.plot(lc[:, 0], lc[:, 1], "k-", lw=1.5)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_aspect("equal")
    ax.set_title("Output-projected input gain of the fitted B(x)")
    fig.tight_layout()
    fig.savefig(figdir / "fig8a_B_gain_map.png", dpi=150)
    plt.close(fig)

    # Analogous figure for the constant-B reduction (bconst, degree=0), the
    # LTI baseline structurally comparable to Korda & Mezic's constant input
    # matrix. Unlike the fitted LPV field, this evaluates the *same* model
    # class the K-M benchmark uses, so it is a fair sanity check: the value
    # is guaranteed spatially uniform (degree-0 fit), but is computed from
    # the actual least-squares fit rather than assumed to equal the exact
    # answer.
    gx = np.linspace(-0.85, 0.85, 60)
    gy = np.linspace(-0.9, 0.9, 60)
    GX, GY = np.meshgrid(gx, gy)
    CB_const = np.full(GX.shape, np.nan)
    pts = np.column_stack([GX.ravel(), GY.ravel()])
    inside = poly.contains_points(pts)
    for i, p in enumerate(pts):
        if inside[i]:
            CB_const.ravel()[i] = (model.C @ bconst(p))[1, 0]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    pc = ax.pcolormesh(GX, GY, CB_const, shading="auto", cmap="RdBu_r", vmin=0.7, vmax=1.3)
    fig.colorbar(pc, ax=ax, label=r"$(C\,B_{\mathrm{const}})_2$   (exact value: 1)")
    ax.plot(lc[:, 0], lc[:, 1], "k-", lw=1.5)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_aspect("equal")
    ax.set_title("Output-projected input gain of the constant-B (LTI) reduction")
    fig.tight_layout()
    fig.savefig(figdir / "fig8b_B_gain_map_const.png", dpi=150)
    plt.close(fig)

    (out / "results_control.json").write_text(json.dumps(results, indent=1))
    print(f"\nresults -> {out/'results_control.json'}")
    print(f"figures -> {figdir}/fig6..fig8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
