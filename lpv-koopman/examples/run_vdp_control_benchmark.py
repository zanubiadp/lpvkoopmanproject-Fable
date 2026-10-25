"""Forced-prediction validation of the LPV input matrix B(x) on the
input-extended Van der Pol, under the Korda & Mezic (2020) test signals.

The plant is ``vdp_nonaffine`` through its exact extension x3 = u, v = udot;
B(x) = dPhi/dx3 is built from the autonomous N = 40+1 model only - no
forced experiments. The paper's controlled benchmark drives the input with a
square wave (300 ms period) and a sine (60 ms period) for 1 s; here the
same waveforms drive the physical input u (realized exactly in the extended
system as piecewise-linear u, i.e. ZOH v = du/Ts), at the paper's unit
amplitude - which reaches past the gain flip at |u| = pi/4 - and at 0.5,
inside the monotone-gain region. This script reports:

  * gradient-quality diagnostics on the extended model (eigenfunction-PDE
    residual: n-D MLS, per-dimension FD (ours), scalar FD steps including
    the thesis default h = 0.1);
  * raw pointwise B(x) magnitudes and the rollout they produce (it explodes);
  * forced rollouts from interior ICs with the two input models every
    figure compares - LPV B(x) (ours) and the Korda-Mezic constant B fitted
    to forced data - plus, in the tables only, the polynomial LPV variant,
    the constant reduction of our field, and raw pointwise B;
  * the input gain each model implies, d xdot2 / du = (C A B)_2 / (C B)_3,
    against the plant's true gain;
  * figures fig6..fig8 + reports/results_control.json.

Run:  python examples/run_vdp_control_benchmark.py   (~6 min)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.path import Path as MplPath
from scipy.spatial import cKDTree

from extended_vdp import KM, LPV, TS, U_MAX, b_sample_points, fd_options, fit_input_models, get_model
from lpvkoopman import BMap, extend_input, pde_residual, predict_forced, simulate_true, u_profile_to_v, vdp_nonaffine
from lpvkoopman.metrics import relative_rmse_percent
from lpvkoopman.nonaffine import level_limit_cycle
from lpvkoopman.vdp_benchmark import sample_interior

PAPER_CONTROLLED = {"square": 6.0, "sine": 2.9}  # Table I, N=20, constant B from forced data
MAX_DX = 0.05  # LPV sub-step bound on |x3| per sub-step: half the plane spacing


def signal(kind: str, amp: float, K: int) -> np.ndarray:
    """u waypoints (K+1,) of the paper's two test waveforms."""
    tg = np.arange(K + 1) * TS
    if kind == "square":
        return amp * np.where((tg % 0.3) < 0.15, 1.0, -1.0)  # paper: 300 ms period
    return amp * np.sin(2 * np.pi * tg / 0.06)  # paper: 60 ms period


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parents[1] / "reports"))
    args = ap.parse_args()
    out = Path(args.outdir)
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    sysna = vdp_nonaffine()
    ext = extend_input(sysna)
    t0 = time.perf_counter()
    model, data, _ = get_model(out / "model_nonaffine.pkl", args.quick)
    cloud = data.flat_points()
    x_clip = (cloud.min(axis=0), cloud.max(axis=0))
    print(f"model ready ({time.perf_counter()-t0:.0f}s), N={model.N}")

    # ---------------- gradient diagnostics --------------------------------
    pts = b_sample_points(args.quick)
    rng = np.random.default_rng(11)
    xq = pts[rng.choice(len(pts), 150 if args.quick else 300, replace=False)]
    opts = fd_options(data)
    grad_table = {}
    print("== eigenfunction-PDE residual ||J f - A z|| / ||A z|| ==")
    for tag, kw in [
        ("mls_nd_k40", dict(method="mls")),
        ("fd_per_dim_ours", dict(method="fd", **opts)),
        ("fd_h1e-3_o2", dict(method="fd", h=1e-3, order=2)),
        ("fd_h0.01_o2", dict(method="fd", h=0.01, order=2)),
        ("fd_h0.1_o2_thesis_default", dict(method="fd", h=0.1, order=2)),
    ]:
        r = pde_residual(model, ext.f, xq, **kw)
        grad_table[tag] = {"median": float(np.median(r)),
                           "p90": float(np.percentile(r, 90))}
        print(f"   {tag:28s} median {np.median(r):9.3e}  p90 {np.percentile(r,90):9.3e}")

    # ---------------- input models -----------------------------------------
    t0 = time.perf_counter()
    Bm = fit_input_models(model, data, args.quick)
    braw = BMap(model, ext.g, method="fd", options=opts)
    nrm = np.linalg.norm(braw.batch(xq)[:, :, 0], axis=1)
    b_mag = {
        "raw_median": float(np.median(nrm)),
        "raw_p90": float(np.percentile(nrm, 90)),
        "raw_max": float(nrm.max()),
        "lpv_knn_median": float(np.median([np.linalg.norm(Bm["lpv_knn"](x)) for x in xq])),
        "km_const_forced": float(np.linalg.norm(Bm["km_const_forced"])),
    }
    print(f"input models ({time.perf_counter()-t0:.0f}s); raw ||B(x)||: median "
          f"{b_mag['raw_median']:.1f}, max {b_mag['raw_max']:.1f}; kNN median "
          f"{b_mag['lpv_knn_median']:.1f}; K-M {b_mag['km_const_forced']:.1f}")
    Bm["raw_fd"] = braw

    # ---------------- forced benchmark -------------------------------------
    K = 100
    n_ic = 100 if args.quick else 250
    results = {"gradient_diagnostics": grad_table, "B_magnitude": b_mag,
               "paper_controlled_N20_2D": PAPER_CONTROLLED, "n_ic": n_ic,
               "horizon_s": K * TS, "max_dx": MAX_DX}
    examples, err_maps = {}, {}
    for amp in (1.0, 0.5):
        for nm in ("square", "sine"):
            t1 = time.perf_counter()
            u_way = signal(nm, amp, K)
            u0, v = u_profile_to_v(u_way, TS)
            lc = level_limit_cycle(sysna, u0)
            x0s = sample_interior(lc, n_ic, np.random.default_rng(7))
            d0 = cKDTree(lc).query(x0s)[0]
            truths = [simulate_true(sysna, x0, u0, v, TS) for x0 in x0s]
            rows, errs, preds = {}, {}, {}
            for bnm, bb in Bm.items():
                idx = range(30) if bnm == "raw_fd" else range(n_ic)  # 30 suffice to show it
                P = [predict_forced(model, bb, np.array([*x0s[j], u0]), v, TS,
                                    x_clip=x_clip, max_dx=MAX_DX) for j in idx]
                e = np.array([relative_rmse_percent(p[:, :2], truths[j][:, :2])
                              for j, p in zip(idx, P)])
                dd = d0[list(idx)]
                rows[bnm] = {
                    "mean": float(e.mean()), "median": float(np.median(e)),
                    "p90": float(np.percentile(e, 90)),
                    "near_lc_mean": float(e[dd < 0.05].mean()) if (dd < 0.05).any() else None,
                    "interior_mean": float(e[dd >= 0.05].mean()),
                }
                errs[bnm], preds[bnm] = e, P
            key = f"{nm}_A{amp:g}"
            results[key] = rows
            print(f"   {key:10s} " + "  ".join(
                f"{b} {r['median']:.1f}%" for b, r in rows.items())
                + f"  [{time.perf_counter()-t1:.0f}s]")
            if amp == 1.0:
                # two showcase ICs: mid-region and near-LC
                j_mid = int(np.argmin(np.abs(d0 - np.percentile(d0, 60))))
                j_lc = int(np.argmin(d0))
                examples[nm] = [(truths[j], {b: preds[b][j] for b in (LPV[0], KM[0])},
                                 u_way, d0[j]) for j in (j_mid, j_lc)]
                err_maps[nm] = (x0s, lc, errs, u0)

    # ---------------- implied input gain d xdot2 / du ----------------------
    A = model.A_continuous()
    C = model.C
    UU, X2 = np.meshgrid(np.linspace(-U_MAX, U_MAX, 47), np.linspace(-0.9, 0.9, 61))
    inside = np.zeros(UU.shape, bool)
    for jc, u in enumerate(UU[0]):
        poly = MplPath(level_limit_cycle(sysna, u))
        inside[:, jc] = poly.contains_points(np.column_stack([np.zeros(len(X2)), X2[:, jc]]))
    g_true = np.full(UU.shape, np.nan)
    gains = {b: np.full(UU.shape, np.nan) for b in ("lpv_knn", "lpv_poly3",
                                                     "const_auto", "km_const_forced")}
    for (i, jc) in zip(*np.nonzero(inside)):
        x = np.array([0.0, X2[i, jc], UU[i, jc]])
        g_true[i, jc] = sysna.dfdu(x[:2], x[2])[1]
        for b in gains:
            B = Bm[b](x) if callable(Bm[b]) else Bm[b]
            gains[b][i, jc] = (C @ A @ B)[1, 0] / (C @ B)[2, 0]
    m = inside
    gain_table = {}
    for b, G in gains.items():
        gain_table[b] = {
            "rmse": float(np.sqrt(np.mean((G[m] - g_true[m]) ** 2))),
            "sign_agreement": float(np.mean(np.sign(G[m]) == np.sign(g_true[m]))),
        }
    gain_table["true_gain_rms"] = float(np.sqrt(np.mean(g_true[m] ** 2)))
    results["input_gain_x1_0"] = gain_table
    print("== implied input gain d xdot2/du on x1 = 0 ==")
    for b, r in gain_table.items():
        print(f"   {b:16s} {r}")

    # ---------------- figures ----------------------------------------------
    t_plot = np.arange(K + 1) * TS
    fig, axes = plt.subplots(3, 4, figsize=(14, 7.6), sharex=True)
    for col_pair, (nm, exs) in enumerate(examples.items()):
        for col_in, (xt, pr, u_way, dd) in enumerate(exs):
            c = 2 * col_pair + col_in
            for row in range(2):
                ax = axes[row, c]
                ax.plot(t_plot, xt[:, row], "b-", lw=1.8, label="true")
                for key, lab, colr in (LPV, KM):
                    ax.plot(t_plot, pr[key][:, row], "--", color=colr, lw=1.5, label=lab)
                if row == 0:
                    e_l = relative_rmse_percent(pr[LPV[0]][:, :2], xt[:, :2])
                    e_k = relative_rmse_percent(pr[KM[0]][:, :2], xt[:, :2])
                    ax.set_title(f"{nm}, d(x0,LC)={dd:.2f}\nLPV {e_l:.1f}%  vs  "
                                 f"const {e_k:.1f}%", fontsize=9)
                if c == 0:
                    ax.set_ylabel(f"$x_{row+1}$")
            ax = axes[2, c]
            ax.plot(t_plot, u_way, "k-", lw=1.2)
            ax.axhspan(np.pi / 4, 1.2, color="orange", alpha=0.15)
            ax.axhspan(-1.2, -np.pi / 4, color="orange", alpha=0.15)
            ax.set_ylim(-1.2, 1.2); ax.set_xlabel("t [s]")
            if c == 0:
                ax.set_ylabel("$u = x_3$")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Forced prediction under the Korda-Mezic test signals applied to $u$ "
                 "(amplitude 1; shaded: reversed-gain region)")
    fig.tight_layout()
    fig.savefig(figdir / "fig6_forced_predictions.png", dpi=150)
    plt.close(fig)

    # spatial error maps for the square wave: LPV vs K-M, shared color scale
    x0s, lc, errs, u0 = err_maps["square"]
    lo = min(errs[LPV[0]].min(), errs[KM[0]].min())
    hi = max(errs[LPV[0]].max(), errs[KM[0]].max())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True)
    for ax, (key, lab, _) in zip(axes, (LPV, KM)):
        ax.plot(lc[:, 0], lc[:, 1], "k-", lw=2, label=f"limit cycle ($u = {u0:+.0f}$)")
        sc = ax.scatter(x0s[:, 0], x0s[:, 1], c=errs[key], s=18, cmap="viridis",
                        norm=LogNorm(lo, hi))
        ax.set_title(f"{lab}\nmedian {np.median(errs[key]):.1f}%", fontsize=10)
        ax.set_xlabel("$x_1$"); ax.set_aspect("equal")
    axes[0].set_ylabel("$x_2$"); axes[0].legend(loc="upper left", fontsize=8)
    fig.colorbar(sc, ax=axes, label="forced prediction error [%] (1 s)")
    fig.suptitle("Square-wave forced prediction error by initial condition")
    fig.savefig(figdir / "fig7_forced_error_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # implied input gain map: truth vs the two models
    vmax = np.nanmax(np.abs(g_true))
    norm = TwoSlopeNorm(0.0, -vmax, vmax)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, (G, ttl) in zip(axes, [
        (g_true, r"true $\partial \dot x_2 / \partial u$"),
        (gains[LPV[0]], LPV[1] + f"\nRMSE {gain_table[LPV[0]]['rmse']:.3f}, sign "
         f"agreement {100*gain_table[LPV[0]]['sign_agreement']:.0f}%"),
        (gains[KM[0]], KM[1] + f"\nRMSE {gain_table[KM[0]]['rmse']:.3f}, sign "
         f"agreement {100*gain_table[KM[0]]['sign_agreement']:.0f}%"),
    ]):
        pc = ax.pcolormesh(UU, X2, G, shading="auto", cmap="RdBu_r", norm=norm)
        ax.contour(UU, X2, np.where(m, g_true, np.nan), levels=[0.0], colors="k",
                   linewidths=1.0)
        ax.set_xlabel("$u = x_3$"); ax.set_title(ttl, fontsize=9)
    axes[0].set_ylabel("$x_2$")
    fig.colorbar(pc, ax=axes, label=r"input gain on $\dot x_2$ (black: true sign flip)")
    fig.suptitle(r"Input gain implied by each lifted model, $(CAB)_2/(CB)_3$, on the "
                 r"slice $x_1 = 0$ (inside each level's limit cycle)", y=1.07)
    fig.savefig(figdir / "fig8_input_gain_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    (out / "results_control.json").write_text(json.dumps(results, indent=1))
    print(f"\nresults -> {out/'results_control.json'}")
    print(f"figures -> {figdir}/fig6..fig8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
