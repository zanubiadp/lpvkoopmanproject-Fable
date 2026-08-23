"""Non-input-affine benchmark: input extension + LPV B(x) vs constant B.

The plant is deliberately NOT control-affine (``vdp_nonaffine``):

    x2dot += w(x, u),   w(x, u) = (0.15 + 0.5 x2) sin(2u) + 0.25 x1 u^2,

with an input gain dw/du that flips sign in u (past the sin(2u) crest at
|u| = pi/4) and in the state (through 0.15 + 0.5 x2). Following the claim
in Korda & Mezic (2020) / Iacob et al. (2024) - used by the thesis for the
vehicle - the system is rewritten exactly as an input-affine one by
extending the state with x3 = u and taking v = udot as the new input.

This script then runs the full pipeline on the extended system and
quantifies the LPV input matrix's advantage over a constant one:

  1. autonomous data collection at constant-u levels (thesis protocol) and
     eigenfunction learning on the 3-D extended state, x3 handled exactly
     (lambda = 0, ``constant_components``);
  2. autonomous validation at *held-out* u levels (never trained on);
  3. input models on the SAME lifted dynamics:
       - ours:   LPV B(x) = dPhi/dx3 from autonomous data only
                 (kNN-smoothed gradient field + global polynomial variant),
       - baseline: Korda & Mezic constant B, least-squares fitted to
                 forced experiments (which the LPV never needs),
       - controls: constant reduction of the LPV field, and B = 0;
  4. three-way forced trajectory comparison (true / LPV / constant-B) on
     three input families: gentle (never leaves the monotone-gain region),
     dwell (sits past the gain flip), sweep (crosses it repeatedly);
  5. figures fig11-fig14 + reports/results_nonaffine.json.

Run:  python examples/run_nonaffine_benchmark.py            (~15 min)
      python examples/run_nonaffine_benchmark.py --quick    (~7 min)
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

from lpvkoopman import (            # function defined inside
    FitConfig,                          # pipeline.py
    extend_input,                       # nonaffine.py
    fit_b_field,                        # lpv.py
    fit_b_knn,                          # lpv.py
    fit_constant_b_forced,              # lpv.py
    fit_koopman_model,                  # pipeline.py
    make_extended_training_data,        # nonaffine.py
    predict_forced,                     # lpv.py
    simulate_true,                      # nonaffine.py
    u_profile_to_v,                     # nonaffine.py
    vdp_nonaffine,                      # nonaffine.py
)
from lpvkoopman.metrics import relative_rmse_percent
from lpvkoopman.nonaffine import frozen, level_limit_cycle
from lpvkoopman.spectrum import dmd_eigenvalues
from lpvkoopman.vdp_benchmark import sample_interior

TS = 0.01
U_MAX = 1.15  # test signals stay just inside the trained range [-1.2, 1.2]


def get_model(cache: Path, quick: bool):
    if cache.exists():
        return pickle.load(open(cache, "rb"))
    n_levels, n_traj = (17, 10) if quick else (25, 12)
    data = make_extended_training_data(
        vdp_nonaffine(),
        np.linspace(-1.2, 1.2, n_levels),
        n_traj_per_level=n_traj,
        horizon=7.0,
        ts=TS,
    )
    base = dmd_eigenvalues(data.X, data.ts)
    base = base[np.abs(base) > 1e-6]  # drop the exact lambda=0 of x3
    model, report = fit_koopman_model(
        data,
        FitConfig(
            n_eig=[20, 20, 1],
            constant_components=(2,),
            alpha_bounds=(-20.0, 1.5),
            base_eigs=base,
            boundary_rcond=1e-3,
        ),
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump((model, data, report.components), open(cache, "wb"))
    return model, data, report.components


def make_u_profile(kind: str, rng, K: int) -> np.ndarray:
    """Piecewise-linear u waypoints (K+1,) for the three test families."""
    tg = np.arange(K + 1) * TS
    if kind == "gentle":  # stays in the monotone-gain region |u| < pi/4
        a = rng.uniform(0.2, 0.4)
        c = rng.uniform(-0.1, 0.1)
        T = rng.uniform(0.8, 2.0)
    elif kind == "dwell":  # sits past the gain flip (|u| > pi/4)
        a = rng.uniform(0.15, 0.3)
        c = rng.choice([-1.0, 1.0]) * rng.uniform(0.85, 1.0)
        T = rng.uniform(1.0, 2.0)
    elif kind == "sweep":  # crosses the flip region repeatedly
        a = rng.uniform(0.6, 1.0)
        c = rng.uniform(-0.15, 0.15)
        T = rng.uniform(0.8, 1.6)
    else:
        raise ValueError(kind)
    ph = rng.uniform(0.0, 2.0 * np.pi)
    return np.clip(c + a * np.sin(2.0 * np.pi * tg / T + ph), -U_MAX, U_MAX)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument(
        "--outdir", default=str(Path(__file__).resolve().parents[1] / "reports")
    )
    args = ap.parse_args()
    out = Path(args.outdir)
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    sysna = vdp_nonaffine()
    ext = extend_input(sysna)

    # ---------------- fig11: the input law -------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    uu = np.linspace(-1.2, 1.2, 241)
    for x2 in [-0.6, -0.3, 0.0, 0.3, 0.6]:
        x = np.array([0.2, x2])
        axes[0].plot(uu, [sysna.f(x, u)[1] - sysna.f(x, 0.0)[1] for u in uu],
                     label=f"$x_2={x2:+.1f}$")
    axes[0].axhline(0.0, color="k", lw=0.6)
    axes[0].set_xlabel("$u$"); axes[0].set_ylabel("$w(x,u) - w(x,0)$")
    axes[0].set_title("input effect on $\\dot x_2$ (at $x_1$=0.2)")
    axes[0].legend(fontsize=8)
    X2, UU = np.meshgrid(np.linspace(-0.8, 0.8, 161), uu, indexing="ij")
    G = 2.0 * (0.15 + 0.5 * X2) * np.cos(2.0 * UU) + 0.5 * 0.2 * UU
    pc = axes[1].pcolormesh(UU, X2, G, shading="auto", cmap="RdBu_r",
                            vmin=-np.abs(G).max(), vmax=np.abs(G).max(),
                            rasterized=True)  # keep the SVG small
    axes[1].contour(UU, X2, G, levels=[0.0], colors="k", linewidths=1.2)
    fig.colorbar(pc, ax=axes[1], label=r"true input gain $\partial \dot x_2/\partial u$")
    axes[1].set_xlabel("$u$"); axes[1].set_ylabel("$x_2$")
    axes[1].set_title("gain map (black: sign flip); $x_1$=0.2")
    fig.suptitle("A deliberately non-affine input: sign-indefinite, state-coupled gain")
    fig.tight_layout()
    fig.savefig(figdir / "fig11_input_law.svg", dpi=150)
    plt.close(fig)

    # ---------------- model ------------------------------------------------
    t0 = time.perf_counter()
    model, data, comps = get_model(out / "model_nonaffine.pkl", args.quick)
    print(f"extended model ready ({time.perf_counter()-t0:.0f}s), "
          f"N={model.N}, {data.n_traj} trajectories")
    lo = data.flat_points().min(axis=0)
    hi = data.flat_points().max(axis=0)
    levels = np.asarray(data.meta["u_levels"])

    results: dict = {
        "protocol": {
            "u_levels": levels.tolist(),
            "n_traj_per_level": data.meta["n_traj_per_level"],
            "horizon": data.meta["horizon"],
            "ts": TS,
            "n_eig": model.block_sizes,
        }
    }

    # ---------------- autonomous validation at held-out levels ------------
    rng = np.random.default_rng(3)
    t_eval = np.arange(101) * TS
    held_out = [-1.15, -0.75, -0.35, 0.15, 0.55, 0.95, 1.15]
    n_ic_auto = 15 if args.quick else 30
    auto_rows = {}
    examples_auto = []
    for u in held_out:
        lc = level_limit_cycle(sysna, u)
        x0s = sample_interior(lc, n_ic_auto, rng, shrink=0.9)
        fro = frozen(sysna, u)
        errs, trajs = [], []
        for x0 in x0s:
            xt = fro.simulate(x0, t_eval)
            xp = model.predict(np.array([*x0, u]), t_eval)[:, :2]
            errs.append(relative_rmse_percent(xp, xt))
            trajs.append((xt, xp))
        errs = np.array(errs)
        auto_rows[f"{u:+.2f}"] = {
            "mean": float(errs.mean()),
            "median": float(np.median(errs)),
            "p90": float(np.percentile(errs, 90)),
            "max": float(errs.max()),
        }
        j = int(np.argsort(errs)[len(errs) // 2])  # median example
        examples_auto.append((u, lc, *trajs[j], errs[j]))
        print(f"  auto u={u:+.2f}: mean {errs.mean():6.2f}%  "
              f"median {np.median(errs):5.2f}%")
    results["autonomous_held_out_levels"] = auto_rows

    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.4))
    for k, (u, lc, xt, xp, e) in enumerate(examples_auto):
        ax = axes.ravel()[k]
        ax.plot(lc[:, 0], lc[:, 1], "k-", lw=1.0, alpha=0.5)
        ax.plot(xt[:, 0], xt[:, 1], "b-", lw=1.6, label="true")
        ax.plot(xp[:, 0], xp[:, 1], "r--", lw=1.4, label="Koopman")
        ax.plot(xt[0, 0], xt[0, 1], "ko", ms=4)
        ax.set_title(f"u = {u:+.2f} (held out), err {e:.1f}%", fontsize=9)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    med_all = np.median([r["median"] for r in auto_rows.values()])
    ax = axes.ravel()[-1]
    us = [float(k) for k in auto_rows]
    ax.semilogy(us, [r["median"] for r in auto_rows.values()], "o-", label="median")
    ax.semilogy(us, [r["p90"] for r in auto_rows.values()], "s--", label="p90")
    ax.set_xlabel("held-out $u$ level"); ax.set_ylabel("error [%]")
    ax.set_title("1 s prediction error vs level", fontsize=9)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "Extended-system autonomous prediction at u levels never seen in training "
        f"(median example per level; overall median {med_all:.1f}%)"
    )
    fig.tight_layout()
    fig.savefig(figdir / "fig12_autonomous_extended.svg", dpi=150)
    plt.close(fig)

    # ---------------- input models ----------------------------------------
    # B(x) sample cloud: interiors of the frozen-u limit cycles at midpoints
    # between training levels (plus the inner levels themselves)
    pts = []
    rng2 = np.random.default_rng(5)
    n_b_levels = 16 if args.quick else 24
    n_b_pts = 80 if args.quick else 150
    for u in np.linspace(-U_MAX, U_MAX, n_b_levels):
        lc = level_limit_cycle(sysna, u)
        xy = sample_interior(lc, n_b_pts, rng2, shrink=0.9)
        pts.append(np.column_stack([xy, np.full(len(xy), u)]))
    fit_pts = np.vstack(pts)
    du = levels[1] - levels[0]
    fd_opts = dict(h=np.array([2e-3, 2e-3, du / 2]), order=2)

    t0 = time.perf_counter()
    b_lpv = fit_b_knn(
        model, ext.g, fit_pts, k=20, f=ext.f, method="fd", options=fd_opts, blend=0.3
    )
    b_poly = fit_b_field(model, ext.g, fit_pts, degree=3, f=ext.f, method="fd", options=fd_opts)
    b_const_auto = fit_b_field(model, ext.g, fit_pts, degree=0, f=ext.f, method="fd", options=fd_opts)
    print(f"B(x) models from autonomous data ({time.perf_counter()-t0:.0f}s)")

    # Korda-Mezic constant B: least squares on forced experiments
    # (white-noise ZOH v decorrelates the input from the autonomous model
    # residual - block inputs gave a visibly worse-biased regression)
    rng3 = np.random.default_rng(9)
    forced = []
    t0 = time.perf_counter()
    for _ in range(60):
        u0 = rng3.uniform(-1.0, 1.0)
        lc = level_limit_cycle(sysna, u0)
        x0 = sample_interior(lc, 1, rng3, shrink=0.8)[0]
        K = 150
        v = rng3.uniform(-3.0, 3.0, K)
        u, vv = u0, v.copy()
        for k in range(K):
            un = u + vv[k] * TS
            if abs(un) > U_MAX:
                vv[k] = 0.0
            else:
                u = un
        forced.append((simulate_true(sysna, x0, u0, vv, TS), vv))
    b_km = fit_constant_b_forced(model, forced, TS)
    print(f"Korda-Mezic constant B from {len(forced)} forced experiments "
          f"({time.perf_counter()-t0:.0f}s)")
    b_zero = np.zeros((model.N, 1))

    candidates = [
        ("lpv_knn", b_lpv),
        ("lpv_poly3", b_poly),
        ("km_const_forced", b_km),
        ("const_auto", b_const_auto),
        ("zero_B", b_zero),
    ]

    # ---------------- forced comparison ------------------------------------
    n_test = 15 if args.quick else 40
    K = 150
    suites = {}
    showcase = {}
    for kind in ["gentle", "dwell", "sweep"]:
        rngt = np.random.default_rng(17)
        rows = {nm: [] for nm, _ in candidates}
        cases = []
        t0 = time.perf_counter()
        for j in range(n_test):
            u_way = make_u_profile(kind, rngt, K)
            u0, v = u_profile_to_v(u_way, TS)
            lc = level_limit_cycle(sysna, u0)
            x0 = sample_interior(lc, 1, rngt, shrink=0.75)[0]
            xt = simulate_true(sysna, x0, u0, v, TS)
            xe0 = np.array([*x0, u0])
            preds = {}
            for nm, bb in candidates:
                xp = predict_forced(model, bb, xe0, v, TS, x_clip=(lo, hi))
                rows[nm].append(relative_rmse_percent(xp[:, :2], xt[:, :2]))
                preds[nm] = xp
            cases.append((xt, preds, u_way))
        suites[kind] = {
            nm: {
                "mean": float(np.mean(e)),
                "median": float(np.median(e)),
                "p90": float(np.percentile(e, 90)),
            }
            for nm, e in rows.items()
        }
        suites[kind]["_errors"] = {nm: list(map(float, e)) for nm, e in rows.items()}
        # showcase: case with median LPV error (representative, not cherry-picked)
        j_show = int(np.argsort(rows["lpv_knn"])[len(cases) // 2])
        showcase[kind] = cases[j_show]
        print(f"  {kind:6s} ({time.perf_counter()-t0:.0f}s): " + "  ".join(
            f"{nm} {np.median(e):.1f}%" for nm, e in rows.items()))
    results["forced_suites"] = {
        k: {nm: v for nm, v in s.items() if nm != "_errors"} for k, s in suites.items()
    }

    # ---------------- fig13: the money plot --------------------------------
    t_plot = np.arange(K + 1) * TS
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 8.2), sharex=True)
    for col, kind in enumerate(["gentle", "dwell", "sweep"]):
        xt, preds, u_way = showcase[kind]
        e_lpv = relative_rmse_percent(preds["lpv_knn"][:, :2], xt[:, :2])
        e_km = relative_rmse_percent(preds["km_const_forced"][:, :2], xt[:, :2])
        for row in range(2):
            ax = axes[row, col]
            ax.plot(t_plot, xt[:, row], "b-", lw=1.8, label="true")
            ax.plot(t_plot, preds["lpv_knn"][:, row], "r--", lw=1.5,
                    label="LPV $B(x)$ (ours, knn, autonomous data)")
            ax.plot(t_plot, preds["lpv_poly3"][:, row], "m--", lw=1.5,
                    label="LPV $B(x)$ (ours, poly, autonomous data)")
            ax.plot(t_plot, preds["km_const_forced"][:, row], "g-.", lw=1.5,
                    label="constant $B$ (Korda-Mezic, forced data)")
            if row == 0:
                ax.set_title(f"{kind}:  LPV {e_lpv:.1f}%  vs  const {e_km:.1f}%",
                             fontsize=10)
            ax.set_ylabel(f"$x_{row+1}$")
        ax = axes[2, col]
        ax.plot(t_plot, u_way, "k-", lw=1.4)
        ax.axhspan(np.pi / 4, 1.25, color="orange", alpha=0.15)
        ax.axhspan(-1.25, -np.pi / 4, color="orange", alpha=0.15)
        ax.set_ylim(-1.25, 1.25)
        ax.set_ylabel("$u$"); ax.set_xlabel("t [s]")
    axes[0, 0].legend(fontsize=8, loc="best")
    axes[2, 0].text(0.02, 0.85, "shaded: reversed-gain region",
                    transform=axes[2, 0].transAxes, fontsize=8)
    fig.suptitle("Forced prediction on the non-affine plant: true vs LPV B(x) vs constant B "
                 "(median-LPV-error case per family)")
    fig.tight_layout()
    fig.savefig(figdir / "fig13_forced_comparison.svg", dpi=150)
    plt.close(fig)

    # ---------------- fig14: error statistics ------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4), sharey=True)
    show = ["lpv_knn", "lpv_poly3", "km_const_forced", "const_auto", "zero_B"]
    labels = ["LPV kNN\n(ours)", "LPV poly3\n(ours)", "const B\n(K-M, forced)",
              "const B\n(auton.)", "B = 0"]
    for ax, kind in zip(axes, ["gentle", "dwell", "sweep"]):
        errs = [suites[kind]["_errors"][nm] for nm in show]
        bp = ax.boxplot(errs, tick_labels=labels, showfliers=False, patch_artist=True)
        for patch, c in zip(bp["boxes"], ["#d62728", "#ff9896", "#2ca02c", "#98df8a", "#7f7f7f"]):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax.set_yscale("log")
        ax.set_title(f"{kind} (n={n_test})")
        ax.grid(alpha=0.3, axis="y")
        ax.tick_params(axis="x", labelsize=7)
    axes[0].set_ylabel("forced prediction error [%] (1.5 s)")
    fig.suptitle("Forced prediction error by input model")
    fig.tight_layout()
    fig.savefig(figdir / "fig14_forced_error_stats.svg", dpi=150)
    plt.close(fig)

    (out / "results_nonaffine.json").write_text(json.dumps(results, indent=1))
    print(f"\nresults -> {out/'results_nonaffine.json'}")
    print(f"figures -> {figdir}/fig11..fig14")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
