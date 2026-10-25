"""Autonomous benchmark of the input-extended Van der Pol model: reproduces
every number and figure in reports/validation_vdp.md.

The plant is ``vdp_nonaffine`` in its exact input-affine extension
(x3 := u, x3dot = v); autonomously x3 is constant, so the extended system's
autonomous flow is the family of frozen-u Van der Pol slices. The model
learns eigenfunctions Phi(x1, x2, x3) on all slices at once, x3 itself
fitted exactly (lambda = 0). Since w(x, 0) = 0, the plane x3 = 0 IS the
scaled Van der Pol of Korda & Mezic (2020), so their Table I applies there
as a published reference.

Protocol:
  * data: 25 u-levels on [-1.2, 1.2] x 12 trajectories from a circle of
    radius 0.05 around each level's equilibrium, Ts = 0.01;
  * paper configuration: 5 s records, N in {4, 8, 12, 16, 20} eigenfunctions
    for (x1, x2) (+1 exact for x3), lattice vs optimized;
  * extended configuration: 7 s records, N in {20, 30, 40} (+1); the N = 40
    optimized model is the one every other script uses;
  * evaluation (a): 500 ICs uniform over the x3 = 0 limit-cycle interior
    (seed 7), 1 s horizon, eq. (55), overall and split into the
    near-limit-cycle band (distance < 0.05) and the rest - Table I protocol;
  * evaluation (b): 30 ICs at each of 7 u levels never seen in training.

Run:  python examples/run_vdp_benchmark.py [--quick] [--outdir reports]
Runtime: ~30 minutes full (one 3-D Delaunay triangulation per fitted model),
~10 min with --quick.
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
from scipy.spatial import cKDTree

from extended_vdp import TS, fit_extended_model, get_model, truncate
from lpvkoopman import vdp_nonaffine
from lpvkoopman.metrics import relative_rmse_percent
from lpvkoopman.nonaffine import frozen, level_limit_cycle
from lpvkoopman.spectrum import lattice_spectrum
from lpvkoopman.vdp_benchmark import sample_interior

# Published reference values: Korda & Mezic (2020), Table I, uncontrolled,
# average over 500 random ICs in the interior of the limit cycle - the
# x3 = 0 plane of the extended system.
PAPER_TABLE1 = {
    "lattice": {4: 100.4, 8: 95.6, 12: 51.34, 16: 13.31, 20: 5.44},
    "optimized": {4: 24.0, 8: 9.7, 12: 5.1, 16: 2.4, 20: 1.4},
}
LC_OMEGA = 1.1074  # rad/s, fundamental frequency of the u = 0 limit cycle
HELD_OUT = [-1.15, -0.75, -0.35, 0.15, 0.55, 0.95, 1.15]


def predict_errors(model, x0s, u, x_true, t):
    """eq. (55) on (x1, x2) per IC; x0s (Q, 2) in the plane x3 = u."""
    X0 = np.column_stack([x0s, np.full(len(x0s), u)])
    Z0 = model.lift(X0)
    fallbacks = model.last_lift_fallbacks
    e = np.array([
        relative_rmse_percent((model.propagate(z0, t) @ model.C.T)[:, :2], xt)
        for z0, xt in zip(Z0, x_true)
    ])
    return e, fallbacks


def evaluate(model, plane0, held, t, fit_s, rep):
    x0s, x_true, d_lc = plane0
    e, fb = predict_errors(model, x0s, 0.0, x_true, t)
    e_held = np.concatenate([predict_errors(model, h[0], u, h[1], t)[0]
                             for u, h in held.items()])
    row = {
        "mean": float(e.mean()),
        "median": float(np.median(e)),
        "p90": float(np.percentile(e, 90)),
        "max": float(e.max()),
        "near_lc_mean": float(e[d_lc < 0.05].mean()),
        "interior_mean": float(e[d_lc >= 0.05].mean()),
        "hull_fallbacks": fb,
        "held_out_mean": float(e_held.mean()),
        "held_out_median": float(np.median(e_held)),
        "fit_seconds": round(fit_s, 1),
        "spectra": [c["spectrum"] for c in rep],
    }
    return row, e


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="reduced sweep for CI")
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parents[1] / "reports"))
    args = ap.parse_args()

    out = Path(args.outdir)
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    sysna = vdp_nonaffine()
    n_test = 150 if args.quick else 500
    n_held = 15 if args.quick else 30
    t_pred = np.arange(101) * TS

    print("== reference model, data and test sets ==")
    t0 = time.perf_counter()
    best_model, data7, best_comps = get_model(out / "model_nonaffine.pkl", args.quick)
    data5 = truncate(data7, 5.0)
    lc = level_limit_cycle(sysna, 0.0)
    x0s = sample_interior(lc, n_test, np.random.default_rng(7))
    x_true = frozen(sysna, 0.0).simulate_many(x0s, t_pred)
    d_lc = cKDTree(lc).query(x0s)[0]
    plane0 = (x0s, x_true, d_lc)
    rng = np.random.default_rng(3)
    held = {}
    for u in HELD_OUT:
        xh = sample_interior(level_limit_cycle(sysna, u), n_held, rng, shrink=0.9)
        held[u] = (xh, frozen(sysna, u).simulate_many(xh, t_pred))
    print(f"   {time.perf_counter()-t0:.1f}s ({data7.n_traj} trajectories, "
          f"{n_test} test ICs on x3 = 0, {n_held} x {len(HELD_OUT)} held-out)")

    results: dict = {"n_test": n_test, "paper_table1": PAPER_TABLE1,
                     "held_out_levels": HELD_OUT, "n_held_per_level": n_held}
    keep = {}  # the models the figures need: (model, errors on x3 = 0, spectrum base)

    print("== paper configuration: 5 s data, Table I sweep ==")
    n_sweep = [12, 20] if args.quick else [4, 8, 12, 16, 20]
    results["paper_config"] = {}
    for n_eig in n_sweep:
        for opt in (False, True):
            tag = "optimized" if opt else "lattice"
            t1 = time.perf_counter()
            model, rep = fit_extended_model(data5, n_eig // 2, optimize=opt)
            row, e = evaluate(model, plane0, held, t_pred,
                              time.perf_counter() - t1, rep.components)
            results["paper_config"][f"N{n_eig}_{tag}"] = row
            print(f"   N={n_eig:<2d}+1 {tag:9s} mean {row['mean']:7.2f}%  "
                  f"interior {row['interior_mean']:6.2f}%  near-LC "
                  f"{row['near_lc_mean']:8.2f}%  held-out med "
                  f"{row['held_out_median']:6.2f}%  (paper: "
                  f"{PAPER_TABLE1[tag].get(n_eig)}%)  [{row['fit_seconds']}s]")
            if n_eig == 20 and opt:
                keep["paper_N20_optimized"] = (model, e, rep.dmd_base)
            del model

    print("== extended configuration: 7 s data ==")
    results["extended_config"] = {}
    ext_sweep = [40] if args.quick else [20, 30, 40]
    for n_eig in ext_sweep:
        for opt in (False, True):
            tag = "optimized" if opt else "lattice"
            t1 = time.perf_counter()
            if n_eig == 40 and opt:  # the shared reference model, cached
                model, comps, base = best_model, best_comps, None
            else:
                model, rep = fit_extended_model(data7, n_eig // 2, optimize=opt)
                comps, base = rep.components, rep.dmd_base
            row, e = evaluate(model, plane0, held, t_pred,
                              time.perf_counter() - t1, comps)
            if base is None and n_eig == 40 and opt:
                row["fit_seconds"] = None  # loaded from model_nonaffine.pkl
            results["extended_config"][f"N{n_eig}_{tag}"] = row
            print(f"   N={n_eig:<2d}+1 {tag:9s} mean {row['mean']:7.2f}%  "
                  f"interior {row['interior_mean']:6.2f}%  near-LC "
                  f"{row['near_lc_mean']:8.2f}%  held-out med "
                  f"{row['held_out_median']:6.2f}%  "
                  f"[{'cached' if row['fit_seconds'] is None else row['fit_seconds']}s]")
            if n_eig == 40 and not opt:
                keep["lattice_base"] = base
            if n_eig == 40 and opt:
                keep["best"] = (model, e, None)

    best_key = "extended config, N=40+1"
    best_err = keep["best"][1]

    # ------------------------------------------------------------------ figures
    print("== figures ==")
    train_pts = data7.flat_points()
    levels = np.asarray(data7.meta["u_levels"])

    # 1. the extended state space and the x3 = 0 plane
    fig = plt.figure(figsize=(13, 5.8))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    sub = train_pts[::37]
    ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], c=sub[:, 2], cmap="coolwarm",
               s=0.6, alpha=0.35, rasterized=True)
    for u in levels[::4]:
        lcu = level_limit_cycle(sysna, u)
        ax.plot(lcu[:, 0], lcu[:, 1], np.full(len(lcu), u), "k-", lw=0.9)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_zlabel("$x_3 = u$")
    ax.set_title(f"Training cloud: {len(levels)} constant-$u$ planes\n"
                 "(black: frozen-$u$ limit cycles, every 4th level)")
    ax.view_init(elev=18, azim=-60)
    ax = fig.add_subplot(1, 2, 2)
    on0 = np.abs(train_pts[:, 2]) < 1e-12
    ax.plot(train_pts[on0][::5, 0], train_pts[on0][::5, 1], ".", ms=1,
            color="0.75", label="training samples, $u = 0$ plane")
    ax.plot(lc[:, 0], lc[:, 1], "k-", lw=2, label="limit cycle ($u = 0$)")
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(0.05 * np.cos(th), 0.05 * np.sin(th), "b-", lw=1.5,
            label="IC circle r=0.05")
    sc = ax.scatter(x0s[:, 0], x0s[:, 1], c=np.log10(best_err), s=14,
                    cmap="viridis", zorder=3)
    fig.colorbar(sc, ax=ax, label=r"$\log_{10}$ prediction error [%]")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title(f"$x_3 = 0$ plane: test errors ({best_key})")
    ax.legend(loc="upper left", fontsize=8); ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(figdir / "fig1_phase_portrait.png", dpi=150)
    plt.close(fig)

    # 2. eigenvalues of the (x1, x2) blocks: lattice start vs learned
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, key, ttl in [
        (axes[0], "paper_N20_optimized", "paper config (5 s, N=20+1)"),
        (axes[1], "best", "extended config (7 s, N=40+1)"),
    ]:
        mdl, _, base = keep[key]
        n_half = mdl.block_sizes[0]
        base = keep["lattice_base"] if base is None else base
        lat = lattice_spectrum(base, n_half).to_complex()
        for k in range(-12, 13):
            ax.axhline(k * LC_OMEGA, color="0.9", lw=0.7, zorder=0)
        ax.scatter(lat.real, lat.imag, marker="s", s=45, facecolor="none",
                   edgecolor="tab:blue", label="DMD lattice (start)")
        for i, mk, col in [(0, "x", "tab:red"), (1, "+", "tab:orange")]:
            ev_ = mdl.components[i].spec.to_complex()
            ax.scatter(ev_.real, ev_.imag, marker=mk, s=50, color=col,
                       label=f"learned, $x_{i+1}$ block")
        ax.set_xlabel(r"Re $\lambda$"); ax.set_title(ttl)
        ax.axvline(0, color="0.8", lw=0.7)
    axes[0].set_ylabel(r"Im $\lambda$")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Continuous-time eigenvalues of the extended model (the $x_3$ "
                 "block is the exact $\\lambda = 0$; lines: $u=0$ limit-cycle "
                 "harmonics $k\\,\\omega_0$)")
    fig.tight_layout(); fig.savefig(figdir / "fig2_eigenvalues.png", dpi=150)
    plt.close(fig)

    # 3. error vs N against the published Table I (x3 = 0 plane)
    if not args.quick:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        Ns = n_sweep
        for tag, color in [("lattice", "tab:blue"), ("optimized", "tab:red")]:
            pc = results["paper_config"]
            ax.semilogy(Ns, [PAPER_TABLE1[tag][n] for n in Ns], "o--",
                        color=color, alpha=0.45, label=f"paper Table I, 2-D ({tag})")
            ax.semilogy(Ns, [pc[f"N{n}_{tag}"]["interior_mean"] for n in Ns],
                        "s-", color=color, label=f"ours, $x_3=0$, d>0.05 ({tag})")
            ax.semilogy(Ns, [pc[f"N{n}_{tag}"]["mean"] for n in Ns], "^:",
                        color=color, alpha=0.8, label=f"ours, $x_3=0$, full ({tag})")
        ec = results["extended_config"]
        ax.semilogy(ext_sweep, [ec[f"N{n}_optimized"]["mean"] for n in ext_sweep],
                    "d-", color="tab:green", lw=2,
                    label="ours, 7 s data, $x_3=0$, full (optimized)")
        ax.semilogy(ext_sweep, [ec[f"N{n}_optimized"]["held_out_mean"] for n in ext_sweep],
                    "d--", color="tab:olive", lw=1.5,
                    label="ours, 7 s data, held-out $u$ levels (optimized)")
        ax.set_xlabel("eigenfunctions for $(x_1, x_2)$, N (+1 exact for $x_3$)")
        ax.set_ylabel("mean prediction error over 1 s [%]")
        ax.set_title("Prediction error vs. lift dimension (extended VdP, autonomous)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(figdir / "fig3_error_vs_N.png", dpi=150)
        plt.close(fig)

    # 4. example predictions with the reference model, x3 = 0 plane
    show_idx = [int(np.argmin(np.abs(d_lc - q))) for q in
                (np.percentile(d_lc, [85, 50, 15, 2]))]
    fig, axes = plt.subplots(2, len(show_idx), figsize=(3.1 * len(show_idx), 5.4),
                             sharex=True)
    for col, j in enumerate(show_idx):
        xp = best_model.predict(np.array([*x0s[j], 0.0]), t_pred)
        for rowi in range(2):
            ax = axes[rowi, col]
            ax.plot(t_pred, x_true[j][:, rowi], "b-", lw=1.8, label="true")
            ax.plot(t_pred, xp[:, rowi], "r--", lw=1.6, label="predicted")
            if rowi == 0:
                ax.set_title(f"d(x0, LC)={d_lc[j]:.3f}\nerr={best_err[j]:.2f}%",
                             fontsize=9)
            ax.set_ylabel(f"$x_{rowi+1}$" if col == 0 else "")
            if rowi == 1:
                ax.set_xlabel("t [s]")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"True vs. predicted trajectories on the $x_3 = u = 0$ plane ({best_key})")
    fig.tight_layout(); fig.savefig(figdir / "fig4_predictions.png", dpi=150)
    plt.close(fig)

    # 5. error over time percentiles: x3 = 0 plane and held-out levels
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for key, color, lab in [("paper_N20_optimized", "tab:orange", "paper config, N=20+1"),
                            ("best", "tab:green", best_key)]:
        mdl = keep[key][0]
        sets = [[(x0s, x_true, 0.0)],
                [(h[0], h[1], u) for u, h in held.items()]]
        for ax, sset in zip(axes, sets):
            errt = []
            for xs, xt, u in sset:
                Z0 = mdl.lift(np.column_stack([xs, np.full(len(xs), u)]))
                for z0, xtj in zip(Z0, xt):
                    xp = (mdl.propagate(z0, t_pred) @ mdl.C.T)[:, :2]
                    errt.append(np.linalg.norm(xp - xtj, axis=1))
            errt = np.array(errt)
            ax.semilogy(t_pred, np.percentile(errt, 50, axis=0), color=color,
                        label=f"{lab} (median)")
            ax.fill_between(t_pred, np.percentile(errt, 10, axis=0),
                            np.percentile(errt, 90, axis=0), color=color, alpha=0.18)
    axes[0].set_title("$x_3 = 0$ plane (500 ICs)")
    axes[1].set_title(f"held-out $u$ levels ({len(HELD_OUT)} x {n_held} ICs)")
    for ax in axes:
        ax.set_xlabel("prediction time [s]")
        ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=9)
    axes[0].set_ylabel(r"$\|(x_1, x_2)_{pred}-(x_1, x_2)_{true}\|$")
    fig.suptitle("State error over the prediction horizon (median, 10-90%)")
    fig.tight_layout(); fig.savefig(figdir / "fig5_error_vs_time.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------ outputs
    np.savez(
        out / "benchmark_arrays.npz",
        x0s=x0s,
        d_lc=d_lc,
        errors_best=best_err,
        errors_paper_n20=keep["paper_N20_optimized"][1],
        limit_cycle=lc,
    )
    (out / "results.json").write_text(json.dumps(results, indent=1))
    print(f"\nresults  -> {out/'results.json'}")
    print(f"figures  -> {figdir}/fig1..fig5")
    b = results["extended_config"]["N40_optimized"]
    print(f"\nHEADLINE ({best_key}, x3 = 0 plane): mean {b['mean']:.2f}%  median "
          f"{b['median']:.2f}%  near-LC {b['near_lc_mean']:.2f}%  held-out median "
          f"{b['held_out_median']:.2f}%  (paper 2-D, N=20: 1.4%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
