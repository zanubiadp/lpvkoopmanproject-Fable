# lpvkoopman — LPV-Koopman modeling, rebuilt from scratch

A clean Python implementation of data-driven Koopman eigenfunction models
for prediction and control, following

* M. Korda, I. Mezić, *Optimal Construction of Koopman Eigenfunctions for
  Prediction and Control*, IEEE TAC 65(12), 2020 — the eigenfunction
  learning method and the Van der Pol benchmark (Sec. VI-A, Table I);
* the thesis *Koopman-based MPC for High Performance Vehicles* (and its
  MATLAB implementation) — used as the behavioral reference, including the
  Appendix-A real conjugate-pair reformulation and the constant-u data
  protocol for input-extended systems;
* Iacob et al. 2024 — state-dependent input matrix B(x) = ∂Φ/∂x · g_c(x),
  built here **from autonomous data only** (no forced experiments), plus
  Koopman-LPV MPC on the lifted model.

**Scope: autonomous prediction, the LPV input matrix B(x), MPC, and
non-input-affine systems via exact input extension** — all validated on
Van der Pol benchmarks. Headline numbers
([`reports/validation_vdp.md`](reports/validation_vdp.md),
[`reports/validation_vdp_control.md`](reports/validation_vdp_control.md),
[`reports/validation_nonaffine.md`](reports/validation_nonaffine.md)):

| capability | result | published reference |
|---|---|---|
| autonomous prediction, paper config (N=20, 5 s) | 1.81 % (away from the limit cycle) | 1.4 % (Korda & Mezić Table I) |
| **autonomous prediction, N=40, 7 s** | **0.83 % over the full limit-cycle interior** | — |
| forced prediction, sine 60 ms, LPV B(x) from autonomous data | **1.30 %** | 2.9 % (constant B fitted to *forced* data) |
| forced prediction, square 300 ms | 8.4 % (LPV) vs 12.6 % (constant-B reduction) | 6.0 % |
| MPC: stabilize the unstable origin from 92 % of the LC radius | ‖x(6 s)‖ = 5.4e-5, 11 ms/solve | — |
| MPC: track x₁ = 0.3 sin(1.2t) | RMS 0.0041 (1.4 % of amplitude) | — |
| **non-affine plant, extended system: autonomous prediction at held-out u levels** | **median 1.3 %** | — |
| **non-affine plant, forced prediction 1.5 s: LPV B(x) vs constant B** | **3.2–7.9 % vs 60–229 % (K-M constant, forced data) / 3.4–14.5 % (best constant)** | — |

Total model fit ~35 s (2-D) / ~3.5 min (extended 3-D); B(x) field fit
seconds; MPC solves ~11 ms — replacing a multi-week CMA-ES run.

## The method in five lines

For an autonomous system ẋ = f(x), a Koopman eigenfunction obeys
φ(x(t)) = e^{λt} φ(x(0)) along trajectories. Given M_t trajectories sampled
every T_s, fix candidate eigenvalues Λ = (λ_1..λ_N) and let the value of
each eigenfunction at each trajectory's initial condition ("boundary
values" g) be free. Then (Korda & Mezić eq. 33–35) the reconstruction of a
state component h along the data is **linear** in g:

    min_g Σ_j ‖h^{(j)} − V(Λ) g^{(j)}‖²,   V(Λ)[k,m] = e^{λ_m kT_s},

solved per trajectory by one shared QR. Eliminating g gives the
variable-projection cost J(Λ) = Σ_j ‖(I − P_{V(Λ)}) h^{(j)}‖² (eq. 39),
minimized over Λ with its exact analytic gradient (eq. 40), starting from
the lattice of integer combinations of the DMD eigenvalues. Off-data
evaluation of Φ (needed to lift a new initial condition) interpolates the
tabulated values e^{λ_m kT_s} g_m^{(j)} over the trajectory cloud.
Prediction is then z(t) = e^{At} Φ(x0), x̂ = C z — all real, in closed form.

## Non-affine inputs: the extension trick and the LPV-vs-constant-B study

Both Korda & Mezić and Iacob et al. observe that a system that is **not**
control-affine — ẋ = f(x, u) with u entering arbitrarily — can be rewritten
*exactly* as a control-affine one by extending the state with the input:

    x₃ := u,   ẋ₃ = v   (v = u̇ is the new input)
    ⇒  ẋ_ext = f_ext(x_ext) + g_ext v,   g_ext = (0, 0, 1) constant.

This is the rewrite the thesis uses for the (naturally non-affine) vehicle.
The whole input nonlinearity w(x, u) moves into the *autonomous* part of the
extended system and is absorbed by eigenfunctions Φ(x₁, x₂, x₃); the LPV
input matrix becomes B(x) = ∂Φ/∂x₃ — state-dependent exactly because the
plant's input gain is.

`examples/run_nonaffine_benchmark.py` validates the full chain on a scaled
Van der Pol with a deliberately hostile input law
(`nonaffine.vdp_nonaffine`),

    ẋ₂ += (0.15 + 0.5 x₂)·sin(2u) + 0.25 x₁ u²,

whose input gain flips sign both in u (past the sin(2u) crest, |u| > π/4)
and in the state (at x₂ = −0.3) — the regime where a constant lifted B is
structurally wrong (fig11):

1. **Data (thesis protocol).** Autonomous experiments of the extended
   system at 25 constant-u levels: ICs on a circle around each level's
   (u-dependent) equilibrium, so the 3-D cloud is a stack of u-planes
   (`make_extended_training_data`). x₃ = u is fitted *exactly* — one λ = 0
   eigenfunction, `FitConfig.constant_components` — and the input
   integration x₃ ← x₃ + T_s v is exact in the lifted model by
   construction.
2. **Autonomous validation.** 1 s predictions at 7 u levels *never seen in
   training*: median 1.3 % (fig12) — the eigenfunction construction and
   its interpolation survive the extra dimension.
3. **The comparison** (fig13 — true / LPV / constant-B trajectories,
   fig14 — statistics). Same lifted A for everyone; only B differs:
   ours = ∂Φ/∂x₃ from **autonomous data only** (FD across the u-planes,
   kNN-smoothed with PDE-residual weights, confidence-blended toward the
   robust constant — `fit_b_knn`); baseline = Korda & Mezić's **constant B
   regressed from 60 forced experiments** (`fit_constant_b_forced`);
   controls = constant reduction of our field and B = 0. Median forced
   prediction error over 1.5 s (three input families ×40 runs):

   | family | LPV B(x) (ours) | K-M const (forced) | const (auton.) | B = 0 |
   |---|---|---|---|---|
   | gentle (\|u\| < π/4) | **3.5 %** | 90.0 % | 7.2 % | 8.5 % |
   | dwell (past the flip) | **3.2 %** | 60.7 % | 3.4 % | 4.6 % |
   | sweep (crosses it) | **7.9 %** | 229.2 % | 14.5 % | 17.6 % |

   The LPV matrix beats the best constant ≈ 2× where the input matters and
   never loses; the forced-data constant is worse than *no input model at
   all*, because its one-step regression soaks up autonomous model error
   correlated with the input (mechanism, and every failure mode hit on the
   way, in [`reports/validation_nonaffine.md`](reports/validation_nonaffine.md)).

## Repository layout

```
src/lpvkoopman/
  systems.py        # ContinuousSystem, VdP variants, trajectory generation
                    # (incl. bidirectional sampling), TrajectoryData
  spectrum.py       # Spectrum (conjugate-closed eigenvalue sets), DMD,
                    # eigenvalue lattice + budget-constrained selection
  varpro.py         # the core: real cos/sin basis, boundary least squares,
                    # projected cost + exact gradient, L-BFGS-B refinement
  model.py          # KoopmanEigenModel: real block-diagonal A, C, Delaunay
                    # interpolation of eigenfunctions (n-D), closed-form
                    # predict
  pipeline.py       # fit_koopman_model(data, FitConfig): incl. exact
                    # constant components (x3 = u) and boundary_rcond
  gradients.py      # dPhi/dx: moving-least-squares (n-D) & FD estimators
                    # (per-dimension steps), eigenfunction-PDE residual
  lpv.py            # B(x) = (dPhi/dx) g_c(x): BMap, residual-weighted
                    # polynomial field fit (n-D), kNN field with confidence
                    # blending, Korda-Mezić constant-B regression from
                    # forced data, exact ZOH discretization, forced LPV
                    # rollout (with scheduling-state clipping)
  nonaffine.py      # non-affine plants, the exact input extension x3 = u,
                    # constant-u-level data collection (thesis protocol),
                    # ground-truth forced simulation, u-profile -> v
  mpc.py            # Koopman-LPV MPC (condensed box-QP via lsq_linear),
                    # closed-loop simulator against the true plant
  vdp_benchmark.py  # Korda-Mezić VdP protocol: data, limit cycle, metrics
  metrics.py        # eq. (55) relative RMSE
examples/
  run_vdp_benchmark.py         # autonomous: every number/figure (~3 min)
  diagnose_near_lc.py          # oracle-lift error-budget experiment (~2 min)
  run_vdp_control_benchmark.py # B(x) + forced prediction vs paper (~3 min)
  run_vdp_mpc.py               # MPC demos: stabilization + tracking (~4 min)
  run_nonaffine_benchmark.py   # non-affine plant: extension + LPV-vs-const
                               # three-way comparison (~15 min)
crosscheck/
  run_crosscheck.py      # numerical cross-check vs. the thesis MATLAB code
  crosscheck.m           # (runs the actual .m files under GNU Octave)
tests/                   # 44 unit/integration tests (~40 s)
reports/
  validation_vdp.md          # autonomous validation report
  validation_vdp_control.md  # B(x) + MPC validation report
  validation_nonaffine.md    # non-affine extension + LPV-vs-const-B report
```

## Install and reproduce

```bash
pip install -e .[dev]
pytest                                        # 31 tests, ~10 s
python examples/run_vdp_benchmark.py          # autonomous tables + figures
python examples/diagnose_near_lc.py           # near-LC error budget
python examples/run_vdp_control_benchmark.py  # B(x) + forced prediction
python examples/run_vdp_mpc.py                # MPC closed-loop demos
python examples/run_nonaffine_benchmark.py    # non-affine extension study
# cross-check vs. the thesis MATLAB implementation (needs octave):
python crosscheck/run_crosscheck.py --matlab-repo /path/to/Koopman-MPC-thesis-MATLAB
```

Dependencies: numpy, scipy, matplotlib. Nothing exotic.

## Design decisions and deviations from the thesis / previous Python port

| # | Choice here | Reference behavior | Why |
|---|---|---|---|
| 1 | Eigenvalues start at the **DMD lattice** and are refined by **L-BFGS-B with the exact variable-projection gradient** (seconds). | Thesis `build_A.m`: same idea via fmincon. Thesis `build_A_adaptive.m` / previous Python port: **random init in [−10,10]^N, CMA-ES, gradient discarded** (weeks of runtime). | This is Korda & Mezić's own recipe; the cost is smooth with a cheap exact gradient, so a quasi-Newton method from an informed start converges in seconds. Verified insensitive to restarts/seeds. |
| 2 | **Real conjugate-pair parametrization everywhere** (pairs → cos/sin columns, real block-diagonal A with [[α, β], [−β, α]] blocks; predictions real by construction; closed-form propagation, no lifted-ODE integration). | Thesis Appendix A introduces the same cos/sin basis for the *cost*; the surrounding pipeline still moved through complex diagonals and `real(C z)`. The paper optimizes unconstrained complex λ (conjugacy not guaranteed). | Halves the optimization variables, guarantees a real model, and removes RK4 integration error in prediction (the reference's lifted RK4 also contained a k4 bug — `x + k1·dt` instead of `x + k3·dt`). Equivalence with the complex formulation is proven in tests to machine precision. |
| 3 | Boundary least squares solved **per trajectory against one shared, column-equilibrated QR** (`numpy.lstsq` on V of size (M_s+1)×N_i). | Reference builds the full sparse block matrix L ((M_s+1)M_t × N_i M_t) and forms `inv(L'L)` explicitly. | Same minimizer (cross-checked to 1e−13), ~1000× smaller problem, and the explicit Gram inverse costs ~8 significant digits (measured in the cross-check). |
| 4 | **Plain 2-D Van der Pol state.** | Previous port carried an input-as-state third dimension even for autonomous modeling. | Out of scope for the autonomous milestone; smaller, better-conditioned problem. The control hook stays in `ContinuousSystem.g`. |
| 5 | Training ICs on the **r = 0.05 circle** (paper protocol); test ICs uniform over the limit-cycle interior. | Previous port sampled box edges at buffer ≈ 3 — far outside the limit cycle, where the interior eigenfunction construction does not even apply (different basin/spectral structure). | Matches the published benchmark and the theory's domain of validity. |
| 6 | **Lattice ordering policy**: ascending degree, pairs before reals within a degree, λ=0 last; budget truncation keeps conjugate closure. | Paper says "first N/2 eigenvalues of Λ_lat" (order unspecified); thesis used `monpowers` order. | Deterministic, conjugate-closed for every budget; for N_i = 10 it reproduces exactly the degree-≤3 set used in the paper. |
| 7 | Optional **fit/tabulation split** (`fit_horizon`) and **bidirectional trajectories** (`generate_data(..., backward=...)`). | Not present in references. | Diagnostic tools that came out of the near-LC analysis (see report §4); both are off by default. |
| 8 | Trajectory data via `solve_ivp` (RK45, rtol 1e−10). | Reference uses a hand-rolled RK4 with the k4 bug above. | Correctness; data accuracy is then far below the model error floor. |
| 9 | Eigenfunction gradients by **moving least squares** (local weighted quadratic on the k nearest samples). | Reference `build_B_numgrad`: central FD with default step h = 0.1. | Measured on the eigenfunction-PDE residual: MLS 2.6 % median vs 62 % for the reference default — h = 0.1 is larger than the eigenfunction features. |
| 10 | **B(x) as a residual-weighted global polynomial field** (`fit_b_field`); raw pointwise gradients are never used in rollouts. | Reference evaluates pointwise numerical gradients directly inside the prediction loop. | Raw B samples reach 2e7 near the limit cycle and destabilize the rollout (measured: 21,556 % median forced error); the smooth field gives 6.5 % / 1.0 % on the paper's two test signals. Still autonomous-data-only. |
| 11 | MPC as condensed **box-constrained least squares** (`scipy.optimize.lsq_linear`), exact ZOH discretization with closed-form blockwise Γ. | Thesis used MATLAB quadprog/CasADi machinery; lifted dynamics integrated by RK4. | No extra dependency, exact discretization (no integration error), 11 ms solves. State constraints would motivate a general QP solver later. |
| 12 | Input-extended systems: **x₃ = u fitted exactly** (`constant_components`: one λ = 0 eigenfunction, boundary value = the level). | Thesis carried the input-as-state through the same generic eigenfunction fit as the physical states. | x₃ is constant along autonomous flows by construction; fitting it generically wastes budget and puts approximation error into the one channel (input integration) that can be exact. |
| 13 | **`boundary_rcond = 1e−3`** on the final boundary fit for extended models. | Reference used the unregularized normal-equations solve throughout. | The refined 20-column exponential basis is near rank-deficient; unregularized, the near-null directions carry |z| ≈ 179 that cancels in C z but wrecks ∂Φ/∂x₃ (measured: forced LPV rollout 36 % → 13 % median from this switch alone). |
| 14 | Cross-plane gradients by **FD with per-dimension steps** (in-plane 2e−3, across planes = spacing/2); B(x) smoothed by **kNN with PDE-residual weights + confidence blending** (`fit_b_knn`), not a global polynomial. | Thesis `build_B_numgrad`: one scalar FD step for all directions; global evaluation of raw gradients. | The n-D MLS estimator is blind across the stacked u-planes (all neighbors share a plane); a single FD step cannot fit both the in-plane feature size and the plane spacing. Global polynomials flatten the true gain variation while importing near-limit-cycle garbage; local averaging with graceful fallback to the robust constant wins every forced suite (report §6). |
| 15 | Constant-B baseline regression **drops steps whose lift left the training hull** (`fit_constant_b_forced`). | — (baseline implemented for this comparison). | The nearest-neighbor lift fallback outside the hull fakes enormous one-step "input responses"; without the filter ‖B‖ inflates by ~5 orders of magnitude. With it, the baseline is as strong as a constant B gets here. |

## Validation summary

* 44 unit/integration tests, including: exact recovery of a linear system
  (machine precision), real ≡ complex cost equivalence, analytic gradient
  vs. finite differences, eigenvalue recovery from perturbed starts,
  closed-form propagation vs. `expm`, and an end-to-end VdP smoke test.
* Octave cross-check against the actual thesis MATLAB functions
  (`getCostGradientKordacc_re_fast.m`, `eigOptim_grad.m`): identical cost
  (≤1e−8 rel., limited by the reference's explicit inverse), identical
  fitted trajectories (≤1e−13), gradients agreeing to ~2e−5 between the two
  independent analytic implementations.
* Full benchmark vs. Korda & Mezić Table I in
  [`reports/validation_vdp.md`](reports/validation_vdp.md).
* Non-affine tests additionally prove: the extension is an *exact* rewrite
  (extended-vs-original simulation to 1e−7), the designed input gain really
  flips sign in u and in the state, x₃ = u is reproduced to machine
  precision, the constant-B regression recovers the true lifted B on a
  linear plant, and the n-D MLS Jacobian recovers a 3-D linear lift.

## Roadmap

1. ~~LPV input matrix B(x)~~ — **done** (gradients via MLS, smooth field
   fit; see `reports/validation_vdp_control.md`).
2. ~~Koopman/LPV MPC~~ — **done** (stabilization + tracking demos).
3. ~~Non-input-affine systems via exact input extension, and the
   LPV-vs-constant-B quantification~~ — **done** (n > 2 MLS included; see
   `reports/validation_nonaffine.md`).
4. Stationarity-bias correction at off-origin setpoints (the one blocker
   for precision regulation; quantified in the control report §5).
5. Other case studies (Duffing, pendulum, single-track vehicle model —
   the vehicle now unblocked by the extension machinery).
6. State/output constraints in the MPC (general QP solver, e.g. OSQP);
   MPC through the extended system's B(x) (rate-penalized v as input).
7. Near-limit-cycle regime at small N: hybrid spectra (explicit ikω₀
   Floquet modes alongside the interior lattice) or region-split models.
