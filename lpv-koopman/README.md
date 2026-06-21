# lpvkoopman — LPV-Koopman modeling, rebuilt from scratch

A clean Python implementation of data-driven Koopman eigenfunction models
for prediction and control, following

* M. Korda, I. Mezić, *Optimal Construction of Koopman Eigenfunctions for
  Prediction and Control*, IEEE TAC 65(12), 2020 — the eigenfunction
  learning method and the Van der Pol benchmark (Sec. VI-A, Table I);
* the thesis *Koopman-based MPC for High Performance Vehicles* (and its
  MATLAB implementation) — used as the behavioral reference, including the
  Appendix-A real conjugate-pair reformulation;
* Iacob et al. 2024 — state-dependent input matrix B(x) = ∂Φ/∂x · g_c(x),
  built here **from autonomous data only** (no forced experiments), plus
  Koopman-LPV MPC on the lifted model.

**Scope: autonomous prediction, the LPV input matrix B(x), and MPC** — all
validated on the Van der Pol benchmark. Headline numbers
([`reports/validation_vdp.md`](reports/validation_vdp.md),
[`reports/validation_vdp_control.md`](reports/validation_vdp_control.md)):

| capability | result | published reference |
|---|---|---|
| autonomous prediction, paper config (N=20, 5 s) | 1.81 % (away from the limit cycle) | 1.4 % (Korda & Mezić Table I) |
| **autonomous prediction, N=40, 7 s** | **0.83 % over the full limit-cycle interior** | — |
| forced prediction, sine 60 ms, LPV B(x) from autonomous data | **1.30 %** | 2.9 % (constant B fitted to *forced* data) |
| forced prediction, square 300 ms | 8.4 % (LPV) vs 12.6 % (constant-B reduction) | 6.0 % |
| MPC: stabilize the unstable origin from 92 % of the LC radius | ‖x(6 s)‖ = 5.4e-5, 11 ms/solve | — |
| MPC: track x₁ = 0.3 sin(1.2t) | RMS 0.0041 (1.4 % of amplitude) | — |

Total model fit ~35 s; B(x) field fit ~5 s; MPC solves ~11 ms — replacing
a multi-week CMA-ES run.

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
                    # interpolation of eigenfunctions, closed-form predict
  pipeline.py       # fit_koopman_model(data, FitConfig)
  gradients.py      # dPhi/dx: moving-least-squares & FD estimators,
                    # eigenfunction-PDE residual diagnostic
  lpv.py            # B(x) = (dPhi/dx) g_c(x): BMap, residual-weighted
                    # polynomial field fit, exact ZOH discretization,
                    # forced LPV rollout
  mpc.py            # Koopman-LPV MPC (condensed box-QP via lsq_linear),
                    # closed-loop simulator against the true plant
  vdp_benchmark.py  # Korda-Mezić VdP protocol: data, limit cycle, metrics
  metrics.py        # eq. (55) relative RMSE
examples/
  run_vdp_benchmark.py         # autonomous: every number/figure (~3 min)
  diagnose_near_lc.py          # oracle-lift error-budget experiment (~2 min)
  run_vdp_control_benchmark.py # B(x) + forced prediction vs paper (~3 min)
  run_vdp_mpc.py               # MPC demos: stabilization + tracking (~4 min)
crosscheck/
  run_crosscheck.py      # numerical cross-check vs. the thesis MATLAB code
  crosscheck.m           # (runs the actual .m files under GNU Octave)
tests/                   # 31 unit/integration tests (~10 s)
reports/
  validation_vdp.md          # autonomous validation report
  validation_vdp_control.md  # B(x) + MPC validation report
```

## Install and reproduce

```bash
pip install -e .[dev]
pytest                                        # 31 tests, ~10 s
python examples/run_vdp_benchmark.py          # autonomous tables + figures
python examples/diagnose_near_lc.py           # near-LC error budget
python examples/run_vdp_control_benchmark.py  # B(x) + forced prediction
python examples/run_vdp_mpc.py                # MPC closed-loop demos
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

## Validation summary

* 31 unit/integration tests, including: exact recovery of a linear system
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

## Roadmap

1. ~~LPV input matrix B(x)~~ — **done** (gradients via MLS, smooth field
   fit; see `reports/validation_vdp_control.md`).
2. ~~Koopman/LPV MPC~~ — **done** (stabilization + tracking demos).
3. Stationarity-bias correction at off-origin setpoints (the one blocker
   for precision regulation; quantified in the control report §5).
4. Other case studies (Duffing, pendulum, single-track vehicle model);
   the MLS gradient kernel needs the trivial n > 2 generalization.
5. State/output constraints in the MPC (general QP solver, e.g. OSQP).
6. Near-limit-cycle regime at small N: hybrid spectra (explicit ikω₀
   Floquet modes alongside the interior lattice) or region-split models.
