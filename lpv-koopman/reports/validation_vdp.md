# Validation report — autonomous Koopman prediction of the input-extended Van der Pol

Reproduce everything here with
`python examples/run_vdp_benchmark.py` (~25 min; figures land in
`reports/figures/`, numbers in `reports/results.json`) and
`python examples/diagnose_near_lc.py` (§4, ~2 min).

## 1. Setup

The plant is the non-affine Van der Pol `vdp_nonaffine`, always in its exact
input-affine extension (x₃ := u, new input v := u̇):

    ẋ1 = 2 x2
    ẋ2 = −0.8 x1 + 2 x2 − 10 x1² x2 + w(x1, x2, x3)
    ẋ3 = v,          w = (0.15 + 0.5 x2) sin(2 x3) + 0.25 x1 x3²

This report covers the **autonomous** part (v = 0): x₃ is then constant, and
the extended flow is the family of frozen-u Van der Pol slices. Each slice
has an unstable focus at its own u-dependent equilibrium and an attracting
limit cycle. The model learns eigenfunctions Φ(x₁, x₂, x₃) for all slices
at once; that autonomous model is what the LPV input matrix
B(x) = ∂Φ/∂x₃ is later differentiated from
([`validation_vdp_control.md`](validation_vdp_control.md),
[`validation_nonaffine.md`](validation_nonaffine.md)).

Because w(x, 0) = 0, **the plane x₃ = 0 is exactly the scaled Van der Pol
of Korda & Mezić 2020, Sec. VI-A**: origin an unstable focus with
eigenvalues 1 ± i√0.6, limit cycle of amplitude ≈ 0.9, period 5.674 s
(ω₀ = 1.1074 rad/s). Their Table I is therefore a valid published reference
for the extended model evaluated on that plane. It is a demanding one,
since their model is fitted to that plane alone.

* **Training** (thesis protocol for input-extended systems): 25 constant-u
  levels on [−1.2, 1.2] (spacing 0.1), 12 trajectories per level from a
  circle of radius 0.05 around that level's equilibrium, Tₛ = 0.01 s,
  **5 s** records ("paper configuration") or **7 s** ("extended
  configuration"). The cloud is a stack of 25 planes; the x₃ = 0 plane
  holds 12 of the 300 trajectories.
* **Model**: N eigenfunctions for (x₁, x₂), N/2 each, plus one exact
  λ = 0 eigenfunction for x₃ (`constant_components`), written "N+1".
  Eigenvalues start at the DMD lattice of the extended data and are
  refined by variable projection; boundary fit with `boundary_rcond = 1e−3`
  (README design decision 12). The N = 40+1, 7 s optimized model is the
  one every other script uses (`examples/extended_vdp.py`).
* **Test (a), the x₃ = 0 plane**: 500 initial conditions uniform over the
  interior of the u = 0 limit cycle (seed 7), 1 s horizon, error metric
  eq. (55) on (x₁, x₂): `100 · ‖x_pred − x_true‖_F / ‖x_true‖_F`. Split
  into the *near-LC band* (distance to the cycle < 0.05, 14 % of the points)
  and the *interior* (the rest), because the two behave differently (§4).
* **Test (b), held-out levels**: 30 interior initial conditions at each of
  the 7 u levels −1.15, −0.75, −0.35, 0.15, 0.55, 0.95, 1.15, none of them
  a training plane.

## 2. Results — paper configuration (5 s data, N ≤ 20)

Mean error over 1 s on the x₃ = 0 plane, in %; last column is the median
over the held-out levels:

| N | paper, lattice | ours, lattice (interior / near-LC / full) | paper, optimized | ours, optimized (interior / near-LC / full) | ours, optimized, held-out median |
|---|---|---|---|---|---|
| 4+1  | 100.4 | 29.6 / 32.3 / 30.0 | 24.0 | 28.1 / 33.1 / 28.8 | 29.8 |
| 8+1  | 95.6  | 23.3 / 28.5 / 24.1 | 9.7  | 14.9 / 28.5 / 16.8 | 16.6 |
| 12+1 | 51.3  | 15.6 / 37.5 / 18.7 | 5.1  | 16.9 / 70.2 / 24.5 | 17.5 |
| 16+1 | 13.3  | 8.6 / 29.4 / 11.5  | 2.4  | 8.2 / 42.8 / 13.1  | 11.5 |
| 20+1 | 5.44  | 7.5 / 57.6 / 14.6  | 1.4  | 8.0 / 86.0 / 19.1  | 9.0 |

Reading this table:

* **At the paper's budget the extended model does not reach Table I on
  x₃ = 0**: 8.0 % in the interior at N = 20+1 against the paper's 1.4 %.
  Two handicaps are built into the extension, and neither is a defect of
  the method. The plane gets 12 trajectories instead of the paper's 100,
  and one spectrum must serve 25 different frozen-u dynamics (the
  linearized Re λ ranges from ≈ 0.5 at u < 0 to ≈ 1.0 at u > 0.5).
* At small N the extended model is *better* than Table I (≈ 30 % vs
  100 % at N = 4): our lattice truncation always keeps the principal
  conjugate pair (design decision 5).
* **Eigenvalue refinement helps less than in the paper.** It optimizes the
  fit over *all* planes, which need not improve the x₃ = 0 plane; at
  N = 12 and 20 the lattice is better there. The held-out levels, which
  measure the cross-plane fit, do favour refinement at N ≥ 8.
* The near-LC band is where the paper configuration breaks, for the reason
  established in §4.

## 3. Results — extended configuration (7 s data, same machinery)

Longer records close the spatial coverage gap near each level's limit
cycle (hull fallbacks drop from 1/500 to 0/500), and a larger budget gives
the shared spectrum room for all the slices:

| N | lattice (full / interior / near-LC) | optimized (full / interior / near-LC) | optimized, held-out median | fit time |
|---|---|---|---|---|
| 20+1 | 10.5 / 10.0 / 13.8 % | 13.0 / 13.2 / 11.6 % | 9.3 % | 78 s |
| 30+1 | 9.1 / 8.4 / 13.8 % | 11.8 / 11.6 / 13.0 % | 7.2 % | 128 s |
| 40+1 | 5.1 / 4.1 / 11.5 % | **2.64 / 2.01 / 6.47 %** | **2.09 %** | ≈ 3.5 min |

**The N = 40+1 model reaches 2.0 % mean error in the interior of the
x₃ = 0 plane and 2.6 % over the full limit-cycle interior** (median 1.18 %,
p90 4.4 %). Its median error at u levels it never saw is 2.1 %
(per-level medians 0.8–6.9 %, fig12 and
[`validation_nonaffine.md`](validation_nonaffine.md) §3). This is the
model that every forced-prediction and MPC result is built on.

The remaining error is a fit residual, not an interpolation or time-drift
effect. fig5 shows it is already present at t = 0 and roughly flat over
the horizon. At the training samples on the x₃ = 0 plane the reconstruction
C Φ(x) misses x by a median 0.008 (≈ 1.5 % of the state's magnitude):
one 20-eigenvalue spectrum per state, shared by 25 planes and regularized
by `boundary_rcond`, cannot reproduce every slice exactly. Removing the
t = 0 offset does not lower the error (median 1.24 % → 1.60 %), so it is
not a pure lift bias either.

Figures (all from `run_vdp_benchmark.py`):

![training data and spatial errors](figures/fig1_phase_portrait.png)

*Fig. 1 — left: the 3-D training cloud, a stack of 25 constant-u planes,
with every fourth level's frozen-u limit cycle. Right: the x₃ = 0 plane,
with its 12 training trajectories, the limit cycle, and the 500 test ICs
colored by log₁₀ error of the N = 40+1 model.*

![eigenvalues](figures/fig2_eigenvalues.png)

*Fig. 2 — learned continuous-time eigenvalues of the x₁ and x₂ blocks vs.
the DMD lattice they start from (the x₃ block is the exact λ = 0). Left:
N = 20+1 / 5 s. Right: N = 40+1 / 7 s. Horizontal lines are the u = 0
limit-cycle harmonics k·ω₀; with a spectrum shared across slices whose
cycles have different frequencies, the learned modes no longer lock onto
one family of harmonics.*

![error vs N](figures/fig3_error_vs_N.png)

*Fig. 3 — error vs. lift dimension on x₃ = 0 against Table I (pale), and
for the 7 s configuration also on the held-out levels (mean, so dominated
by the u = +0.95 and +1.15 tails).*

![predictions](figures/fig4_predictions.png)

*Fig. 4 — true vs. predicted trajectories of the N = 40+1 model on the
x₃ = 0 plane, from deep interior (left) to the limit cycle (right).*

![error vs time](figures/fig5_error_vs_time.png)

*Fig. 5 — error over the prediction horizon (median and 10–90 % band),
paper vs. extended configuration, on x₃ = 0 (left) and the held-out levels
(right).*

## 4. Why the 5 s / N = 20+1 model fails near the limit cycle

Established by the **oracle-lift experiment**
(`examples/diagnose_near_lc.py`). On a trained u-plane, replace the
interpolated lift Φ̂(x₀, u) by the *exact* construction: integrate the
frozen-u slice backward from x₀ to that level's boundary circle (transit
time τ, boundary angle θ), then φ(x₀) = e^{λτ} g(θ) with the level's learned
boundary values (periodic cubic spline over its 12 angles). This removes
spatial interpolation, in-plane and across planes, from the error budget.
Result (N = 20+1, 5 s, optimized; 250 ICs on each of u = −0.6, 0, +0.6):

| lift | interior mean | near-LC mean |
|---|---|---|
| interpolated (data-driven) | 25.5 % | 134 % |
| oracle (exact flow) | 84.6 % | 1525 % |

| time depth τ | ICs | oracle / interpolated (median ratio) | interpolated median error |
|---|---|---|---|
| < 3 s | 242 | 1.3 | 7.1 % |
| 3–5 s | 412 | 1.5 | 5.1 % |
| 5–7 s | 87 | 12 | 128 % |
| ≥ 7 s | 4 | 157 | 163 % |

1. **Interpolation over the 3-D cloud is not the bottleneck.** The exact
   lift is nowhere better than the interpolated one. Inside the fitted
   window (τ < 5 s) the two agree to within the oracle's own error from
   having only 12 boundary angles to spline between.
2. **The failure is the time depth.** Errors jump by more than an order of
   magnitude once τ exceeds the 5 s fitting window, for both lifts. Points
   near a level's limit cycle sit at τ ≈ 5…40 s from its boundary circle,
   so the model must evaluate e^{λτ} far outside the window it was
   identified on. With every lattice eigenvalue at Re λ > 0, growing
   exponentials extrapolate away from the bounded, quasi-periodic truth.
   Equivalently, the Koopman spectrum *on* a limit cycle is {i k ω₀}
   (plus transversal Floquet decay), which no small truncation of the
   interior lattice contains; this is the obstruction formalized by
   Colbrook & Mezić (*Limits and Powers of Koopman Learning*). The 3-D
   interpolation actually *helps* beyond the window: where one plane's
   records end, neighbouring planes' longer-reaching data still bracket
   the point, which the single-plane oracle cannot use.

The extended configuration resolves it constructively: 7 s of data place
samples, and the fitting window, on the cycles themselves, and N = 40+1
gives the shared spectrum room for both regimes. That is why the near-LC
band falls from 86 % to 6.5 %.

## 5. Cross-check against the thesis MATLAB implementation

A numerical check of the core routines, independent of the study's plant.
`crosscheck/run_crosscheck.py` runs the *actual* thesis `.m` files
(`getCostGradientKordacc_re_fast.m` — Appendix-A real formulation;
`eigOptim_grad.m` — discrete complex formulation) under GNU Octave on the
same data and compares against this implementation. The data is a 2-D
autonomous Van der Pol set (the x₃ = 0 slice of the extended plant); the
check concerns the numerical routines, not the plant:

| quantity | rel. difference |
|---|---|
| projected cost J (real conjugate-pair formulation) | 7.6e−9 |
| fitted trajectory values L·q vs. V·G | 2.3e−14 |
| cost J, discrete complex formulation at μ = e^{λTₛ} | 1.3e−11 |
| analytic gradient, ours vs. reference analytic | 2.0e−5 |
| analytic gradient, ours vs. FD of the reference cost | 4.7e−4 (FD-limited) |

The residual J/gradient differences are fully explained by the reference's
explicit Gram-matrix inversion (`inv(L'L)`), which loses ≈ 8 digits — its
own J disagrees with its own repo's `eigOptim_grad` at the same 5e−8 level
at which it disagrees with ours. The two formulations (real cos/sin pairs
vs. complex conjugate exponentials; continuous e^{λkTₛ} vs. discrete μᵏ)
are confirmed numerically equivalent, validating the thesis Appendix-A
construction itself.

## 6. Performance

Data generation ≈ 13 s (300 trajectories). Each model build is dominated
by the 3-D Delaunay triangulation of the cloud (≈ 25 s for 5 s data,
≈ 60 s for 7 s data); eigenvalue refinement adds seconds to ≈ 2.5 min
(N = 40+1). The full sweep (15 fits) runs in ≈ 25 min on one core. The
previous CMA-ES pipeline (random init in [−10,10]^N, gradient discarded)
ran for ~25 days without convergence on a *2-D* problem; the cost landscape
is the same — the difference is the lattice initialization and the exact
gradient.

## 7. Limitations and open questions for the next iteration

* **Per-plane data density.** The extension spreads the trajectory budget
  over 25 planes. On x₃ = 0 the paper's 100 trajectories become 12, and the
  paper-budget model stays at 8 % interior error. More trajectories per
  level, or fewer levels with the cross-plane FD step adapted, are the
  obvious levers; the plane spacing also sets the B(x) accuracy
  ([`validation_nonaffine.md`](validation_nonaffine.md) §6), so the two
  trade off.
* **One spectrum for all slices.** Positive-u slices are more unstable and
  are fitted with more compromise (held-out error 2.8–6.9 % at u ≥ 0.55 vs
  ≤ 1.3 % below). u-dependent eigenvalues would mean leaving the
  linear-lift model class; a cheaper option is a larger budget.
* **Optimizer alignment.** Refinement minimizes the all-plane fit cost,
  which at N = 12 and 20 worsened the x₃ = 0 plane relative to the
  lattice; at N = 40+1 it helps everywhere.
* **Near-LC band at small N** is out of reach of the model class (§4).
  Candidate remedies (not implemented): pinning a few eigenvalues to
  measured i k ω₀ Floquet modes, region-split models, or isostable
  coordinates.
* **Noise**: all data here is noiseless simulation, as in the benchmark;
  regularized variants (the paper's δ₁/δ₂ terms) are not yet implemented.
