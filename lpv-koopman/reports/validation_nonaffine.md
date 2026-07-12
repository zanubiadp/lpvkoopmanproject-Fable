# Validation report: non-affine input, the extension trick, and LPV vs constant B

Companion to `examples/run_nonaffine_benchmark.py`; all numbers from a full
run (`reports/results_nonaffine.json`), figures `reports/figures/fig11..14`.

## 1. The claim being tested

Korda & Mezić (2020) and Iacob et al. (2024) both note that a system that is
**not** control-affine,

    x1' = f1(x1, x2, u),    x2' = f2(x1, x2, u),

can be rewritten *exactly* as a control-affine system by extending the state
with the input,

    x3 := u,    x3' = v   (v = u̇ is the new input),

after which every control-affine Koopman method applies unchanged — the
thesis uses exactly this rewrite for the vehicle model, which is naturally
non-affine in its input. Two things needed validating:

1. the autonomous eigenfunction pipeline still works on the extended 3-D
   state, trained only at a finite set of constant-u levels (the thesis
   data protocol: "experiments from different initial conditions at
   different values of u_constant");
2. on the extended system, the LPV input matrix B(x) = ∂Φ/∂x₃ built from
   **autonomous data only** beats the **constant** lifted B of the Korda &
   Mezić control pipeline (fitted to *forced* data) when the input enters
   strongly nonlinearly.

Both hold; the second by a factor ≈ 2 against the best constant and by an
order of magnitude against the forced-data constant. Details and the
failure modes met along the way are below.

## 2. The plant (fig11)

`vdp_nonaffine`: the scaled Van der Pol of the paper benchmark with a
deliberately hostile input channel,

    x1' = 2 x2
    x2' = −0.8 x1 + 2 x2 − 10 x1² x2 + w(x, u)
    w(x, u) = (0.15 + 0.5 x2) · sin(2u) + 0.25 x1 u²

designed so that a model linear in u is *as wrong as possible* over
u ∈ [−1.2, 1.2] while every frozen-u slice keeps the Van der Pol structure
(unstable focus + attracting limit cycle — verified numerically across the
whole range, so autonomous experiments still explore the state space at
every level):

* ∂w/∂u flips **sign in u** past the sin(2u) crest (|u| > π/4 ≈ 0.785):
  beyond it, increasing u pushes the state the *other* way;
* the gain is multiplied by (0.15 + 0.5 x2), which flips **sign in the
  state** at x2 = −0.3: the same input action reverses effect depending on
  where the trajectory is;
* 0.25 x1 u² has zero slope at u = 0 — invisible to any linearization in u —
  and couples the input to x1.

No constant lifted B can represent a sign-indefinite, state-dependent gain;
that is precisely the regime the LPV matrix is for.

## 3. Extended-system protocol and autonomous validation (fig12)

* Data: 25 u-levels uniform on [−1.2, 1.2] (spacing 0.1), 12 trajectories
  per level starting on a circle of radius 0.05 around the *frozen-u
  equilibrium* (which shifts with u), 7 s at Ts = 0.01 — 300 trajectories,
  ≈ 210k samples of the 3-D extended state. x3 is constant along every
  trajectory, so the cloud is a stack of 25 planes.
* Fit: budgets [20, 20, 1]; x3 is declared a `constant_component` and
  fitted *exactly* (single eigenvalue λ = 0, boundary value = the level, no
  optimizer budget spent). DMD base from the extended data with the exact
  λ ≈ 0 mode of x3 removed; `boundary_rcond = 1e−3` (see §6). Fit ≈ 3.5 min.
* Validation: 1 s predictions from 30 interior initial conditions at each
  of 7 u levels **never seen in training** (midpoints of the training
  levels). Per-level median error (eq. 55):

| held-out u | −1.15 | −0.75 | −0.35 | +0.15 | +0.55 | +0.95 | +1.15 |
|---|---|---|---|---|---|---|---|
| median | 1.0 % | 0.9 % | 0.8 % | 1.3 % | 2.8 % | 6.9 % | 5.2 % |
| p90 | 7.6 % | 3.2 % | 2.1 % | 6.8 % | 14.1 % | 18.6 % | 25.2 % |

Overall median 1.3 % — the same ballpark as the 2-D autonomous benchmark,
i.e. **the pipeline survives the extension**: eigenfunction values
interpolate cleanly *across* the u-planes. Positive levels are harder: the
frozen-u slices there are more strongly unstable (linearization Re λ up to
≈ 1.0 vs ≈ 0.5 at u < 0), so one shared spectrum across all slices fits
them with more compromise; the u = +0.95/+1.15 tails come from
initial conditions near those levels' limit cycles.

## 4. Input models compared

All five run on the **same** lifted dynamics (same eigenfunctions, same A);
they differ only in the lifted input matrix:

| tag | B | data used |
|---|---|---|
| `lpv_knn` (**ours**) | LPV B(x) = ∂Φ/∂x₃: per-dimension-step FD gradients of the interpolated eigenfunctions across the u-planes, sampled at 3 600 points, kNN-smoothed (k = 20) with PDE-residual reliability weights and confidence blending (blend = 0.3) toward the weighted-mean constant | autonomous only |
| `lpv_poly3` (ours, variant) | same samples, one global degree-3 polynomial per lifted row | autonomous only |
| `km_const_forced` | **Korda & Mezić recipe**: one constant B by least squares on one-step lifted residuals z⁺ − A_d z = Γ B v over 60 forced experiments (white-noise ZOH v, 1.5 s each) | forced |
| `const_auto` | degree-0 reduction of the LPV field (the best constant available from autonomous data) | autonomous only |
| `zero_B` | B = 0 in the (x1, x2) channels (x3 still integrates v exactly) | — |

`zero_B` matters as a control: on the extended system the *nonlinearity of
the input has moved into the autonomous part*, so even with no input model
at all, the lifted A does much of the work. Any candidate must beat it to
be adding value.

## 5. Results (fig13, fig14)

Forced prediction over 1.5 s from interior initial conditions, 40 runs per
family, error = eq. (55) on (x1, x2). Input families: **gentle** stays in
the monotone-gain region |u| < π/4; **dwell** sits past the gain flip;
**sweep** crosses it repeatedly. Median (mean) %:

| input family | LPV B(x) **(ours, autonomous)** | K-M constant B (forced) | constant B (autonomous) | B = 0 |
|---|---|---|---|---|
| gentle | **3.5** (6.9) | 90.0 (98.2) | 7.2 (14.5) | 8.5 (16.1) |
| dwell | **3.2** (9.6) | 60.7 (70.6) | 3.4 (7.4) | 4.6 (7.6) |
| sweep | **7.9** (21.0) | 229.2 (276.0) | 14.5 (25.5) | 17.6 (29.8) |

* **LPV vs best constant: ≈ 2× better on gentle and sweep**, at parity on
  dwell (where trajectories converge to the limit cycle and the input
  contribution is smallest — B = 0 at 4.6 % says most of that family is
  autonomous dynamics).
* **The K-M constant B is 10–30× worse than ours** — and worse than doing
  nothing (B = 0). This is not a strawman implementation: its x3-row is
  recovered exactly (= 1), and §6 explains the mechanism. With a scalar
  input, ridge-regularizing their regression can only *shrink* B toward 0,
  i.e. toward the `zero_B` row — there is no constant in between that
  fixes it.
* The trajectory-level picture (fig13, median-LPV-error cases, not
  cherry-picked): the LPV rollout is barely distinguishable from the truth;
  the constant-B rollout bends the wrong way exactly when u dwells in the
  reversed-gain band.

## 6. What made it work (numbers from the prototyping runs)

The naive port of the 2-D control pipeline **failed** — the first attempt
had the LPV field *worse* than its own constant reduction (median 48 % vs
10 % on sweeps). Three changes turned it around, all now defaults or
options in the library:

1. **Boundary-fit regularization** (`FitConfig.boundary_rcond = 1e−3`).
   With 20 refined eigenvalues the exponential basis V is nearly
   rank-deficient; at lstsq's default cutoff the near-null directions get
   arbitrary huge boundary values (lifted coordinates reaching |z| ≈ 179,
   jumping by ≈ 23 between adjacent u-planes). Those coordinates cancel in
   the reconstruction C z — autonomous predictions barely notice — but
   ∂Φ/∂x₃ inherits the full roughness, and the input response lives
   exactly in those coordinates. Cutting them costs ≈ 0.6 pp of autonomous
   median error and buys a 3× cleaner one-step input response.
2. **Dense u-levels + per-dimension FD steps.** The cross-plane derivative
   is a secant between planes (the in-plane MLS estimator is blind across
   planes — its neighbors all share one plane, see
   `gradients.lift_jacobian_mls`); halving the plane spacing (0.2 → 0.1,
   FD step = spacing/2) halved the secant's curvature bias: raw one-step
   response error 19 % → 14 % median.
3. **Local smoothing with confidence blending** (`KnnBField`). Gradient
   sample quality is spatially bimodal: excellent in the interior, garbage
   near each level's limit cycle (the known 2-D failure mode). A global
   polynomial couples the two regions — degree 3 both *flattens* the true
   gain variation and lets near-LC outliers bend the interior. Averaging
   the k = 20 nearest samples with PDE-residual weights keeps the local
   structure; blending toward the global weighted mean where the
   neighborhood's mean reliability is low (α = w_loc/(w_loc + 0.3 w̄))
   rescues the near-LC region, where a local average of bad samples is
   still bad. Without the blend, dwell-family error was 27–43 %; with it,
   3.2 %.

And the mechanism behind the K-M constant's collapse: the one-step
regression residual z⁺ − A_d z contains the *autonomous* model error, which
on the extended system is far larger than the one-step input response
(median lifted response norm ≈ 5 vs autonomous residuals of comparable
size). Whatever part of that error correlates with v through the closed
u-trajectory leaks into B; the regressed direction then excites unstable
lifted modes coherently at every step of a rollout. Filtering regression
steps outside the training hull (`fit_constant_b_forced` does this — the
nearest-neighbor lift fallback otherwise fakes enormous input responses,
inflating ‖B‖ by ~5 orders of magnitude) and white-noise excitation both
help, but the bias is structural: a single B must average a gain field
whose true sign flips over the operating range.

## 7. Honest caveats

* The headline LPV numbers use kNN + blend hyperparameters (k = 20,
  c = 0.3) selected on a 30-run tuning set with the same generator but a
  different seed than the reported 40-run suites; sensitivity is mild
  (k ∈ [10, 40], c ∈ [0.3, 1] all beat every constant at 1.5 s).
* At longer horizons (2.5 s) dwell-family trajectories live on the limit
  cycle where the gradients are unreliable; there the blended field
  degrades toward the constant (by design) and the constant becomes
  competitive again (5.0 % vs 8.7–11 % for LPV variants). The LPV advantage
  is a *transient/interior* phenomenon — consistent with the 2-D findings.
* The comparison gives the constant-B baseline *more* information than
  ours (60 forced experiments); the asymmetry is deliberate and makes the
  result stronger, not weaker.
