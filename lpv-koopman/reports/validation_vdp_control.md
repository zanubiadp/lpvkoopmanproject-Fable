# Validation report — LPV input matrix B(x) and Koopman-LPV MPC (Van der Pol)

Part 1 (§1–§7) covers the input-affine plant; part 2 (§8–§11) covers the
non-affine plant through its exact input extension, and the LPV-vs-constant-B
controller comparison.

# Part 1 — the input-affine plant

Phase-2 results, building on the autonomous model of
[`validation_vdp.md`](validation_vdp.md) (N = 40, 7 s data, 0.83 %
autonomous prediction error). Reproduce with
`python examples/run_vdp_control_benchmark.py` (~3 min) and
`python examples/run_vdp_mpc.py` (~4 min); numbers land in
`reports/results_control.json` / `results_mpc.json`.

**Headline: the thesis's central claim is validated end-to-end.** The
state-dependent input matrix B(x) = ∂Φ/∂x·g_c(x) built **exclusively from
autonomous trajectory data** (no forced experiments anywhere) delivers
forced-prediction accuracy competitive with — and on one of two benchmark
signals better than — Korda & Mezić's constant B that was *fitted to
forced data*, and it supports a real-time MPC that stabilizes the unstable
origin and tracks moving references on the true nonlinear plant.

## 1. Eigenfunction gradients (the make-or-break ingredient)

B(x) needs ∂Φ/∂x; Φ is known as tabulated values on the trajectory cloud.
Quality is measured by the eigenfunction-PDE residual
‖(∂Φ/∂x)f − AΦ‖/‖AΦ‖, which is exactly zero along trajectories for the
ideal construction (200 interior test points):

| gradient estimator | median residual | p90 |
|---|---|---|
| **moving least squares (k=40, quadratic)** | **0.026** | 2.4 |
| central FD on interpolant, h=1e-3, order 4 | 0.093 | 2.8 |
| central FD, h=0.01, order 4 | 0.051 | 3.0 |
| central FD, **h=0.1, order 2 (the thesis/MATLAB default)** | **0.62** | 43 |

Two findings: (i) MLS is the best estimator and is used throughout;
(ii) the reference implementation's `build_B_numgrad` default step
h = 0.1 destroys the gradients (62 % median error) — eigenfunction
features near the origin are smaller than the stencil. This is another
structural defect of the original pipeline, now quantified.

## 2. Raw pointwise B(x) is exact-in-principle but unusable — and the fix

Raw MLS gradients give ‖B(x)‖ with median ≈ 1.5e4 in the interior,
≈ 4e5 within 0.05 of the limit cycle, max ≈ 2e7: the individual
eigenfunction surfaces are steep (their reconstruction relies on
cancellation), so pointwise B estimates are huge and noisy. Rolling the
lifted model forward with them injects O(‖B_err‖) into expanding modes
every step — forced predictions **explode** (median error 21,556 % on the
square wave).

The fix, still autonomous-data-only (`fit_b_field`): fit one global
low-order polynomial per lifted row of B(x) over the operating region,
weighting samples by PDE-residual quality. This denoises and bounds the
field while preserving its state dependence; its output projection
C·B(x) lands within ~10 % of the exact value [0, 1] everywhere
(fig8_B_gain_map.png). Degree 0 of the same fit provides the constant-B
(LTI) baseline.

## 3. Forced prediction vs. the published controlled benchmark

Paper protocol (Sec. VI-A controlled): unit-amplitude square wave
(300 ms period) and sine (60 ms period), 1 s horizon, eq. (55) error;
250 ICs uniform over the limit-cycle interior. Reference: Table I,
N = 20 optimized, with **constant B identified from a dedicated forced
dataset**: square 6.0 %, sine 2.9 %.

| input model (ours: autonomous data only) | square: mean / median | sine: mean / median |
|---|---|---|
| **LPV B(x), polynomial deg 4** | **8.4 % / 6.5 %** | **1.30 % / 1.03 %** |
| constant B (deg-0 reduction) | 12.6 % / 8.4 % | 1.61 % / 1.17 % |
| raw pointwise MLS B(x) | 2.9e7 % / 21,556 % | 1.9e7 % / 1.15 % |
| paper (constant B from forced data) | 6.0 % | 2.9 % |

* **Sine: 1.30 % vs. the paper's 2.9 %** — better, with zero forced data.
* Square: 8.4 % vs. 6.0 % — same ballpark; the square wave's ±1 plateaus
  push trajectories across (and partly beyond) the trained region, where
  the near-LC error band of the autonomous model (see Phase-1 report §4)
  is re-excited at every switch. Error is concentrated near the limit
  cycle (fig7_forced_error_map.png), exactly as in the paper's own Fig. 4.
* **The LPV field beats its own constant reduction by ~1.5× on the square
  wave** — direct evidence of the value of state dependence, isolated
  from every other pipeline ingredient (same model, same fit, same data).

![forced predictions](figures/fig6_forced_predictions.png)
![forced error map](figures/fig7_forced_error_map.png)
![B gain map](figures/fig8_B_gain_map.png)

## 4. Koopman-LPV MPC on the true plant

Controller: lift measured state, freeze B(x_k) over the horizon
(exact ZOH discretization, closed-form blockwise Γ = A⁻¹(e^{ATs}−I)),
solve the condensed tracking problem as a box-constrained least-squares
(`scipy.optimize.lsq_linear` — no extra QP dependency), apply u₀,
repeat at 100 Hz. Solve time ≈ 11 ms/step (horizon 80, N = 40, pure
Python) — real-time-rate feasible.

| task | result |
|---|---|
| **Stabilize the unstable origin** from 92 % of the limit-cycle radius, \|u\| ≤ 1 | ‖x‖: 0.69 → 3.7e-2 (2 s) → 1.4e-3 (4 s) → **5.4e-5 (6 s)**; open loop diverges to the cycle |
| **Track x₁ = 0.3 sin(1.2t)** | **RMS error 0.0041 (1.4 % of amplitude)**, max error 0.0069 |
| Piecewise-constant setpoints x₁ = ±0.25 (offset-free variant) | settled RMS 0.066; first setpoint settles cleanly (u → 0.20, the exact equilibrium input), later segments ring after reference jumps — see §5 |

![stabilization](figures/fig9_mpc_stabilization.png)
![tracking](figures/fig10_mpc_tracking.png)

## 5. Documented limitation: stationarity bias at off-origin setpoints

Constant-setpoint regulation exposes a structural weakness, with a clean
diagnosis:

* Open-loop, from the true equilibrium (0.25, 0) under the true
  equilibrium input u* = 0.2, the lifted model holds position for
  ≈ 0.5 s and then drifts (to (0.19, −0.18) by 1 s).
* Inverting the model for the stationary input at (±0.25, 0) gives
  u_ff ≈ ±0.78 versus the true ±0.2 — a ~4× bias.

Root cause: the model is trained (and excellent) at predicting **along
flows**; pointwise stationarity C(Az + B(x)u) = 0 requires exact
cancellation between large lifted terms, which magnifies small fitting
errors. Feedback hides most of this (the loop holds the setpoint to
within 0.03–0.06), an output-disturbance estimator (`offset_free=True`)
removes part of the rest, but ~0.05 of bias/ringing survives reference
jumps. Integral action with naive anti-windup made it worse (tested).
Remedies for the next iteration: equilibrium-anchored local correction of
(A, B) around target setpoints, or a disturbance model in the lifted
state rather than the output.

## 6. What carries over from the references — and what was changed

* Iacob et al. (2024): B(x) = ∂Φ/∂x·g_c(x) — implemented as stated; the
  *practical* contribution here is the residual-weighted smooth field fit
  (§2), without which the exact formula is numerically unusable on a
  system with expanding modes.
* Thesis: gradient-based B from autonomous data — vindicated; its
  finite-difference defaults (h = 0.1) and the absence of field smoothing
  explain why this step underperformed in the original pipeline. The
  thesis's stable-equilibrium case studies (Duffing, pendulum) were also
  far more forgiving: contracting A absorbs B-noise injections, VdP's
  expanding A amplifies them.
* MPC: standard qLPV-Koopman scheme (freeze B at the measured state per
  step). Box-constrained-input QP solved exactly via bounded least
  squares; state constraints and a proper QP solver (e.g. OSQP) are the
  natural extension when needed.

## 7. Open items for the next iteration

* Stationarity-bias correction at setpoints (§5) — the one blocker for
  precision regulation away from the origin.
* Square-wave-class inputs that drive the state outside the trained
  region: extend training coverage outward (the bidirectional sampler in
  `systems.generate_data` is ready for this) or add input-aware data.
* State/output constraints in the MPC (switch to a general QP solver).
* Multi-input systems and the vehicle models from the thesis: the entire
  B(x)/MPC path is written for m ≥ 1, only the 2-D MLS gradient kernel
  assumes n = 2 and needs the trivial generalization.
* Iacob et al.'s LTI-vs-LPV error bounds: quantify them on this example
  and compare with the measured 1.5× square-wave gap.

---

# Part 2 — MPC on the *non*-affine plant (LPV B(x) vs constant B)

Everything above concerns the input-affine plant `vdp_scaled`. This part
runs the same three tasks on `vdp_nonaffine` through its exact input
extension (x₃ = u, v = u̇), with the *same* lifted dynamics driven by two
competing input models: the LPV field B(x) from autonomous data, and the
Korda–Mezić constant B fitted to 60 forced experiments. Reproduce with
`python examples/run_vdp_mpc.py`; figures fig15–fig17, numbers under the
`nonaffine` key of `results_mpc.json`.

**Headline, in two halves.** On *stabilization* the separation is large and
one-sided, but it is **transient, not asymptotic**: the LPV field pulls the
state to ‖x‖ = 1.4e-2 and holds it near the origin for ~9 s before losing
it, while the constant B never gets closer to the origin than its own
initial condition and failed in all 432 of its closed-loop runs, under every
one of 72 weight configurations. On *off-origin reference tracking* the LPV
field does not win at all — the constant's own optimum is better on the sine
task, and both arms fail on switching setpoints. The advantage this route
buys is real but considerably narrower than the forced-prediction study
alone would suggest (§9).

Getting to a comparison that measures model quality rather than actuator
saturation required three corrections to the original setup, each of which
is a finding in its own right (§8), and the residual tracking error is then
model error rather than a plant limit (§10).

## 8. Three corrections to the non-affine MPC setup

### 8.1 The controller was boxing the wrong variable

For the extended plant the MPC's decision variable is v = u̇; the physical
actuator command is the extra **state** x₃ = u. The original setup left
`MPCConfig`'s `u_min/u_max` at their ∓1 defaults, which therefore imposed a
*slew limit* of |u̇| ≤ 1 s⁻¹ — the loop could move u by only 0.01 per
step — while leaving u itself entirely unconstrained. The symptom is
unmissable in hindsight: every run in the old `results_mpc.json` reports
`u_max = 1.0000` exactly, i.e. the controller sat pinned on that bound for
its whole duration, and meanwhile u wandered outside the identified range
x₃ ∈ [−1.2, 1.2], where `KoopmanEigenModel.lift` silently degrades to
nearest-neighbour values and every prediction downstream is meaningless.
(`predict_forced` already guards this with `x_clip`; `KoopmanMPC` did not.)

The fix is a change of decision variable rather than a re-tuning.
Since x₃ integrates the input exactly, v_k = (u_{k+1} − u_k)/ts, the map
from the actuator sequence to the rate sequence is linear and bidiagonal.
Substituting it keeps the problem a *box-constrained* least-squares one —
so `lsq_linear` still applies, no QP dependency is added — but the box now
constrains u itself:

```
decision variable  w = [u_1 … u_Np],   bounds [−U_MAX, U_MAX]
v = (D w − e₀u₀)/ts    (D = I − shift)
```

Implemented as `MPCConfig.integrator_state`; it is opt-in, so the affine
sections are untouched. Verified to reproduce the direct solve to 1e-13
when the box is inactive, confirming the substitution is exact and not an
approximation. `KoopmanMPC` also gained the `x_clip` guard that
`predict_forced` already had.

### 8.2 The horizon was five times too short

With the input bounded correctly the loop still failed. The reason is
geometric: the origin is an unstable **focus** with eigenvalues 1 ± 0.775i,
and the extremal controls that trace the boundary of its null-controllable
region switch on the half-rotation period π/β = **4.06 s**. A horizon of
0.8 s cannot represent the manoeuvre that stabilization requires. Horizon
length, not weighting, was the binding constraint:

| control rate ts | horizon | LPV ∫‖x‖dt | settled ‖x‖ |
|---|---|---|---|
| 0.01 | 0.8 s | 4.25 | 0.66 |
| 0.05 | 8 s | 4.21 | 0.76 |
| 0.10 | 4 s | 4.61 | 0.88 |
| **0.10** | **8 s** | **1.11** | **0.163** |
| 0.15 | 8 s | 1.70 | 0.41 |
| 0.10 | 12 s | 5.58 | 0.75 |

Two things worth noting. The horizon must be bought by coarsening the
control rate — at ts = 0.01 an 8 s horizon is an 800-step condensed QP per
solve, which is not real-time — and 12 s is *worse* than 8 s, because model
error accumulates over the prediction faster than the extra foresight is
worth. The chosen operating point is ts = 0.10 s, Np = 80.

### 8.3 All three tasks were outside the plant's reach

This is the finding that explains the original figures. The affine plant
has g = (0, 1): authority ±1.0 on ẋ₂, with |u| ≤ 1. The non-affine plant's
input enters as

```
w(x,u) = (0.15 + 0.5 x₂) sin(2u) + 0.25 x₁ u²,     |u| ≤ 1.15
```

which at the origin is worth `0.15 sin(2u)` — **±0.15, about 1/7 of the
affine plant's authority** — against the identical unstable focus with
Re λ = 1. The three tasks were inherited verbatim from the affine
benchmark and none of them survives the change of plant (fig17):

| task | affine spec | demanded | achievable | verdict |
|---|---|---|---|---|
| 1. stabilize from 0.92·lc | ‖x₀‖ = 0.692 | — | null-controllable reach **0.313** | **2.25× outside**; unreachable |
| 2. sine A=0.3, ω=1.2 | — | ‖w‖ up to **0.360** | ≤ 0.253, and only ±0.06 at the worst instant | infeasible over **54 %** of the period |
| 3. setpoints ±0.25 | — | w = **0.2000** | **0.1989** | infeasible by 0.0011 |

The null-controllable region (the set of states steerable to the origin) is
computed by reversing time and integrating the extremal bang-bang branches
outward over a spread of switching times — for a planar single-input system
that traces its boundary exactly. It reaches 0.313 in ‖x‖; the affine x₀
sits at 0.692. This is confirmed independently by an *oracle* controller —
a receding-horizon MPC given the true plant, a 3 s horizon and full
knowledge — which pulls ‖x‖ to 0.28 by 2 s and then loses it back to 0.78
by 6 s. **No controller can perform the original task 1**, so the
bang-bang saturation visible in the old fig15/fig16 was the controller
correctly failing at an impossible demand, not a tuning defect.

#### The sharper limit: recoverable margin collapses at off-origin setpoints

Pointwise feasibility of the equilibrium input is necessary but *not*
sufficient. Holding a setpoint at +r already spends most of the push-up
authority, so the region from which that setpoint can still be recovered
shrinks asymmetrically — computed around each (r, 0) at its equilibrium u*:

| r | u* | w margin | reach −x₁ | **reach +x₁** |
|---|---|---|---|---|
| 0.00 | 0.000 | 0.150 | 0.313 | **0.248** |
| 0.10 | 0.274 | 0.087 | 0.413 | **0.148** |
| 0.15 | 0.427 | 0.057 | 0.463 | **0.098** |
| 0.20 | 0.613 | 0.027 | 0.513 | **0.048** |
| 0.25 | — | −0.001 | 0.563 | **0.000** |

At r = 0.20 any overshoot beyond 0.048 is unrecoverable — and an MPC
carrying 10–40 % forced-prediction error *will* overshoot that. This is why
the setpoint amplitude is 0.10 rather than the pointwise-feasible 0.20, and
it is a genuine limitation of the input-extension approach on a plant whose
gain is this weak, not an artifact of the Koopman model.

Rescaled tasks: x₀ = 0.25·lc (‖x₀‖ = 0.19), sine A = 0.10 / ω = 0.8,
setpoints ±0.10 with a 4 s dwell.

## 9. Weight selection under a bistable closed loop

The tuned closed loop is **bistable**: it either holds the reference or
escapes to the limit cycle, with essentially nothing in between. Adjacent
weight settings give RMS 0.169 and 0.44; the error distribution is bimodal,
not unimodal-with-noise. Taking the argmin of a single run would therefore
be fitting the noise, and would report a controller that works by luck.

Weights are instead selected by **robustness**: for each task, 72 (Q, R, S)
configurations are run from 6 perturbed initial conditions per input model
(864 closed-loop runs per task), ranked first by how many of the 6 stayed
bounded and only then by median error.

Crucially the shared weights are picked by a **neutral** criterion — best
combined LPV + constant score — not by the LPV arm. Selecting on the LPV arm
and reporting whatever the constant happens to score at those weights would
rig the ablation, and on the sine task that is not hypothetical: the
constant's own optimum holds 4/6 where it holds 0/6 at the LPV optimum.
Both the shared-weight result and each arm's own optimum are given below.

| task | shared Q / R / S | LPV @ shared | const @ shared | LPV own best | const own best |
|---|---|---|---|---|---|
| 1. stabilization | (1, 0.05, 0.1) / 1e-4 / 1e-2 | **2/6**, med 0.840 | **0/6**, med 0.930 | 2/6, med 0.840 | 0/6, med 0.885 |
| 2. sine A=0.10 | (1, 0.2, 0.5) / 1e-2 / 1e-3 | 1/6, med 0.497 | **4/6**, med 0.102 | 3/6, med 0.169 | **4/6, med 0.102** |
| 3. steps ±0.10 | (1, 0.2, 0.5) / 1e-2 / 1e-3 | 0/6, med 0.664 | 1/6, med 0.650 | 1/6, med 0.667 | 1/6, med 0.650 |

The q₃ entry weights the actuator channel x₃ = u. Under `integrator_state`
the box already keeps u legal, so q₃ is a control-effort term — the
extension's analogue of the affine problem's R‖u‖², since R here weighs
u̇ instead. Larger q₃ and R both help by keeping the loop gentle and the
state inside the region where the model is trustworthy.

### What the three tasks actually show

The results are **task-dependent, and only task 1 is a clean LPV win.**
Aggregated over all 72 configurations × 6 ICs = 432 runs per arm per task:

| task | LPV runs held | const runs held | configs where const held ≥ once |
|---|---|---|---|
| 1. stabilization | **27 / 432** | **0 / 432** | **0 / 72** |
| 2. sine | 40 / 432 | 10 / 432 | 4 / 72 |
| 3. steps | 1 / 432 | 1 / 432 | 1 / 72 |

* **Task 1 — a large but *transient* LPV win.** Scored over 20 s the
  separation is unambiguous, but it is not asymptotic stabilization:

  | arm | min ‖x‖ reached | at t | dwell ‖x‖<0.1 | ‖x‖ at 20 s |
  |---|---|---|---|---|
  | LPV B(x) | **1.4e-2** (from ‖x₀‖ = 0.188, a 13× reduction) | 8.7 s | **8.7 s** (t = 1.0 → 9.6 s) | 0.93 |
  | constant B | **1.84e-1** — its own starting value | 0.2 s | **never** | 0.56 |

  The LPV field pulls the state to within 0.1 of the origin within 1 s,
  holds it there for **8.7 s** (reaching 1.4e-2 at t = 8.7 s), and then
  loses it around t = 10 s. The constant B **never gets closer to the
  origin than its own initial condition** — it never reaches ‖x‖ < 0.15 at
  any time, and it failed in every one of its 432 runs under every
  weighting tried. So the claim supported by the data is "the LPV field
  reaches and transiently holds the origin, and the constant never
  approaches it at all", not "the LPV field stabilizes the plant".

  The escape has a consistent explanation. Relative forced-prediction
  error stays at 15–40 % as ‖x‖ → 0 (§10), so once the state is small
  enough the feedback is acting largely on model error while the unstable
  focus keeps its Re λ = 1 growth — the loop cannot win that race
  indefinitely. Mechanism for the *advantage*: the task holds u near 0 and
  the state near the origin, exactly where the constant's error is worst
  and where the LPV field is best resolved.

  **Methodological note.** An earlier version of this scoring used the mean
  of ‖x‖ over the last 40 % of an 8 s run and reported 5/6 successes. That
  metric was wrong twice over: 8 s ends before some escapes occur, and
  averaging across an escape lets a run that finishes on the limit cycle
  score as converged. The numbers above use the *maximum* over the final
  20 % of a 20 s run, which cannot be gamed that way.
* **Task 2 — no LPV advantage; the constant's best is better.** The LPV
  arm holds more often across the sweep (40 vs 10 runs), but at its own
  optimum the constant is both more reliable (4/6 vs 3/6) and more
  accurate (median 0.102 vs 0.169). Neither arm is dependable. This is
  consistent with the caveat already recorded in `validation_nonaffine.md`
  — the LPV advantage is a transient/interior phenomenon, and a reference
  that keeps the state cycling through the region where the eigenfunction
  gradients are unreliable erodes it.
* **Task 3 — both arms fail.** 71 of 72 configurations held zero times for
  each arm. This is not a tuning failure: §8.3 shows the recoverable
  margin above a setpoint collapses to 0.148 at r = 0.10, and the models
  carry 10–40 % (LPV) to 100–500 % (constant) forced-prediction error in
  that region, so overshoot past the recoverable boundary is essentially
  guaranteed. Switching setpoint regulation on this plant is beyond what
  either input model currently supports; reducing the amplitude to ±0.05
  did not rescue it either.

The honest summary is that the input-extension + LPV-B route buys a real
and large advantage for **stabilization**, and does not currently buy one
for **off-origin reference tracking** on this plant.

## 10. The residual tracking error is model error, not plant limitation

It would be easy to attribute the remaining tracking error to the plant's
weak actuator. It is not that. An oracle controller that knows the true
plant tracks the A = 0.10 sine at **RMS 0.0012 (1.2 % of amplitude)**,
using only u ∈ [−0.30, +0.68] of the ±1.15 available — the task has ample
margin. The best Koopman-MPC result anywhere in the sweep is RMS 0.062 —
**~50× worse than achievable** — and the shared-weight runs that fig16
reports are worse still (0.085 constant, 0.505 LPV).

The gap is the lifted model's forced-prediction accuracy in the region
these tasks occupy. Measured over 0.8 s rollouts near the origin:

| initial ‖x‖ | LPV B(x) | constant B |
|---|---|---|
| 0.38 | 9–38 % | 72–174 % |
| 0.19 | 15–24 % | 108–296 % |
| 0.09 | 17–40 % | 159–**519 %** |

The autonomous model is not the problem (0.9–9.5 % over the same horizons);
the input channel is. Note also that the constant-B error *grows* as the
state approaches the origin, which is exactly where all three tasks live —
a single averaged B cannot represent a gain field whose magnitude and sign
both vary, and the mismatch is worst where the gain is smallest.

This bounds what weight tuning can achieve: the controller is limited by
B-model fidelity near the origin, so the next improvement must come from
the input model (denser autonomous sampling near the equilibrium, or a
gradient estimator that stays accurate where Φ's features are smallest) —
not from the MPC.

## 11. Bugs fixed in `run_vdp_mpc.py`

The original non-affine section also carried a set of defects that made
fig15 unreadable independently of control quality; recorded here because
several produced *plausible-looking* output:

* both `semilogy` calls plotted the same `nx` array, so the LPV curve was
  hidden underneath the constant-B one — the figure showed one run drawn
  twice;
* both actuator traces plotted `stab.U`, the **affine** section's result;
* `lc_lpv` leaked from the B-fitting loop, so the "limit cycle" drawn and
  the x₀ placed on it belonged to the u = +1.15 level, not u = 0;
* `ax.plot(*x0, "bo")` with a 3-vector — matplotlib parses the third
  positional as data, producing the two spurious `$x_0$` legend entries;
* the results dict used the key `"stabilization LPV nonaffine"` twice, so
  the constant-B entry silently overwrote the LPV one;
* `results = {}` mid-function discarded the affine results, which were then
  overwritten in `results_mpc.json` (both halves are now nested under
  `affine` / `nonaffine`);
* ‖x‖ was taken over all three states, mixing the actuator into the state
  norm;
* the tracking figure labelled `res.U` as "u" — that is v = u̇; the
  actuator is `res.X[:, 2]`, now plotted on its own row;
* `get_model_na` was a verbatim copy of `run_nonaffine_benchmark.get_model`
  referencing five names this module never imported — a guaranteed
  `NameError` on any cache miss. It now imports the shared function.
