"""End-to-end fitting pipeline: data -> DMD -> lattice -> refinement ->
boundary fit -> interpolable Koopman model."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .model import ComponentFit, KoopmanEigenModel, lifted_values
from .spectrum import Spectrum, dmd_eigenvalues, lattice_spectrum
from .systems import TrajectoryData
from .varpro import basis_matrix, fit_boundary, projection_cost, refine_spectrum

Array = np.ndarray


@dataclass
class FitConfig:
    n_eig: int | list[int] = 20  # total lift dimension (split across states)
    optimize: bool = True  # refine eigenvalues (vs. lattice only)
    interpolation: str = "linear"  # "linear" (paper) or "cubic"
    alpha_bounds: tuple[float, float] = (-20.0, 20.0)
    n_restarts: int = 0  # extra perturbed L-BFGS-B starts
    maxiter: int = 1000
    seed: int = 0
    base_eigs: Array | None = None  # override DMD base (e.g. linearization)
    constant_components: tuple[int, ...] = ()
    """State components known to be constant along every training trajectory
    (e.g. x3 = u in an input-extended system, where x3dot = 0 autonomously).
    Such a component is fitted exactly - single eigenvalue lambda = 0,
    boundary value = the trajectory's constant - instead of burning
    optimizer budget rediscovering it. Its entry in a per-component
    ``n_eig`` list is ignored (the block always has size 1)."""
    boundary_rcond: float = 1e-12
    """lstsq cutoff for the *final* boundary fit (eigenvalue refinement is
    untouched). The refined basis V can be nearly rank-deficient; at the
    default cutoff the near-null directions receive arbitrary huge
    coefficients (|z| ~ 1e2 observed on input-extended data). Those
    coordinates cancel in the reconstruction C z, so autonomous predictions
    barely notice - but eigenfunction *gradients* (the LPV B(x)) inherit
    the full roughness. A cutoff ~1e-3 zeroes the ill-determined directions
    at a small autonomous cost (measured on the non-affine benchmark:
    median 1.6% -> 2.3% autonomous, forced LPV rollout 36% -> 13%)."""
    fit_horizon: float | None = None
    """If set, eigenvalues and boundary values are fitted on the trajectory
    prefix t <= fit_horizon only, while eigenfunction values are tabulated
    (hence interpolable) along the full trajectories. Rationale: the
    exponential basis fits the transient well but fights the asymptotic
    limit-cycle portion; the prefix keeps the fit clean while the longer
    tabulation extends spatial coverage of the interpolation cloud toward
    the limit cycle, where convex-hull gaps otherwise dominate test error."""


@dataclass
class FitReport:
    dmd_base: Array = field(default_factory=lambda: np.empty(0, complex))
    components: list[dict] = field(default_factory=list)
    fit_seconds: float = 0.0


def fit_koopman_model(
    data: TrajectoryData, config: FitConfig | None = None
) -> tuple[KoopmanEigenModel, FitReport]:
    """Learn a Koopman eigenfunction model from autonomous trajectory data.

    Follows Korda & Mezic (2020) Sec. IV with the budget of ``n_eig``
    eigenfunctions split across state components; eigenvalues start at the
    DMD lattice and are optionally refined per component by variable
    projection. Returns the assembled model and a diagnostic report.
    """
    cfg = config or FitConfig()
    t0 = time.perf_counter()
    n = data.n_states
    rng = np.random.default_rng(cfg.seed)

    budgets = (
        [cfg.n_eig // n] * n if isinstance(cfg.n_eig, int) else list(cfg.n_eig)
    )
    if isinstance(cfg.n_eig, int) and cfg.n_eig % n:
        raise ValueError("scalar n_eig must be divisible by the state dimension")

    # Optionally restrict the *fit* to the samples with t <= fit_horizon
    # (tabulation below always uses the full data).
    if cfg.fit_horizon is not None:
        fit_mask = data.t <= cfg.fit_horizon + 1e-12
        if fit_mask.sum() < 2:
            raise ValueError("fit_horizon outside the data window")
    else:
        fit_mask = np.ones(data.n_samples, dtype=bool)
    t_fit = data.t[fit_mask]

    base = (
        np.asarray(cfg.base_eigs, dtype=complex)
        if cfg.base_eigs is not None
        else dmd_eigenvalues(data.X[:, fit_mask], data.ts)
    )

    report = FitReport(dmd_base=base)
    components: list[ComponentFit] = []
    for i in range(n):
        H = data.component(i)[fit_mask]  # (n_fit, M_t)
        if i in cfg.constant_components:
            vals = H[0].copy()  # (M_t,) the per-trajectory constants
            resid = float(((H - vals[None, :]) ** 2).sum())
            spec = Spectrum(pairs=np.empty((0, 2)), reals=np.array([0.0]))
            comp_info = {
                "constant_component": True,
                "lattice_cost": resid,
                "optimized_cost": resid,
                "spectrum": str(spec),
            }
            components.append(
                ComponentFit(spec=spec, G=vals[None, :], cost=resid, info=comp_info)
            )
            report.components.append(comp_info)
            continue
        spec0 = lattice_spectrum(base, budgets[i])
        J_lattice = projection_cost(spec0, t_fit, H)
        if cfg.optimize:
            res = refine_spectrum(
                spec0,
                t_fit,
                H,
                alpha_bounds=cfg.alpha_bounds,
                maxiter=cfg.maxiter,
                n_restarts=cfg.n_restarts,
                rng=rng,
            )
            spec, J = res.spectrum, res.cost
            comp_info = {
                "lattice_cost": J_lattice,
                "optimized_cost": J,
                "n_iter": res.n_iter,
                "converged": res.converged,
            }
        else:
            spec, J = spec0, J_lattice
            comp_info = {"lattice_cost": J_lattice, "optimized_cost": None}

        V = basis_matrix(spec, t_fit)
        G, _ = fit_boundary(V, H, rcond=cfg.boundary_rcond)
        components.append(ComponentFit(spec=spec, G=G, cost=J, info=comp_info))
        comp_info["spectrum"] = str(spec)
        report.components.append(comp_info)

    # Tabulate lifted coordinates at every training sample, in flat point
    # order (trajectory-major) to match TrajectoryData.flat_points().
    tables = []
    for comp in components:
        Z = lifted_values(comp.spec, data.t, comp.G)  # (N_i, T, M_t)
        tables.append(Z.transpose(2, 1, 0).reshape(-1, comp.spec.size))
    lifted_at_points = np.concatenate(tables, axis=1)

    model = KoopmanEigenModel(
        components=components,
        points=data.flat_points(),
        lifted_at_points=lifted_at_points,
        interpolation=cfg.interpolation,
    )
    report.fit_seconds = time.perf_counter() - t0
    return model, report
