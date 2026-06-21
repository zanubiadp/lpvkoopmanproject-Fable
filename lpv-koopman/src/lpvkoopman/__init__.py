"""LPV-Koopman modeling toolkit.

Scope: autonomous Koopman eigenfunction models learned from trajectory
data (Korda & Mezic 2020) with a real conjugate-pair formulation, the
LPV input matrix B(x) = (dPhi/dx) g_c(x) derived from eigenfunction
gradients with no forced experiments (Iacob et al. 2024 / thesis), and
Koopman-LPV model predictive control on the lifted model.
"""
from .gradients import lift_jacobian, pde_residual
from .lpv import BMap, PolynomialBField, discretize, fit_b_field, predict_forced
from .model import KoopmanEigenModel
from .mpc import KoopmanMPC, MPCConfig, run_closed_loop
from .pipeline import FitConfig, FitReport, fit_koopman_model
from .spectrum import Spectrum, dmd_eigenvalues, lattice, lattice_spectrum
from .systems import (
    ContinuousSystem,
    TrajectoryData,
    circle_initial_conditions,
    generate_data,
    linear_system,
    vdp_classic,
    vdp_scaled,
)

__all__ = [
    "BMap",
    "ContinuousSystem",
    "KoopmanMPC",
    "MPCConfig",
    "PolynomialBField",
    "discretize",
    "fit_b_field",
    "lift_jacobian",
    "pde_residual",
    "predict_forced",
    "run_closed_loop",
    "FitConfig",
    "FitReport",
    "KoopmanEigenModel",
    "Spectrum",
    "TrajectoryData",
    "circle_initial_conditions",
    "dmd_eigenvalues",
    "fit_koopman_model",
    "generate_data",
    "lattice",
    "lattice_spectrum",
    "linear_system",
    "vdp_classic",
    "vdp_scaled",
]

__version__ = "0.1.0"
