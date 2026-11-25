"""Cooling of a moulded wall against the mould: a 1D finite-volume solver and the exact series.

The wall of a moulded part (thickness s) starts at the melt temperature. Both faces lose heat to the
mould through a contact heat-transfer coefficient h. By symmetry only half the wall is solved:
x = 0 is the mid-plane (no flux) and x = s/2 is the face against the mould.

    H'(T) dT/dt = d/dx (k dT/dx)            0 < x < s/2
    dT/dx = 0                               x = 0
    -k dT/dx = h (T - T_mould)              x = s/2

H(T) is the enthalpy per volume. It holds the sensible heat rho*c*T and the latent heat of
crystallisation, released over a Gaussian band around T_CRYST. With zero latent heat the problem is
linear and has the exact series solution in `analytic_theta`, which the tests and `verify.py` use.

The quantity of interest is the cooling time: the time until the mid-plane reaches T_EJECT.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import brentq
from scipy.special import ndtr

# Material and process constants (a generic semi-crystalline polymer, polypropylene-like).
RHO = 900.0  # kg/m^3
CP = 2000.0  # J/(kg K)
RHO_CP = RHO * CP  # J/(m^3 K)
T_CRYST = 120.0  # C, centre of the crystallisation band
SIGMA_CRYST = 6.0  # K, width of the crystallisation band
T_EJECT = 100.0  # C, the mid-plane must be below this before the part can be ejected

PARAM_NAMES = (
    "melt_temp_c",
    "mould_temp_c",
    "thickness_mm",
    "diffusivity_mm2_s",
    "htc_w_m2k",
    "latent_heat_kj_kg",
)

# Discretisation used for the dataset. `verify.py` measures what these choices cost in accuracy.
N_CELLS = 32  # cells across the half wall, so 33 nodes from mid-plane to face
DFO = 0.005  # time step in Fourier units (a * dt / (s/2)^2) while searching for the cooling time
STEPS_TO_EJECT = 400  # time steps from 0 to the cooling time in the second pass
STARTUP_STEPS = 4  # backward Euler steps before Crank-Nicolson, to damp the start (Rannacher)
PICARD_MAX_ITERATIONS = 12
PICARD_TOLERANCE_K = 1e-7

# Output times, as fractions of the cooling time. They start at 5 %: before that the cold layer at
# the mould face is only a few cells thick and the solver's own error there exceeds 1 K.
FRACTIONS = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])


@dataclass(frozen=True)
class Case:
    """Per-case quantities in SI units, each an array of shape (B,)."""

    t_melt: np.ndarray
    t_mould: np.ndarray
    half: np.ndarray  # half thickness, m
    diffusivity: np.ndarray  # m^2/s
    htc: np.ndarray  # W/(m^2 K)
    latent: np.ndarray  # latent heat per volume, J/m^3

    @property
    def conductivity(self) -> np.ndarray:
        return self.diffusivity * RHO_CP

    @property
    def biot(self) -> np.ndarray:
        return self.htc * self.half / self.conductivity

    @classmethod
    def from_params(cls, params: np.ndarray) -> Case:
        p = np.atleast_2d(np.asarray(params, dtype=np.float64))
        if p.shape[1] != len(PARAM_NAMES):
            raise ValueError(f"expected {len(PARAM_NAMES)} parameters per case, got {p.shape[1]}")
        case = cls(
            t_melt=p[:, 0],
            t_mould=p[:, 1],
            half=p[:, 2] * 0.5e-3,
            diffusivity=p[:, 3] * 1e-6,
            htc=p[:, 4],
            latent=p[:, 5] * 1e3 * RHO,
        )
        if np.any(case.t_mould >= T_EJECT) or np.any(case.t_melt <= T_EJECT):
            raise ValueError(f"need mould < {T_EJECT} C < melt for every case")
        if np.any(case.half <= 0) or np.any(case.diffusivity <= 0) or np.any(case.htc <= 0) or np.any(case.latent < 0):
            raise ValueError("thickness, diffusivity and heat-transfer coefficient must be positive")
        return case

    def take(self, idx: np.ndarray) -> Case:
        return Case(*(getattr(self, f)[idx] for f in ("t_melt", "t_mould", "half", "diffusivity", "htc", "latent")))


def enthalpy(temp: np.ndarray, latent: np.ndarray) -> np.ndarray:
    """Enthalpy per volume (J/m^3), relative to 0 C. `latent` broadcasts against `temp`."""
    return RHO_CP * temp + latent * ndtr((temp - T_CRYST) / SIGMA_CRYST)


def _apparent_capacity(temp: np.ndarray, latent: np.ndarray) -> np.ndarray:
    z = (temp - T_CRYST) / SIGMA_CRYST
    return RHO_CP + latent * np.exp(-0.5 * z * z) / (SIGMA_CRYST * np.sqrt(2.0 * np.pi))


def _chord_capacity(t0: np.ndarray, t1: np.ndarray, latent: np.ndarray) -> np.ndarray:
    """(H(t1) - H(t0)) / (t1 - t0): with it each step conserves enthalpy exactly, peak or no peak."""
    dt = t1 - t0
    small = np.abs(dt) < 1e-6
    safe = np.where(small, 1.0, dt)
    chord = (enthalpy(t1, latent) - enthalpy(t0, latent)) / safe
    return np.where(small, _apparent_capacity(t0, latent), chord)


def node_weights(n_cells: int) -> np.ndarray:
    """Control-volume widths of the nodes as fractions of the half thickness (trapezoid rule)."""
    w = np.full(n_cells + 1, 1.0 / n_cells)
    w[0] = w[-1] = 0.5 / n_cells
    return w


def stored_heat(temp: np.ndarray, case: Case, n_cells: int) -> np.ndarray:
    """Enthalpy per unit face area (J/m^2) of profiles `temp` of shape (B, N) or (B, F, N)."""
    shape = (-1,) + (1,) * (temp.ndim - 1)
    lat = case.latent.reshape(shape)
    half = case.half.reshape(shape[:-1])
    return half * np.sum(node_weights(n_cells) * enthalpy(temp, lat), axis=-1)


def removable_heat(case: Case) -> np.ndarray:
    """Heat per face area (J/m^2) between a wall at melt temperature and one at mould temperature."""
    return case.half * (enthalpy(case.t_melt, case.latent) - enthalpy(case.t_mould, case.latent))


class _Stepper:
    """One theta-scheme step for a batch of independent cases, solved as one banded system."""

    def __init__(self, case: Case, n_cells: int):
        self.case = case
        self.n = n_cells + 1
        dx = case.half / n_cells
        self.g = (case.conductivity / dx)[:, None]  # conductance between neighbouring nodes
        self.w = node_weights(n_cells)[None, :] * case.half[:, None]
        self.lat = case.latent[:, None]
        self.h = case.htc
        self.tw = case.t_mould

    def _k_times(self, temp: np.ndarray) -> np.ndarray:
        """Heat leaving each node per unit time and area (conduction, plus the face to the mould)."""
        g, out = self.g[:, 0], np.empty_like(temp)
        out[:, 0] = g * (temp[:, 0] - temp[:, 1])
        out[:, 1:-1] = self.g * (2.0 * temp[:, 1:-1] - temp[:, :-2] - temp[:, 2:])
        out[:, -1] = g * (temp[:, -1] - temp[:, -2]) + self.h * (temp[:, -1] - self.tw)
        return out

    def step(self, temp: np.ndarray, dt: np.ndarray, theta: float) -> tuple[np.ndarray, np.ndarray]:
        """Advance every case by its own dt. Returns the new profiles and the heat (J/m^2) that left."""
        b, n = temp.shape
        g = self.g
        k_diag = np.repeat(2.0 * g, n, axis=1)
        k_diag[:, 0] = g[:, 0]
        k_diag[:, -1] = g[:, 0] + self.h
        upper = np.repeat(-theta * g, n, axis=1)
        upper[:, 0] = 0.0  # no coupling between the last node of a case and the first of the next
        lower = np.repeat(-theta * g, n, axis=1)
        lower[:, -1] = 0.0
        explicit = (1.0 - theta) * self._k_times(temp)
        explicit[:, -1] -= theta * self.h * self.tw  # implicit part of the mould temperature term

        # Picard iterations on the heat capacity: start from the tangent at the old temperature, then
        # use the chord between old and new temperature until the new temperature stops moving.
        new = temp
        for it in range(PICARD_MAX_ITERATIONS):
            cap = _apparent_capacity(temp, self.lat) if it == 0 else _chord_capacity(temp, new, self.lat)
            m = self.w * cap / dt[:, None]
            ab = np.vstack([upper.ravel(), (m + theta * k_diag).ravel(), lower.ravel()])
            prev = new
            new = solve_banded((1, 1), ab, (m * temp - explicit).ravel(), check_finite=False).reshape(b, n)
            if it > 0 and np.max(np.abs(new - prev)) < PICARD_TOLERANCE_K:
                break
        face = theta * new[:, -1] + (1.0 - theta) * temp[:, -1]
        return new, self.h * (face - self.tw) * dt


def cooling_time(case: Case, n_cells: int = N_CELLS, dfo: float = DFO, max_fo: float = 60.0) -> np.ndarray:
    """First pass: march every case until its mid-plane falls below T_EJECT. Returns seconds."""
    b = case.t_melt.shape[0]
    result = np.full(b, np.nan)
    active = np.arange(b)
    sub = case
    stepper = _Stepper(sub, n_cells)
    dt = dfo * sub.half**2 / sub.diffusivity
    temp = np.repeat(sub.t_melt[:, None], n_cells + 1, axis=1)
    max_steps = int(np.ceil(max_fo / dfo))
    for step in range(max_steps):
        theta = 1.0 if step < STARTUP_STEPS else 0.5
        new, _ = stepper.step(temp, dt, theta)
        old_mid, new_mid = temp[:, 0], new[:, 0]
        done = new_mid <= T_EJECT
        if done.any():
            frac = (old_mid[done] - T_EJECT) / (old_mid[done] - new_mid[done])
            result[active[done]] = (step + frac) * dt[done]
            keep = ~done
            if not keep.any():
                return result
            active, new, dt = active[keep], new[keep], dt[keep]
            sub = case.take(active)
            stepper = _Stepper(sub, n_cells)
        temp = new
    raise RuntimeError(f"{active.size} cases did not reach the ejection temperature by Fo = {max_fo}")


@dataclass(frozen=True)
class Solution:
    t_cool: np.ndarray  # (B,) seconds
    field: np.ndarray  # (B, F, N) temperatures in C at FRACTIONS * t_cool, nodes from mid-plane to face
    heat: np.ndarray  # (B, F) heat that left through one face by each output time, J/m^2


def solve(
    params: np.ndarray, n_cells: int = N_CELLS, dfo: float = DFO, steps_to_eject: int = STEPS_TO_EJECT
) -> Solution:
    """Cooling time, the profiles at FRACTIONS of it, and the heat removed, for a batch of cases."""
    case = Case.from_params(params)
    t_cool = cooling_time(case, n_cells, dfo)
    record = np.rint(FRACTIONS * steps_to_eject).astype(int)
    if not np.allclose(record, FRACTIONS * steps_to_eject):
        raise ValueError("steps_to_eject must put every output fraction on a whole step")

    b, n = t_cool.shape[0], n_cells + 1
    stepper = _Stepper(case, n_cells)
    dt = t_cool / steps_to_eject
    temp = np.repeat(case.t_melt[:, None], n, axis=1)
    removed = np.zeros(b)
    field = np.empty((b, FRACTIONS.size, n))
    heat = np.empty((b, FRACTIONS.size))
    slot = 0
    for step in range(1, steps_to_eject + 1):
        theta = 1.0 if step <= STARTUP_STEPS else 0.5
        temp, q = stepper.step(temp, dt, theta)
        removed += q
        while slot < record.size and record[slot] == step:
            field[:, slot] = temp
            heat[:, slot] = removed
            slot += 1
    return Solution(t_cool=t_cool, field=field, heat=heat)


# ---------------------------------------------------------------------------------------------
# Exact solution for zero latent heat (constant properties), used to verify the solver.


def robin_eigenvalues(biot: float, n_terms: int) -> np.ndarray:
    """Roots of lambda * tan(lambda) = Bi, one in each interval (j*pi, j*pi + pi/2)."""

    def f(lam: float) -> float:
        return lam * np.sin(lam) - biot * np.cos(lam)

    return np.array([brentq(f, j * np.pi + 1e-12, j * np.pi + np.pi / 2) for j in range(n_terms)])


def analytic_theta(xi: np.ndarray, fo: np.ndarray, biot: float, n_terms: int = 200) -> np.ndarray:
    """(T - T_mould) / (T_melt - T_mould) at xi = x / (s/2) from the mid-plane and Fourier number fo.

        theta = sum_n C_n cos(lambda_n xi) exp(-lambda_n^2 Fo),  C_n = 4 sin(l) / (2 l + sin(2 l))

    Returns shape fo.shape + xi.shape.
    """
    lam = robin_eigenvalues(biot, n_terms)
    coef = 4.0 * np.sin(lam) / (2.0 * lam + np.sin(2.0 * lam))
    fo = np.asarray(fo, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)
    decay = np.exp(-np.multiply.outer(fo, lam**2))  # fo.shape + (n,)
    modes = np.cos(np.multiply.outer(lam, xi))  # (n,) + xi.shape
    return np.tensordot(decay * coef, modes, axes=1)


def analytic_cooling_time(params: np.ndarray) -> np.ndarray:
    """Exact cooling time (s) for cases with zero latent heat."""
    case = Case.from_params(params)
    if np.any(case.latent > 0):
        raise ValueError("the exact solution only holds without latent heat")
    out = np.empty(case.t_melt.shape[0])
    for i in range(out.size):
        # The mid-plane only drops below T_EJECT at Fo > 0.05 for these inputs, where 40 terms are
        # exact to machine precision (exp(-lambda_40^2 * 0.05) < 1e-300).
        lam = robin_eigenvalues(float(case.biot[i]), 40)
        coef = 4.0 * np.sin(lam) / (2.0 * lam + np.sin(2.0 * lam))
        target = (T_EJECT - case.t_mould[i]) / (case.t_melt[i] - case.t_mould[i])

        def mid_minus_target(fo: float, lam: np.ndarray = lam, coef: np.ndarray = coef, target: float = target):
            return float(np.sum(coef * np.exp(-(lam**2) * fo))) - target

        fo_cool = brentq(mid_minus_target, 0.05, 200.0, xtol=1e-14, rtol=1e-13)
        out[i] = fo_cool * case.half[i] ** 2 / case.diffusivity[i]
    return out
