# -*- coding: utf-8 -*-
"""
Magnetic Hysteresis Loop Fitting
=================================
Fits experimental VSM (Vibrating Sample Magnetometer) data for bilayer
magnetic thin-film systems using a macrospin energy minimisation model.

Physical model
--------------
The total energy of the two-layer system in an applied field B:

    E = E_Zeeman + E_Exchange + E_Anisotropy

    E_Zeeman    = -M·B·d·(cos θ₁ + cos θ₂)
    E_Exchange  = -cos(θ₁−θ₂)·(J₁ + J₂·cos(θ₁−θ₂))
    E_Aniso     = -Kᵤ·d·(cos²(a₁−θ₁) + cos²(a₂−θ₂))

where θ₁, θ₂ are the magnetisation angles of the two layers,
J₁ is the bilinear, J₂ the biquadratic exchange constant,
Kᵤ is the uniaxial anisotropy constant, and a₁, a₂ its easy-axis angles.

Fitting strategy
----------------
Replaces the original grid-search (brute-force) with SciPy
``differential_evolution``, giving the same physics at ~10–100× speed.

Usage
-----
1. Edit the ``# --- Configuration ---`` block below (file path, sample
   parameters, search bounds).
2. Run:  ``python fit_hysteresis.py``
3. Results are printed to the console and saved to ``results/fit_<name>.txt``.
   A plot is shown interactively and saved to ``results/fit_<name>.png``.

Author: Oleksii Kozlov
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_FILE   = "data/380.txt"   # two-column file: H [Oe], M [any units]
FIELD_UNIT  = "Oe"             # "Oe" → divides by 10 to get mT; "mT" → as-is

d = 2.0       # layer thickness [nm]
M = 1100.0    # saturation magnetisation [A/m]

XLIM_PLOT = 150   # ±field range shown on the plot [mT]

# Parameter search bounds: (min, max)
# Order: Ku [uJ/m³], a1 [rad], a2 [rad], J2 [uJ/m²], J1 [uJ/m²]
BOUNDS = [
    (20.0,  23.0),          # Ku  – uniaxial anisotropy
    (-np.pi, np.pi),        # a1  – easy-axis angle, layer 1
    (-np.pi, np.pi),        # a2  – easy-axis angle, layer 2
    (-15.0,  -5.0),         # J2  – biquadratic exchange
    (-80.0, -60.0),         # J1  – bilinear exchange
]

DE_MAXITER = 100    # differential evolution generations (increase for accuracy)
DE_POPSIZE = 15     # population size per parameter
DE_TOL     = 1e-4   # convergence tolerance
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Physics
# ---------------------------------------------------------------------------

def energy(angles: np.ndarray, B: float, params: tuple) -> float:
    """Total energy of the two-layer macrospin system."""
    theta1, theta2 = angles
    Ku, a1, a2, J2, J1 = params

    zeeman    = -M * B * d * (np.cos(theta1) + np.cos(theta2))
    dth       = theta1 - theta2
    exchange  = -np.cos(dth) * (J1 + J2 * np.cos(dth))
    anisotropy= -Ku * d * (np.cos(a1 - theta1)**2 + np.cos(a2 - theta2)**2)

    # 1e3 factor: unit rescaling so all terms are comparable in magnitude
    return zeeman + 1e3 * (exchange + anisotropy)


def equilibrium_angles(B: float, params: tuple, x0: np.ndarray) -> np.ndarray:
    """Find magnetisation angles that minimise energy at field B."""
    res = minimize(energy, x0, args=(B, params), method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 5000})
    return res.x


def magnetisation(angles: np.ndarray) -> float:
    """Normalised projection of magnetisation onto the field axis."""
    return 0.5 * (np.cos(angles[0]) + np.cos(angles[1]))


def simulate_branch(params: tuple, B_branch: np.ndarray,
                    x0: np.ndarray = None) -> np.ndarray:
    """
    Simulate one branch of the hysteresis loop.

    Parameters
    ----------
    params   : model parameters (Ku, a1, a2, J2, J1)
    B_branch : field values in mT, in the order they were measured
    x0       : initial angle guess (default: aligned with first field)

    Returns
    -------
    M_sim : normalised magnetisation array, same length as B_branch
    """
    if x0 is None:
        x0 = np.array([0.0, 0.0] if B_branch[0] > 0 else [np.pi, np.pi])

    angles  = x0.copy()
    M_sim   = np.empty(len(B_branch))
    for i, B in enumerate(B_branch):
        angles   = equilibrium_angles(B, params, angles)
        M_sim[i] = magnetisation(angles)
    return M_sim


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def objective(params: np.ndarray, B1, Y1, B2, Y2) -> float:
    """Sum of squared residuals over both hysteresis branches."""
    M1 = simulate_branch(params, B1)
    M2 = simulate_branch(params, B2, x0=np.array([np.pi, np.pi]))
    return float(np.sum((M1 - Y1)**2) + np.sum((M2 - Y2)**2))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str, field_unit: str = "Oe"):
    """Load two-column VSM data and split into two hysteresis branches."""
    z = np.loadtxt(path)
    H = z[:, 0] / 10.0 if field_unit.upper() == "OE" else z[:, 0]  # → mT
    m = z[:, 1] / np.max(z[:, 1])                                    # → M/Ms

    # Split into monotonically-decreasing (branch 1) and -increasing (branch 2)
    B1, Y1, B2, Y2 = [], [], [], []
    for i in range(len(H) - 1):
        if H[i] >= H[i + 1]:
            B1.append(H[i]); Y1.append(m[i])
        else:
            B2.append(H[i]); Y2.append(m[i])

    return (np.array(B1), np.array(Y1),
            np.array(B2), np.array(Y2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print(" Magnetic Hysteresis Fitting — Differential Evolution")
    print("=" * 60)

    # Load data
    data_path = Path(DATA_FILE)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file '{DATA_FILE}' not found.\n"
            "Place your two-column VSM file in the data/ directory and update DATA_FILE."
        )
    B1, Y1, B2, Y2 = load_data(DATA_FILE, FIELD_UNIT)
    print(f"Loaded '{DATA_FILE}': {len(B1)+len(B2)} points, "
          f"field range [{min(np.min(B1),np.min(B2)):.1f}, "
          f"{max(np.max(B1),np.max(B2)):.1f}] mT\n")

    # Fit
    t0 = datetime.now()
    print("Running differential evolution …")
    result = differential_evolution(
        objective,
        BOUNDS,
        args=(B1, Y1, B2, Y2),
        strategy="best1bin",
        maxiter=DE_MAXITER,
        popsize=DE_POPSIZE,
        tol=DE_TOL,
        seed=42,
        disp=True,
        workers=1,   # set to -1 to use all CPU cores
    )
    elapsed = datetime.now() - t0

    p = result.x
    Ku, a1, a2, J2, J1 = p
    print(f"\nFitting complete in {elapsed}")
    print(f"  Ku  = {Ku:.4f}  uJ/m³")
    print(f"  a1  = {a1:.4f}  rad  ({np.degrees(a1):.2f}°)")
    print(f"  a2  = {a2:.4f}  rad  ({np.degrees(a2):.2f}°)")
    print(f"  J2  = {J2:.4f}  uJ/m²  (biquadratic)")
    print(f"  J1  = {J1:.4f}  uJ/m²  (bilinear)")
    print(f"  SSR = {result.fun:.6f}")

    # Simulate best-fit curves
    M1_fit = simulate_branch(p, B1)
    M2_fit = simulate_branch(p, B2, x0=np.array([np.pi, np.pi]))

    # Save fit data
    stem = data_path.stem
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_txt = out_dir / f"fit_{stem}.txt"
    with open(out_txt, "w") as f:
        f.write("# B[mT]  M_exp  M_fit\n")
        for b, me, mf in zip(B1, Y1, M1_fit):
            f.write(f"{b:.4f}  {me:.6f}  {mf:.6f}\n")
        for b, me, mf in zip(B2, Y2, M2_fit):
            f.write(f"{b:.4f}  {me:.6f}  {mf:.6f}\n")
    print(f"\nFit data saved → {out_txt}")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(B1, Y1, "o", ms=2, color="steelblue", alpha=0.4, label="Exp ↓")
    ax.plot(B2, Y2, "o", ms=2, color="steelblue", alpha=0.4, label="Exp ↑")
    ax.plot(B1, M1_fit, color="crimson",   lw=2, label="Fit ↓")
    ax.plot(B2, M2_fit, color="darkorange",lw=2, label="Fit ↑")
    ax.set_xlim(-XLIM_PLOT, XLIM_PLOT)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("B [mT]")
    ax.set_ylabel("M / Mₛ")
    ax.set_title(
        f"T = {stem} K     "
        f"Ku={Ku:.2f}  J₁={J1:.2f}  J₂={J2:.2f}  "
        f"a₁={np.degrees(a1):.1f}°  a₂={np.degrees(a2):.1f}°"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = out_dir / f"fit_{stem}.png"
    fig.savefig(out_png, dpi=150)
    print(f"Plot saved        → {out_png}")
    plt.show()


if __name__ == "__main__":
    main()
