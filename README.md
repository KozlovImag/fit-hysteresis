# Magnetic Hysteresis Loop Fitting

Fits experimental VSM data for **bilayer magnetic thin-film** systems using a two-macrospin energy-minimisation model. Parameters — bilinear exchange *J*₁, biquadratic exchange *J*₂, uniaxial anisotropy *K*ᵤ, and easy-axis angles *a*₁/*a*₂ — are recovered via **Differential Evolution** (SciPy), replacing the original brute-force grid search while keeping the same physical model.

This code was used in the analysis behind the publication:

> V.Iurchuck, O. Kozlov, S.Sorokin *et al.*, **"All-electrical operation of a Curie switch at room temperature"**, *Physical Review Applied* (accepted).

---

## Physical model

The total energy of the two-layer system in an external field *B*:

$$E = E_\text{Zeeman} + E_\text{Exchange} + E_\text{Anisotropy}$$

$$E_\text{Zeeman} = -M B d\,(\cos\theta_1 + \cos\theta_2)$$

$$E_\text{Exchange} = -\cos(\theta_1-\theta_2)\bigl(J_1 + J_2\cos(\theta_1-\theta_2)\bigr)$$

$$E_\text{Aniso} = -K_u d\,\bigl(\cos^2(a_1-\theta_1)+\cos^2(a_2-\theta_2)\bigr)$$

At each field step the equilibrium angles (θ₁, θ₂) are found with Nelder-Mead minimisation, tracking the state from the previous step to reproduce hysteresis.

---

## Repository layout

```
hysteresis-fitting/
├── fit_hysteresis.py   # main script — configure & run this
├── requirements.txt
├── data/               # put your .txt VSM files here
│   └── 380.txt         # example: two columns — H [Oe], M [any unit]
└── results/            # output plots and fit data (auto-created)
```

---

## Quick start

```bash
git clone https://github.com/<your-handle>/hysteresis-fitting.git
cd hysteresis-fitting
pip install -r requirements.txt
```

Place your VSM data file (two columns: *H* and *M*) in `data/`, then edit the **Configuration** block at the top of `fit_hysteresis.py`:

```python
DATA_FILE  = "data/380.txt"   # your file
FIELD_UNIT = "Oe"             # "Oe" (÷10 → mT) or "mT"

d = 2.0       # layer thickness [nm]
M = 1100.0    # saturation magnetisation [A/m]

BOUNDS = [
    (20.0, 23.0),          # Ku  [uJ/m³]
    (-np.pi, np.pi),       # a1  [rad]
    (-np.pi, np.pi),       # a2  [rad]
    (-15.0,  -5.0),        # J2  [uJ/m²]  biquadratic
    (-80.0, -60.0),        # J1  [uJ/m²]  bilinear
]
```

Then run:

```bash
python fit_hysteresis.py
```

The script prints the best-fit parameters, saves a plot to `results/fit_<name>.png`, and writes the fit data to `results/fit_<name>.txt`.

---

## Input data format

Plain text, two columns, space- or tab-separated, no header required:

```
 2000.0   1098.3
 1980.0   1097.1
 ...
-2000.0  -1099.0
```

Column 1 — applied field (*H*) in Oe or mT (set `FIELD_UNIT` accordingly).  
Column 2 — measured magnetisation in any unit (normalised to *M*ₛ internally).

---

## Output

| File | Contents |
|------|----------|
| `results/fit_<name>.png` | overlay plot: experiment vs. model |
| `results/fit_<name>.txt` | three-column table: B [mT], M_exp, M_fit |

Console output example:

```
Ku  = 21.3812  uJ/m³
a1  = -0.6283  rad  (-36.00°)
a2  = -3.1416  rad  (-180.00°)
J2  =  -8.4971  uJ/m²  (biquadratic)
J1  = -74.2103  uJ/m²  (bilinear)
SSR =  0.002341
```

---

## Tuning the optimiser

| Parameter | Where | Effect |
|-----------|-------|--------|
| `DE_MAXITER` | config block | more generations → higher accuracy, slower |
| `DE_POPSIZE` | config block | larger population → better global search |
| `BOUNDS` | config block | tighter bounds → faster, needs prior knowledge |
| `workers=-1` | `differential_evolution` call | parallelise over all CPU cores |

For a fast first run keep `DE_MAXITER=50`, `DE_POPSIZE=10`. For publication-quality fits use `DE_MAXITER=300`, `DE_POPSIZE=20`.

---

## Dependencies

- Python ≥ 3.9
- NumPy ≥ 1.24
- SciPy ≥ 1.11
- Matplotlib ≥ 3.7

---

## License

MIT — see `LICENSE`.

---

## Citation

If you use this code in your work, please cite the associated paper:

```bibtex
@article{kozlov2025curieswitch,
  title   = {All-electrical operation of a {Curie} switch at room temperature},
  author  = {Kozlov, Oleksii and others},
  journal = {Physical Review Applied},
  year    = {2025},
  note    = {accepted}
}
```
