# PPPF-mixture-RBPF (reference implementation)

This repo implements a minimal, end-to-end version of:

- **Pathwise Perturbation Particle Filtering (PPPF)** as a **Gaussian-mixture transition model** constructed from:
  - **stochastic perfect-foresight / extended-path solves** (Fair & Taylor, 1983)
  - **pathwise first-order linearization** (local Jacobians along the solved path)
  - **anticipated-shock quadrature** over a low-dimensional auxiliary variable via **unscented transform (UT) sigma points** (Julier & Uhlmann, 2004)
- A **Rao–Blackwellized particle filter (RBPF)** (Doucet et al., 2000) that:
  - samples discrete mixture indices (UT nodes)
  - runs a Kalman filter over the continuous state conditional on those indices

The goal is correctness and reproducibility (not a speed record).

## What’s implemented

### 1) Burnside (1998-style Jensen/IRF validation)

We implement the classic recursion

```
v_t = β E_t[ exp((1−γ) x_{t+1}) (1 + v_{t+1}) ]
x_t = (1−ρ) μ + ρ x_{t−1} + ε_t,   ε_t ~ N(0, σ^2)
```

and compute:

- **Exact** `v(x)` via a truncated analytic infinite series (Gaussian AR(1) → log-normal moments).
- **Certainty-equivalence (CE)** approximation (set future shocks to zero inside the expectation).
- **UT one-step quadrature** approximation (integrate only the next shock with UT, keep the continuation deterministic).

We then produce IRFs for `v_t` after a shock to `x_0` of `{0.5σ, 1σ, 2σ}` and write CSV/PNG under `output/burnside/`.

This is a Jensen check: UT tracks the exact IRF better than CE because it captures curvature from `exp(·)`.

### 2) New Keynesian (NK) model with an ELB kink

We implement a minimal 3-equation NK model with an occasionally binding constraint:

```
x_t = E x_{t+1} - σ^{-1}( i_t - E π_{t+1} - r_t^n )
π_t = β E π_{t+1} + κ x_t
i_t = max( i_lower , φπ π_t + φx x_t + ν_t )
```

Exogenous states:

```
r_t^n = ρ_r r_{t-1}^n + σ_r ε_{r,t}
ν_t   = ρ_ν ν_{t-1}   + σ_ν ε_{ν,t}
```

Implementation note: to make the ELB *occasionally* binding (rather than binding at the steady state), the code uses a consistent steady state with a positive natural rate `r_ss` and policy intercept `i_ss` (default `i_ss=r_ss`), and treats the state `r_t^n` as a **deviation** around `r_ss` inside the IS equation.

#### Baseline: OccBin-style piecewise linear + bootstrap PF

We implement an **OccBin-style regime iteration** (Guerrieri & Iacoviello, 2015) on a finite horizon `H` with terminal conditions `x_{t+H}=π_{t+H}=0`, producing a piecewise-linear mapping and using a **bootstrap PF** as a likelihood estimator.

#### Our method: PPPF-mixture-RBPF

We construct, at each filter step, a **UT-indexed mixture** over a scalar anticipated shock:

```
ω_t = ε_{r,t+1}^a   (scalar, p=1)
```

For each UT node `k`, we solve the NK OccBin system with the node’s anticipated shock inserted at `t+1`, linearize the one-step mapping to get a linear-Gaussian component, and then run an **RBPF** over the node index with a **Kalman filter** for the continuous state.

Implementation note: `include/quadrature/sigma_points.hpp` implements the commonly cited UT defaults `α=1e-2, κ=0, β=2`. In 1D these weights can be negative (fine for deterministic quadrature, but incompatible with sampling mixture indices). For the **probability mixture** used by the RBPF we therefore use a UT parameterization with `λ=2` (equivalently `α=1, κ=2` in 1D), yielding positive weights `{2/3, 1/6, 1/6}` and points `{μ, μ±sqrt(3)σ}`.

#### “True/global” benchmark IRF (stochastic, expectation-consistent)

To answer “which IRF is closer to the true nonlinear one?”, the repo also computes a **global stochastic solution** for the NK-ELB model and uses it as an IRF benchmark.

- We discretize the exogenous AR(1) states `(r_t^n, ν_t)` with **Tauchen** into a finite Markov chain.
- We solve for policy functions `x(r,ν), π(r,ν), i(r,ν)` on that discrete state space as an **expectation-consistent equilibrium** with the ELB enforced as a **complementarity condition** using a Fischer–Burmeister semi-smooth Newton method.
- The “global IRF” reported in `output/nk/irf.csv` is an **expected IRF** `E[y_t]` under this Markov chain solution starting from an impulse to `ε_{r,0}`.

This is “global” in the sense of being solved on the full discretized state space (not a local linearization and not a deterministic perfect-foresight path). It is still an **approximation** because of discretization; grid sizes are set in `include/models/nk_global.hpp` (`make_default_global_grid`).

We compare (on the same simulated dataset):

- runtime
- log-likelihood variance across `R=30` repeated PF runs
- ESS over time
- an IRF comparison near the ELB (including the global benchmark)

All outputs go to `output/nk/`.

## Build

Requirements:

- C++17 compiler
- CMake ≥ 3.16
- Eigen3 (found via `find_package(Eigen3)`)
- Python 3 (plots only). If `matplotlib` is present it will be used; otherwise plots are rendered via a small pure-stdlib fallback in `experiments/simple_png_plot.py`.

Build:

```bash
cd pppf-mixture-rbpf
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run everything

From `pppf-mixture-rbpf/`:

```bash
experiments/run_all.sh
```

Artifacts are written under `output/`:

- `output/burnside/irf_*.csv` and `output/burnside/irf_*.png`
- `output/nk/loglik_repeats.csv`, `output/nk/ess.csv`, `output/nk/summary.json`, `output/nk/irf.csv` (includes `*_global`, `*_occbin`, `*_pppf` columns), `output/nk/*.png`

## Notes on scope / defaults

- Horizon is fixed to `H=20` for the NK extended-path/OccBin solves (documented in code).
- The global NK benchmark uses a Tauchen grid for `(r,ν)` (defaults are `17×13` with width `m=3.5`).
- Particle counts and sample sizes are chosen to keep `run_all.sh` fast while still producing stable diagnostics.
- Seeds are explicit for deterministic reproducibility.

## References (text only)

- Fair & Taylor (1983) Extended Path
- Guerrieri & Iacoviello (2015) OccBin
- Aruoba et al. (2021) piecewise-linear approximations and filtering for OBC DSGEs
- Doucet et al. (2000) Rao–Blackwellized particle filtering
- Julier & Uhlmann (2004) unscented transform
- Alspach & Sorenson (1972) Gaussian-sum filtering
