# A) Critical issues

- Location: Section `Introduction`, pp. 1-3. What was wrong: the old draft described “PPPF” as if it were one approximation object rather than a stack of horizon truncation, expectation approximation, local linearization, and localization choices. Why it matters: without that separation, likelihood gaps could not be attributed cleanly and the benchmark risked sounding more definitive than it was. Specific fix: the introduction now defines PPPF as a layered construction and states explicitly that the true kernel, the finite-horizon kernel, the linearized kernel, and the quadrature kernel are distinct objects.

- Location: Section `Nonlinear state-space model`, Table 1, p. 4. What was wrong: the old notation used `x_t` both for the generic latent state and for the NK output gap. Why it matters: the collision made the algorithm and benchmark sections hard to parse and obscured which state the filter actually propagated. Specific fix: the paper now reserves `z_t` for the generic latent state and keeps `x_t` for the NK output gap only.

- Location: Section `Nonlinear state-space model`, eqs. (1)-(3), p. 4. What was wrong: the old manuscript imposed a globally linear measurement equation in the generic exposition and then quietly treated the policy rate differently in the NK benchmark. Why it matters: the paper was implicitly claiming exact conditional Gaussian filtering where the implementation actually uses a moment-projected censored update for the interest-rate measurement. Specific fix: the revised text introduces a generic measurement density first, then a linear-Gaussian special case, and explicitly states that the NK policy-rate measurement is a separate approximation held fixed across ablations.

- Location: Section `From stacked equilibrium conditions to a conditional linear-Gaussian law`, eq. (4), p. 5. What was wrong: the old draft defined the stacked solve without a regime argument and therefore differentiated through a kink-prone object. Why it matters: at an ELB kink the Jacobian of the max operator is not the right object; one must differentiate the regime-conditioned smooth system. Specific fix: the stacked residual map now includes the localization/regime index `s_t` directly in eq. (4).

- Location: Section `From stacked equilibrium conditions to a conditional linear-Gaussian law`, eqs. (5)-(7), p. 5. What was wrong: the old manuscript suppressed the local Taylor remainder and treated the conditional Gaussian law as if it were exact once a path had been solved. Why it matters: that hides where local-linearization bias enters and makes later “convergence” language too strong. Specific fix: eq. (5) now includes the remainder term `r_t^{lin}`, and eqs. (6)-(7) are stated explicitly as the approximation obtained after dropping it.

- Location: Section `Integrating out anticipated shocks`, eqs. (8)-(9), p. 6. What was wrong: the previous exposition did not clearly distinguish integration over the numerical proxy `\omega_t` from integration over the realized shock `\varepsilon_{t+1}`. Why it matters: readers could easily misread the method as double-counting uncertainty or as adding fictitious shocks to the model. Specific fix: the revised kernel definition states the measure `\mu_\Omega(d\omega)` explicitly and emphasizes that `\omega_t` is a numerical integration variable, not an extra structural innovation.

- Location: Section `Integrating out anticipated shocks`, eq. (9), p. 6. What was wrong: the old draft allowed sigma-point quadrature to slide back and forth between deterministic quadrature and probabilistic mixture language. Why it matters: negative-weight rules are admissible for deterministic integration but invalid for sampling mixture indices in an RBPF. Specific fix: eq. (9) now defines a probability mixture only when the component weights are nonnegative and normalized, and the surrounding prose states that signed rules require deterministic branching instead.

- Location: Section `Discrete localization and globalization`, p. 6. What was wrong: the old manuscript leaned too hard on a generic “globalization by partition refinement” theorem without separating smooth-region refinement from kink handling. Why it matters: the NK application does not implement a general partition-refinement scheme, and no single Taylor expansion resolves an ELB kink. Specific fix: the revised section distinguishes cell refinement away from kinks from explicit regime conditioning near kinks and shifts the discussion from over-broad theorem language to a precise approximation taxonomy.

- Location: Section `Approximation taxonomy`, Table 2, p. 7. What was wrong: the draft lacked a single place where the four core approximation axes were stated side by side. Why it matters: without that table, the empirical section could not diagnose which layer was responsible for which likelihood gap. Specific fix: Table 2 now lists the exact object being replaced, the computable surrogate, and the main error mechanism for each axis.

- Location: Section `PPPF-mixture-RBPF recursion`, eq. (10), pp. 7-8. What was wrong: the old recursion mixed an implementation-specific optimal proposal with a generic weight formula and never clarified when the proposal correction collapses to the predictive normalizing constant. Why it matters: a reader trying to reproduce the algorithm could easily code the wrong weight update. Specific fix: the revised pseudocode states the general weight formula in eq. (10) and then gives the auxiliary-proposal simplification explicitly.

- Location: Section `Approximation theory`, Proposition 1 and eq. (11), p. 9. What was wrong: the old theory section blurred convergence to the quadrature-refined PPPF kernel with convergence to the true DSGE kernel. Why it matters: that overstates what the analysis shows and invites a referee objection on mathematical overclaiming. Specific fix: Proposition 1 now states only convergence of the numerical quadrature to the PPPF mixture kernel, and eq. (11) decomposes the remaining error into the four conceptually distinct layers.

- Location: Section `Approximation theory`, Assumption 2 and Proposition 2, pp. 9-10. What was wrong: the old perturbation argument used a strong global minorization condition without flagging how unrealistic it is for DSGEs. Why it matters: the paper risked sounding as though it had proved a general likelihood theorem for models where the assumption is often false. Specific fix: the revised text labels the condition as a strong stability device and reframes the result as a standard perturbation implication used only to motivate the empirical diagnostics.

- Location: Section `Benchmark: a New Keynesian model with an ELB`, eqs. (12a)-(12d), pp. 11-12. What was wrong: the previous benchmark section listed `i_t` as part of the latent state even though the implementation propagates `z_t=(x_t,\pi_t,r_t^n,\nu_t)` and computes the rate algebraically. Why it matters: that mismatch made the measurement mapping and Kalman conditioning look different from the code. Specific fix: the benchmark now defines the latent state correctly and states that the policy rate is an algebraic measurement transformation.

- Location: Section `Benchmark: a New Keynesian model with an ELB`, pp. 12-14. What was wrong: the old benchmark language treated the “exact” comparator too casually and described the PLC/COPF comparator too strongly. Why it matters: the benchmark likelihood is exact only for the discretized Markov-chain benchmark, and the implemented PLC comparison is PLC-inspired rather than a full reproduction of Aruoba et al. Specific fix: the revised prose now labels the benchmark as exact for the discretized model only and recasts the PLC object as a comparator in the same spirit rather than the same algorithm.

- Location: Section `Illustrative results` and `Conclusion`, pp. 13-14. What was wrong: the old narrative mixed variance-reduction claims with accuracy claims and sometimes implied that higher ESS meant the approximation was better. Why it matters: that confuses Monte Carlo stability with transition-kernel fidelity. Specific fix: the revised results and conclusion sections separate variance reduction from approximation bias and identify kink handling as the main practical source of PPPF error in the NK calibration.

# B) Algorithm spec v2

1. Inputs: parameter vector `\theta`, data `y_{1:T}`, particle count `N`, horizon `H`, anticipated-shock proxy law `\mu_\Omega`, localization/regime set `\mathcal{S}_t`, quadrature rule `{(\omega^{(k)}, w_k)}_{k=1}^K`, initial Gaussian summaries `{(\mu_0^{(i)}, P_0^{(i)})}_{i=1}^N`, and an index proposal family `q_t^{(i)}(j)`.
2. Initialize particle weights at `W_0^{(i)}=1/N`.
3. For each date `t=1,\ldots,T`, choose an anchor.
4. Shared-anchor version: set `z_t^{ref}=\sum_i W_{t-1}^{(i)}\mu_{t-1}^{(i)}`.
5. Per-particle version: set `z_t^{ref,(i)}=\mu_{t-1}^{(i)}` for each particle.
6. For each required anchor and each candidate pair `j=(s,k)`, solve the finite-horizon stacked system `\Psi_H(U_t; z_t^{ref}, \omega^{(k)}, s, \theta)=0`.
7. Compute the pathwise Jacobians `A_t^{(j)}` and `B_t^{(j)}`, set `Q_t^{(j)}=B_t^{(j)}\Sigma B_t^{(j)\top}`, and set the affine term `a_t^{(j)}=\bar z_{t+1}^{(j)}-A_t^{(j)} z_t^{ref}`.
8. Form the component weights `\tilde w_t^{(j)}`. If the rule is used for sampling, require `\tilde w_t^{(j)}\ge 0` and `\sum_j \tilde w_t^{(j)}=1`. If not, switch to deterministic branching rather than sampling.
9. For each particle `i` and candidate component `j`, compute the Gaussian prediction
   `\mu_{t|t-1}^{(i,j)}=A_t^{(j)}\mu_{t-1}^{(i)}+a_t^{(j)}`
   and
   `P_{t|t-1}^{(i,j)}=A_t^{(j)}P_{t-1}^{(i)}A_t^{(j)\top}+Q_t^{(j)}`.
10. Evaluate the predictive density `\ell_t^{(i)}(j)=p(y_t\mid y_{1:t-1}, j, \mu_{t-1}^{(i)}, P_{t-1}^{(i)})`. Under linear-Gaussian measurements this is the Kalman predictive density. Under the NK rate measurement it is the moment-projected censored Gaussian predictive density.
11. Choose a proposal for `j_t^{(i)}`.
12. Prior proposal: `q_t^{(i)}(j)=\tilde w_t^{(j)}`.
13. Auxiliary proposal: `q_t^{(i)}(j)\propto \tilde w_t^{(j)}\ell_t^{(i)}(j)`.
14. Sample `j_t^{(i)}\sim q_t^{(i)}(\cdot)`.
15. Update the particle weight by
   `W_t^{(i)} \propto W_{t-1}^{(i)} \ell_t^{(i)}(j_t^{(i)}) \tilde w_t^{(j_t^{(i)})} / q_t^{(i)}(j_t^{(i)})`.
16. Under the auxiliary proposal, replace the ratio in step 15 by the normalizing constant `\sum_j \tilde w_t^{(j)}\ell_t^{(i)}(j)`.
17. Conditional on `j_t^{(i)}`, perform the measurement update and obtain the posterior Gaussian summary `(\mu_t^{(i)}, P_t^{(i)})`.
18. Normalize the particle weights, compute ESS, resample if ESS falls below the threshold, and accumulate the log-likelihood with the log normalizing constant.
19. Output the estimated log-likelihood together with diagnostics on ESS, mixture usage, and the relative contribution of the approximation axes.

# C) Definitions and notation refactor

Proposed notation table:

| Symbol | Meaning | Status |
|---|---|---|
| `z_t` | Generic latent state propagated by the filter | Use everywhere outside application-specific equations |
| `y_t` | Observable vector | Keep |
| `g_\theta(y_t\mid z_t)` | Generic measurement density | Introduce before the linear-Gaussian special case |
| `M, R` | Linear measurement matrix and covariance in the Gaussian special case | Keep, but only after `g_\theta` is introduced |
| `\varepsilon_{t+1}` | Realized one-step innovation that moves the state from `t` to `t+1` | Keep and label as random |
| `\omega_t` | Anticipated-shock proxy used only for numerical integration of future uncertainty | Keep and label as a numerical integration variable |
| `\mu_\Omega(d\omega)` | Gaussian reference measure for `\omega_t` | State every time `\omega_t` is integrated out |
| `H` | Finite-horizon stacked-solve length | Keep |
| `s_t` | Localization or regime index | Keep, but say whether it is deterministic, sampled, or mixed over |
| `k_t` | Quadrature node index for `\omega^{(k)}` | Keep |
| `j_t=(s_t,k_t)` | Combined discrete index propagated by the RBPF | Keep |
| `z_t^{ref}` | Anchor/reference state used to solve the stacked system and compute Jacobians | Keep |
| `x_t` | Output gap in the NK model only | Reserve for the application section |
| `\pi_t` | Inflation in the NK model only | Reserve for the application section |
| `i_t` | Policy rate in the NK model only | Reserve for the application section |

First-use rewrites:

- `z_t`: “Let `z_t\in\mathbb{R}^{n_z}` denote the generic latent state used by the filter. We reserve `x_t` for the output gap in the New Keynesian application below.”
- `y_t`: “Let `y_t\in\mathbb{R}^{n_y}` denote the observed data, with generic measurement density `g_\theta(y_t\mid z_t)`.”
- `\varepsilon_{t+1}`: “The realized innovation `\varepsilon_{t+1}\sim\mathcal{N}(0,\Sigma)` is random and moves the state from date `t` to `t+1`.”
- `\omega_t`: “The anticipated-shock proxy `\omega_t\sim\mathcal{N}(0,\Omega)` is a numerical integration variable used to approximate future conditional expectations; it is distinct from `\varepsilon_{t+1}`.”
- `s_t`: “The localization index `s_t` selects either a smooth partition cell or a regime-conditioned branch of the equilibrium system.”
- `k_t`: “The node index `k_t` labels the quadrature point `\omega^{(k)}`.”
- `H`: “The horizon `H` is the length of the finite-horizon stacked equilibrium system used at each filtering step.”

# D) Approximation taxonomy

> **Four core PPPF approximations**
>
> 1. **Finite horizon / terminal condition.**
>    Exact object: the infinite-horizon DSGE transition kernel `K_\theta`.
>    Computable surrogate: the transition induced by the `H`-period stacked equilibrium problem with a terminal condition.
>    Error source: truncation of future expectations and terminal misspecification.
>
> 2. **Expectation approximation via anticipated-shock quadrature.**
>    Exact object: the finite-horizon kernel with the full future shock law integrated out.
>    Computable surrogate: a kernel based on a low-dimensional proxy `\omega_t` integrated with respect to `\mu_\Omega(d\omega)`, then approximated numerically by quadrature.
>    Error source: omitted future uncertainty, an inadequate proxy state, or coarse quadrature.
>
> 3. **Local linearization.**
>    Exact object: the finite-horizon path solve conditional on `(\omega_t,s_t)`.
>    Computable surrogate: the affine-Gaussian component obtained by first-order expansion around the solved path.
>    Error source: the Taylor remainder and, when used, anchor-reuse error from extrapolating away from `z_t^{ref}`.
>
> 4. **Localization / regime handling.**
>    Exact object: the correct smooth branch selected by the true state or true regime law.
>    Computable surrogate: partition-based localization or an explicit regime mixture over `s_t`.
>    Error source: misclassified cells, hard regime selection at kinks, or too-coarse regime mixtures near the ELB boundary.

Interpretation:

- Quadrature refinement converges only to the PPPF mixture kernel implied by the chosen horizon, localization rule, and local-linearization scheme.
- It does not by itself converge to the true DSGE kernel.
- In the NK implementation there is also a measurement-side approximation for the observed policy rate; that approximation is kept fixed in the PPPF ablations so the benchmark still isolates the transition-side errors above.

# E) Example/benchmark fixes

- Burnside verification exercise: keep it, but state explicitly that it diagnoses only the expectation-approximation layer. It is not a full PPPF validation because there is no regime issue and no local-linearization error.

- Linear-Gaussian sanity check: keep it and emphasize that it isolates the filtering layer. PPPF with `K=1` and degenerate `\omega` should coincide with the Kalman filter exactly for the same approximate model.

- Minimal toy example: add it before the NK benchmark, exactly as in the revised manuscript, because it gives a reader with no DSGE background a clean view of what “quadrature as mixture” means and why moment-matching the mixture is not likelihood preserving.

- NK no-ELB verification table: keep it, but describe the Kalman baseline as exact for the same no-ELB approximate model evaluated by the filters, not as exact for the nonlinear DSGE. That removes an unnecessary identification risk.

- NK benchmark summary table: keep it, but frame it as a bias-versus-variance decomposition rather than a horse race. High ESS and low likelihood dispersion identify Monte Carlo stability, not transition accuracy.

- PLC comparator: keep it only if it is described as PLC-inspired rather than a complete PLC/COPF replication. Otherwise a referee can object that the comparison is not apples-to-apples.

- PPPF ablation table: elevate it from a side result to the main diagnostic table. The paper should explicitly map each row to one term in the kernel decomposition.

- IRF figure near the ELB: keep it, but interpret it jointly with the likelihood ablations. An IRF miss at the kink is useful because it visually diagnoses regime mishandling even when ESS looks excellent.

- Add one benchmark with adaptive regime mixture only at the first ELB decision: this directly tests whether the first-order PPPF bias comes from the boundary itself rather than from deeper-horizon regime uncertainty.

- Add one benchmark with the same ELB data but a richer anticipated-shock proxy and unchanged particle count: this isolates whether the dominant remaining gap is in expectation approximation or in regime handling.

# F) Path to making it work

- Replace hard first-period regime selection with a cheap two-component bind/slack mixture near the ELB boundary. This is the highest-return change because the current ablations already show kink handling is the dominant bias source.

- Keep the anticipated-shock proxy genuinely low dimensional, but let it include the specific shock directions that move ELB incidence most strongly. In the NK benchmark that usually means prioritizing the natural-rate shock and only adding the policy shock if the marginal gain is visible in the ablations.

- Use adaptive quadrature only when the filtered mean is close to the ELB boundary. Away from the kink, a single-node or very low-node rule is usually enough and keeps runtime under control.

- Move from a single shared anchor to a hybrid anchor rule: shared anchor in calm regions, per-particle anchor for the small subset of particles whose predicted policy rule is near the ELB cutoff. That targets anchor-reuse bias without paying full per-particle solve cost.

- Condition Jacobians on explicit regime paths rather than differentiating through endogenous hard decisions. The regime should be part of the state of the approximation, not an incidental side effect of the path solve.

- Increase the horizon only adaptively when the continuation value of the ELB state is material. A static large horizon is expensive; a short baseline horizon with boundary-triggered extension is more plausible computationally.

- Improve the terminal condition for ELB episodes. Even a simple regime-consistent terminal mapping is likely to outperform a blanket zero-gap terminal condition when the economy is near the constraint.

- Track and report component usage probabilities and realized regime frequencies in the filter output. If one mixture component dominates almost all the time, additional quadrature nodes are wasted and the bias is probably elsewhere.

- Keep the rate-measurement approximation fixed while improving the transition first. Otherwise the benchmark will confound transition-side fixes with a separate measurement-side approximation problem.

- Make the ablation table the acceptance test for each algorithmic change. Any proposed “improvement” should be judged by whether it reduces the specific kernel-error component it targets, not by whether it raises ESS alone.
