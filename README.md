# Probability of stability of the repressilator: MATLAB code for Section 3

Code accompanying **Section 3, "Probability of Stability: The Symmetric and Non-Symmetric Repressilator"**
of the research report *Integrated Structural and Probabilistic Approaches for Biological and
Epidemiological Systems* 

The question studied in this section is: when the parameters of a three-gene repressilator are
uncertain and uniformly distributed, what is the probability that its equilibrium is stable,
i.e. that the maximum real part of the eigenvalues of the Jacobian is negative? The code estimates
this probability with Monte Carlo (reference), Gauss-Legendre quadrature on the true model, and
several estimates built from a third-order generalized polynomial chaos (gPCE) surrogate.

## Which file produces which result

### Settings of `repressilator_stability_probability.m`

The two toggles at the top of the script select the case:

| `n_hill` | `run_symmetric` | Case | Report |
|---|---|---|---|
| 2 | `true` | 2D symmetric, n = 2 | Table 6, Section 3.1.2 |
| 3 | `true` | 2D symmetric, n = 3 | Table 7, Section 3.1.3 |
| 3 | `false` | 6D asymmetric, n = 3 | Table 8 and the figure of Section 3.3 |

### Helper functions for the Smolyak/multinomial script

These files are in `stability_probability/` and are called by `repressilator_smolyak_multinomial.m`
(Appendix D of the report): `precompute_MomMats.m`, `multinomial_moment.m`,
`generate_multi_indices.m`, `get_mi_recursive.m`, `patterns.m`, `compute_bk_exact.m`,
`multinomial_coeff.m`, `evaluate_legendre_1d.m` and `gauss_legendre_1d.m`.


## Requirements
- Statistics and Machine Learning Toolbox (for `normpdf`).
- The **PoCET** toolbox (Polynomial Chaos Expansion Toolbox for Matlab) by Petzke, Mesbah and
  Streif, used to build the orthogonal polynomial basis:
  [github.com/MrFelixP/PoCET](https://github.com/MrFelixP/PoCET).The helper
  `get_PSImap.m` lives in its `auxiliary/` subfolder, so the whole toolbox folder (with all
  subfolders) must be on the MATLAB path.

## Running time and memory

- The Monte Carlo reference runs evaluate the true model (a root-finding step plus an eigenvalue
  computation) for every sample. The 1,000,000-sample reference takes from about 20 s to a minute,
  and the 5,000,000-sample ground truth in the time-constrained and steady-state scripts takes from
  about 2 to 10 minutes depending on the computer. 
- The first run of `repressilator_smolyak_multinomial.m` precomputes the moment matrices (the
  message "No cache found" is printed) and saves them to `InhibitoryRingMoments.mat`. Later runs load the file and take well under a
  second. The cache file is not included in this repository; it is created automatically.


## Reference

F. Petzke, A. Mesbah and S. Streif, "PoCET: a Polynomial Chaos Expansion Toolbox for Matlab",
IFAC-PapersOnLine, 2020.
