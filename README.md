# Adventure with Julia and Turing: Modelling the Effects of Parasites on Behavioral Phenotypes

A hands-on workshop on **phenotypic plasticity** and **Bayesian stable-isotope mixing models**, using Julia's [Turing.jl](https://turinglang.org/) probabilistic-programming framework.

## Motivation

Stable-isotope mixing models (SIMMs) let us estimate the proportional contribution of different food sources to a consumer's diet from tracer data (e.g. δ¹³C and δ¹⁵N). The R package [MixSIAR](https://github.com/brianstock/MixSIAR) is the current standard, but it has practical limitations:

- Interactions between categorical and continuous covariates are difficult to specify.
- The JAGS back-end is largely a "black box" — users have limited control over model structure, priors, and sampler tuning.

By re-implementing these models in **Turing.jl** we gain:

- Full transparency: every line of the model is readable Julia code.
- Flexible model specification: arbitrary covariate structures, random effects, and derived quantities.
- Modern automatic-differentiation back-ends (ForwardDiff, ReverseDiff, Mooncake) with easy benchmarking.
- Multi-threaded MCMC sampling out of the box.

## Case Study — American Alligator Diet

The dataset comes from the MixSIAR alligator example (Stock & Semmens, PeerJ 2016). It contains:

| File | Description |
|---|---|
| `data/alligator_consumer.csv` | Individual alligators with δ¹³C, δ¹⁵N, body length, sex, and size class |
| `data/alligator_sources.csv` | Mean and SD of δ¹³C and δ¹⁵N for Marine and Freshwater end-members |
| `data/alligator_TEF.csv` | Trophic enrichment (discrimination) factors for each source × tracer |

The scientific question: **How does the proportion of marine vs. freshwater prey change with body size, sex, and their interaction?**

## Workshop Structure

### Part 1 — R / MixSIAR Baseline (`R/Alligators.R`)

Run the alligator example through MixSIAR to see the standard workflow:
load data → write JAGS model → sample → post-process → plot.
This establishes a reference result and exposes the limitations we want to overcome.

### Part 2 — Julia / Turing Models (`Julia/MixingModels.jl`)

Build the same mixing model from scratch in Turing, progressing through four levels of complexity:

| Model | What it adds |
|---|---|
| `Mixing0` | Simplest two-source model — single global proportion `p_m`, fixed TEF, residual error |
| `Mixing1` | Propagates source uncertainty (σ) into the likelihood |
| `Mixing2` | Incorporates trophic discrimination factors with their uncertainty |
| `Mixing3` | Full hierarchical model — logit-linear predictor with body length, sex, their interaction, and individual random effects |

### Part 3 — Posterior Analysis

- Trace plots and convergence diagnostics
- Posterior predictive curves of marine-diet proportion vs. body size
- Individual-level diet specialization index (ε)

## Requirements

### R

```r
install.packages("MixSIAR")
# or
remotes::install_github("brianstock/MixSIAR", dependencies = TRUE)
```

Additional R packages: `tidyr`, `ggplot2`, `R2jags`, `RColorBrewer`, `reshape2`.

### Julia (≥ 1.10)

```julia
using Pkg
Pkg.add([
    "Turing", "Distributions", "MCMCChains",
    "AlgebraOfGraphics", "CairoMakie",
    "FillArrays", "LinearAlgebra",
    "CSV", "DataFrames", "StatsModels",
    "CategoricalArrays",
    "DifferentiationInterface", "ADTypes",
    "ForwardDiff", "ReverseDiff", "Mooncake"
])
```

## Quick Start

```bash
# 1. Run the R baseline (optional)
Rscript R/Alligators.R

# 2. Run the Julia workshop script
cd Julia/
julia --threads=auto MixingModels.jl
```

Or work interactively in the Julia REPL / VS Code for a step-by-step walkthrough.

## File Overview

```
PlasticityWorkshop/
├── README.md
├── data/
│   ├── alligator_consumer.csv   # Consumer isotope data (N = 181)
│   ├── alligator_sources.csv    # Source end-member statistics
│   └── alligator_TEF.csv        # Trophic enrichment factors
├── Julia/
│   ├── MixingModels.jl          # Main workshop script (models + analysis)
│   └── Functions.jl             # Helper functions (invlogit, HDI, prediction, etc.)
└── R/
    └── Alligators.R             # MixSIAR reference analysis
```

## Key Concepts

- **Stable-isotope mixing model**: A system of equations that links observed consumer isotope values to a weighted mixture of source signatures, corrected for trophic discrimination.
- **Trophic discrimination factor (TDF)**: The systematic shift in isotope ratios between a consumer and its diet, due to metabolic fractionation.
- **Logit-linear predictor**: In `Mixing3`, diet proportions are modelled on the logit scale so that covariates (length, sex) map smoothly to the (0, 1) probability space.
- **Specialization index (ε)**: A measure of how far an individual's diet departs from a perfect generalist (equal proportions); ranges from 0 (generalist) to 1 (specialist).

## References

- Stock, B.C. & Semmens, B.X. (2016). MixSIAR GUI User Manual. Version 3.1. [doi:10.5281/zenodo.1209993](https://doi.org/10.5281/zenodo.1209993)
- Stock, B.C. et al. (2018). Analyzing mixing systems using a new generation of Bayesian tracer mixing models. *PeerJ*, 6, e5096.
- Ge, H., Xu, K. & Ghahramani, Z. (2018). Turing: A Language for Flexible Probabilistic Inference. *AISTATS*.

## License

Workshop materials for educational use.
