# ═══════════════════════════════════════════════════════════════════════════════
# Adventure with Julia and Turing
# Modelling the Effects of Parasites on Behavioral Phenotypes
# ───────────────────────────────────────────────────────────────────────────────
# Bayesian Stable-Isotope Mixing Models — from scratch in Turing.jl
#
# This script walks through four progressively complex mixing models that
# estimate the proportion of marine vs. freshwater prey in American alligator
# diets using δ¹³C and δ¹⁵N stable-isotope tracers.
#
# Inspired by the MixSIAR R package (Stock & Semmens 2016), but implemented
# in Turing.jl for full transparency, flexible covariate structures, and
# modern AD-based MCMC sampling.
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1. Package Loading ───────────────────────────────────────────────────────
# Uncomment the next two lines the first time you run this to install packages:
# using Pkg
# Pkg.add(["Turing", "MCMCChains", "AlgebraOfGraphics", "CairoMakie",
#           "FillArrays", "LinearAlgebra", "CSV", "DataFrames", "StatsModels",
#           "CategoricalArrays", "DifferentiationInterface", "ADTypes",
#           "ForwardDiff", "ReverseDiff", "Mooncake", "FlexiChains", "Enzyme"])

using Turing, Distributions
using MCMCChains, AlgebraOfGraphics, CairoMakie
using FillArrays
using LinearAlgebra
using CSV, DataFrames
using StatsModels
using CategoricalArrays

using Random
Random.seed!(0)

# AD back-ends for benchmarking sampler performance
using DifferentiationInterface
using ADTypes
using DynamicPPL.TestUtils.AD: run_ad, ADResult
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
import Mooncake: Mooncake
using FlexiChains
import Enzyme: Enzyme

# ── 2. Trace-Plot Helper ─────────────────────────────────────────────────────

"""
    tracer_plots(chain; params=nothing, plot_size=(500, 1000))

Produce faceted trace plots (one row per parameter, coloured by chain) using
AlgebraOfGraphics. If `params` is nothing every non-log-posterior column is
plotted; pass a vector of strings to select specific parameters.
"""
function tracer_plots(chain; params=nothing, plot_size = (500, 1000))
    df = DataFrame(chain)

    if isnothing(params)
        params = [name for name in names(df) if !startswith(string(name), "lp__")]
    end

    n_iter = size(df, 1)/maximum(df.chain)
    df.iteration = repeat(1:n_iter, outer = maximum(df.chain))

    longdf = stack(df, params, variable_name = :Parameter, value_name = :Value)
    longdf.chain = categorical(longdf.chain)

    plt = data(longdf) * mapping(:iteration, :Value, row = :Parameter, color = :chain) * visual(Lines, linewidth = 0.75)
    draw(plt, facet=(; linkxaxes=:minimal, linkyaxes=:minimal), 
    figure = (; size = plot_size))
end


# ── 3. Load Data ─────────────────────────────────────────────────────────────
# Consumer data: each row is an individual alligator with its δ¹³C, δ¹⁵N,
# body length (cm), sex, habitat, and size class.
mix = CSV.read("data/alligator_consumer.csv", DataFrame);

# Source end-members: mean and SD of δ¹³C and δ¹⁵N for Marine and Freshwater.
sources = CSV.read("data/alligator_sources.csv", DataFrame);

# Trophic Discrimination Factors (TDF): the metabolic offset between diet and
# consumer tissue, reported as mean ± SD for each source × tracer combination.
tdf = CSV.read("data/alligator_TEF.csv", DataFrame);

include("Functions.jl")


# ── 4. Data Inspection ───────────────────────────────────────────────────────

first(mix, 5)
names(mix)
first(sources, 5)
first(tdf, 5)






# plot the stable isotpe data 

plt1 = data(mix) * mapping(:d13C => "δ13C", :d15N => "δ15N") * visual(Scatter, alpha = 0.3, markersize = 5)


sources.minC .= sources.Meand13C .- sources.SDd13C;
sources.maxC .= sources.Meand13C .+ sources.SDd13C;

sources.minN .= sources.Meand15N .- sources.SDd15N;
sources.maxN .= sources.Meand15N .+ sources.SDd15N;

plt2 = data(sources) * (

mapping(:Meand13C, :Meand15N, color = :Source) * visual(Scatter,  markersize = 15) +

mapping(:Meand15N, :minC, :maxC, color = :Source) * visual(Rangebars; direction = :x) +

mapping(:Meand13C, :minN, :maxN, color = :Source) * visual(Rangebars; direction = :y)

);

draw(plt1  + plt2, figure = (;title= "Alligator dataset"))


# ═══════════════════════════════════════════════════════════════════════════════
# PART A — Building the Models
# ═══════════════════════════════════════════════════════════════════════════════
#
# A stable-isotope mixing model relates observed consumer isotope values (Y)
# to a weighted sum of source signatures corrected for trophic discrimination:
#
#   Y_ij = Σₛ pₛ · (μₛⱼ + λⱼ) + error
#
# where:
#   i = individual, j = tracer (δ¹³C or δ¹⁵N), s = source
#   pₛ = proportion of source s in the diet (Σ pₛ = 1)
#   μₛⱼ = mean isotope value of source s for tracer j
#   λⱼ  = trophic discrimination factor for tracer j
# ═══════════════════════════════════════════════════════════════════════════════

# ── 5. Prepare Response Matrix Y ─────────────────────────────────────────────
# Y is N × 2: each row is an individual, columns are δ¹³C and δ¹⁵N.
Y_ij = Matrix(mix[!, [:d13C, :d15N]]);

# ── 6. Design Matrix X (for the full model later) ────────────────────────────
# Covariates: centred body length, sex (Male=1), and their interaction.
# Centring length improves sampler geometry and makes the intercept interpretable
# as the expected logit-proportion at the mean body size.
# X = zeros(size(mix, 1), 3)
# X[:, 1] .= mix[!, :Length] .- mean(mix[!, :Length])
# X[:, 2] .= ifelse.(mix[!, :sex] .== "Male", 1, 0)
# X[:, 3] .= X[:, 1] .* X[:, 2]

# Alternative: build the same matrix programmatically with StatsModels:

minimum(mix.Length), maximum(mix.Length)
mix.Length_centered = mix.Length .- mean(mix.Length)

f = @formula(d13C ~ 1 + Length_centered * sex)
f = apply_schema(f, schema(f, mix))
_, X = modelcols(f, mix)
X = X[:, 2:end]   # drop the intercept (handled separately in the model)

# ── 7. Source Signature Matrices ──────────────────────────────────────────────
# μ_mat[source, tracer]: mean isotope value per source per tracer.
# Layout: rows = {Marine, Freshwater}, columns = {δ¹³C, δ¹⁵N}
μ_mat = zeros(2, 2) 
μ_mat[1, 1] = sources[1, :Meand13C]
μ_mat[2, 1] = sources[2, :Meand13C]
μ_mat[1, 2] = sources[1, :Meand15N]
μ_mat[2, 2] = sources[2, :Meand15N]

# σ_mat[source, tracer]: SD of source isotope signatures.
σ_mat = zeros(2, 2)
σ_mat[1, 1] = sources[1, :SDd13C]
σ_mat[2, 1] = sources[2, :SDd13C]
σ_mat[1, 2] = sources[1, :SDd15N]
σ_mat[2, 2] = sources[2, :SDd15N]

# tdf_mat[source, tracer]: mean trophic discrimination factors.
tdf_mat = zeros(2, 2)
tdf_mat[1, 1] = tdf[1, 3]
tdf_mat[2, 1] = tdf[2, 3]
tdf_mat[1, 2] = tdf[1, 5]
tdf_mat[2, 2] = tdf[2, 5]


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 0 — Simplest mixing model
# ─────────────────────────────────────────────────────────────────────────────
# Single global diet proportion p_m (marine), fixed TEF values,
# and a per-tracer residual variance. No source uncertainty.

@model function Mixing0(Y, μᵢⱼ)
    N, k = size(Y)

    # Fixed trophic discrimination factors (from literature)
    λᵢⱼ = [0.61, 1.22]    # mean TDF for δ¹³C and δ¹⁵N
    τ = [0.12, 0.08]       # SD of TDF (unused here, propagated in later models)

    # Prior: uniform on diet proportion (marine)
    p_m ~ Uniform(0, 1)
    p_f = 1 - p_m           # freshwater proportion (closure constraint)

    # Per-tracer residual variance
    σ² ~ filldist(Gamma(1, 2), k)

    # Likelihood: each tracer column is normally distributed around the
    # proportion-weighted source means, shifted by the discrimination factor.
    for j in 1:k
        Y[:, j] ~ Normal.(p_m * (μᵢⱼ[1, j] + λᵢⱼ[j]) + p_f * (μᵢⱼ[2, j] + λᵢⱼ[j]), sqrt(σ²[j]))
    end
end;

# Quick test run (short chains for checking the model compiles and samples)
@time chain0 = sample(Mixing0(Y_ij, μ_mat), NUTS(0.65), MCMCThreads(), 100, 4);

# Production run with longer warm-up and more draws
@time chain0 = sample(Mixing0(Y_ij, μ_mat), NUTS(50_000, 0.65), MCMCThreads(), 50_000, 4);
describe(chain0)
tracer_plots(chain0; params=["p_m", "σ²[1]", "σ²[2]"], plot_size = (500, 500))


# using FlexiChains

@time chain0 = sample(Mixing0(Y_ij, μ_mat), NUTS(50_000, 0.65), MCMCThreads(), 50_000, 4; chain_type=VNChain);

plot(chain0)
ss = summarystats(chain0)         # mean, std, mcse, ess, rhat for all variables
# ss[@varname(x), stat=At(:mean)]  # -> Float64 (the mean of x)

mean(chain0)              # just the mean for all variables
mean(chain0)[@varname(p_m)] # -> Float64



# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — Propagating source uncertainty
# ─────────────────────────────────────────────────────────────────────────────
# Same as Model 0 but the variance of each source's isotope signature is
# propagated through the mixing equation, so the likelihood width reflects
# both residual and source-level uncertainty.

@model function Mixing1(Y, μ_mat, σ_mat)
    p_m ~ Uniform(0, 1)
    p_f = 1 - p_m

    # Expected consumer values (proportion-weighted source means)
    μC = p_m * μ_mat[1, 1] + p_f * μ_mat[2, 1] 
    μN = p_m * μ_mat[1, 2] + p_f * μ_mat[2, 2]

    # Variance propagation: Var(Y_j) ≈ Σₛ pₛ² · σₛⱼ²
    σC = p_m^2 * σ_mat[1, 1]^2 + p_f^2 * σ_mat[2, 1]^2
    σN = p_m^2 * σ_mat[1, 2]^2 + p_f^2 * σ_mat[2, 2]^2

    Y[:,1] .~ Normal.(μC, σC)
    Y[:,2] .~ Normal.(μN, σN)
end

chain1 = sample(Mixing1(Y_ij, μ_mat, σ_mat), NUTS(1000, 0.65), 1000)
describe(chain1)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — Adding trophic discrimination factors with uncertainty
# ─────────────────────────────────────────────────────────────────────────────
# Extends Model 1 by shifting each source mean by its trophic discrimination
# factor (TDF) and including TDF variance in the error propagation.

@model function Mixing2(Y, μ_mat, σ_mat, tdf)
    p_m ~ Uniform(0, 1)
    p_f = 1 - p_m

    # Source mean + TDF correction
    μC = p_m * (μ_mat[1, 1] + tdf[1,3]) + p_f * (μ_mat[2, 1] + tdf[2,3]) 
    μN = p_m * (μ_mat[1, 2] + tdf[1,5]) + p_f * (μ_mat[2, 2] + tdf[2,5])

    # Combined source + TDF variance
    σC = p_m^2 * (σ_mat[1, 1]^2 + tdf[1,4]) + p_f^2 * (σ_mat[2, 1]^2 + tdf[2,4])
    σN = p_m^2 * (σ_mat[1, 2]^2 + tdf[1,5]) + p_f^2 * (σ_mat[2, 2]^2 + tdf[2,5])

    Y[:,1] .~ Normal.(μC, σC)
    Y[:,2] .~ Normal.(μN, σN)

    return p_m, p_f
end

chain2 = sample(Mixing2(Y_ij, μ_mat, σ_mat, tdf), NUTS(1000, 0.65), 1000)
describe(chain2)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3 — Full hierarchical model with covariates
# ─────────────────────────────────────────────────────────────────────────────
# This is the main model of the workshop. Diet proportions vary across
# individuals via a logit-linear predictor:
#
#   logit(p_mᵢ) = α + β₁·length_cᵢ + β₂·Maleᵢ + β₃·length_c×Male + bᵢ
#
# where bᵢ ~ Normal(0, σ²_id) is an individual random effect that captures
# unexplained among-individual variation (i.e., individual "personality" in
# diet choice — a form of phenotypic plasticity).
#
# This covariate structure — including interactions between a continuous
# variable (length) and a categorical variable (sex) plus individual random
# effects — is exactly the kind of model that is difficult to set up in
# MixSIAR's JAGS back-end, motivating the move to Turing.jl.

tdf = Matrix(tdf[!, 3:end])

@model function Mixing3(Y, X, μ_mat, σ_mat, tdf)
    N, M = size(X)
    
    # Residual scaling parameters (one per tracer)
    ϵ_c ~ truncated(Cauchy(0, 10.0); lower=0)
    ϵ_n ~ truncated(Cauchy(0, 1.5); lower=0)

    # SD of the individual random effect
    σ²_id ~ truncated(Cauchy(0, 20); lower=0)
    
    # Fixed effects: intercept and slopes on the logit scale
    α ~ Normal(0, 3.0)
    β ~ MvNormal(zeros(M), 2.0 .* I)

    # Per-individual diet proportion via the logit link
    p_m = invlogit.(α .+ X * β )
    p_f = 1 .- p_m

    # Expected isotope values (source means + TDF, weighted by proportions)
    @views μC = p_m .* (μ_mat[1, 1] + tdf[1,2]) .+ p_f .* (μ_mat[2, 1] + tdf[2,2]) 
    @views μN = p_m .* (μ_mat[1, 2] + tdf[1,4]) .+ p_f .* (μ_mat[2, 2] + tdf[2,4])

    # Combined variance from source uncertainty and TDF uncertainty
    @views σC = (p_m.^2 .* sqrt(σ_mat[1, 1].^2 + tdf[1, 3].^2)) .+ (p_f.^2 .* sqrt(σ_mat[2, 1].^2 + tdf[2, 3].^2))
    @views σN = (p_m.^2 .* sqrt(σ_mat[1, 2].^2 + tdf[1, 4].^2)) .+ (p_f.^2 .* sqrt(σ_mat[2, 2].^2 + tdf[2, 4].^2))

    # Likelihood with additional residual scaling
    Y[:,1] ~ MvNormal(μC, σC .* ϵ_c)
    Y[:,2] ~ MvNormal(μN, σN .* ϵ_n)
end;


# ── 8. AD Backend Benchmarking ────────────────────────────────────────────────
# Turing supports multiple automatic-differentiation backends. We benchmark
# ForwardDiff and ReverseDiff to pick the fastest for this model.

Threads.nthreads()


result1 = run_ad(Mixing3(Y_ij, X, μ_mat, σ_mat, tdf), AutoForwardDiff(); benchmark=true);
result2 = run_ad(Mixing3(Y_ij, X, μ_mat, σ_mat, tdf), AutoReverseDiff(); benchmark=true);
result3 = run_ad(Mixing3(Y_ij, X, μ_mat, σ_mat, tdf), AutoMooncake(); benchmark=true);
# result4 = run_ad(Mixing3(Y_ij, X, μ_mat, σ_mat, tdf), AutoEnzyme(); benchmark=true);



# Gradient-to-primal time ratio (lower = more efficient)
result1.grad_time/result1.primal_time
result2.grad_time/result2.primal_time   # typically faster for models with many parameters
result3.grad_time/result3.primal_time

fastAD =  AutoMooncake();


# ── 9. Sampling Model 3 ──────────────────────────────────────────────────────

f = @formula(d13C ~ 1 + Length_centered * sex)
f = apply_schema(f, schema(f, mix))
_, X = modelcols(f, mix)
X = X[:, 2:end]   # drop the intercept (handled separately in the model)



model3 = Mixing3(Y_ij, X, μ_mat, σ_mat, tdf);

# Quick diagnostic run
@time chain3 = sample(model3, NUTS(50, 0.65; adtype=fastAD), MCMCThreads(), 100, 4; chain_type=VNChain);

# Full production run (4 chains in parallel)
@time chain3 = sample(model3, NUTS(1000, 0.8; adtype=fastAD), MCMCThreads(), 1000, 4; chain_type=VNChain);

ss3 = summarystats(chain3);

plot(chain3)

unique(mix.ID)

# ═══════════════════════════════════════════════════════════════════════════════
# PART B — Posterior Analysis
# ═══════════════════════════════════════════════════════════════════════════════

# ── 10. Posterior Predictions: Marine Diet Proportion vs. Body Size ──────────

α = DataFrame(chain3[@varname(α)]); α = α.value;
β = DataFrame(chain3[@varname(β)]); β= Matrix(hcat(β.value...)');

# Prediction grid: body lengths spanning the observed range
z = collect(minimum(mix.Length):1:maximum(mix.Length))
z_c = z .- mean(mix.Length)


mix.sex = categorical(mix.sex);
levels(mix.sex)
# Make prediction dataset
pred_Males = DataFrame(Length = z, Length_centered = z_c, sex = "Male")
pred_Females = DataFrame(Length = z, Length_centered = z_c, sex = "Female")
pred_data = vcat(pred_Males, pred_Females)
pred_data.sex = categorical(pred_data.sex);
levels(pred_data.sex)



fxp = @formula(Length ~ 1 + Length_centered * sex)
fxp = apply_schema(fxp, schema(fxp, pred_data))
_, Xp = modelcols(fxp, pred_data)
Xp = Xp[:, 2:end]   # drop the intercept (handled separately in 

# Predict p_m on the logit scale for Males (Male = 1), then summarise

pp = invlogit.(α .+ β * Xp');

m = mean(pp, dims = 1);

p_zp_ci = map(x -> (HDI(x, credible_mass=0.95)), eachcol(pp))
p_zp_ci = Matrix(hcat(p_zp_ci...)')


pred_data.p_m .= m';
pred_data.l95 .= p_zp_ci[:, 1];
pred_data.u95 .= p_zp_ci[:, 2];

# Plot: marine diet proportion (%) vs. body size with 95 % credible band
plot2 = data(pred_data) * (mapping(:Length, :p_m, color = :sex) * visual(Lines, linewidth=2) + 
mapping(:Length, :l95, :u95, color = :sex) * visual(Band, alpha=0.3))
draw(plot2, axis = (; xlabel = "Length (cm)", ylabel = "Marine diet proportion", title = "Marine diet proportion vs. body size"))

