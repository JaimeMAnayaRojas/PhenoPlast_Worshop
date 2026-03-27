# ─────────────────────────────────────────────────────────────────────────────
# Functions.jl — Helper utilities for the Bayesian mixing-model workshop
# ─────────────────────────────────────────────────────────────────────────────

"""
    invlogit(x)

Map a real-valued linear predictor to the (0, 1) probability scale.
This is the inverse of the logit (log-odds) function:  1 / (1 + exp(-x)).
"""
function invlogit(x)
    return 1 / (1 + exp(-x))
end


"""
    Post_summary(df; digits=3)

Summarise posterior draws stored column-wise in a DataFrame.

Returns a DataFrame with one row per parameter containing:
- `median` — posterior median
- `l99`, `l95`, `l68` / `u68`, `u95`, `u99` — bounds of 99.7 %, 95 %, and
  68 % highest-density intervals (HDI)
- `PP` — "level of support", i.e. the percentage of posterior mass above zero
"""
function Post_summary(df, digits = 3)
    ci68 = (mapcols(x -> HDI(x, credible_mass=0.68), df))
    ci95 = (mapcols(x -> HDI(x, credible_mass=0.95), df))
    ci99 = (mapcols(x -> HDI(x, credible_mass=0.997), df))
    PostP = Vector(((mapcols(x -> LOS(x), df)))[1,:])
    df = DataFrame(Parameters = names(df), median = round.(median.(eachcol(df)), digits=digits),
    l99 = round.(Vector(ci99[1, :]), digits =digits),
    l95 = round.(Vector(ci95[1, :]), digits =digits), 
    l68 = round.(Vector(ci68[1, :]), digits =digits),
    u68 = round.(Vector(ci68[2, :]), digits =digits),
    u95 = round.(Vector(ci95[2, :]), digits =digits),
    u99 = round.(Vector(ci99[2, :]), digits =digits),
    PP = round.(PostP, digits =digits))

    return df
end


"""
    predict_p_m(size, Male, Β)

Build a design matrix from body-length values and a sex indicator, then
project every posterior draw of the coefficient vector through the linear
predictor  α + β₁·length_c + β₂·Male + β₃·length_c×Male.

Returns a matrix (n_draws × n_sizes) of linear-predictor values (logit scale).
Apply `invlogit` to convert to diet proportions.
"""
function predict_p_m(size, Male, Β)
    Xc = zeros(length(size), 4)
    Xc[:, 1] .= 1
    Xc[:, 2] .= size .- mean(mix[!, :Length])
    Xc[:, 3] .= Male
    Xc[:, 4] .= Xc[:, 2] .* Xc[:, 3]
    
    return (Xc * Β')'
end


"""
    HDI(samples; credible_mass=0.95)

Compute the Highest Density Interval — the shortest contiguous interval that
contains `credible_mass` proportion of the posterior samples.

Adapted from: https://stackoverflow.com/questions/22284502/highest-posterior-density-region-and-central-credible-region
"""
function HDI(samples; credible_mass=0.95)
	sorted_points = sort(samples)
	ciIdxInc = Int(ceil(credible_mass * length(sorted_points)))
	nCIs = length(sorted_points) - ciIdxInc
	ciWidth = repeat([0.0],nCIs)
	for i in range(1, stop=nCIs)
		ciWidth[i] = sorted_points[i + ciIdxInc] - sorted_points[i]
	end
	HDImin = sorted_points[findfirst(isequal(minimum(ciWidth)),ciWidth)]
	HDImax = sorted_points[findfirst(isequal(minimum(ciWidth)),ciWidth)+ciIdxInc]
	return([HDImin, HDImax])
end


"""
    LOS(v; b=0)

Level of Support — percentage of posterior samples that exceed the threshold `b`.
Analogous to a one-sided Bayesian p-value: LOS ≈ 100 % means strong evidence
the parameter is positive; LOS ≈ 0 % means strong evidence it is negative.
"""
function LOS(v, b = 0)
	return 100*length(findall(v .> b)) ./length(v)
end


"""
    ϵ(x)

Individual diet specialization index (Bolnick et al. 2002 style).

For a two-source system with proportions p and (1-p):
  ε = √[(p − 0.5)² + ((1−p) − 0.5)²] / √[(1 − 0.5)² + (0 − 0.5)²]

Ranges from 0 (perfect generalist, p = 0.5) to 1 (complete specialist, p ∈ {0,1}).
Vectorised: works on scalars, vectors, or matrices of posterior draws.
"""
ϵ = function(x)
    a = (x .- 0.5).^2 + ((1 .- x) .- 0.5).^2
    b = (1 - 0.5)^2 + (0 - 0.5)^2
    s = sqrt.(a) ./ √b
    return s
end
