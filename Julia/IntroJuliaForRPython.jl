# Intro to Julia for R/Python users
# =================================
#
# Goal:
# 1) Read tabular data from CSV files
# 2) Work with DataFrames (select, filter, mutate, group summaries, joins)
# 3) Build plots with AlgebraOfGraphics + CairoMakie
#
# Run from anywhere:
#   julia Julia/IntroJuliaForRPython.jl
#
# If needed, install packages once:
# using Pkg
# Pkg.add(["CSV", "DataFrames", "Statistics", "AlgebraOfGraphics", "CairoMakie"])

using CSV
using DataFrames
using Statistics
using AlgebraOfGraphics
using CairoMakie

println("=== Julia intro script (R/Python friendly) ===")

# Use paths relative to this script file, not the current terminal location.
project_root = normpath(joinpath(@__DIR__, ".."))
data_dir = joinpath(project_root, "data")

consumer_path = joinpath(data_dir, "alligator_consumer.csv")
sources_path = joinpath(data_dir, "alligator_sources.csv")
tdf_path = joinpath(data_dir, "alligator_TEF.csv")

# -----------------------------------------------------------------------------
# 1) Reading data
# -----------------------------------------------------------------------------
println("\n1) Reading CSV files...")
mix = CSV.read(consumer_path, DataFrame)
sources = CSV.read(sources_path, DataFrame)
tdf = CSV.read(tdf_path, DataFrame)

println("Rows in mix: ", nrow(mix), " | Columns: ", ncol(mix))
println("Rows in sources: ", nrow(sources), " | Columns: ", ncol(sources))
println("Rows in tdf: ", nrow(tdf), " | Columns: ", ncol(tdf))

println("\nFirst 5 rows of consumer data:")
show(first(mix, 5), allcols=true)
println()

# -----------------------------------------------------------------------------
# 2) DataFrames basics
# -----------------------------------------------------------------------------
#
# R analogies:
# - dplyr::select(df, col1, col2)     ~ select(df, [:col1, :col2])
# - dplyr::filter(df, cond)           ~ filter(:col => ByRow(...), df)
# - dplyr::mutate(df, new = ...)      ~ transform(df, ..., :new => ...)
# - group_by(...) |> summarize(...)   ~ combine(groupby(df, ...), ...)
#
# Python (pandas) analogies:
# - df[["col1", "col2"]]              ~ select(df, [:col1, :col2])
# - df[df["Length"] .> 250]           ~ filter(:Length => ByRow(>(250)), df)
# - df.assign(new = ...)              ~ transform(df, ..., :new => ...)
# - df.groupby("sex").agg(...)        ~ combine(groupby(df, :sex), ...)

println("\n2) DataFrames operations...")

# Select columns
small = select(mix, [:ID, :sex, :Length, :d13C, :d15N, :habitat])
println("Selected columns: ", names(small))

# Filter rows: keep larger alligators
large_gators = filter(row -> row.Length >= 250, small)
println("Alligators with Length >= 250: ", nrow(large_gators))

# Mutate/transform: centered length and a simple isotope index
mix2 = transform(
    mix,
    :Length => (x -> x .- mean(x)) => :Length_centered,
    [:d13C, :d15N] => ((c13, n15) -> n15 .- c13) => :isotope_gap,
)

# Group + summarize
summary_by_sex = combine(
    groupby(mix2, :sex),
    nrow => :n,
    :Length => mean => :mean_length,
    :d13C => mean => :mean_d13C,
    :d15N => mean => :mean_d15N,
    :isotope_gap => mean => :mean_gap,
)

println("\nSummary by sex:")
show(summary_by_sex, allcols=true)
println()

# Join example: attach source means with TDF values
# (This is just to demonstrate leftjoin; it is not a model fit step.)
source_lookup = leftjoin(
    sources,
    rename(tdf, :source => :Source),
    on = :Source,
    makeunique = true,
)

println("\nJoined source table (sources + tdf):")
show(source_lookup, allcols=true)
println()

# -----------------------------------------------------------------------------
# 3) Plotting with AlgebraOfGraphics
# -----------------------------------------------------------------------------
println("\n3) Plotting with AlgebraOfGraphics...")

# Make output folder for figures.
fig_dir = joinpath(project_root, "figures")
mkpath(fig_dir)

# Plot A: scatter d13C vs d15N by sex
plt_scatter =
    data(mix2) *
    mapping(:d13C, :d15N, color=:sex) *
    visual(Scatter, markersize=10, alpha=0.7)

fig1 = draw(
    plt_scatter;
    figure=(; size=(800, 520)),
    axis=(; xlabel="d13C", ylabel="d15N", title="Alligator isotopes by sex"),
)
save(joinpath(fig_dir, "intro_scatter_d13C_d15N.png"), fig1)

# Plot B: d15N over length, colored by sex
plt_trend =
    data(mix2) *
    mapping(:Length, :d15N, color=:sex) *
    visual(Scatter, alpha=0.45, markersize=8)

fig2 = draw(
    plt_trend;
    figure=(; size=(800, 520)),
    axis=(; xlabel="Length (cm)", ylabel="d15N", title="d15N vs Length"),
)
save(joinpath(fig_dir, "intro_trend_d15N_length.png"), fig2)

# Plot C: quick bar chart from summary table
plt_bar =
    data(summary_by_sex) *
    mapping(:sex, :mean_d15N, color=:sex) *
    visual(BarPlot)

fig3 = draw(
    plt_bar;
    figure=(; size=(700, 500)),
    axis=(; xlabel="Sex", ylabel="Mean d15N", title="Mean d15N by sex"),
)
save(joinpath(fig_dir, "intro_bar_mean_d15N_by_sex.png"), fig3)

println("Saved figures in: ", fig_dir)
println("- intro_scatter_d13C_d15N.png")
println("- intro_trend_d15N_length.png")
println("- intro_bar_mean_d15N_by_sex.png")

println("\nDone. This script covered:")
println("  * Reading CSV data")
println("  * Core DataFrames workflows")
println("  * Plotting with AlgebraOfGraphics")
