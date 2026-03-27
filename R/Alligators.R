# ═══════════════════════════════════════════════════════════════════════════════
# MixSIAR Reference Analysis — American Alligator Diet
# ═══════════════════════════════════════════════════════════════════════════════
#
# This script runs the alligator stable-isotope mixing model through MixSIAR
# (Stock & Semmens, PeerJ 2018) to establish a baseline result before
# reimplementing the same analysis in Julia/Turing.jl.
#
# The dataset includes:
#   - Consumer data: δ¹³C and δ¹⁵N for 181 alligators, with body length and sex
#   - Source data: Marine and Freshwater end-member isotope signatures
#   - TEF data: Trophic enrichment (discrimination) factors
#
# Original script by Brian Stock (April 21, 2017).
# Recreates Figures 6 and 8 from: https://peerj.com/articles/5096/
# ═══════════════════════════════════════════════════════════════════════════════

# ── Install MixSIAR (uncomment if needed) ────────────────────────────────────
# install.packages("devtools")
# remotes::install_github("brianstock/MixSIAR", dependencies = TRUE)
# install.packages("quantreg")

library(MixSIAR)
library(tidyr)
library(ggplot2)

# ── 1. Locate the built-in example data shipped with MixSIAR ─────────────────

mix.filename    <- system.file("extdata", "alligator_consumer.csv",            package = "MixSIAR")
source.filename <- system.file("extdata", "alligator_sources_simplemean.csv",  package = "MixSIAR")
discr.filename  <- system.file("extdata", "alligator_TEF.csv",                package = "MixSIAR")

# ── 2. Load and configure data objects ────────────────────────────────────────
# Note: covariates (cont_effects, factors) are set to NULL here for the
# simplest baseline. To add a continuous effect of body length or a random
# effect of individual, pass them as arguments — but MixSIAR cannot easily
# model interactions between categorical and continuous variables, which is
# one of the motivations for using Turing.jl instead.

mix <- load_mix_data(
  filename    = mix.filename,
  iso_names   = c("d13C", "d15N"),
  factors     = NULL,
  fac_random  = NULL,
  fac_nested  = NULL,
  cont_effects = NULL
)

source <- load_source_data(
  filename       = source.filename,
  source_factors = NULL,
  conc_dep       = FALSE,
  data_type      = "means",
  mix            = mix
)

discr <- load_discr_data(filename = discr.filename, mix = mix)

# ── 3. Define and write the JAGS model ────────────────────────────────────────
# MixSIAR auto-generates a JAGS model file. The user has limited control
# over the model structure — this "black box" aspect is one limitation we
# address in the Turing reimplementation.

model_filename <- "MixSIAR_model_cont_ind.txt"
resid_err   <- TRUE
process_err <- FALSE

write_JAGS_model(model_filename, resid_err, process_err, mix, source)

# ── 4. MCMC sampling ─────────────────────────────────────────────────────────

run <- list(
  chainLength = 100000,
  burn        = 50000,
  thin        = 50,
  chains      = 4,
  calcDIC     = TRUE
)

start_time <- Sys.time()
jags.mod <- run_model(run = run, mix, source, discr, model_filename, alpha.prior = 1)
end_time <- Sys.time()
cat("Time taken to run model:", end_time - start_time, "\n")

# ── 5. Isospace plot ──────────────────────────────────────────────────────────
# Shows consumer data in δ¹³C–δ¹⁵N space together with source polygons,
# useful as a visual sanity check.

plot_data(
  filename      = "isospace_plot",
  plot_save_pdf = TRUE,
  plot_save_png = FALSE,
  mix, source, discr
)

graphics.off()

# ═══════════════════════════════════════════════════════════════════════════════
# Post-processing — Figure 6: Diet proportion vs. body length
# ═══════════════════════════════════════════════════════════════════════════════

R2jags::attach.jags(jags.mod)
n.sources    <- source$n.sources
source_names <- source$source_names

# Specialization index (Bolnick et al. 2002 style)
# ε = 0 → generalist (equal use of sources), ε = 1 → complete specialist
calc_eps <- function(f) {
  n.sources <- length(f)
  gam <- rep(1 / n.sources, n.sources)   # generalist reference
  phi <- rep(0, n.sources)
  phi[1] <- 1                             # specialist reference
  sqrt(sum((f - gam)^2)) / sqrt(sum((phi - gam)^2))
}

# Reconstruct the continuous-effect posterior from ILR-space coefficients
ce    <- 1
label <- mix$cont_effects[ce]
cont  <- mix$CE[[ce]]
ilr.cont <- get(paste("ilr.cont", ce, sep = ""))

n.plot    <- 200
chain.len <- dim(p.global)[1]
Cont1.plot <- seq(from = round(min(cont), 1), to = round(max(cont), 1), length.out = n.plot)

# Build the ILR-space linear predictor across the length gradient
ilr.plot <- array(NA, dim = c(n.plot, n.sources - 1, chain.len))
for (src in 1:n.sources - 1) {
  for (i in 1:n.plot) {
    ilr.plot[i, src, ] <- ilr.global[, src] + ilr.cont[, src] * Cont1.plot[i]
  }
}

# ILR → proportion transform (inverse additive-log-ratio via the Helmert sub-composition)
e <- matrix(rep(0, n.sources * (n.sources - 1)), nrow = n.sources, ncol = (n.sources - 1))
for (i in 1:(n.sources - 1)) {
  e[, i] <- exp(c(rep(sqrt(1 / (i * (i + 1))), i), -sqrt(i / (i + 1)), rep(0, n.sources - i - 1)))
  e[, i] <- e[, i] / sum(e[, i])
}

cross  <- array(data = NA, dim = c(n.plot, chain.len, n.sources, n.sources - 1))
tmp    <- array(data = NA, dim = c(n.plot, chain.len, n.sources))
p.plot <- array(data = NA, dim = c(n.plot, chain.len, n.sources))

for (i in 1:n.plot) {
  for (d in 1:chain.len) {
    for (j in 1:(n.sources - 1)) {
      cross[i, d, , j] <- (e[, j]^ilr.plot[i, j, d]) / sum(e[, j]^ilr.plot[i, j, d])
    }
    for (src in 1:n.sources) {
      tmp[i, d, src] <- prod(cross[i, d, src, ])
    }
    for (src in 1:n.sources) {
      p.plot[i, d, src] <- tmp[i, d, src] / sum(tmp[i, d, ])
    }
  }
}

# Posterior quantiles for credible intervals
get_high <- function(x) { quantile(x, .95) }
get_low  <- function(x) { quantile(x, .05) }

p.low    <- apply(p.plot, c(1, 3), get_low)
p.high   <- apply(p.plot, c(1, 3), get_high)
p.median <- apply(p.plot, c(1, 3), median)
colnames(p.median) <- source_names

# Back-transform the x-axis to the original length scale
Cont1.plot <- Cont1.plot * mix$CE_scale + mix$CE_center

df <- data.frame(
  reshape2::melt(p.median)[, 2:3],
  rep(Cont1.plot, n.sources),
  reshape2::melt(p.low)[, 3],
  reshape2::melt(p.high)[, 3]
)
colnames(df) <- c("source", "median", "x", "low", "high")

# Individual-level posterior proportions
original <- combine_sources(jags.mod, mix, source, alpha.prior = 1,
  groups = list(Freshwater = "Freshwater", Marine = "Marine"))

col.ind.marine <- grep("p.ind\\[.*,2]", colnames(original$post))
p.ind.marine   <- as.data.frame(original$post[, col.ind.marine])
lengths         <- mix$CE_orig[[1]]
colnames(p.ind.marine) <- paste0("ind", 1:length(lengths))

df.ind        <- p.ind.marine %>% gather(Ind, p)
df.ind$Length <- rep(lengths, each = dim(original$post)[1])
df.ind$Ind    <- factor(df.ind$Ind)

# ── Figure 6: Marine diet proportion vs. total body length ────────────────────
cols <- RColorBrewer::brewer.pal(9, "Blues")

png("fig6_diet_length_ind.png", width = 7, height = 4.5, units = "in", res = 300)
print(ggplot(df.ind) +
  geom_ribbon(data = df[df$source == "Marine", ], mapping = aes(x = x, ymin = low, ymax = high), alpha = 0.35, fill = cols[6]) +
  geom_line(data = df[df$source == "Marine", ], mapping = aes(x = x, y = median), size = 1.5, color = cols[9]) +
  geom_linerange(mapping = aes(x = Length, y = p, group = Ind), stat = "summary", color = cols[6], size = 1,
    fun.min = function(z) { quantile(z, 0.05) },
    fun.max = function(z) { quantile(z, 0.95) }) +
  geom_point(mapping = aes(x = Length, y = p, group = Ind), stat = "summary", shape = 21, color = cols[9], fill = cols[9], size = 3,
    fun = function(z) { quantile(z, 0.5) }) +
  scale_y_continuous(limits = c(0, 1), expand = c(0.01, 0.01)) +
  ylab(expression(italic(p)[marine])) +
  xlab("Total length (cm)") +
  theme_bw() +
  theme(
    panel.border     = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.line        = element_line(colour = "black"),
    axis.title       = element_text(size = 16),
    axis.text        = element_text(size = 14),
    legend.text      = element_text(size = 14),
    legend.position  = c(.02, 1),
    legend.justification = c(0, 1),
    legend.title     = element_blank()
  ))
dev.off()

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 8 — Histogram of individual specialization index (ε)
# ═══════════════════════════════════════════════════════════════════════════════

med.p.ind <- apply(p.ind.marine, 2, median)
med.q.ind <- 1 - med.p.ind
df.fig8   <- data.frame(pmarine = med.p.ind, pfresh = med.q.ind)
df.fig8$eps <- apply(df.fig8, 1, calc_eps)

png("fig8_eps_hist.png", width = 7, height = 4.5, units = "in", res = 500)
print(ggplot(df.fig8, aes(x = eps)) +
  geom_histogram(bins = 20) +
  xlab(expression(paste("Specialization index (", epsilon[ind], ")", sep = ""))) +
  ylab("Count") +
  labs(title = "") +
  scale_y_continuous(expand = c(0, 0), limits = c(0, 125)) +
  theme_bw() +
  theme(
    panel.border     = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.line        = element_line(colour = "black"),
    axis.title       = element_text(size = 16),
    axis.text        = element_text(size = 14),
    legend.text      = element_text(size = 14),
    legend.position  = c(.02, 1),
    legend.justification = c(0, 1),
    legend.title     = element_blank()
  ))
dev.off()
