library(tidyverse)
library(ggplot2)
library(brms)
library(tidybayes)
library(bayesplot)

data <- read.csv("data/snails/workshop-plasticity/2026-03-19-life-history-dataset.csv", sep = ";")


# see data structure
str(data)

# Histograms for key life-history variables

# 1. egg_no_in_1_Clutch
ggplot(data, aes(x = egg_no_in_1_Clutch, fill = treatment)) +
  geom_histogram() +
  labs(title = "Histogram of egg_no_in_1_Clutch") -> hist_egg_no
ggsave("figures/hist_egg_no_in_1_Clutch.png", hist_egg_no)


data$treatment <- factor(data$treatment, levels = c("W","AC"))
levels(data$treatment) 


data$treatmentNumeric <- as.numeric(data$treatment == "AC") - 0.5
unique(data$treatmentNumeric)


data$generation <- as.factor(data$generation)
levels(data$generation)

data$generationNumeric <- as.numeric(data$generation == "I") - 0.5
unique(data$generationNumeric)


names(data)
f1 <- bf(egg_no_in_1_Clutch ~  treatment * generation + weight_at_1_Clutch_g + (1 | family.id) + (1 | replicate) + (1|tank.id), family = negbinomial())
f2 <- bf(weight_at_1_Clutch_g ~  treatment * generation + (1 | family.id) + (1 | replicate) + (1|tank.id), family = gaussian())

model1 <- brm(f1 + f2, data = data, backend = "cmdstanr")
summary(model1)


# create a new dataframe with the treatment and generation as factors
new_data <- expand.grid(treatment = levels(data$treatment), 
            generation = levels(data$generation),
            weight_at_1_Clutch_g = mean(data$weight_at_1_Clutch_g)
            )


preds <- epred_draws(model1, newdata = new_data, re_formula = NA)

levels(new_data$treatment)

post <- as_draws_df(model1)
names(post)

post %>%
  rename(b_treatmentW_generationF6 = `b_weightat1Clutchg_treatmentAC:generationF6`) %>%
  summarise(pp = sum(b_treatmentW_generationF6 > 0)/n())

post %>%
  rename(b_treatmentW_generationF6 = `b_weightat1Clutchg_treatmentAC:generationF6`) %>%
  ggplot(aes(x = b_treatmentW_generationF6)) + 
  geom_vline(xintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Density of b_treatmentW_generationF6") +
  geom_density() -> density_b_treatmentW_generationF6



names(preds)

preds %>%
  ggplot(aes(x = treatment, y = .epred, fill = generation, color = generation)) +
  stat_pointinterval(position = "dodge", .width = c(0.68, 0.95)) +
  scale_fill_manual(values = c("#111212", "#2e90dc")) +
  scale_color_manual(values = c("#111212", "#2e90dc")) +
  ylab("Egg number") +
  xlab("Treatment") + facet_wrap(. ~ .category, scales = "free_y") +
  theme_bw() +
  theme(legend.position = "top") +
  labs(title = "Predicted Egg number (1st Clutch)") -> plot_preds




## put the plots together using ggarrange
library(ggpubr)
ggarrange(plot_preds, density_b_treatmentW_generationF6, labels = c("A", "B"), ncol = 2) -> plot_preds_and_density_b_treatmentW_generationF6

ggsave("figures/plot_preds_and_density_b_treatmentW_generationF6.png", plot_preds_and_density_b_treatmentW_generationF6, width = 10, height = 5)

