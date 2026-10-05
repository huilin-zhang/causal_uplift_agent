# Causal forest

## Idea
A causal forest grows trees that split where the treatment effect differs, not where the outcome differs (Athey, Tibshirani and Wager 2019). The econml CausalForestDML version first removes the part of outcome and treatment that features predict (double machine learning), then fits the forest on the residuals.

## Honest splitting
Each tree uses one half of its sample to choose splits and the other half to estimate leaf effects. This keeps the leaf estimates free of the split search and allows valid confidence intervals.

## When to use
Good default for randomized tests of moderate size with weak, smooth effect heterogeneity. It is slower than meta-learners; on very large data fit on a subsample or use a T-learner or IPW learner.
