# Choosing an estimator by data type

## Randomized and balanced
All learners are valid and the propensity is known. Use the causal forest as the default and compare it with X- and T-learners on held-out data.

## Randomized but unbalanced arms
When one arm holds more than 70% of customers, the X-learner uses the large arm to improve the model of the small one. Use the known assignment share as the propensity.

## Observational data
Treatment depends on features, so a plain difference or T-learner is biased. Estimate the propensity, check overlap, and use IPW or doubly robust methods such as the causal forest with a propensity model. State the no-unmeasured-confounding assumption.

## Rare outcomes
When the outcome rate is below 5%, effect estimates are noisy. Prefer pooled learners and report wide intervals rather than over-reading small differences.

## Very large data
Above about 200,000 rows, fit fast learners (T-learner, IPW) or subsample for the forest.
