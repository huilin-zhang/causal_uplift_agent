# A/B test checks

## Sample ratio mismatch (SRM)
If the test was planned as 50/50, the arm sizes should be close to 50/50. A chi-square test compares the observed counts with the planned split. A p-value below 0.001 means the split is broken: assignment, logging, or data loss is treating the arms differently. Stop and fix the pipeline before reading any effect. An effect estimate from a test with SRM cannot be trusted.

## Covariate balance
Randomization should make the arms look alike before treatment. For each pre-treatment feature compute the standardized mean difference (SMD): the difference in means divided by the pooled standard deviation. |SMD| below 0.1 is the usual bar. SMD does not depend on sample size, unlike a t-test, so it shows whether a gap is big enough to matter.

## Average treatment effect
In a randomized test the difference in churn rates between control and treated is an unbiased estimate of the average effect. Use the Neyman standard error, sqrt(p1(1-p1)/n1 + p0(1-p0)/n0). Regression adjustment with centered covariates and treatment interactions (Lin 2013) keeps the estimate consistent and usually narrows the interval.

## Power
With a 12% base churn rate and 25,000 customers per arm, the standard error of the difference is about 0.29 percentage points. The minimum detectable effect at 80% power and 5% two-sided significance is about 2.8 standard errors, roughly 0.8 pp. Smaller true effects will often not reach significance.
