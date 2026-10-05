# IPW and propensity scores

## Propensity score
The propensity score e(x) is the chance that a customer with features x receives the action. In a randomized test it is known (for example 0.5). In observational data, where staff choose who to contact, it must be estimated, and it is usually far from constant.

## Inverse probability weighting
Weight treated customers by 1/e(x) and control customers by 1/(1-e(x)). Both weighted arms then look like the full population, so their difference estimates the average effect. Normalized weights (the Hajek form) are more stable. Clip e(x) to [0.05, 0.95] so a few customers cannot dominate.

## Transformed outcome learner
Y* = y (t - e) / (e (1 - e)) has expected value equal to the effect given x. Any regression model fit to Y* estimates uplift. It is unbiased but very noisy, because every churner gets a large positive or negative value.

## Overlap and confounding
IPW only removes bias from features that are measured. If staff used information not in the data, the estimate stays biased. Check overlap: if many customers have e(x) near 0 or 1, the data cannot tell what would have happened to them under the other choice.
