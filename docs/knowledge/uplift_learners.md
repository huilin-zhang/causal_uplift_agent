# Uplift learners

## What uplift means
Uplift (CATE) is the change in a customer's outcome caused by the action: churn without outreach minus churn with outreach. We never see both for one customer, so it must be estimated from groups. Churn risk and uplift are different things. A customer can have high risk and zero uplift (a lost cause) or low risk and negative uplift (a sleeping dog).

## T-learner
Fit one churn model on the control arm and one on the treated arm. The uplift estimate is the difference of their predictions. It is simple, but each model sees half the data and their errors do not cancel, so the difference is noisy when the effect is small.

## X-learner
Fit the two outcome models, then impute an effect for every customer: for treated customers, observed churn minus predicted churn under control; for control customers, predicted churn under treatment minus observed churn. Regress these imputed effects on features and blend the two effect models with the propensity score. It helps when one arm is much larger than the other (Kuenzel et al. 2019).

## k-means segment baseline
Cluster customers on their features, then use the A/B difference inside each cluster as the effect for everyone in that cluster. It is easy to explain, but clusters are formed on features that describe behaviour, not on features that change the effect, so it often misses the real effect pattern.
