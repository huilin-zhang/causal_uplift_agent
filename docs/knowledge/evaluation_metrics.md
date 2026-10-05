# Evaluation metrics for uplift

## PEHE
Precision in estimation of heterogeneous effects: the root mean squared error between estimated and true effects. It needs the true effect, so it only works on simulated data. A useful reference is the PEHE of predicting the average effect for everyone; a model worse than that has learned noise.

## Qini curve and coefficient
Sort customers by predicted uplift. At each cut-off, the Qini curve counts extra retained customers: treated retained minus control retained, with the control count scaled to the treated group size. The Qini coefficient is the average gap between this curve and the straight line of random targeting. Zero means no better than random. Qini works on real A/B data because it uses observed outcomes only, but it is noisy when effects are small.

## Policy value
If we contact the top k% by a score, how many extra customers do we keep? On real data it is estimated from the A/B difference inside the selected group. On simulated data we can also sum the true effects. Comparing uplift targeting with churn-risk targeting at the same coverage is the business question.

## Bootstrap intervals
Resample with replacement and recompute the metric many times. The 2.5th and 97.5th percentiles give a 95% interval. Be clear about what was resampled: resampling only the test set with frozen models covers evaluation noise; refitting on new training data also covers training noise.
