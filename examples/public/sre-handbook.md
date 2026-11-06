# Rollback and error budgets (public reference)

A rollback window is the period during which a deployment can be reverted without data loss.
Error budgets are typically expressed as a percentage of failed requests over a rolling 30-day period.
A canary release sends a small share of traffic to the new version before full rollout.
Blue-green deployment keeps two identical environments and switches traffic between them.
