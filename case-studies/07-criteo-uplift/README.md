# Case 07: Propensity vs uplift targeting on real randomized data (Criteo)

**Method:** randomized-trial readout of three targeting rules (random, propensity, T-learner
uplift) on the public Criteo uplift dataset.
**Question:** *Case 06's simulation says that targeting by conversion propensity inflates
platform-credited results and makes marginal returns fall below average returns. Does that hold
on real data where treatment was randomized, and how much does an uplift model add?*
**Essay:** [The targeting model already knows who would convert](https://github.com/wavde/wavde/blob/main/writing/propensity-vs-uplift-targeting-criteo.md)

> **Replication:** [case 08](../08-hillstrom-email-uplift/) repeats these checks on the randomized Hillstrom e-mail experiment. Credit inflation and falling marginal returns replicate, but the concentration of uplift in high-propensity users does not: the top 20% by propensity reach 25.9% of incremental visits there.

## TL;DR

All numbers are from [`results.md`](results.md), computed on 6,988,166 held-out users. Every
targeted set is read out against its own randomized control group.

- **Credit inflation replicates.** For visits, platform-credited visits are 4.71x the
  incremental visits when everyone is targeted. That rises to 5.90x for the top 20% by
  propensity and 8.58x for the top 5%. Targeting by uplift lowers it (3.94x at 5%). For
  conversions the effect is smaller: 2.58x for everyone and 3.37x for the top 5% by
  propensity.
- **Marginal returns fall far faster than case 06 assumed.** Under propensity targeting,
  incremental visits per 1,000 extra users from 10% to 40% reach are 7.30, against 49.75 per
  1,000 in the top 10%: a ratio of 0.15. For conversions the ratio is 0.04. Case 06's
  simulation used 0.81. A lift test run on a tightly targeted campaign badly overprices the
  next dollar.
- **High-propensity users are also high-uplift users here,** in absolute terms. The top 20% by
  visit propensity hold 72.2% of all incremental visits, and the top 20% by conversion
  propensity hold 86.2% of all incremental conversions. This is the opposite of the "would
  have converted anyway" story for absolute effects: relative lift may fall with propensity,
  but absolute uplift rises.
- **An uplift model helps only at tight budgets, and only for the common outcome.** For
  visits, the T-learner's top 5% reaches 45.7% of all incremental visits, against 28.9% for
  propensity. By 30% reach the two are within 2 points (85.3% vs 83.5%). For rare
  conversions, propensity matches or beats the T-learner at every reach (59.1% vs 53.1% at
  5%).

**Decision rule.**
- Don't use platform-credited results as incrementality, especially for tightly targeted
  campaigns.
- Price budget increases with a marginal test (case 06's two-cell split), never with the
  average lift at the current reach.
- Before replacing propensity targeting with an uplift model, test the uplift model against
  propensity on a randomized holdout at the reach you actually buy. On this data it only pays
  at tight reach on a common outcome.

## How this changes case 06

Case 06's simulation made the lift effect *fall* slightly with baseline intent. On Criteo,
absolute uplift *rises* steeply with propensity, so the same two warnings apply with more
force: credit inflation is real, and marginal returns collapse faster. Case 06's claim that a
two-cell budget split is needed to price the next dollar is strengthened. What the simulation
didn't capture is how much of the incremental effect sits in a small high-propensity
segment.

## Design

- **Data:** Criteo uplift dataset v2.1 (Diemert et al. 2018). 13,979,592 users from
  incrementality tests in which a random 15% of users were withheld from targeting. It has 12
  anonymized features and `visit` and `conversion` labels. The download is SHA-256 verified.
  License CC-BY-NC-SA 4.0; the data is not committed.
- **Split:** a seeded 50/50 split into train and test.
- **Models** (ridge logistic regression on the standardized features and their squares,
  fit by Newton's method in numpy, in [`src/criteo_uplift.py`](src/criteo_uplift.py)):
  - **Propensity:** `P(outcome | treated, x)`, fit on 2,000,000 treated training users. This
    is what a platform optimizing for outcomes learns.
  - **Uplift:** a T-learner, `P1(x) − P0(x)`, using a second model fit on the 1,048,318
    control training users.
- **Readout:** for each rule and reach, the treated-minus-control rate within the targeted
  test users. That is unbiased because treatment was randomized. Standard errors come from 20
  random folds of the test set.

## Reproduce

```bash
pip install -r requirements.txt
python case-studies/07-criteo-uplift/src/run.py   # downloads ~300 MB on first run; ~1 minute after
```

This uses seed `20261127`. Two runs produce byte-identical `results.md`. CI runs the same
pipeline with `--smoke`, on synthetic data with the real schema. Tests are in
[`tests/test_criteo_uplift.py`](../../tests/test_criteo_uplift.py).

## Limitations

1. **Intent-to-treat.** `treatment` means "eligible to be targeted". Only 3.1% of all users
   were exposed, so per-exposure effects are much larger. Platform credit here counts every
   outcome among treated users, which is a simplification of real attribution windows.
2. **One advertiser's tests, anonymized features.** The concentration of uplift in
   high-propensity users may not carry over to other advertisers or products.
3. **Simple models.** A stronger uplift learner (causal forests, X-learner, direct uplift
   objectives) could beat propensity by more, especially for conversions, where the T-learner
   differences two noisy rare-event models.
4. **Standard errors exclude model-fitting noise.** They reflect test-set sampling only. The
   models are fit once on the training half.
5. **Shares are relative to targeting everyone.** "Share of all incremental reached" compares
   against treating all users, not against a budget-optimal policy.

## References

- Diemert, E., Betlei, A., Renaudin, C. & Amini, M.-R. (2018). *A large scale benchmark for
  uplift modeling.* AdKDD.
- Künzel, S. R., Sekhon, J. S., Bickel, P. J. & Yu, B. (2019). *Metalearners for estimating
  heterogeneous treatment effects using machine learning.* PNAS.
- Radcliffe, N. J. (2007). *Using control groups to target on predicted lift.* Direct
  Marketing Analytics Journal.
