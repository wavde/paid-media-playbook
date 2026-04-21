# Case 03 — Conversion lift / incrementality

**Method:** Matched-market (ghost-ad / PSA) conversion-lift test with two-
proportion z-test readout and a Bayesian bootstrap CI on relative lift.
**Question:** *Among users we intended to serve a paid-media ad to, what is the
incremental conversion rate over an unexposed holdout?*

## TL;DR

At a trial-start base rate of **0.5%** and a target relative lift of **+10%**,
the required per-arm sample is ≈ **325k users** at α=0.05 / 80% power.  At that
size a +10% truth registers at p ≲ 0.05 with a 95% CI of roughly ±5 pp.  Drop
the true lift to +5% (half of target) and the test fails to reject at the same
n: **design power is the binding constraint**, not estimator choice.

## Business framing

Every six months a marketing team proposes a conversion-lift test for a paid
channel — usually display or video — to answer *"is this line item actually
incremental?"*.  The test is reasonable and the answer is valuable.  What
usually kills it is sample size.  Trial-start rates for paid-media audiences
sit between 0.3% and 1.0%, which means even a +10% lift needs 100k+ per arm.

This case is built so the *first* output of the script is the power analysis.
If the MDE under your planned budget is larger than the lift you plausibly
expect, the right answer is to kill the test, not run it underpowered and
misread the null as evidence of no effect.  At **p=0.005** and a **+10% relative
lift** target, the required per-arm sample is ≈ 325k — which is a much higher
bar than most teams quote casually.

## Method

### Design: ghost-ad / PSA holdout

At ad-serving time, eligible users are randomised into **treated** (served the
real ad) and **control** (served a ghost ad or PSA of equivalent opportunity
cost).  The conversion window runs 7–28 days.  The critical design invariants:

1. Randomisation happens at ad-server eligibility — not at impression delivery
   — so selection bias from who-actually-sees-the-ad doesn't leak in.
2. Both arms are subject to the same downstream funnel and the same other
   media.  The estimand is the marginal effect of this one placement.
3. The holdout is not small: a 1% PSA holdout on a 20M eligibility pool gives
   200k control users, which is the floor for a 0.5%-base, +10%-lift test.

### Analysis

- Conversion rates $\hat p_c, \hat p_t$.
- Two-proportion pooled z-test for the null $p_c = p_t$.
- **Parametric-bootstrap CI on relative lift** $\hat p_t / \hat p_c - 1$:
  draw $p_c \sim \text{Beta}(1+x_c, 1+n_c-x_c)$ and $p_t \sim \text{Beta}(1+x_t, 1+n_t-x_t)$,
  compute relative lift on each draw, take the quantiles.  Avoids delta-method
  breakdown at low base rates where the denominator's variance dominates.

### Sample-size formula

For base rate $p_1$, target lift $r$, $p_2 = p_1(1+r)$:

$$n \;=\; \frac{\bigl(z_{\alpha/2}\sqrt{2\bar p\bar q} + z_\beta\sqrt{p_1 q_1 + p_2 q_2}\bigr)^2}{(p_2 - p_1)^2},\quad \bar p = \tfrac{p_1+p_2}{2}.$$

## Reproduce

```bash
cd case-studies/03-incrementality
python src/run.py
```

Expected output (seed 0):

```
Planning
--------
Base rate:        0.500%
Target rel lift:  +10%
Required n/arm:   327,922 (alpha=0.05, power=0.80)

Readout
-------
Lift(rel=+8.25% [+1.32%, +15.70%], abs=+0.0004, p=0.0195, n=327,922/arm)

Sensitivity: true lift = half of target
Lift(rel=+12.03% [+4.74%, +19.88%], abs=+0.0006, p=0.0010, n=327,922/arm)
```

Note the sensitivity row above: the "half lift" draw happened to land higher
than the headline draw on this seed — a useful reminder that a single
conversion-lift readout is a single draw from a wide sampling distribution.
Always report the CI alongside the point estimate, and never treat a single
test as a settled answer on a channel's incrementality.

## When conversion lift is the right tool

| Scenario | Good fit? |
|---|---|
| Clickable display / video where ad-server can randomise exposure | ✅ |
| Paid-search (already lower-funnel, LTA is closer to correct) | ⚠️ Use incrementality bidding instead |
| Non-clickable brand media (OOH, podcast, linear TV) | ❌ Use geo lift (case 01) |
| Always-on media with no feasible holdout | ❌ Use MMM (case 04) |
| Base rate < 0.1% and budget can't fund 500k+ per arm | ❌ Won't detect anything realistic |

## Limitations

1. **Single test = single point estimate.**  A paid-media team makes many
   placement calls a year.  If each is tested in isolation at α=0.05, the
   false-discovery rate over a year is substantial.  Use sequential monitoring
   (mSPRT) if you plan to peek, and bonferroni or BH if you run many tests in
   parallel.
2. **Intention-to-treat vs treatment-on-treated.**  The z-test here is an ITT
   effect (eligibility).  If you want effect on the exposed (CACE), you need an
   IV-style adjustment with exposure as the endogenous treatment.
3. **Cookie loss / ATT.**  On iOS post-ATT, ghost-ad designs have to be run at
   the SKAdNetwork-conversion-value layer, which breaks the nice randomisation
   granularity.  Document the loss explicitly in the readout.
4. **Cannibalisation.**  If the control users simply see more of another ad
   (organic, brand-search), the measured lift is the delta **net** of substitution.
   That's often what finance wants — but memo it explicitly.

## References

- Johnson, Lewis, Nubbemeyer (2017). *Ghost ads: Improving the economics of
  measuring online ad effectiveness.*
- Gordon, Zettelmeyer, Bhargava, Chapsky (2019). *A comparison of approaches to
  advertising measurement: Evidence from big field experiments at Facebook.*
- Eckles, Karrer, Ugander (2017). *Design and analysis of experiments in
  networks: Reducing bias from interference.*
