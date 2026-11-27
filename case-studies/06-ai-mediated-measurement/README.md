# Case 06: Measurement when targeting, creative and pacing are model-driven

**Methods:** ghost-ads holdouts, a two-cell budget-split lift test, audience-randomized
creative tests, and MMM calibrated with a lift test.
**Question:** *When the ad platform's algorithm picks who sees the ads, which creative they
see and how fast budget is spent (Advantage+ / Performance Max style), what can platform
reports, lift tests and MMM each tell you, and which decision does each support?*

## TL;DR

On synthetic data where the truth is known by construction (all numbers from
[`results.md`](results.md)):

- **Platform-credited conversions are not incrementality, and the gap grows as targeting
  sharpens.** Credited conversions are 2.50x the true incremental conversions under random
  targeting and 3.43x when the platform's score is 0.95-correlated with baseline intent. A
  ghost-ads holdout stays unbiased at every sharpness (bias within ±0.7%, coverage 92–96%).
- **A lift test prices the budget it was run at, not the next dollar.** At targeting
  sharpness 0.8, the average iROAS at 10% reach is 1.10 but the marginal iROAS from 10% to
  40% is 0.90. A lift test at 10% reach says "scale up" in 94% of replications, and
  platform-credited numbers say so in 100%. A two-cell budget-split test says so in 0%, and
  its marginal estimate (0.90 vs true 0.90) has 98% CI coverage. Under random targeting,
  average and marginal match (marginal / average = 1.00).
- **An in-platform creative A/B test compares audiences, not creatives.** With identical
  per-person creative effects, served CVR picks creative A in 100% of replications because
  the platform delivers A to high-baseline users. Per-arm holdouts correctly show that
  campaign B *as delivered* is more incremental (−0.0233 against a true −0.0232). Only a
  test that fixes the audience first and then randomizes creative answers the creative
  question: it declares no winner in 93%.
- **MMM is biased when the platform paces spend with demand, and a lift-test prior only
  halves the damage.** True ROI is 0.71. At a spend–demand correlation of 0.6, the MMM
  reports 1.40 (+98%) and says ROI > 1 in 100% of replications. Using the lift test as a
  precision-weighted prior still gives 1.05 (+48%, ROI > 1 in 63%). Replacing the
  coefficient with the lift-test estimate gives 0.71 with 95% coverage. A simple MMM-vs-lift
  conflict test flags 94% of those panels.

**Decision rule.**
- Use platform-reported numbers for within-policy pacing only.
- Budget changes need a holdout at *both* budget levels: a two-cell split.
- Creative decisions need audience-randomized tests.
- When spend is algorithm-driven, an MMM coefficient that conflicts with a lift test should
  be replaced by the lift result, not averaged with it.

## Business framing

Automated campaign types now choose the audience, the creative mix and the bid in real
time. This is good for performance and bad for measurement, because every quantity a team
looks at is now an output of the same optimizer it is trying to evaluate:

1. The platform's conversion count is driven by *who* the optimizer finds, and it finds
   people who would have converted anyway.
2. The platform's creative "A/B test" lets the optimizer route each creative to the audience
   where it looks best. Braun & Schwartz (2023) call this *divergent delivery*.
3. The optimizer raises spend when it sees high intent, so spend and latent demand move
   together and the MMM can't separate them.

## Design

All three mechanisms live in [`src/ai_measurement.py`](src/ai_measurement.py). Each user
has a baseline conversion probability `p0` and a treated probability `p1`, so true
incremental conversions are known exactly.

| Part | Simulator | Truth | Estimators compared |
|---|---|---|---|
| Targeting | Baseline intent `z`. The ad raises log-odds by `max(0, 0.7 − 0.2z + noise)`. The platform score has correlation *sharpness* with `z` | `sum(p1 − p0)` over served users | Platform credit; 50% ghost-ads holdout |
| Budget transport | Same users; reach 10% vs 40% | Exact average and marginal uplift | Platform average and marginal; lift at each reach; two-cell budget split |
| Creative | Two segments. Segment 1 has +1 baseline logit and a lift of 0.15; segment 0 has a lift of 0.60. The creative has no per-person effect | Creative effect 0; exact per-arm uplift | Served CVR; per-arm ghost-ads lift; audience-fixed creative randomization |
| MMM | 104 weeks; AR(1) latent demand; log spend correlated with demand; media transforms known exactly | `beta`; true ROI ≈ 0.71 | OLS MMM (Newey-West SE); 8-week 20% user holdout; lift-as-prior; lift-only |

The MMM is given its best case: the true adstock and saturation transforms. The only thing
it can't see is demand.

## Reproduce

```bash
pip install -r requirements.txt
python case-studies/06-ai-mediated-measurement/src/run.py
```

This uses seed `20261127`, takes about 3 minutes, and writes [`results.md`](results.md).
Two runs produce byte-identical output. Known-answer tests are in
[`tests/test_ai_measurement.py`](../../tests/test_ai_measurement.py).

## When each tool is the right one

| Decision | Platform report | Single lift test | Two-cell budget split | Audience-randomized creative test | MMM |
|---|---|---|---|---|---|
| Is this campaign, as it runs today, incremental? | ❌ (2.50–3.43x inflated) | ✅ | ✅ | — | ⚠️ only if spend isn't demand-paced |
| Should we raise or cut budget? | ❌ | ❌ (prices the average, not the margin) | ✅ | — | ⚠️ calibrate the coefficient to a lift test |
| Which creative is better? | ❌ (picks the audience) | ⚠️ answers "which campaign as delivered" | — | ✅ | ❌ |
| Within-policy pacing and day-to-day tuning | ✅ | — | — | — | — |

## Limitations

1. **Synthetic, stylized data.** Each mechanism is isolated and switched on one at a time.
   Real campaigns combine all three, and the platform's model drifts during a test.
2. **Effect sizes are chosen, not estimated.** The break-even cost in the budget-transport
   test is placed halfway between true average and marginal uplift so the decision flips.
   With a different cost, the average-vs-marginal gap may or may not change the call. The
   size of the gap (marginal / average 0.81–0.85 for sharpness ≥ 0.5) is the transportable
   part.
3. **Ghost-ads holdouts are assumed feasible and clean.** The platform must log who it
   *would* have served, and there is no interference between holdout and treated users.
   Many platforms only offer a coarser holdout.
4. **The two-cell split needs scale.** The decision result uses 2,000,000 users per test.
   Smaller tests will separate average from marginal less reliably.
5. **MMM is OLS with known transforms.** A Bayesian MMM with learned transforms would add
   its own error. Calibration here acts on a single coefficient, and the lift window is
   assumed to measure the same coefficient the MMM estimates (no time-varying effect).
6. **Point decisions.** Wrong-decision rates use point estimates against break-even.
   Teams that require the CI to clear break-even will make fewer wrong calls and more
   "no decision" calls.

## References

- Braun, M. & Schwartz, E. M. (2023). *Where A/B testing goes wrong: How divergent delivery
  affects what online experiments cannot (and can) tell you about how customers respond to
  advertising.* Journal of Marketing.
- Johnson, G. A., Lewis, R. A. & Nubbemeyer, E. I. (2017). *Ghost ads: Improving the
  economics of measuring online ad effectiveness.* Journal of Marketing Research.
- Gordon, B. R., Zettelmeyer, F., Bhargava, N. & Chapsky, D. (2019). *A comparison of
  approaches to advertising measurement: Evidence from big field experiments at Facebook.*
  Marketing Science.
- Jin, Y., Wang, Y., Sun, Y., Chan, D. & Koehler, J. (2017). *Bayesian methods for media mix
  modeling with carryover and shape effects.* Google.
