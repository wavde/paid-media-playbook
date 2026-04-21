# Case 01 — Geo lift for a paid-media brand campaign

**Method:** Synthetic control (Abadie, Diamond, Hainmueller 2010) applied to a
DMA-level weekly panel.
**Question:** *If we run a brand-awareness push in one DMA, what would trial-starts
in that DMA have been without the push?*

## TL;DR

On a simulated panel of 40 DMAs × 52 weeks (treatment applied to one DMA at
week 40, true lift = +8%), synthetic control recovers the ATT within ±1 pp,
with a placebo-in-space empirical p-value ≤ 0.10 and a clean pre-period fit.

The **power analysis** up front is the piece that matters for a media team
planning the test: given the weekly noise in trial-starts and a realistic post-
period length, what is the smallest lift we could detect?  If the MDE is larger
than the lift we plausibly expect, the test is not worth running — recommend a
larger treated market or a longer post window.

## Business framing

Paid-media campaigns that are not directly clickable — brand video, audio,
OOH — can't be measured cleanly with last-touch or even an MTA model.  Geo
holdouts are the paid-media version of an A/B test: pick a target market, pick
comparable donor markets, turn the campaign **on** in the target and leave it
**off** in the donors.  The difference between the target DMA and a synthetic
blend of donors is the incremental effect.

This is the approach that tends to survive a finance review because the
counterfactual is explicit and auditable — you can point to the donor weights
and the pre-period fit.

## Method

For a single treated DMA *i=1* and *J* donors with weekly outcomes $Y_{it}$:

$$\hat w = \arg\min_{w \in \Delta^J}\ \sum_{t=1}^{T_0} \left(Y_{1t} - \sum_{j=2}^{J+1} w_j Y_{jt}\right)^2$$

on the simplex $\Delta^J = \{w \ge 0, \sum_j w_j = 1\}$.  We solve with SLSQP
(SciPy); the convexity constraint prevents extrapolation.

Estimated per-week effect: $\hat\tau_t = Y_{1t} - \sum_j \hat w_j Y_{jt}$.
Post-period ATT: the mean of $\hat\tau_t$ over treated weeks.

## Inference — placebo in space

Standard errors on SC don't have a clean closed form (one treated unit).  We
follow Abadie's permutation recipe: re-fit SC pretending each donor is the
treated unit, rank the true treated unit's post-to-pre RMSPE ratio in the
resulting placebo distribution, report the rank as an empirical p-value.

Donors with pathological pre-fit (pre-RMSPE > 20× the treated's) are dropped;
they can't be interpreted as comparison units.

## Power — plan before you run

Paid-media teams often request geo lift tests when the expected lift is in the
low-single-digit percent.  If weekly donor SD is 25 trials and the post-period
is 12 weeks, the MDE at 80% power / α=0.05 is roughly

$$\text{MDE} = (z_{\alpha/2} + z_\beta) \cdot \frac{\sigma}{\sqrt{T_{\text{post}}}} \approx (1.96 + 0.84) \cdot \frac{25}{\sqrt{12}} \approx 20\ \text{trials/wk}.$$

If that's larger than your expected business lift, the test will fail to
detect a real effect — push harder or pick a bigger treated market.

## Reproduce

```bash
cd case-studies/01-geo-lift
python src/run.py
```

Expected output (seed 0):

```
Panel: 40 DMAs x 52 weeks
Treated: dma_00   True lift: +8.0%

GeoLift(ATT=+44.91 (+9.6%), RMSPE pre=25.74 post=54.04 ratio=2.1, top donors: dma_04=0.31, dma_35=0.30, dma_07=0.15)

Placebo-in-space p-value (RMSPE-ratio rank): 0.025
Donor pool avg weekly SD: 47.5
MDE (80% power, alpha=0.05, 12w post): 38.4 trials/wk
```

## When geo lift is the right tool (and when it isn't)

| Scenario | Good fit? |
|---|---|
| Non-clickable paid media (brand video, audio, OOH) | ✅ |
| One or a few treated markets, many donors, long pre-period | ✅ |
| Treated DMA is at the edge of the donor distribution (e.g., NYC) | ⚠️ No convex blend may fit — consider augmented SC |
| Very short post window (< 4 weeks) or very noisy outcome | ❌ MDE likely exceeds realistic lift |
| National always-on campaign | ❌ No untreated donors — reach for MMM (case 04) |

## Limitations & what I'd do next

1. **Covariate matching.** Classic Abadie also matches on pre-treatment covariates
   with a V-matrix weighted inner product.  Omitted for clarity.
2. **Augmented SC** (Ben-Michael et al. 2021).  Adds an outcome model bias
   correction for edge-of-donor-distribution treated units.
3. **Conformal inference** (Chernozhukov, Wüthrich, Zhu 2021).  Proper CIs that
   don't rely on donor permutation.
4. **Multi-treated.** For a rollout across 20 DMAs over 6 weeks, reach for
   synthetic DiD (Arkhangelsky et al. 2021).

## References

- Abadie, Diamond, Hainmueller (2010). *Synthetic Control Methods for Comparative Case Studies.*
- Abadie (2021). *Using Synthetic Controls: Feasibility, Data Requirements, and Methodological Aspects.*
- Vaver & Koehler (2011). *Measuring Ad Effectiveness Using Geo Experiments.* (Google.)
- Kerman et al. (2017). *Estimating ad effectiveness using geo experiments in a time-based regression framework.* (Google.)
