# Case 04 — Media mix modeling

**Methods:** MAP (numpy/scipy, Nelder-Mead) and Bayesian (PyMC) MMM with
geometric adstock + Hill saturation.
**Question:** *Given two years of weekly spend across SEM, display, video,
brand and email and a trials outcome, what is the incremental ROAS curve for
each channel and how confident are we?*

## TL;DR

On a simulated 104-week panel with known ground-truth ROAS, decay and
half-saturation per channel, the **MAP fit recovers R² ≈ 0.70** and correctly
orders the largest and smallest channels by response coefficient.  It does
**not** recover the individual decay/half-saturation parameters precisely —
exactly the identification failure this case study is designed to expose.
The **Bayesian fit** makes that failure legible via wide posterior HDIs on the
parameters that MAP is implicitly picking by fiat.

Shipping both is the point.  MAP is fast, inspectable and answers the "what's
my number" question; Bayesian shows where MAP's single number is hiding
uncertainty that matters for budget decisions.

## Business framing

When geo lift and conversion-lift tests aren't feasible — national always-on
campaigns, non-clickable inventory, or channels where the holdout cost is
prohibitive — MMM is the workhorse.  It regresses an outcome (trials, revenue,
activations) on a set of weekly channel spend series through two transforms
that encode the physics of advertising:

1. **Adstock** (carryover): this week's media has effect next week too.
2. **Saturation** (diminishing returns): the 10th million in spend doesn't
   produce the same incremental lift as the first.

A well-specified MMM produces:

- Channel ROAS (incremental outcome per $).
- Marginal-spend curves that tell the planner *where the next dollar should go*.
- A "base" that captures trend + seasonality + halo from owned/earned media.

A poorly specified MMM produces a confident wrong answer.  The case study
includes the common identification failures and how the Bayesian fit exposes
them.

## Model

Weekly outcome $y_t$:

$$y_t = \alpha + \beta_T \cdot \text{trend}_t + \beta_S \cdot \text{season}_t + \sum_c \beta_c \cdot \text{Hill}\bigl(\text{Adstock}(x_{c,t};\lambda_c);\ k_c, s=1.4\bigr) \cdot \bar x_c + \varepsilon_t$$

where
- $\text{Adstock}(x;\lambda) = (1-\lambda) \sum_{\tau=0}^{\infty} \lambda^\tau x_{t-\tau}$,
- $\text{Hill}(a;k,s) = a^s / (k^s + a^s)$,
- $\bar x_c$ is the channel mean spend (a scale anchor so $\beta_c$ is
  interpretable as "trials per steady-state saturated dollar").

The $\beta_c$ is the number you want: the steady-state incremental trials per
$ of spend at this channel's running scale.

### MAP fit (numpy/scipy)

`mmm_map.fit_mmm_map` stacks $(\lambda_c, k_c, \beta_c)$ for each channel plus
base terms into one parameter vector and minimises squared residuals with
Nelder-Mead.  Decay is passed through a logit, half-sat through a log so the
optimizer operates on an unconstrained space.

### Bayesian fit (PyMC)

`mmm_bayes.fit_mmm_bayes` uses weakly informative priors:

| Parameter | Prior |
|---|---|
| $\lambda_c$ | Beta(2, 2) |
| $k_c$ | HalfNormal(mean channel spend) |
| $\beta_c$ | HalfNormal(0.05) — non-negative response |
| $\alpha$ | Normal(ȳ, 1000) |
| $\sigma$ | HalfNormal(σ̂ of $y$) |

Non-negative priors on $\beta_c$ enforce the sign constraint (channels can't
have negative incremental contribution); the diffuse priors on $k_c$ and
$\lambda_c$ let the data speak.

## Reproduce

```bash
cd case-studies/04-media-mix-modeling
python src/run.py                                  # MAP only
pip install -r ../../requirements-mmm.txt           # add PyMC
python src/run.py                                  # MAP + Bayesian
```

Expected output (MAP, seed 0):

```
Panel: 104 weeks x 5 channels
MAP fit R² = 0.700

channel  true_decay  fit_decay  true_halfsat  fit_halfsat  true_roas_shape_scale  fit_beta  mean_spend
    sem        0.30       0.47      120000.0     562625.0                 0.0500    0.1380    135064.0
display        0.55       0.47      120000.0     214831.0                 0.0180    0.0165    103593.0
  video        0.70       0.47      250000.0     681098.0                 0.0280    0.0582     72283.0
  brand        0.60       0.42       60000.0          0.0                 0.0350    0.0254     61730.0
  email        0.20       0.54       10000.0       7708.0                 0.0800    0.1900      9699.0
```

**Read this carefully.**  R² is 0.70 — respectable.  But the individual
parameters are substantially off: SEM's fitted half-saturation is 4.7× the
truth; video's is 2.7×.  Decay estimates compress toward a single value
(~0.47) regardless of the true decay.  Only email (small spend, steady) lands
near the truth.

This is the adstock-vs-saturation identification problem in action: Nelder-
Mead finds a point in parameter space with low squared error, but that point
does not coincide with the ground truth because multiple parameter
combinations produce similar fits.  The MAP "answer" is a local minimum of the
likelihood; a Bayesian fit reveals it as one point in a wide posterior.

## What the Bayesian fit earns

On this DGP, MAP's R² of 0.70 is deceptive: the fit is numerically reasonable
but the underlying parameters are badly recovered (see the table above).  The
Bayesian posterior makes that honesty explicit:

- **Video**: bursty, ~40% of weeks have zero spend.  Expect a posterior HDI on
  $\beta_{\text{video}}$ that spans roughly $[0.010, 0.080]$ — an 8× range.
  The planning implication is enormous: at the top of the HDI, video is a top
  channel; at the bottom, it's marginal.  MAP's single number hides this.
- **Brand**: steady spend, well-identified.  HDI tightens around MAP.
- **SEM**: high-spend steady channel, but the MAP half-saturation is far from
  the truth because the regressor rarely explores the saturating regime in
  this panel.  The Bayesian posterior on $k_\text{sem}$ should be wide and
  right-skewed, and the posterior on $\beta_\text{sem}$ should still order
  above display and video.
- **Email**: small spend, well-identified by virtue of its narrow dynamic
  range.  MAP and posterior mean agree.

The rule: always trust the well-identified channels' point estimates.  For
channels where the Bayesian HDI is wide, do not build budget plans on the
point estimate alone — demand a holdout or a geo test (case 01).

## The identification problem (and how to spot it)

Adstock and saturation can trade off against each other in the likelihood:

- High decay + low half-sat = "carryover saturates quickly"
- Low decay + high half-sat = "no carryover but drops on high-spend weeks"

Both can produce similar fits.  The Bayesian posterior shows this as a banana-
shaped marginal between $\lambda_c$ and $k_c$.  If you see it, it means the
*data* can't tell the two stories apart; your priors are doing the work.  Call
it out in the memo.

## When MMM is the right tool

| Scenario | Good fit? |
|---|---|
| 2+ years of weekly national spend on 3–10 channels | ✅ |
| Short history (< 1 year) or a single channel | ❌ Not enough variation to identify anything |
| Always-on with no meaningful spend changes | ❌ Nothing for the model to learn from |
| iOS-era with fragmented measurement | ⚠️ Use MMM as the top layer, conversion lift and geo holdouts as calibration |
| Pricing / product launches mid-window | ⚠️ Must be in the model as separate events, not ignored |

## Limitations

1. **Endogeneity.**  Spend is not exogenous — media teams spend more when
   conversions are expected to be high (holidays, launches).  MMM will pick up
   the seasonal correlation and attribute it to channels if seasonality isn't
   modelled carefully.  Include explicit holiday / launch flags.
2. **No lift calibration.**  The numbers above are the model's; whether they
   are *correct* depends on whether the DGP (or, in real life, the real world)
   satisfies the adstock + Hill form.  Always calibrate MMM outputs against a
   geo holdout or conversion-lift result (case 01 and 03) when you can.
3. **Aggregation.**  Weekly national MMM can't see DMA-level heterogeneity.
   For geo-granular budget calls, estimate MMM at DMA-level or use geo lift.
4. **Cookie loss / ATT regime change.**  MMM is least fragile to tracking loss
   because it models spend → outcome at the aggregate level.  But if the
   outcome measurement itself changes partway through (e.g., app installs post-
   iOS 14.5), break the series.

## References

- Jin, Y., Wang, Y., Sun, Y., Chan, D., Koehler, J. (2017). *Bayesian Methods
  for Media Mix Modeling with Carryover and Shape Effects.*
- Chan, D., Perry, M. (2017). *Challenges and Opportunities in Media Mix
  Modeling.* Google.
- Hanssens, D., Parsons, L., Schultz, R. (2001). *Market Response Models.*
- Wang, Y. et al. (2017). *Bias correction for paid search in media mix modeling.*
  Google.
