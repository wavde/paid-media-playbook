# Case 08: Does it replicate? Propensity vs uplift on a randomized e-mail test (Hillstrom)

**Method:** the [case 07](../07-criteo-uplift/) targeting checks, run on a second public
randomized experiment from a different advertiser and channel, with cross-fitted models and
bootstrap SEs.
**Question:** *Case 07 found, on Criteo display ads, that the users a propensity model ranks
highest are also the users the ad moves most, so propensity targeting captures most of the
incremental outcomes. Does that hold for a retailer's e-mail campaign?*

## TL;DR

All numbers are from [`results.md`](results.md). The data has 64,000 customers, randomized one
third each to a men's e-mail, a women's e-mail, or no e-mail. Every customer is scored
out-of-fold.

- **The concentration finding does not replicate.**
  - For visits, the top 20% of customers by propensity reach 25.9% (SE 2.1%) of all
    incremental visits from sending any e-mail. Random targeting would reach 20%.
  - The uplift model reaches 27.8% (1.9%).
  - The same comparison on Criteo was 72.2% for propensity (case 07).
- **Marginal returns still fall, but far less steeply.** Under propensity targeting, the
  marginal incremental visits per 1,000 extra customers from 10% to 40% reach are 0.67 (0.11)
  of the top 10%'s average. Case 07 found 0.15.
- **Credit inflation is still real, but it barely grows with targeting.** Credited visits are
  2.74x (0.11) the incremental visits when everyone is e-mailed, and 3.15x (0.34) for the top
  10% by propensity.
- **An uplift model adds little.** The differences from propensity are within about one SE at
  10% and 20% reach. The one clear gain is at 50% reach: 64.4% (2.3%) of incremental visits
  against 56.7% (2.2%).
- **Conversions are too rare to say much.** There are about 212 incremental conversions from
  sending any e-mail, so most conversion ratios are not estimable, and every share has an SE of
  5 points or more.

**What this changes.** Case 07's conclusion was that propensity targeting already finds the
persuadable users, so an uplift model mostly isn't worth it. That turns out to be a property of
that dataset, not a general rule. On e-mail to existing customers, the incremental effect is
spread across the base. Neither score concentrates it much, and the case for targeting at all
is weaker. What did replicate is that credited results overstate incrementality, and that the
marginal return is below the average under propensity targeting.

## Design

- **Data:** Kevin Hillstrom, *The MineThatData E-Mail Analytics And Data Mining Challenge*
  (2008).
  - 64,000 customers who purchased in the last twelve months, with recency, past spend,
    men's/women's purchase flags, a new-customer flag, zip type and channel.
  - Visits, conversions and spend are tracked for two weeks after the e-mail.
  - Published for public analysis with no stated license, so it is downloaded at runtime
    from the author's site, checked against a SHA-256, and not committed.
- **Comparisons:** any e-mail vs no e-mail (main), plus each campaign vs no e-mail.
- **Models:** ridge logistic regression, fit with the same numpy IRLS as case 07
  ([`src/paid_media/uplift.py`](../../src/paid_media/uplift.py)).
  - **Propensity:** `P(outcome | e-mailed, x)`.
  - **Uplift:** a T-learner, which subtracts a model fit on the control group.
- **Cross-fitting:** with only 64,000 customers, a 50/50 split would leave the top 5% at about
  1,600 people. Instead, scores come from 5-fold cross-fitting, so every customer is scored by
  models that never saw them.
- **SEs:** a 300-replicate Poisson bootstrap over customers, with the ranking held fixed.
  Ratios whose denominator has a bootstrap coefficient of variation above 0.3 are reported as
  not estimable.

## Reproduce

```bash
pip install -r requirements.txt
python case-studies/08-hillstrom-email-uplift/src/run.py   # downloads ~4 MB on first run
```

This uses seed `20261127` and runs in seconds. Two runs produce byte-identical `results.md`.
CI runs the pipeline with `--smoke`, on synthetic data with the real schema. Tests are in
[`tests/test_hillstrom_uplift.py`](../../tests/test_hillstrom_uplift.py).

## Limitations

1. **Small for this question.** 64,000 customers and about 212 incremental conversions. Visit
   results are reasonably precise, but conversion results are mostly noise.
2. **Different outcome window and treatment.** A single e-mail with a two-week window, against
   Criteo's display campaigns. The two datasets differ in channel, advertiser, audience
   (existing customers only here) and features, so this case can show that case 07's
   concentration result isn't general, but not explain why.
3. **Simple models, as in case 07.** A stronger uplift learner might do better. The Hillstrom
   literature reports modest gains from uplift models on this data.
4. **SEs exclude model-fitting noise,** and the ranking is held fixed in the bootstrap.
5. **Data provenance.** The file comes from the author's site over plain HTTP. The SHA-256
   check guards against changes, but the dataset could disappear.

## References

- Hillstrom, K. (2008). *The MineThatData E-Mail Analytics And Data Mining Challenge.*
  blog.minethatdata.com.
- Radcliffe, N. J. (2008). *Hillstrom's MineThatData Email Analytics Challenge: An Approach
  Using Uplift Modelling.* Stochastic Solutions.
- Künzel, S. R., Sekhon, J. S., Bickel, P. J. & Yu, B. (2019). *Metalearners for estimating
  heterogeneous treatment effects using machine learning.* PNAS.
- Chernozhukov, V. et al. (2018). *Double/debiased machine learning for treatment and
  structural parameters.* Econometrics Journal (on cross-fitting).
