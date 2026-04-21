# Paid Media Playbook

> End-to-end measurement case studies for paid media: geo lift, multi-touch
> attribution, conversion-lift testing, media mix modeling, and web attribution.
> Each paired with a decision memo.

![CI](https://github.com/wavde/paid-media-playbook/actions/workflows/ci.yml/badge.svg)

The measurement methods that actually drive paid-media budget decisions —
written as reproducible case studies.  Each study pairs a simulator (so
reviewers can regenerate data and verify that the method recovers the known
ground truth) with a memo-style writeup covering the business framing, the
method, the limitations, and a "when to use which" rule for the broader stack.

All data here is **synthetic**.  The framing is inspired by paid-media work on
productivity-software advertising (SEM + display as the primary paid channels,
trials and seat activations as the outcome), but no proprietary numbers,
segments, or accounts are used.

## Case studies

| # | Case study | Methods | Status |
|---|---|---|---|
| 00 | [Method selection — when to use which](case-studies/00-method-selection/) | Decision matrix across all four methods | Complete |
| 01 | [Geo lift for a paid-media brand campaign](case-studies/01-geo-lift/) | Synthetic control, placebo-in-space, MDE power analysis | Complete |
| 02 | [MTA: rule-based vs data-driven](case-studies/02-mta-comparison/) | LTA, FTA, position, time-decay, Markov removal, Shapley + click/view credit weighting | Complete |
| 03 | [Conversion lift / incrementality](case-studies/03-incrementality/) | Ghost-ad matched-market, two-proportion z, Bayesian bootstrap CI, sample-size planning | Complete |
| 04 | [Media mix modeling](case-studies/04-media-mix-modeling/) | MAP (numpy/scipy) + Bayesian (PyMC) with adstock + Hill saturation | Complete |
| 05 | [Web attribution deep dive](case-studies/05-web-attribution/) | Session stitching, journey reconciliation, window sensitivity, iOS/ATT loss | Complete |

Each case includes:

1. A business question the way a paid-media or finance partner would phrase it.
2. An explicit estimand with the assumptions stated.
3. A simulator that generates data with known ground truth.
4. A method implemented in a single inspectable module (100–300 lines).
5. A memo that would survive an interview take-home review.

Start with [case 00](case-studies/00-method-selection/) for the "when to use
which" matrix; it makes the rest of the playbook easier to navigate.

## Stack

- **numpy / scipy / pandas** as the default toolkit.
- **statsmodels** and **scikit-learn** where they fit naturally.
- **PyMC** for the Bayesian MMM in case 04 (optional — the MAP reference
  implementation runs without it).  Install with:
  `pip install -r requirements-mmm.txt`.
- **[experiment-toolkit](https://github.com/wavde/experiment-toolkit)** — the
  sibling PyPI package — for sequential monitoring when a conversion-lift
  test peeks.

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate          # macOS / Linux
pip install -r requirements.txt
pytest -q                          # smoke tests regenerate data and validate methods
```

Each case has its own reproducer:

```bash
python case-studies/01-geo-lift/src/run.py
python case-studies/02-mta-comparison/src/run.py
python case-studies/03-incrementality/src/run.py
python case-studies/04-media-mix-modeling/src/run.py
python case-studies/05-web-attribution/src/run.py
```

## Intended audience

- Data scientists and analysts moving from experimentation or product analytics
  into paid-media measurement.
- Marketing-science and marketing-analytics practitioners looking for a
  cross-method reference with memo-style writeups.
- Hiring managers who want to see method + memo + reproducer side by side.

The top-level READMEs are written generalist-first; specialist depth (priors,
identification arguments, window conventions, consent regimes) lives inside
each case study.

## Sibling repositories

- **[causal-inference-playbook](https://github.com/wavde/causal-inference-playbook)** —
  experimentation + quasi-experimental methods that underpin geo lift and
  conversion-lift (CUPED, synthetic control, DiD, propensity score, mSPRT).
- **[experiment-toolkit](https://github.com/wavde/experiment-toolkit)** —
  small PyPI package for sample sizing, CUPED, sequential tests, and DiD.
- **[product-analytics-deepdive](https://github.com/wavde/product-analytics-deepdive)** —
  funnel, retention, segmentation, north-star metric work.
- **[analytics-sandbox](https://github.com/wavde/analytics-sandbox)** — SQL
  reference, including a last-touch attribution problem.

## Out of scope

No ad-server integration code, no orchestration framework, no proprietary
data, no vendor-specific APIs (Meta CAPI / Google Enhanced Conversions / etc.
are referenced in memos but not implemented).  Simulators are structured to
behave realistically, not to replicate any specific company's numbers.

## License

MIT.  See [LICENSE](LICENSE).

## Selected references

- Chan, D., Perry, M. (2017). *Challenges and Opportunities in Media Mix Modeling.* Google.
- Gordon, Zettelmeyer, Bhargava, Chapsky (2019). *A comparison of approaches to advertising measurement: Evidence from big field experiments at Facebook.*
- Jin, Y. et al. (2017). *Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects.*
- Johnson, G., Lewis, R., Nubbemeyer, E. (2017). *Ghost ads: Improving the economics of measuring online ad effectiveness.*
- Vaver, J., Koehler, J. (2011). *Measuring Ad Effectiveness Using Geo Experiments.* Google.
- Berman, R. (2018). *Beyond the Last Touch: Attribution in Online Advertising.*
- Abadie, A., Diamond, A., Hainmueller, J. (2010). *Synthetic Control Methods for Comparative Case Studies.*
- Apple (2021). *User Privacy and Data Use / SKAdNetwork.*
