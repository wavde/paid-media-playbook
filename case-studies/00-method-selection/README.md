# Case 00 — When to use which: a decision matrix for paid-media measurement

The four methods in this playbook answer different questions.  Picking the
wrong one produces an answer that looks precise and is systematically biased.

## The question → method map

| Business question | Right tool | Case |
|---|---|---|
| "Is this non-clickable brand campaign driving incremental trials in a market?" | Geo holdout / synthetic control | **01** |
| "Which channels is last-touch over-crediting, and by how much?" | Data-driven MTA (Markov / Shapley) | **02** |
| "Is this specific ad placement incremental?" | Conversion lift (ghost ad / PSA) | **03** |
| "How should we allocate budget across channels at steady state?" | Media mix modeling | **04** |
| "Is our measurement stack seeing what we think it's seeing?" | Session stitching + window sensitivity | **05** |

## The comparative view

| Dimension | Geo lift | MTA | Conversion lift | MMM |
|---|---|---|---|---|
| Grain | Market-week | User-touch | User-eligibility | Channel-week |
| Estimand | ATT on treated DMA | Fractional credit per channel | ITT lift for one placement | Marginal ROAS curve |
| Randomisation | Quasi (geo) | Observational | True randomisation | Observational |
| Time to read | 8–16 weeks | Any | 2–8 weeks | 6–12 months of data |
| Cost | Lost incremental in donor DMAs | Path data infra | Holdout opportunity cost | Modeling time |
| Bias risk | Wrong donors / spillover | Confounded paths / no counterfactual | Interference | Endogeneity, co-movement |
| Works on non-clickable | ✅ | ❌ | ❌ | ✅ |
| Works on always-on | ❌ | ✅ | ❌ | ✅ |
| Answers counterfactual "what if I cut X?" | ✅ (for treated DMA) | ❌ | ✅ (for tested placement) | ✅ (via marginal curve) |

## How they fit together

A paid-media measurement stack that earns its keep usually looks like this:

1. **MMM** sits at the top and answers the quarterly budget allocation question.
   Fast, always-on, everyone can use the number.  But it's also the least
   experimentally rigorous, so it has to be calibrated.

2. **Geo lift and conversion lift** provide the calibration anchors.  Every
   6–12 months a major channel is tested experimentally; the result is used to
   adjust the MMM's channel beta (not the model structure — a scale factor is
   enough).  The best MMM teams publish their calibration ratios.

3. **MTA** runs daily on the tracked paths and answers tactical questions
   (creative, audience, bidding).  It is **explicitly not** the source of
   truth for budget reallocation — that role is MMM's.  MTA's value is speed
   and granularity, not causal precision.

4. **Stitching + window sensitivity** is the health check on the MTA layer.
   If stitching quality drops (cookie loss rises, iOS share shifts), MTA's
   numbers drift; the calibration ratio catches the drift against MMM / lift
   results.

## Anti-patterns

The following are common, named, and always wrong:

| Anti-pattern | Why it's wrong |
|---|---|
| Using LTA as the budget-reallocation source | Systematically under-credits upper-funnel display + video (case 02) |
| Running a conversion-lift test on 1% of traffic at a 0.1% base rate | Underpowered; reads null as "no effect" (case 03) |
| Trusting MMM beta without a lift calibration | Confuses spend co-movement with causal effect (case 04) |
| Quoting LTA share on a 1-day window on post-ATT iOS | All three biases compound (case 05) |
| Comparing MMM and MTA numbers and averaging them | They answer different questions; the average is wrong |

## Escalation rules

A short set of rules that map questions to methods, in priority order:

1. **If you can randomise, do.**  Conversion lift > geo lift > MMM > MTA.
2. **If the channel is non-clickable or geo-concentrated, geo lift.**  Period.
3. **Always-on, diverse channel mix, budget-allocation question: MMM.**
   Calibrate against lift results every 6 months.
4. **Last-touch is a reporting layer, not a decision layer.**  If anyone on
   the team proposes reallocating budget based on LTA share, point them at
   case 02.
5. **Stitching and window sensitivity come before any attribution rule.**  If
   30% of journeys span multiple user ids, no rule will save you.

## References (consolidated)

- Gordon, Zettelmeyer, Bhargava, Chapsky (2019). *A comparison of approaches to
  advertising measurement: Evidence from big field experiments at Facebook.*
- Chan, D., Perry, M. (2017). *Challenges and Opportunities in Media Mix
  Modeling.*  Google.
- Vaver, J., Koehler, J. (2011). *Measuring Ad Effectiveness Using Geo
  Experiments.*  Google.
- Johnson, G., Lewis, R., Nubbemeyer, E. (2017). *Ghost ads: Improving the
  economics of measuring online ad effectiveness.*
- Berman, R. (2018). *Beyond the Last Touch: Attribution in Online Advertising.*
