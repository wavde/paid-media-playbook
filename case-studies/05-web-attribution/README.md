# Case 05 — Web attribution deep dive

**Methods:** Session stitching, journey identity reconciliation, attribution-
window sensitivity analysis, iOS/ATT visibility loss quantification.
**Question:** *Before we apply any attribution rule, how much of the user
journey is the stack actually seeing?*

## TL;DR

On a simulated event log with 18% user-level cookie-loss and realistic iOS
app visibility loss, ~15% of stitched journeys span more than one user id on
web, and **iOS journeys collapse to 1 event each under ATT** (vs ~6.4 on
web/android).  Shortening the attribution window from 28 days to 1 day
doubles brand+direct's share of display+video credit and cuts upper-funnel
credit roughly in half.

The point of this case is that **the attribution rule is downstream of
stitching**, and stitching errors dominate attribution errors in the post-ATT
regime.

## Business framing

Every ad-measurement stack has the same question: *what did this user see
before they converted?*  The path from raw event log to answer passes through:

1. **Session stitching**: turn a stream of events into sessions.
2. **Journey reconciliation**: stitch a user's sessions across device / cookie
   changes into a single journey.
3. **Attribution window**: decide how far back a touch is credited.
4. **Rule or model**: LTA / MTA / Markov / Shapley (see case 02).

Steps 1–3 are plumbing.  Step 4 is what everyone argues about.  In practice
**the plumbing is where most of the measurement bias lives**, especially since
iOS 14.5/ATT and Safari ITP reset how web cookies persist.

## The three mechanisms that break visibility

### 1. Cookie loss on web

A user's first-party cookie expires, gets blocked by ITP, or is cleared.  Their
next session appears under a new anonymous id.  If the conversion happens in
that new session, the pre-conversion touches look like a journey of zero.

Mitigations (in order of effectiveness):

- Authenticated user id (highest signal; requires login).
- Device/IP probabilistic bridging (IAB standard; expect 60–70% match rates).
- UTM persistence in localStorage with server-side join.

### 2. iOS 14.5 ATT / app identifiers

On iOS post-ATT, most users opt out of IDFA sharing.  App events come in with
no stable cross-session identifier; each session is a new "user".  A 7-touch
app journey appears to the stack as 7 separate one-touch journeys.

This case simulates that by giving iOS app users a ``_ios`` suffix on every
session.

### 3. Attribution-window truncation

Shorter windows under-count upper-funnel influence.  Display and video
impressions that ran 14 days before conversion are visible in a 28-day window
but not in a 1-day window.  iOS SKAdNetwork's postback window is ~24 hours for
conversion values; this is the floor for app measurement.

## Reproduce

```bash
cd case-studies/05-web-attribution
python src/run.py
```

Expected output (seed 0):

```
Event log
---------
Events:           32,317
Simulated cookie-loss rate (ground truth): 18.0%

After session stitching
-----------------------
Distinct user_ids:  10,407
Distinct sessions:  32,216
Distinct journeys:  9,068
Share of journeys spanning >1 user_id (cookie loss): 14.8%

Attribution-window sensitivity (last-touch share)
-------------------------------------------------
window_days     1      7      28
channel
brand        0.253  0.250  0.230
direct       0.218  0.208  0.191
sem          0.197  0.192  0.190
email        0.163  0.149  0.146
organic      0.102  0.106  0.117
display      0.043  0.055  0.076
video        0.023  0.039  0.050

iOS attribution loss (events per journey by device)
---------------------------------------------------
             events  unique_users  unique_journeys  events_per_journey
device
web_mobile    13326          2787             2056                6.48
web_desktop   10942          2317             1709                6.40
ios_app        4800          4800             4800                1.00
android_app    3249           503              503                6.46
```

## Reading the table

- **iOS is a journey of one.**  4,800 iOS events map 1:1 to 4,800 "users" and
  4,800 "journeys" — because under ATT the stack cannot stitch sessions
  back to a stable identifier, so every event looks like a new user's first
  touch.  Compare to android / web at ~6.4 events per journey.  This is the
  ATT visibility floor; any attribution result quoted on iOS traffic is
  measuring the floor, not the channel.
- **1-day window**: brand + direct capture 47% of credit; display + video
  capture under 7% combined.  That is the paid-media team's measurement
  worst case — upper funnel is invisible, budget shifts toward the
  cheapest-looking channels, and total spend efficiency silently collapses.
- **28-day window**: credit rebalances.  Display rises from 4.3% → 7.6%
  (+3.3 pp, ~77% relative); video from 2.3% → 5.0% (+2.7 pp, ~2.2×
  relative).  Brand / direct give up ~5 pp combined.  The absolute pp
  shifts are small because brand/direct dominate last-touch under any
  window, but the relative shifts are exactly what drives channel-level
  ROAS decisions.
- **7-day window**: the usual industry default.  Close to 28-day for
  well-stitched journeys; closer to 1-day for fragmented ones (iOS, post-
  ATT mobile).

The delta between 1-day and 28-day is a **lower bound** on the measurement
loss from ATT/cookie-loss — the upper bound is "you can't see the touch at
all".

## What a production stack should expose

Beyond the rule-based dashboards most teams start with:

1. **Stitching quality metric**: share of journeys spanning >1 user id.  Track
   this weekly; investigate any week-over-week change > 3 pp.
2. **Window sensitivity view**: every LTA / MTA dashboard should carry a
   toggle for 1/7/28-day windows and show the channel-share diff.
3. **iOS / Safari segment**: separate the funnels.  Pooled numbers hide the
   ATT regime change until it's too late.
4. **Calibration anchors**: every 6 months run a conversion-lift test (case 03)
   or geo holdout (case 01) on a major channel.  Calibrate the stack's
   reported ROAS against the experimental truth.  If they diverge > 20%,
   something in the stack is miscalibrated; don't let the dashboard be the
   source of truth by default.

## Limitations

1. **The journey reconciliation here is trivial** — a shared ``u_NNNNN`` prefix.
   In production you'd use a deterministic join on authenticated user id (login)
   plus a probabilistic fallback (IP + UA + geo bucket within a time window).
2. **No privacy constraints modelled.**  ATT opt-in rates are ~25% on average
   but vary by app category and prompt timing.  Model this as a binomial opt-in
   per user and you see the ATT effect in each channel's path-length distribution.
3. **Conversion window ≠ attribution window.**  The simulation here collapses
   the two.  Real stacks use separate windows for conversion eligibility and
   touch credit.
4. **No server-to-server measurement.**  Meta's CAPI / Google's Enhanced
   Conversions / LinkedIn's CAPI all restore some visibility by hashing PII
   server-side.  A full treatment would model the match rate (typically 65–85%)
   and compare pixel-only vs CAPI journeys.

## References

- Apple (2021). *User Privacy and Data Use.* (ATT framework.)
- Apple (2021). *SKAdNetwork Documentation.*
- Kireyev, P., Pauwels, K., Gupta, S. (2016). *Do display ads influence search?
  Attribution and dynamics in online advertising.*
- IAB Tech Lab (2022). *Identity Solutions Landscape.*
- Lada, A., Wang, H., Yan, T. (2019). *A Closer Look at Incrementality Testing
  at Meta.*
