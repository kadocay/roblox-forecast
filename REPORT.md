# REPORT — roblox-up-and-coming-forecast-01

Entry for the make.fun coding bounty **"Predict Roblox's Up-and-Coming top 10"**
($500 prize; deadline 2026-11-13T06:00:00Z; scored window 2026-11-13T06:00:00Z →
2026-11-16T06:00:00Z, 72 hourly reads).

## What was built

- `main.py` — Python 3.13, standard library only, no network, no API keys,
  no model calls, fully deterministic (fixed seed `20261113`; no wall clock,
  no unseeded randomness). Reads `$MAKEFUN_INPUT_DIR` (else `/input`) and
  writes `$MAKEFUN_OUTPUT_DIR/forecast.json` (else `/output/forecast.json`),
  reading the window, places and output name from `task.json` (never
  hard-coded).
- `LICENSE` — MIT.

The program is a **latent-score Monte Carlo diffusion** forecaster:

1. Every game seen on the chart gets a latent score from its last rank,
   `s = -log(rank)`. The top of the chart is spread out in score, the tail
   is dense — which matches how the real chart moves (sticky top, churning
   tail).
2. Each hour, every game's score takes a Gaussian step (σ = 0.20 per hour,
   ×0.7 for the games at the top, whose scores sit further apart) plus its
   own recent drift: the least-squares slope of its score over the last
   48 h, shrunk 0.5× and capped at |0.02|/h. Games off the chart keep their
   last on-chart score; games unseen for over a week are dropped; the pool
   is capped at 220 games by score.
3. Games whose `details` carry a `created` date (the house recorder's reads)
   leave the chart 28 days after creation: a game that ages out mid-window
   is removed from the simulation at that hour, and one already past 28 days
   at the last read is excluded entirely.
4. At each of the window's 72 hourly slots the simulated scores are ranked
   (1,500 replicates). The forecast at a place is the fraction of
   (replicate, slot) pairs in which each game held that place, averaged over
   all slots.
5. Rows are written only for probabilities ≥ 0.007, and only while they stay
   above the residual share an unnamed game would receive at that place —
   so a named game always earns more than the unnamed share it displaces,
   and brand-new games (never seen in the history) are covered by the mass
   left unnamed at each place. `top10` orders games by expected rank
   (probability mass outside the top 10 counted as rank 15).

## Spec followed

- Kit: `~/workspace/makefun-werewolf/kits/roblox-up-and-coming-forecast-01/roblox-up-and-coming-forecast-01-kit/`
  (`RULES.txt` read first, then `competition.yaml`, `input/task.json`,
  `starter/main.ts`; the scorer's canonical rule confirmed against
  `harness/harness-src.tar.gz` → `grader/forecast/ranked.ts`:
  `P_FLOOR = 1e-4`, unnamed share = `(1 − place sum) / unnamed games on
  that read's chart`, score = mean over reads of Σ places ln max(P, 1e-4),
  hits/agreement as in the rules).
- Output: exactly two keys `probabilities` / `top10`; rows exactly
  `{universe_id, place, p}`; 1 ≤ place ≤ 10; each place ≤ 100 rows;
  each place's sum ≤ 1; `top10` exactly 10 distinct universe ids;
  file ≪ 1,048,576 bytes (12,184 bytes on the practice input).
- Limits: ~19–20 s runtime on this machine (limit 300 s), 1 CPU, stdlib only.

## Verification (commands and results)

Scorer replica implemented from `ranked.ts` and checked against the rules
text; the house baseline's starter logic was replicated (`baseline_forecast`)
and backtested on the same protocol, so every number below is measured with
the competition's own log-score rule.

### 1. Local run + determinism + format

```sh
cd ~/workspace/makefun-werewolf/roblox-forecast
mkdir -p out1 out2
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out1 python3 main.py   # ~32 s
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out2 python3 main.py   # ~36 s
cmp out1/forecast.json out2/forecast.json   # => identical, no output (byte-identical)
```

Practice-input output (final code): 236 rows (place 1: 13 rows, sum 0.9205;
place 10: 34 rows, sum 0.7218; all place sums ≤ 1), `top10` 10 distinct ids,
12,184 bytes.

### 2. Tuning — rolling-origin backtests (numpy prototype, same model)

Protocol: for each of 19 cutoffs (reads 14–32 of the 34-read practice
history), forecast from the reads up to the cutoff and score the reads in
the following 72 h with the log-score rule; mean over cutoffs.
Baseline (persistence replica) on the same protocol: **−43.99**
(hits 5.99, agreement 48.9). The house states the baseline scores
−43.512039 on the true practice window.

| config | mean log score | hits | agreement |
|---|---:|---:|---:|
| MC σ=0.06, no drift | −47.40 | 5.99 | 48.9 |
| MC σ=0.09, no drift | −42.02 | 5.99 | 48.9 |
| MC σ=0.12, drift 48 h | −39.20 | 5.99 | 47.2 |
| MC σ=0.16, drift 48 h | −36.99 | 5.99 | 47.2 |
| MC σ=0.20, drift 48 h | −36.57 | 6.02 | 47.7 |
| MC σ=0.20, drift 48 h, top σ ×0.7 | **−36.08** | 5.98 | 47.2 |
| MC σ=0.25, drift 48 h | −38.74 | 6.07 | 48.1 |
| MC σ=0.40, drift 48 h | −43.09 | 6.04 | 48.0 |
| sqrt / root-4 score mapping (σ=0.20) | −39.13 / −42.86 | — | — |
| OU mean-reversion θ=0.005 / 0.01 / 0.02 | −36.05 / −36.24 / −37.29 | — | — |
| empirical rank-transition kernel alone | −38.45 | 5.99 | 47.9 |
| MC+kernel blend 0.7 / 0.5 | −35.93 / −36.37 | 6.20 / 6.02 | 49.3 / 48.4 |
| p_min sweep (σ=0.20, drift, top ×0.7): 0.0005 / 0.0015 / 0.003 / 0.006 / 0.008 / 0.010 / 0.015 / 0.020 | −37.34 / −36.08 / −35.67 / **−35.23** / −35.32 / −35.34 / −35.55 / −36.00 | 5.98 | 47.2 |

Chosen configuration: σ=0.20, top σ ×0.7, drift 48 h (shrink 0.5, cap
0.02), log-rank score, `P_MIN = 0.007` (the 0.006–0.008 plateau;
differences inside it are within MC noise). Rejected: OU mean-reversion
(no gain), the empirical kernel and its blends (kernel alone −38.45;
blend 0.7 at p_min 0.006 scored −37.03, worse than pure MC), `playing`
growth as a drift signal (correlation with rank change 0.015 — no signal).

Data notes that shaped the model: top-10 retention is 68% at +24 h,
49% at +48 h, 38% at +72 h; 287 of 330 top-10 games stay top-10 at the
next read (~6 h later); only 1 of 39 games that ever held the top 10
entered it having never been on the chart; `log(playing)` correlates 0.45
with rank but playing *growth* does not predict rank change.

### 3. End-to-end backtest of the actual entry (`main.py`)

`python3 /tmp/rbf/e2e.py` — runs the shipped `main.py` (1,500 replicates,
`P_MIN = 0.007`) through the same 19-cutoff rolling-origin protocol.
307 s total, all 19 cutoffs:

| cutoff (history ends) | entry log score | hits | agreement | baseline log score |
|---|---:|---:|---:|---:|
| 14, 2026-11-02 | −37.1736 | 5.70 | 44.1 | −48.4114 |
| 15, 2026-11-02 | −41.6715 | 5.40 | 40.5 | −50.8858 |
| 16, 2026-11-03 | −38.8836 | 5.30 | 38.5 | −52.7055 |
| 17, 2026-11-03 | −37.3829 | 6.40 | 49.3 | −47.4361 |
| 18, 2026-11-03 | −37.5817 | 5.40 | 43.2 | −47.3430 |
| 19, 2026-11-04 | −39.6336 | 4.90 | 38.6 | −49.5913 |
| 20, 2026-11-04 | −36.6688 | 5.00 | 37.6 | −50.2692 |
| 21, 2026-11-04 | −38.7658 | 5.18 | 36.5 | −50.8845 |
| 22, 2026-11-04 | −41.7083 | 4.82 | 30.1 | −55.8516 |
| 23, 2026-11-05 | −43.1970 | 5.40 | 33.5 | −56.9113 |
| 24, 2026-11-05 | −40.0860 | 5.89 | 45.3 | −51.9673 |
| 25, 2026-11-05 | −37.0529 | 6.00 | 47.0 | −52.1253 |
| 26, 2026-11-06 | −34.7349 | 6.57 | 56.3 | −50.7787 |
| 27, 2026-11-06 | −28.6431 | 7.33 | 63.2 | −37.7127 |
| 28, 2026-11-06 | −28.6151 | 7.20 | 59.4 | −32.9181 |
| 29, 2026-11-06 | −27.5298 | 6.75 | 60.0 | −30.0082 |
| 30, 2026-11-07 | −28.4929 | 5.00 | 47.3 | −30.4348 |
| 31, 2026-11-07 | −26.5574 | 7.50 | 65.0 | −29.0921 |
| 32, 2026-11-07 | −22.9326 | 7.00 | 61.0 | −10.5434 |
| **mean** | **−35.1217** | **5.93** | **47.2** | **−43.9932** |

**Entry beats the baseline by +8.8715 nats on average, and beats it at 18
of the 19 cutoffs** (the exception is cutoff 32, which has a single scoring
read ~2 h after the cutoff, where persistence wins). Early cutoffs
(Nov 2–5) are much harder than late ones for both forecasts — the chart
was churning faster then — which is why the mean is dominated by the
volatile period. The shipped entry's −35.1217 matches the numpy prototype's
−35.23 at the same configuration, confirming the pure-Python port.

### 4. 28-day expiry unit test (synthetic input with `details`)

`python3 /tmp/rbf/expiry_test.py` — a game expiring 1 h after the last
read gets P = 0 everywhere (it is gone before the window starts); a game
already past 28 days is excluded from `top10`; a game expiring 30 h after
the last read keeps P(at #1) = 0.303 and top-10 mass 0.378, matching the
~28/72 in-window share. **Passed.**

## Known weaknesses (honest)

- **The true practice window was never scored locally.** The kit's `input/`
  holds only the 34-read history; the 10 archive reads in the practice
  window (2026-11-08 → 2026-11-10) are not in the kit, and fetching all ten
  full snapshot files from the public archive would have needed ~100
  paginated page reads (each file is ~9,300 lines; only the up-and-coming
  top 10 of each was needed). A GitHub API trees request for the file list
  was refused with HTTP 403 and was not retried. So the house baseline's
  stated practice score (−43.512039) could not be reproduced; all tuning
  and verification used rolling-origin backtests inside the practice
  history instead (19 cutoffs, reported above). The backtest baseline on
  that protocol is −43.99, close to but not the same as the stated practice
  number, and the two are not directly comparable.
- **Sparse history vs. dense final history.** All tuning used archive reads
  3.6–11.0 h apart. The final run's history adds ~48 hourly house-recorder
  reads, so per-game drift estimates will rest on up to 48 points instead
  of ~8, and the model was never tested on hourly-spaced data. The
  parameters (σ, drift window, P_MIN) were chosen on sparse data and may
  not be optimal for dense data; the direction of any error is unknown.
- **Brand-new games are only covered by the unnamed residual.** A game
  that has never been on the chart cannot be named, so if one takes a top-10
  place it scores only the residual share (about 0.002–0.005 per place at
  the final threshold). In the history this happened once in 8 days
  ("Ragdoll Wars!" jumped straight to #5), so the expected cost is small
  but real.
- **Per-cutoff variance is large.** Entry log scores ranged from −43.20 to
  −22.93 across cutoffs; the +8.87 mean improvement over baseline is
  consistent (18/19 cutoffs) but the final 72 h could resemble the hard
  early-November period rather than the calm late one.
- **`P_MIN` was tuned on the same backtests used for reporting** (0.006–0.008
  plateau; the 0.006/0.008 endpoints differ by ~0.1 nats, within MC noise),
  so a small amount of selection overfitting is possible. The chosen 0.007
  sits mid-plateau rather than at the single best point.
- **No use of `playing`, votes, genre, or top-trending/top-playing-now
  context.** Playing *growth* showed no predictive correlation (0.015) and
  was dropped; the other fields were not tested as signals.
- **The 28-day expiry rule was unit-tested synthetically only** — the
  practice history carries no `details`, so no real game was ever observed
  aging out in the tuning data.

## What was NOT done (per the task)

- No GitHub repo created, nothing submitted, no wallets/keypairs created,
  no private keys or API keys handled. No live browser work was needed.
  The entry is `main.py` + `LICENSE` + this `REPORT.md`, ready to be copied
  into the `kadocay/roblox-forecast` repository by the parent agent.
