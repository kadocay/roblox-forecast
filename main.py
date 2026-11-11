#!/usr/bin/env python3
"""Forecast Roblox's Up-and-Coming chart top 10 for make.fun competition
roblox-up-and-coming-forecast-01.

Reads /input/history.ndjson (one chart read per line, oldest first) and
/input/task.json, writes /output/forecast.json: for each of the top 10
places, a probability for each game named being at that place, plus the
top-10 guess in order. Python 3.13, standard library only, no network,
fully deterministic (fixed seed; no wall clock, no unseeded randomness).

Method: latent-score Monte Carlo diffusion (v2, see REPORT.md).
  * Every game seen on the chart gets a latent score from its last rank,
    s = -log(rank): the top of the chart is spread out in score, the tail
    is dense, which matches how the real chart moves (sticky top, churning
    tail).
  * Each hour every game's score takes a heavy-tailed step (Student-t with
    3 degrees of freedom, scaled to the same per-hour sd) plus its own
    recent drift: the least-squares slope of its score over the last 48 h,
    shrunk and capped. Scores are capped at 0 (a game cannot sit above
    the #1 score), and games already in the top 3 get no upward rank drift
    (they cannot climb further; empirically they rotate out, not up).
    Off the chart, a game's score is its last on-chart score; games
    unseen for over a week are dropped.
  * Playing-growth boost: a game whose player count surged over the last
    48 h (log growth above 0.5) gets extra upward drift, min(0.04,
    0.024 * (growth - 0.5)) per hour. The chart is sorted by user growth,
    and mid-chart games with a 48 h playing surge reach the top 10 within
    72 h about 5x more often than the base rate (21% vs 4.5% in the
    archive), so the boost names the climbers the rank history alone
    misses. The boost applies to every rank: top games with a surge hold
    their place better too.
  * Games whose creators give a `created` date (the house recorder's reads
    carry games/v1 details) leave the chart 28 days after creation, so a
    game that ages out mid-window is removed from the simulation at that
    hour.
  * At each of the window's hourly slots the simulated scores are ranked;
    the forecast at a place is the fraction of (replicate, slot) pairs in
    which each game held that place, averaged over all 72 slots. Games
    nobody has seen (brand-new entries) are covered by the mass left
    unnamed at each place: rows are only written for probabilities above
    the residual share an unnamed game would otherwise receive, so a named
    game always earns more than the unnamed share it displaces.
  * top10 orders games by expected rank (probability mass outside the top
    10 counted as a mid-chart rank).

Tuned and validated on three evaluations, all scored with the
competition's own log-score rule (see REPORT.md): the true practice
window (10 archive reads, 2026-11-08 to 2026-11-10, locally
reconstructed from the public archive; the local scorer reproduces the
house baseline's stated practice score -43.512039 exactly), 19
rolling-origin backtests over the practice history, and 13
rolling-origin backtests over the recent regime (cutoffs 2026-11-07 to
2026-11-10, archive reads to 2026-11-11). v2 scores about -32.1 on the
practice window, -34.2 on the 19-cutoff backtests and -33.2 on the
recent-regime backtests; the v1 entry scored -35.8 (official practice),
-35.1 and -33.6 on the same three evaluations.
"""

import heapq
import json
import math
import os
import random
from datetime import datetime, timedelta

# --- tuned constants (from the backtests in REPORT.md) --------------------
SEED = 20261113
REPS = 2000
SIGMA = 0.20            # per-hour sd of a game's score step (log-rank units)
SIGMA_TOP_MULT = 0.7    # top-10 games move more slowly in score units
DF = 3                  # Student-t degrees of freedom for the hourly step
SCORE_CAP = 0.0         # latent scores cannot exceed the #1 score
DRIFT_TOP_ZERO = True   # top-3 games get no upward rank drift
DRIFT_HOURS = 48        # window for each game's recent score slope
DRIFT_SHRINK = 0.5      # shrinkage of the slope toward 0
DRIFT_CAP = 0.02        # cap on |slope| per hour
PLAY_G0 = 0.5           # 48 h log-playing growth above this earns a boost
PLAY_K = 0.024          # boost per unit of excess log-playing growth (/h)
PLAY_CAP = 0.040        # cap on the playing-growth drift boost (/h)
PLAY_WINDOW_H = 48      # lookback for the playing-growth reference read
PLAY_WINDOW_TOL_H = 12  # reference read must be 48 h +/- this (earliest 36 h)
MAX_AGE_DAYS = 7        # drop games unseen for longer than this
EXPIRY_DAYS = 28        # games leave the chart this long after creation
MAX_GAMES = 220         # simulation pool, best scores first
OFF_RANK = 110          # score floor for games off the chart
P_MIN = 0.007           # smallest probability written as a row
OUTSIDE_RANK = 15.0     # rank credited to probability mass outside top 10
# ---------------------------------------------------------------------------

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def map_score(rank):
    return -math.log(max(rank, 1))


def estimate_drift(series, t0):
    """Least-squares slope of a game's score against time over the last
    DRIFT_HOURS, shrunk toward 0 and capped. `series` is a list of
    (time, rank), oldest first."""
    pts = [
        (t, map_score(r))
        for (t, r) in series
        if (t0 - t).total_seconds() <= DRIFT_HOURS * 3600
    ]
    if len(pts) < 3:
        return 0.0
    n = float(len(pts))
    mt = sum((t - t0).total_seconds() / 3600.0 for t, _ in pts) / n
    ms = sum(s for _, s in pts) / n
    num = sum(((t - t0).total_seconds() / 3600.0 - mt) * (s - ms) for t, s in pts)
    den = sum(((t - t0).total_seconds() / 3600.0 - mt) ** 2 for t, _ in pts)
    if den <= 0:
        return 0.0
    slope = num / den
    slope *= DRIFT_SHRINK * n / (n + 2.0)
    return max(-DRIFT_CAP, min(DRIFT_CAP, slope))


def playing_growth(play_series, t0):
    """Log growth of a game's player count over the last PLAY_WINDOW_H:
    log(playing at the last read / playing at the read nearest t0-48 h
    within +/- PLAY_WINDOW_TOL_H). 0 when either count is missing."""
    if not play_series:
        return 0.0
    t_now, p_now = play_series[-1]
    if not p_now:
        return 0.0
    target = PLAY_WINDOW_H * 3600.0
    tol = PLAY_WINDOW_TOL_H * 3600.0
    best = None
    for t, p in play_series:
        dt = (t0 - t).total_seconds()
        if not p or abs(dt - target) > tol:
            continue
        if best is None or abs(dt - target) < abs((t0 - best[0]).total_seconds() - target):
            best = (t, p)
    if best is None or not best[1]:
        return 0.0
    return math.log(p_now / best[1])


def load_history(input_dir):
    task_path = os.path.join(input_dir, "task.json")
    with open(task_path, "r", encoding="utf-8") as fh:
        task = json.load(fh)
    reads = []
    hist_path = os.path.join(input_dir, "history.ndjson")
    with open(hist_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d["_t"] = parse_time(d["at"])
            reads.append(d)
    reads.sort(key=lambda r: r["_t"])
    return task, reads


def build_games(reads):
    """Return (games, last_chart_size): games is a list of
    (universe_id, score, drift, expiry_step_or_None, last_rank) with
    expiry measured in hourly steps from the last read; the caller fills
    the step once the forecast start is known."""
    last = reads[-1]
    t0 = last["_t"]
    last_chart = sorted(last["up_and_coming"], key=lambda x: x["rank"])
    hist = {}
    play_hist = {}
    for r in reads:
        for x in r["up_and_coming"]:
            hist.setdefault(x["universe_id"], []).append((r["_t"], x["rank"]))
            play_hist.setdefault(x["universe_id"], []).append(
                (r["_t"], x.get("playing"))
            )
    created = {}
    for r in reads:
        details = r.get("details") or {}
        for key, val in details.items():
            if val and val.get("created"):
                try:
                    created[int(key)] = parse_time(val["created"])
                except (ValueError, TypeError):
                    pass

    def expiry_of(uid):
        c = created.get(uid)
        if c is None:
            return None
        return c + timedelta(days=EXPIRY_DAYS)

    def make_game(uid, rank):
        drift = estimate_drift(hist[uid], t0)
        if DRIFT_TOP_ZERO and rank <= 3:
            drift = min(drift, 0.0)
        growth = playing_growth(play_hist.get(uid, []), t0)
        if growth > PLAY_G0:
            drift += min(PLAY_CAP, PLAY_K * (growth - PLAY_G0))
        return (uid, map_score(rank), drift, expiry_of(uid), rank)

    games = []
    off_floor = map_score(OFF_RANK) - 0.5
    for x in last_chart:
        games.append(make_game(x["universe_id"], x["rank"]))
    on_chart = {x["universe_id"] for x in last_chart}
    for uid, series in hist.items():
        if uid in on_chart:
            continue
        t_last, r_last = series[-1]
        age_h = (t0 - t_last).total_seconds() / 3600.0
        if age_h > MAX_AGE_DAYS * 24:
            continue
        s0 = map_score(r_last)
        if s0 < off_floor:
            continue
        games.append(make_game(uid, r_last))
    games.sort(key=lambda g: -g[1])
    return games[:MAX_GAMES], len(last_chart)


def forecast(task, reads):
    places = int(task.get("places", 10))
    window = task.get("window", {})
    slots = int(window.get("slots", 72))
    slot_minutes = int(window.get("slot_minutes", 60))
    window_start = parse_time(window["from"]) if window.get("from") else reads[-1]["_t"]
    window_end = window_start + timedelta(minutes=slots * slot_minutes)

    games, last_size = build_games(reads)
    if not games:
        return {"probabilities": [], "top10": []}
    uids = [g[0] for g in games]
    s0 = [g[1] for g in games]
    drift = [g[2] for g in games]
    sig = [
        SIGMA * SIGMA_TOP_MULT if s > map_score(11) else SIGMA for s in s0
    ]
    t0 = reads[-1]["_t"]

    # Hourly step at which each game ages out of the chart (None = never).
    # A game already past its 28 days at the last read is out: drop it.
    expiry_step = []
    keep_games = []
    for g in games:
        exp = g[3]
        if exp is None:
            expiry_step.append(None)
            keep_games.append(g)
        else:
            h = (exp - t0).total_seconds() / 3600.0
            step = int(math.floor(h))
            if step <= 0:
                continue
            expiry_step.append(step)
            keep_games.append(g)
    games = keep_games
    if not games:
        return {"probabilities": [], "top10": []}
    uids = [g[0] for g in games]
    s0 = [g[1] for g in games]
    drift = [g[2] for g in games]
    sig = [
        SIGMA * SIGMA_TOP_MULT if s > map_score(11) else SIGMA for s in s0
    ]
    t0 = reads[-1]["_t"]
    n_steps = int(math.ceil((window_end - t0).total_seconds() / 3600.0)) + 1
    if n_steps < 1:
        n_steps = 1
    # Which hourly step's state each slot reads (last grid time <= slot start).
    slot_of_step = {}
    for s in range(slots):
        slot_t = window_start + timedelta(minutes=s * slot_minutes)
        step = int((slot_t - t0).total_seconds() // 3600)
        step = min(max(step, 0), n_steps - 1)
        slot_of_step.setdefault(step, []).append(s)

    rng = random.Random(SEED)
    gi = range(len(games))
    counts = [[0] * places for _ in gi]
    scores = [list(s0) for _ in range(REPS)]
    # Tie-break jitter: the score cap pins several games at exactly 0.0,
    # and heapq.nlargest would break those exact ties by game index,
    # favouring the incumbent #1 every time. A tiny per-replicate jitter
    # (far below any genuine score gap, so it only ever decides exact
    # ties) splits tied places fairly instead. It is deterministic: its
    # own seeded stream, fixed per replicate, so two runs agree byte for
    # byte. (Without it the practice score is ~0.35 worse and the
    # incumbent #1 is over-credited; see REPORT.md.)
    jit_rng = random.Random(SEED ^ 0x5EED)
    jitter = [[jit_rng.random() * 1e-9 for _ in gi] for _ in range(REPS)]
    # Student-t(df) step, scaled to sd SIGMA: t = Z / sqrt(chi2/df),
    # chi2(df) = 2 * Gamma(df/2, 1); Var(t) = df / (df - 2).
    t_scale = math.sqrt((DF - 2.0) / DF) if DF > 2 else 1.0
    half_df = DF / 2.0
    for step in range(n_steps):
        if step > 0:
            expiring = [i for i, e in enumerate(expiry_step) if e == step]
            for r in range(REPS):
                row = scores[r]
                row = [
                    s + d + sg * t_scale * rng.gauss(0.0, 1.0)
                    / math.sqrt(rng.gammavariate(half_df, 2.0) / DF)
                    for s, d, sg in zip(row, drift, sig)
                ]
                if SCORE_CAP is not None:
                    row = [s if s < SCORE_CAP else SCORE_CAP for s in row]
                for i in expiring:
                    row[i] = -1e18
                scores[r] = row
        if step in slot_of_step:
            weight = len(slot_of_step[step])
            for r in range(REPS):
                row = scores[r]
                jrow = jitter[r]
                keyed = [s + j for s, j in zip(row, jrow)]
                top = heapq.nlargest(places, gi, key=keyed.__getitem__)
                for j, idx in enumerate(top):
                    counts[idx][j] += weight
    total = float(REPS * slots)

    probabilities = []
    for j in range(places):
        col = [(counts[i][j] / total, uids[i]) for i in range(len(games))]
        col.sort(key=lambda p_u: (-p_u[0], p_u[1]))
        keep = [(p, u) for (p, u) in col if p >= P_MIN][:100]
        # Drop rows whose probability is below the share an unnamed game
        # would receive at this place instead.
        for _ in range(3):
            s_sum = sum(p for p, _ in keep)
            unnamed_n = max(1, last_size - len(keep))
            residual = max(0.0, 1.0 - s_sum) / unnamed_n
            new_keep = [(p, u) for (p, u) in keep if p >= residual * 0.9]
            if len(new_keep) == len(keep):
                break
            keep = new_keep
        for p, u in keep:
            probabilities.append(
                {"universe_id": int(u), "place": j + 1, "p": round(float(p), 6)}
            )
    probabilities.sort(key=lambda row: (row["place"], -row["p"], row["universe_id"]))

    # top10: order by expected rank, mass outside the top 10 as OUTSIDE_RANK.
    exp_rank = {}
    for i, u in enumerate(uids):
        inside = sum(counts[i][j] / total * (j + 1) for j in range(places))
        mass = sum(counts[i][j] / total for j in range(places))
        exp_rank[u] = inside + (1.0 - mass) * OUTSIDE_RANK
    top10 = [
        int(u) for u, _ in sorted(exp_rank.items(), key=lambda kv: (kv[1], kv[0]))[:places]
    ]
    return {"probabilities": probabilities, "top10": top10}


def main():
    task, reads = load_history(INPUT_DIR)
    result = forecast(task, reads)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, task.get("output", "forecast.json"))
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(result, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
