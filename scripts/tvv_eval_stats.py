"""
uncertainty and robustness checks for the TVV (verifier_mc) held-out numbers, from saved scores only.

reads data/labeled/v5_eval_scores.json (TVV's per-turn q_value on the 300 v5_eval_ood episodes) and
reports/failure_detection_mc*.json. no model is loaded and nothing is retrained or re-scored, so this
runs on CPU in seconds. it does six things:

1. reproduces the per-game, pooled and macro-averaged turn-level AUC already reported.
2. puts 95% intervals on them with an episode-level cluster bootstrap: turns from one episode share
   one label and are strongly correlated, so episodes (not turns) are resampled, within each game.
3. a game-stratified AUC that only compares won and lost turns from the same game, weighted by each
   game's number of won-lost pairs. the pooled AUC also compares across games, so it can reward a
   score that merely separates easy games from hard ones.
4. drops evaluation episodes whose start state matches the training episode with the same index
   (same seed number; the env wrapper passes seeds 1, 2, 3, ... for both splits, and for some games
   the "test" fold reproduces the training game) and recomputes AUC on what is left.
5. lead time for lost episodes, replaying scripts/failure_detection_eval.py's online detector
   (5-turn trailing mean, flag at the first drop below threshold) with the thresholds it saved.
6. reads back how the saved F1 / precision / recall summaries were aggregated.

usage, from the repo root:
    python -m scripts.tvv_eval_stats [--resamples 2000] [--seed 0] [--out reports/tvv_eval_stats.md]
"""

import argparse
import json
import os
import random
import statistics

import numpy as np

from scripts.failure_detection_eval import first_flag_turn, trailing_mean

GAMES = ["coin", "mapreader", "peckingorder", "simonsays"]
SCORES_PATH = "data/labeled/v5_eval_scores.json"
EVAL_DIR = "data/rollouts/v5_eval_ood"
TRAIN_DIR = "data/rollouts/v5_train"
FD_PATH = "reports/failure_detection_mc.json"
FD_PERGAME_PATH = "reports/failure_detection_mc_pergame.json"
K = 5  # trailing-mean window, as in failure_detection_eval.py's --k default and the saved results
FEW_LOST_EPISODES = 10  # below this many lost episodes in a game, its numbers are flagged as unstable


# -- auc ------------------------------------------------------------------------------------------

def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks with ties sharing their average rank (what the mann-whitney u statistic needs)."""
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.empty(len(values), dtype=float)
    boundaries = np.flatnonzero(np.diff(sorted_vals)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(values)]))
    for s, e in zip(starts, ends):
        ranks[order[s:e]] = (s + e + 1) / 2.0
    return ranks


def auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """probability a random won turn outscores a random lost turn, ties counted half. None if either
    class is missing (a game, or a bootstrap resample of it, with no lost turns has no AUC)."""
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _average_ranks(scores)
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def n_pairs(labels: np.ndarray) -> int:
    n_pos = int(labels.sum())
    return n_pos * (len(labels) - n_pos)


# -- data -----------------------------------------------------------------------------------------

def load_episodes() -> dict[str, list[dict]]:
    """{game: [episode]}, each episode with numpy arrays of its per-turn q_value and labels.

    games are kept in the order they first appear in the scores file. that order matters for
    reproducing the saved calibration/test splits: failure_detection_eval.py shares one rng across
    games while shuffling, so each game's split depends on the games shuffled before it."""
    with open(SCORES_PATH) as f:
        raw = json.load(f)
    by_game: dict[str, list[dict]] = {}
    for ep in raw:
        by_game.setdefault(ep["game"], [])
    for ep in raw:
        scores = np.array([t["q_value"] for t in ep["turns"]], dtype=float)
        by_game[ep["game"]].append({
            **ep,
            "scores": scores,
            "labels": np.full(len(scores), int(bool(ep["won"])), dtype=int),
        })
    return by_game


def concat(episodes: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    if not episodes:
        return np.array([]), np.array([], dtype=int)
    return np.concatenate([e["scores"] for e in episodes]), np.concatenate([e["labels"] for e in episodes])


def summary_aucs(by_game: dict[str, list[dict]]) -> dict[str, float | None]:
    """per-game, pooled, macro-averaged and game-stratified AUC for one set of episodes."""
    out: dict[str, float | None] = {}
    pooled_s, pooled_l = [], []
    weighted, total_pairs = 0.0, 0
    for game in GAMES:
        s, l = concat(by_game[game])
        a = auc(s, l)
        out[game] = a
        pooled_s.append(s)
        pooled_l.append(l)
        if a is not None:
            weighted += a * n_pairs(l)
            total_pairs += n_pairs(l)
    out["pooled"] = auc(np.concatenate(pooled_s), np.concatenate(pooled_l))
    per_game = [out[g] for g in GAMES]
    out["macro"] = None if any(a is None for a in per_game) else sum(per_game) / len(per_game)
    out["stratified"] = weighted / total_pairs if total_pairs else None
    return out


def bootstrap(by_game: dict[str, list[dict]], resamples: int, seed: int) -> dict[str, dict]:
    """episode-level cluster bootstrap, resampling whole episodes with replacement within each game."""
    rng = random.Random(seed)
    draws: dict[str, list[float]] = {}
    undefined: dict[str, int] = {}
    for _ in range(resamples):
        sample = {g: [rng.choice(eps) for _ in eps] for g, eps in by_game.items()}
        for key, value in summary_aucs(sample).items():
            if value is None:
                undefined[key] = undefined.get(key, 0) + 1
            else:
                draws.setdefault(key, []).append(value)
    result = {}
    for key in GAMES + ["pooled", "macro", "stratified"]:
        values = sorted(draws.get(key, []))
        if values:
            lo = values[int(0.025 * (len(values) - 1))]
            hi = values[int(0.975 * (len(values) - 1))]
        else:
            lo = hi = None
        result[key] = {"lo": lo, "hi": hi, "undefined": undefined.get(key, 0), "valid": len(values)}
    return result


# -- overlap with training ------------------------------------------------------------------------

def start_state(path: str) -> tuple[str, str] | None:
    """(task goal, first observation), the same comparison the audit used. None if the episode has
    no turns, which then cannot be said to match anything."""
    with open(path) as f:
        traj = json.load(f)
    if not traj["turns"]:
        return None
    return traj["task_goal"], traj["turns"][0]["obs_before"]


def matches_training(game: str, task_id: str) -> bool:
    """does eval episode eval_ood_i start exactly like training episode train_i (same seed number)?"""
    index = task_id.rsplit("_", 1)[-1]
    eval_state = start_state(os.path.join(EVAL_DIR, game, f"{task_id}.json"))
    train_path = os.path.join(TRAIN_DIR, game, f"train_{index}.json")
    if eval_state is None or not os.path.exists(train_path):
        return False
    return eval_state == start_state(train_path)


# -- lead time ------------------------------------------------------------------------------------

def calibration_test_split(by_game: dict[str, list[dict]], seed: int) -> tuple[list[dict], list[dict]]:
    """the exact per-game half split failure_detection_eval.run_multi_seed() makes for this seed:
    one rng shared across games, visited in scores-file order (see load_episodes)."""
    rng = random.Random(seed)
    calibration, test = [], []
    for game in by_game:
        eps = sorted(by_game[game], key=lambda e: e["task_id"])
        rng.shuffle(eps)
        half = len(eps) // 2
        calibration.extend(eps[:half])
        test.extend(eps[half:])
    return calibration, test


def flag_turn(ep: dict, threshold: float) -> int | None:
    return first_flag_turn(trailing_mean(list(ep["scores"]), K), threshold)


def lead_times(by_game: dict[str, list[dict]], fd: dict) -> tuple[dict, list[dict]]:
    """per game, lead times of lost test-half episodes pooled over the five saved splits, plus a
    check that this replay reproduces the saved confusion counts for each split.

    lead time is the script's own definition,
    total_steps - 1 - flag_turn. the script leaves episodes of 2 or fewer steps out of its lead-time
    mean; they are counted here as well so nothing is hidden."""
    per_game: dict[str, dict] = {
        g: {"leads": [], "flag_turns": [], "never": 0, "lost": 0, "short": 0, "won": 0, "false_alarms": 0} for g in GAMES
    }
    checks = []
    for run in fd["per_seed_test_results"]:
        _, test = calibration_test_split(by_game, run["seed"])
        tp = fp = tn = fn = 0
        for ep in test:
            flagged = flag_turn(ep, run["threshold"])
            lost = not ep["won"]
            tp += flagged is not None and lost
            fp += flagged is not None and not lost
            fn += flagged is None and lost
            tn += flagged is None and not lost
            stats = per_game[ep["game"]]
            if not lost:
                stats["won"] += 1
                stats["false_alarms"] += flagged is not None
                continue
            stats["lost"] += 1
            if flagged is not None:
                stats["flag_turns"].append(flagged)
            if flagged is None:
                stats["never"] += 1
            elif ep["total_steps"] <= 2:
                stats["short"] += 1
            else:
                stats["leads"].append(ep["total_steps"] - 1 - flagged)
        saved = (run["tp"], run["fp"], run["tn"], run["fn"])
        checks.append({"seed": run["seed"], "threshold": run["threshold"], "replayed": (tp, fp, tn, fn), "saved": saved})
    return per_game, checks


# -- report ---------------------------------------------------------------------------------------

def fmt(x: float | None, digits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def ci(b: dict) -> str:
    text = f"[{fmt(b['lo'])}, {fmt(b['hi'])}]"
    if b["undefined"]:
        text += f" ({b['undefined']} of {b['undefined'] + b['valid']} resamples undefined)"
    return text


def build_report(resamples: int, seed: int) -> str:
    by_game = load_episodes()
    lines: list[str] = []
    add = lines.append
    lost_eps = {g: sum(1 for e in by_game[g] if not e["won"]) for g in GAMES}
    few = [g for g in GAMES if lost_eps[g] < FEW_LOST_EPISODES]

    add("# TVV held-out evaluation: uncertainty and robustness\n")
    add(f"Scores: `{SCORES_PATH}`. Bootstrap: {resamples} resamples, seed {seed}, whole episodes resampled "
        "within each game. Intervals are 2.5th to 97.5th percentiles.\n")

    # 1 and 2
    point = summary_aucs(by_game)
    boot = bootstrap(by_game, resamples, seed)
    expected = {"coin": 0.709, "mapreader": 0.782, "peckingorder": 0.695, "simonsays": 0.562, "pooled": 0.844, "macro": 0.687}
    add("## 1-3. AUC with 95% cluster-bootstrap intervals\n")
    add("| Scope | Episodes (won / lost) | Lost turns | AUC | Expected | Reproduced | 95% CI |")
    add("|---|---|---|---|---|---|---|")
    for game in GAMES:
        eps = by_game[game]
        _, l = concat(eps)
        flag = " (few lost)" if game in few else ""
        add(f"| {game}{flag} | {len(eps)} ({len(eps) - lost_eps[game]} / {lost_eps[game]}) | {int((l == 0).sum())} | "
            f"{fmt(point[game])} | {expected[game]:.3f} | {'yes' if round(point[game], 3) == expected[game] else 'NO'} | {ci(boot[game])} |")
    for key, label in (("pooled", "pooled (all turns)"), ("macro", "macro (mean of 4 games)"), ("stratified", "game-stratified")):
        exp = f"{expected[key]:.3f}" if key in expected else ""
        rep = ("yes" if round(point[key], 3) == expected[key] else "NO") if key in expected else ""
        add(f"| {label} | 300 | {sum(int((concat(by_game[g])[1] == 0).sum()) for g in GAMES)} | {fmt(point[key])} | {exp} | {rep} | {ci(boot[key])} |")
    add("")
    add("Game-stratified AUC compares won and lost turns only within the same game; each game is weighted by its "
        "number of won-lost turn pairs.")
    pairs = {g: n_pairs(concat(by_game[g])[1]) for g in GAMES}
    total = sum(pairs.values())
    add("Weights: " + ", ".join(f"{g} {100 * pairs[g] / total:.1f}%" for g in GAMES) + ".")
    if few:
        add(f"\n**Flag:** {', '.join(few)} has fewer than {FEW_LOST_EPISODES} lost episodes, so its AUC and interval rest "
            "on very few independent failures. A bootstrap resample with no lost episode has no AUC and is dropped "
            "from that game's interval and from the macro average (counted in the table).")

    # 4
    add("\n## 4. Excluding evaluation episodes that start like their same-index training episode\n")
    kept = {g: [e for e in by_game[g] if not matches_training(g, e["task_id"])] for g in GAMES}
    kept_point = summary_aucs(kept)
    kept_boot = bootstrap(kept, resamples, seed)
    add("| Scope | Kept / total | Kept (won / lost) | AUC, all episodes | AUC, kept only | 95% CI, kept only |")
    add("|---|---|---|---|---|---|")
    for game in GAMES:
        k_lost = sum(1 for e in kept[game] if not e["won"])
        add(f"| {game} | {len(kept[game])} / {len(by_game[game])} | {len(kept[game]) - k_lost} / {k_lost} | "
            f"{fmt(point[game])} | {fmt(kept_point[game])} | {ci(kept_boot[game])} |")
    for key, label in (("pooled", "pooled"), ("macro", "macro"), ("stratified", "game-stratified")):
        add(f"| {label} | {sum(len(v) for v in kept.values())} / 300 | | {fmt(point[key])} | {fmt(kept_point[key])} | {ci(kept_boot[key])} |")
    add("\nA match means the same task goal and an identical first observation as `v5_train/<game>/train_<i>.json` "
        "for `eval_ood_<i>`. It is strong evidence of the same game instance, not proof.")

    # 5
    with open(FD_PATH) as f:
        fd = json.load(f)
    per_game, checks = lead_times(by_game, fd)
    add("\n## 5. Lead time for lost episodes (test halves of the 5 saved splits, pooled)\n")
    add("| Game | Lost test episodes (over 5 splits) | Never flagged | Flagged, steps <= 2 | Lead time, mean | Lead time, median | First flag, median turn index (0 = first turn) | Flagged on the first turn | Won episodes falsely flagged |")
    add("|---|---|---|---|---|---|---|---|---|")
    all_leads, all_lost, all_never = [], 0, 0
    for game in GAMES:
        s = per_game[game]
        all_leads += s["leads"]
        all_lost += s["lost"]
        all_never += s["never"]
        flag = " (few lost)" if game in few else ""
        share = f"{s['never']} ({100 * s['never'] / s['lost']:.1f}%)" if s["lost"] else "0"
        first = sum(1 for t in s["flag_turns"] if t == 0)
        add(f"| {game}{flag} | {s['lost']} | {share} | {s['short']} | "
            f"{fmt(statistics.mean(s['leads']) if s['leads'] else None, 1)} | "
            f"{fmt(statistics.median(s['leads']) if s['leads'] else None, 1)} | "
            f"{fmt(statistics.median(s['flag_turns']) if s['flag_turns'] else None, 0)} | "
            f"{first} of {len(s['flag_turns'])} | {s['false_alarms']} of {s['won']} |")
    all_flags = [t for g in GAMES for t in per_game[g]["flag_turns"]]
    all_won = sum(per_game[g]["won"] for g in GAMES)
    all_fa = sum(per_game[g]["false_alarms"] for g in GAMES)
    add(f"| all | {all_lost} | {all_never} ({100 * all_never / all_lost:.1f}%) | | "
        f"{fmt(statistics.mean(all_leads), 1)} | {fmt(statistics.median(all_leads), 1)} | "
        f"{fmt(statistics.median(all_flags), 0)} | {sum(1 for t in all_flags if t == 0)} of {len(all_flags)} | {all_fa} of {all_won} |")
    add("\nLead time is the script's own definition, `total_steps - 1 - flag_turn`: turns between the first flag and "
        "the last turn. Each lost episode appears once per split whose test half contains it, so an episode can be "
        "counted up to 5 times. Thresholds are each split's saved threshold (applied to the same split's test half).")
    add("\nReplay check against the saved confusion counts (tp, fp, tn, fn):\n")
    add("| Split seed | Threshold | Replayed | Saved | Match |")
    add("|---|---|---|---|---|")
    for c in checks:
        add(f"| {c['seed']} | {c['threshold']:.4f} | {c['replayed']} | {c['saved']} | {'yes' if c['replayed'] == c['saved'] else 'NO'} |")

    # 6
    add("\n## 6. How the saved F1 / precision / recall were aggregated\n")
    runs = fd["per_seed_test_results"]
    f1s = [r["f1"] for r in runs]
    precs = [r["precision"] for r in runs]
    recs = [r["recall"] for r in runs]
    tp, fp, fn = (sum(r[k] for r in runs) for k in ("tp", "fp", "fn"))
    micro_p, micro_r = tp / (tp + fp), tp / (tp + fn)
    mean_p, mean_r = statistics.mean(precs), statistics.mean(recs)
    add("| Quantity | Value | How it is computed |")
    add("|---|---|---|")
    add(f"| Saved `f1_mean` | {fd['f1_mean']:.4f} | mean over the 5 splits of each split's F1 (recomputed: {statistics.mean(f1s):.4f}) |")
    add(f"| Saved `precision_mean` | {fd['precision_mean']:.4f} | mean over the 5 splits (recomputed: {mean_p:.4f}) |")
    add(f"| Saved `recall_mean` | {fd['recall_mean']:.4f} | mean over the 5 splits (recomputed: {mean_r:.4f}) |")
    add(f"| F1 range over splits | {min(f1s):.4f} to {max(f1s):.4f} | saved `f1_min`, `f1_max` |")
    add(f"| F1 standard deviation over splits | {statistics.stdev(f1s):.4f} | sample SD of the 5 split F1s (not saved; computed here) |")
    add(f"| F1 of the mean precision and recall | {2 * mean_p * mean_r / (mean_p + mean_r):.4f} | not what was reported; shown because it differs from `f1_mean` |")
    add(f"| F1 from summed counts over splits | {2 * micro_p * micro_r / (micro_p + micro_r):.4f} | not what was reported |")
    add("\nEach split's pooled F1 treats all four games' test-half episodes as one set at that split's single threshold "
        "(`scripts/failure_detection_eval.py`, `evaluate_at_threshold` on the pooled `test` list). No bootstrap was "
        "saved; the only spread available is across the 5 splits, and the splits share episodes, so that spread "
        "understates the true uncertainty.")
    if os.path.exists(FD_PERGAME_PATH):
        with open(FD_PERGAME_PATH) as f:
            pg = json.load(f)
        same = [round(a["threshold"], 6) for a in pg["per_seed_test_results"]] == [round(r["threshold"], 6) for r in runs]
        add(f"\n`{FD_PERGAME_PATH}` uses the same seeds and {'the same' if same else 'DIFFERENT'} thresholds. Per game, "
            "over the 5 splits:\n")
        add("| Game | F1 mean (min to max) | Precision mean | Recall mean | Lost test episodes per split |")
        add("|---|---|---|---|---|")
        for game in GAMES:
            s = pg["per_game"][game]
            flag = " (few lost)" if game in few else ""
            add(f"| {game}{flag} | {s['f1_mean']:.3f} ({s['f1_min']:.3f} to {s['f1_max']:.3f}) | "
                f"{s['precision_mean']:.3f} | {s['recall_mean']:.3f} | {s['n_test_lost_per_seed']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="reports/tvv_eval_stats.md")
    args = parser.parse_args()
    report = build_report(args.resamples, args.seed)
    print(report)
    with open(args.out, "w") as f:
        f.write(report)
    print(f"written to {args.out}")
