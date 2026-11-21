"""Steps 6 and 7: the results tables.

Scores every method on one group of test users, under both markings of the
test answers, and prints the tables that go in the README.

    uv run python src/final_report.py                 # choosing group
    uv run python src/final_report.py --group final   # the reported numbers

Honesty rule: every decision (which model, how many epochs, which baseline)
was made on the CHOOSING group. The final group is scored here only to report
what was already decided. Use the choosing group while checking this script.

Methods:
  random            what picking videos at random scores on average
  shortest first    everyone gets the shortest videos (a pure length shortcut)
  longest first     everyone gets the longest videos
  like rate         step 5's popularity baseline (same list for everyone)
  similar users     a counting method, no neural network: find training users
                    whose likes agree with this user's, and recommend what
                    they liked (explained in taste_check.py)
  model rows        saved two-tower models from outputs/. The main model was
                    trained five times with different seeds; its row shows the
                    average, and after the +- how much the five runs differ.
                    Step 7 adds four rows: the in-batch model and the mixed
                    model, each without and with the logQ correction (all IDs
                    only, so the correction is the only difference in a pair).

Markings (defined in prepare_data.py):
  liked       the main marking. Its bars come from training data only.
  liked_even  second table: the test answers evened out by video length.
  finished    the rejected "watched to the end" label. Shown last, to explain
              why it was rejected.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import load_user_features, load_video_tags
from metrics import (candidate_videos, grade_scores, load_liked_by_user,
                     split_test_users)
from model import TwoTower
from popularity import popularity_table
from taste_check import centre, neighbour_scores

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs"

KS = (10, 50, 100)
SHORT_SECONDS = 5  # the short-video check introduced in step 5
MAIN_RUNS = [f"main_seed{seed}" for seed in range(5)]
# Seed 0 ran for 40 epochs to see the whole curve; its best epoch was 9.
# Seeds 1-4 ran for 12 epochs, which covers that peak.
LOGQ_RUNS = ["s7_mixed_logq"] + [f"s7_mixed_logq_seed{seed}" for seed in range(1, 5)]
# (row name, saved runs in outputs/). Several runs = same recipe, other seeds.
MODEL_ROWS = [
    ("model: in-batch (run 1)", ["inbatch"]),
    ("model: + 1 skipped (run 2)", ["hardneg"]),
    ("in-batch, IDs only", ["s7_inbatch"]),
    ("in-batch + logQ", ["s7_inbatch_logq"]),
    ("mixed", ["s7_mixed"]),
    ("mixed + logQ", LOGQ_RUNS),
    ("model: liked vs skipped", MAIN_RUNS),
    ("same model, old label", ["oldlabel"]),
]
MARKINGS = [
    ("liked", "MAIN MARKING (bars from training data only)", None),
    ("liked_even", "SECOND MARKING (test answers evened out by video length)", None),
    ("finished", 'OLD LABEL ("watched to the end"), for the explanation only',
     ["shortest first", "model: liked vs skipped", "same model, old label"]),
]


def random_expected(users, n_candidates, liked_by_user):
    """What random picks score on average. No simulation needed: each pick is
    a hit with chance (videos this user liked) / (all candidate videos)."""
    worth = [1 / math.log2(position + 1) for position in range(1, 11)]
    recall, ndcg = {k: [] for k in KS}, []
    for user in users:
        n_liked = len(liked_by_user[user])
        if n_liked == 0:
            continue  # can't be graded, same rule as the scorer
        chance = n_liked / n_candidates
        for k in KS:
            recall[k].append(k * chance / min(n_liked, k))
        ndcg.append(chance * sum(worth) / sum(worth[:min(n_liked, 10)]))
    return {**{k: float(np.mean(v)) for k, v in recall.items()},
            "auc": 0.5, "ndcg@10": float(np.mean(ndcg))}


def similar_user_scores(train, test, users, candidates):
    """The "similar users" counting method from taste_check.py, for any group.

    Match each test user to training users on the videos OUTSIDE the test set,
    then score each test video by what those matched users thought of it.
    Only training data is used.
    """
    n_users, n_videos = train["user_id"].max() + 1, train["video_id"].max() + 1
    cand = np.array(candidates)
    other = np.setdiff1d(np.arange(n_videos), cand)
    # Training users who are not test users: the only ones who watched the
    # test videos in the training data.
    them = np.setdiff1d(np.arange(n_users), test["user_id"].unique())
    watched = np.zeros((n_users, n_videos), dtype=bool)
    watched[train["user_id"], train["video_id"]] = True
    liked = np.zeros_like(watched)
    yes = train[train["liked"] == 1]
    liked[yes["user_id"], yes["video_id"]] = True
    me = np.array(users)
    x_me, _ = centre(liked[np.ix_(me, other)], watched[np.ix_(me, other)])
    x_them, _ = centre(liked[them], watched[them])
    watched_them = watched[np.ix_(them, cand)].astype(np.float32)
    return neighbour_scores(x_me, x_them[:, other], x_them[:, cand], watched_them)


@torch.no_grad()
def model_scores(name, users, candidates, user_codes, feature_sizes,
                 video_tags, n_tag_codes):
    """Load a saved model and score every user against every candidate."""
    info = json.loads((OUT / f"{name}.json").read_text())
    # Runs from before the "IDs only" option existed always used the features.
    use_features = info.get("config", {}).get("use_features", True)
    model = TwoTower(len(user_codes), feature_sizes, user_codes,
                     len(video_tags), n_tag_codes, video_tags,
                     use_features=use_features)
    model.load_state_dict(torch.load(OUT / f"{name}.pt", map_location="cpu"))
    model.eval()
    scores = model.score_matrix(torch.tensor(users), torch.tensor(candidates))
    return scores.numpy(), info.get("best_epoch")


def show(values):
    """One run: the number. Several runs: their average, +- their spread
    (standard deviation: the typical distance of one run from the average)."""
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{np.mean(values):.3f} +-{np.std(values, ddof=1):.3f}"


def print_table(title, rows):
    """rows: list of (name, [one score dict per run])."""
    columns = [(10, "recall@10"), (50, "recall@50"), (100, "recall@100"),
               ("ndcg@10", "NDCG@10"), ("auc", "AUC")]
    print(f"\n{title}")
    print(f"  {'method':28s}" + "".join(f"{label:>15s}" for _, label in columns))
    for name, runs in rows:
        print(f"  {name:28s}" +
              "".join(f"{show([r[key] for r in runs]):>15s}" for key, _ in columns))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--group", choices=["choosing", "final"], default="choosing")
    args = p.parse_args()

    train = pd.read_parquet(PROCESSED / "train.parquet",
                            columns=["user_id", "video_id", "video_duration", "liked"])
    test = pd.read_parquet(PROCESSED / "test.parquet")
    choosing, final = split_test_users(test)
    users = sorted(final if args.group == "final" else choosing)
    candidates = [int(v) for v in candidate_videos(test)]
    answers = {label: {u: {int(v) for v in s}
                       for u, s in load_liked_by_user(test, label).items()}
               for label, _, _ in MARKINGS}
    print(f"{args.group.upper()} GROUP: {len(users)} users, "
          f"{len(candidates):,} candidate videos")

    # --- one grid of scores [users, candidates] per method (or per run) ---
    table = popularity_table(train, candidates)
    seconds = table["seconds"].to_numpy()

    def same_for_everyone(video_scores):
        return np.tile(video_scores, (len(users), 1))

    like_rate = np.nan_to_num(table["like_rate"].to_numpy(), nan=-1.0)  # untrusted last
    grids = [
        ("shortest first", [same_for_everyone(-seconds)]),
        ("longest first", [same_for_everyone(seconds)]),
        ("like rate (popularity)", [same_for_everyone(like_rate)]),
        ("similar users (counting)",
         [similar_user_scores(train, test, users, candidates)]),
    ]
    user_codes, feature_sizes = load_user_features()
    video_tags, n_tag_codes = load_video_tags()
    best_epochs = {}
    for row_name, runs in MODEL_ROWS:
        row = []
        for run in runs:
            scores, best_epochs[run] = model_scores(
                run, users, candidates, user_codes, feature_sizes,
                video_tags, n_tag_codes)
            row.append(scores)
        grids.append((row_name, row))

    # --- the tables ---
    results = {}
    for label, title, only in MARKINGS:
        rows = [("random (expected)",
                 [random_expected(users, len(candidates), answers[label])])]
        for name, row in grids:
            if only is None or name in only:
                rows.append((name, [grade_scores(g, users, candidates,
                                                 answers[label], KS)[0] for g in row]))
        print_table(f"{title}, {args.group} group", rows)
        results[label] = {name: [{str(k): v for k, v in r.items()} for r in runs]
                          for name, runs in rows}

    # --- rows trained several times: each run on its own, to see the spread ---
    for row_name, runs in MODEL_ROWS:
        if len(runs) == 1:
            continue
        print(f"\nTHE {len(runs)} RUNS OF \"{row_name}\" (main marking, {args.group} group)")
        for run, r in zip(runs, results["liked"][row_name]):
            print(f"  {run}: best epoch {best_epochs[run]:2d}   recall@10 {r['10']:.3f}   "
                  f"NDCG@10 {r['ndcg@10']:.3f}   AUC {r['auc']:.3f}")

    # --- what each method recommends ---
    short = seconds < SHORT_SECONDS
    short_ids = set(np.array(candidates)[short].tolist())
    kept = [v for v in candidates if v not in short_ids]
    watch_count = table["watched"].to_numpy()
    print(f"\nWHAT EACH METHOD PUTS IN THE TOP 10 ({args.group} group). "
          f"For reference: {short.sum()} of {len(candidates):,} candidates "
          f"({short.mean():.1%}) are under {SHORT_SECONDS}s; the median candidate "
          f"is {np.median(seconds):.1f}s long and was watched "
          f"{np.median(watch_count):,.0f} times in training.")
    print(f"  {'method':28s}{'under 5s':>10s}{'median length':>15s}"
          f"{'median watches':>16s}{'distinct videos':>17s}"
          f"{'recall@10 without <5s':>24s}")
    picks_report = {}
    for name, row in grids:
        stats = []
        for g in row:
            top10 = np.argsort(-g, axis=1, kind="stable")[:, :10]
            without = {}
            for label in ["liked", "liked_even"]:
                liked_kept = {u: answers[label][u] - short_ids for u in users}
                without[label] = grade_scores(g[:, ~short], users, kept,
                                              liked_kept, KS)[0][10]
            stats.append({"short_share": float(short[top10].mean()),
                          "median_seconds": float(np.median(seconds[top10])),
                          "median_watches": float(np.median(watch_count[top10])),
                          "distinct": int(len(np.unique(top10))),
                          "recall@10_without_short": without["liked"],
                          "recall@10_without_short_even": without["liked_even"]})
        mean = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
        picks_report[name] = stats
        print(f"  {name:28s}{mean['short_share']:>10.1%}{mean['median_seconds']:>14.1f}s"
              f"{mean['median_watches']:>16,.0f}{mean['distinct']:>17,.0f}"
              f"{mean['recall@10_without_short']:>24.3f}")

    out = OUT / f"final_report_{args.group}.json"
    out.write_text(json.dumps({"group": args.group, "n_users": len(users),
                               "scores": results, "picks": picks_report,
                               "best_epochs": best_epochs}, indent=2))
    print(f"\nsaved to outputs/{out.name}")


if __name__ == "__main__":
    main()
