"""Compare three definitions of "liked" before committing to one.

A. watch_ratio >= 1.0                         (watched to the end once)
B. length-aware: total engagement in the top third for videos of similar length
C. watch_ratio >= 2.0                         (watched twice)

For each, on the test table:
  - liked rate by video length (flat = length no longer drives the label)
  - recall@10 of random picks
  - recall@10 of the "10 shortest videos for everyone" shortcut
If a dumb length rule scores far above random, the label rewards length, not taste.

Run:  uv run python -I src/compare_labels.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "KuaiRec" / "data"
PAIR = ["user_id", "video_id"]
# Length groups for option B; override with:  ... compare_labels.py 50
N_GROUPS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
TOP_SHARE = 1 / 3  # option B: the top third within each length group "wins"
K = 10


def load_train_engagement():
    """Training pairs with total engagement = sum of watch_ratio over all watches."""
    cols = ["user_id", "video_id", "video_duration", "timestamp", "watch_ratio"]
    big = pd.read_csv(RAW / "big_matrix.csv", usecols=cols).drop_duplicates()
    return big.groupby(PAIR).agg(
        total=("watch_ratio", "sum"),
        max_ratio=("watch_ratio", "max"),
        video_duration=("video_duration", "first"),
    ).reset_index()


def length_group_edges(train):
    """Split the training VIDEOS (not watches) into N_GROUPS equal-size length groups."""
    durations = train.groupby("video_id")["video_duration"].first()
    edges = np.unique(np.quantile(durations, np.linspace(0, 1, N_GROUPS + 1)))
    edges[0], edges[-1] = -np.inf, np.inf  # catch any video outside the training range
    return edges


def score_label(test, liked, name):
    """Print the length breakdown, random baseline, and shortest-video shortcut."""
    t = test.assign(liked=liked.astype(int))
    seconds = t["video_duration"] / 1000
    coarse = pd.cut(seconds, [0, 7, 15, 30, 60, np.inf],
                    labels=["<7s", "7-15s", "15-30s", "30-60s", ">60s"])
    by_len = t.groupby(coarse, observed=True)["liked"].mean()

    n_liked = t.groupby("user_id")["liked"].sum()
    n_seen = t.groupby("user_id").size()
    denom = np.minimum(n_liked, K)
    ok = denom > 0  # users with no likes can't be scored

    # Random picks: expected hits = K * (share of the user's videos they liked).
    random_recall = (K * n_liked / n_seen / denom)[ok].mean()

    # Shortcut: the same K shortest candidate videos for every user.
    shortest = (t.drop_duplicates("video_id")
                 .nsmallest(K, "video_duration")["video_id"])
    hits = t[t["video_id"].isin(shortest)].groupby("user_id")["liked"].sum()
    hits = hits.reindex(n_liked.index, fill_value=0)
    shortcut_recall = (hits / denom)[ok].mean()

    print(f"\n--- {name} ---")
    print(f"liked share overall: {t['liked'].mean():.1%}   "
          f"users with 0 likes: {(~ok).sum()}, with <{K}: {(n_liked < K).sum()}")
    print("liked rate by length: " +
          "  ".join(f"{k} {v:.1%}" for k, v in by_len.items()))
    print(f"recall@{K}: random {random_recall:.3f}   "
          f"'{K} shortest videos' {shortcut_recall:.3f}")


def main():
    train = load_train_engagement()
    test = pd.read_parquet(ROOT / "data" / "processed" / "test.parquet")

    # Option B thresholds, learned from TRAINING data only.
    edges = length_group_edges(train)
    train_group = pd.cut(train["video_duration"], edges, labels=False)
    bars = train.groupby(train_group)["total"].quantile(1 - TOP_SHARE)
    test_group = pd.cut(test["video_duration"], edges, labels=False)
    # In the test table each pair has one watch, so total = watch_ratio.
    liked_b = test["watch_ratio"] >= test_group.map(bars)

    print("Option B bars (top-third engagement per length group, from training):")
    lo = edges[:-1] / 1000
    hi = edges[1:] / 1000
    for g, bar in bars.items():
        print(f"  {lo[g]:6.1f}s to {hi[g]:6.1f}s : liked if total watch_ratio >= {bar:.2f}")

    score_label(test, test["watch_ratio"] >= 1.0, "A: watch_ratio >= 1.0")
    score_label(test, liked_b, "B: top third within length group")
    score_label(test, test["watch_ratio"] >= 2.0, "C: watch_ratio >= 2.0")

    liked_b_train = train["total"] >= train_group.map(bars)
    print(f"\n(option B on the training table: {liked_b_train.mean():.1%} liked)")


if __name__ == "__main__":
    main()
