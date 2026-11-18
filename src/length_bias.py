"""How much of the short-video effect is mechanical bias vs real preference?

Idea: completion depends on video length mechanically (short videos finish
before you swipe, and they loop). Pressing like / save / share takes the same
effort on any length of video. So compare, by video length:
    completion rate   vs   like, save, share, comment, follow, "see less" rates
If completion falls steeply with length but the explicit actions don't, most
of the gap is mechanical bias.

Source: item_daily_features.csv, platform-wide counts per video per day (all
Kuaishou users, not only the 7,176 in our tables). We sum each video's days.

Run:  uv run python -I src/length_bias.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "KuaiRec" / "data"
PROCESSED = ROOT / "data" / "processed"

# Rate name -> column counting users who did that action (per day).
ACTIONS = {
    "completed": "complete_play_user_num",
    "liked": "like_user_num",
    "saved": "collect_user_num",
    "shared": "share_user_num",
    "commented": "comment_user_num",
    "followed": "follow_user_num",
    "see_less": "reduce_similar_user_num",
}
BUCKETS = [0, 7, 15, 30, 60, np.inf]
LABELS = ["<7s", "7-15s", "15-30s", "30-60s", ">60s"]


def spearman(a, b):
    """Spearman correlation = ordinary correlation of the two columns' ranks."""
    return a.rank().corr(b.rank())


def video_rates():
    """One row per video: its length and each action as a share of its players."""
    cols = ["video_id", "play_user_num"] + list(ACTIONS.values())
    daily = pd.read_csv(RAW / "item_daily_features.csv", usecols=cols)
    totals = daily.groupby("video_id").sum()

    # Use the same video length as the rest of the project (from the watch logs).
    train = pd.read_parquet(PROCESSED / "train.parquet")
    length = train.groupby("video_id")["video_duration"].first() / 1000
    totals = totals.join(length.rename("seconds"), how="inner")
    totals = totals[totals["play_user_num"] > 0]

    for name, col in ACTIONS.items():
        totals[name] = totals[col] / totals["play_user_num"]
    return totals


def by_length(v):
    """Pooled rate per length bucket: total actions / total players in the bucket."""
    bucket = pd.cut(v["seconds"], BUCKETS, labels=LABELS)
    g = v.groupby(bucket, observed=True)
    out = pd.DataFrame({"videos": g.size()})
    for name, col in ACTIONS.items():
        out[name] = g[col].sum() / g["play_user_num"].sum()
    return out


def main():
    v = video_rates()
    print(f"videos analysed: {len(v):,}  (platform-wide plays summed: "
          f"{v['play_user_num'].sum():,.0f})")

    print("\nRATES BY VIDEO LENGTH (share of players who did each action)")
    table = by_length(v)
    fmt = {c: "{:.2%}".format for c in ACTIONS}
    fmt["videos"] = "{:,}".format
    print(table.to_string(formatters=fmt))

    print("\nSHORTEST vs LONGEST: how many times higher is the rate for <7s than >60s?")
    for name in ACTIONS:
        ratio = table.loc["<7s", name] / table.loc[">60s", name]
        print(f"  {name:10s} {ratio:6.1f}x")

    print("\nSPEARMAN RANK CORRELATION between video length and each rate")
    print("(-1 = longer always lower, 0 = unrelated, +1 = longer always higher)")
    for name in ACTIONS:
        r = spearman(v["seconds"], v[name])
        print(f"  {name:10s} {r:+.2f}")

    # Which label tracks deliberate actions better, per video?
    # Share of test users each video counts as "liked" under each label.
    test = pd.read_parquet(PROCESSED / "test.parquet")
    test["label_A"] = test["engagement"] >= 1.0
    test["label_B"] = label_b_for_test(test)
    per_video = test.groupby("video_id")[["label_A", "label_B"]].mean()
    joined = per_video.join(v, how="inner")
    print(f"\nWHICH LABEL TRACKS DELIBERATE ACTIONS? ({len(joined):,} test videos)")
    print("Spearman correlation between a video's test 'liked' share and its "
          "platform rate:")
    print(f"  {'':10s} {'A (>=1.0)':>10s} {'B (length-aware)':>17s}")
    for name in ["liked", "saved", "shared", "commented", "followed", "see_less"]:
        a = spearman(joined["label_A"], joined[name])
        b = spearman(joined["label_B"], joined[name])
        print(f"  {name:10s} {a:+10.2f} {b:+17.2f}")


def label_b_for_test(test, n_groups=100, top_share=1 / 3):
    """Option B label for the test table (same method as src/compare_labels.py).

    Bars = the top-third cutoff of total engagement per length group, learned
    from TRAINING data only, then applied to test watches.
    """
    cols = ["user_id", "video_id", "video_duration", "timestamp", "watch_ratio"]
    big = pd.read_csv(RAW / "big_matrix.csv", usecols=cols).drop_duplicates()
    train = big.groupby(["user_id", "video_id"]).agg(
        total=("watch_ratio", "sum"), video_duration=("video_duration", "first"))
    durations = train.groupby("video_id")["video_duration"].first()
    edges = np.unique(np.quantile(durations, np.linspace(0, 1, n_groups + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    bars = train.groupby(pd.cut(train["video_duration"], edges, labels=False))[
        "total"].quantile(1 - top_share)
    group = pd.cut(test["video_duration"], edges, labels=False)
    return test["engagement"] >= group.map(bars)


if __name__ == "__main__":
    main()
