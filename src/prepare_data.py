"""Step 3: turn raw watch logs into a training set and a test set.

Each row is one (user, video) pair with a label: liked = 1 or 0.

The "liked" rule (option B, length-aware), agreed in step 3:
  1. Split videos into groups of similar length: 100 equal-size groups, with
     any group spanning more than 25% (longest vs shortest) cut further,
     keeping at least 25 videos per group.
  2. Engagement of a pair = SUM of watch_ratio over all of that user's watches
     of that video (so rewatching counts).
  3. In each length group, the "bar" is the engagement only the top third of
     training pairs reach.
  4. liked = engagement >= the bar for that video's length group.
Bars are computed from TRAINING data only, then applied to both tables.

Why not "watched to the end"? It mostly measured video length: under 7s, 70%
of watches finished; over 60s, 0.3%. See src/compare_labels.py and
src/length_bias.py for the evidence.

Training set: big_matrix (after removing exact duplicate log rows).
Test set:     small_matrix (one watch per pair; nearly every user saw every video).

Run:  uv run python src/prepare_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "KuaiRec" / "data"
OUT = ROOT / "data" / "processed"

N_LENGTH_GROUPS = 100  # how finely to split videos by length
MAX_GROUP_RATIO = 1.25 # longest video in a group at most 25% longer than shortest
MIN_GROUP_VIDEOS = 25  # ...but never cut a group into pieces smaller than this
TOP_SHARE = 1 / 3      # share of pairs in each length group that count as "liked"
FINISHED = 1.0         # watch_ratio meaning "watched to the end" (used in check 1)

COLS = ["user_id", "video_id", "video_duration", "timestamp", "watch_ratio"]
DTYPES = {"user_id": "int32", "video_id": "int32", "video_duration": "int64",
          "timestamp": "float64", "watch_ratio": "float64"}
PAIR = ["user_id", "video_id"]


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def check_grows_on_you(big):
    """How often is a video skipped first, then finished on a later watch?"""
    section("CHECK 1: DO VIDEOS GROW ON PEOPLE?")

    # Put each pair's watches in time order, then number them 0, 1, 2, ...
    b = big.sort_values(PAIR + ["timestamp"])
    b["watch_no"] = b.groupby(PAIR).cumcount()

    first = b[b["watch_no"] == 0].set_index(PAIR)["watch_ratio"]
    later_best = b[b["watch_no"] > 0].groupby(PAIR)["watch_ratio"].max()
    rep = pd.DataFrame({"first": first.loc[later_best.index], "later_best": later_best})

    n_pairs = len(first)
    first_skipped = rep[rep["first"] < FINISHED]
    grew = first_skipped[first_skipped["later_best"] >= FINISHED]

    print(f"(user, video) pairs in total:         {n_pairs:,}")
    print(f"pairs watched more than once:         {len(rep):,} "
          f"({len(rep) / n_pairs:.1%} of pairs)")
    print(f"  ...first watch skipped:             {len(first_skipped):,}")
    print(f"      ...then finished on a later one: {len(grew):,} "
          f"({len(grew) / len(first_skipped):.1%} of those)")


def length_group_edges(train):
    """Boundaries that split the training videos into groups of similar length.

    Start with N_LENGTH_GROUPS groups holding equal numbers of videos. Long
    videos are rare, so the last equal-size group stretched from 77s to 5 min,
    and one bar can't be fair across that range (20% of 90s is easy, 20% of
    5 min is hard). So any group whose longest video is more than
    MAX_GROUP_RATIO times its shortest is cut into smaller pieces.

    But each piece keeps at least MIN_GROUP_VIDEOS videos: in a group with
    one video, "top third" would force that video to ~33% liked no matter how
    good it is, which erases the very signal we want to measure.
    """
    durations = train.groupby("video_id")["video_duration"].first()
    edges = np.unique(np.quantile(durations, np.linspace(0, 1, N_LENGTH_GROUPS + 1)))
    refined = [edges[0]]
    for lo, hi in zip(edges[:-1], edges[1:]):
        wanted = int(np.ceil(np.log(hi / lo) / np.log(MAX_GROUP_RATIO)))
        in_group = durations[(durations > lo) & (durations <= hi)]
        pieces = min(wanted, len(in_group) // MIN_GROUP_VIDEOS)
        if pieces > 1:
            # Cut at quantiles so every piece holds about the same number of videos.
            cuts = np.quantile(in_group, np.linspace(0, 1, pieces + 1))[1:-1]
            refined.extend(cuts)
        refined.append(hi)
    edges = np.array(refined)
    edges[0], edges[-1] = -np.inf, np.inf  # catch any video outside the training range
    return edges


def print_group_sizes(train, edges):
    """Show how many groups there are and the smallest ones (fewest examples)."""
    group = pd.cut(train["video_duration"], edges, labels=False)
    sizes = train.groupby(group).agg(videos=("video_id", "nunique"),
                                     pairs=("video_id", "size"))
    lo = np.maximum(edges[:-1], 0) / 1000
    hi = edges[1:] / 1000
    sizes["range"] = [f"{lo[g]:.1f}s-{hi[g]:.1f}s" for g in sizes.index]
    print(f"length groups: {len(sizes)} (with training data)")
    print("smallest groups by number of videos:")
    print(sizes.nsmallest(6, "videos").to_string(
        formatters={"pairs": "{:,}".format}))


def build_train(big):
    """One row per pair. engagement = total watch_ratio across all watches."""
    return big.groupby(PAIR).agg(
        engagement=("watch_ratio", "sum"),
        n_watches=("watch_ratio", "size"),
        video_duration=("video_duration", "first"),
    ).reset_index()


def build_test(small):
    """The small table already has one watch per pair, so engagement = watch_ratio."""
    test = small[PAIR + ["video_duration"]].copy()
    test["engagement"] = small["watch_ratio"]
    return test


def add_liked_label(df, edges, bars):
    """liked = 1 if engagement reaches the bar for this video's length group."""
    group = pd.cut(df["video_duration"], edges, labels=False)
    df["liked"] = (df["engagement"] >= group.map(bars)).astype("int8")


def check_short_video_effect(df, name):
    """Liked rate by video length. Should now be roughly flat."""
    seconds = df["video_duration"] / 1000
    buckets = pd.cut(seconds, [0, 7, 15, 30, 60, 90, 120, np.inf],
                     labels=["<7s", "7-15s", "15-30s", "30-60s", "60-90s",
                             "90-120s", ">120s"])
    rates = df.groupby(buckets, observed=True)["liked"].mean()
    print(f"{name}: " + "  ".join(f"{k} {v:.1%}" for k, v in rates.items()))


def main():
    big = pd.read_csv(RAW / "big_matrix.csv", usecols=COLS, dtype=DTYPES)
    small = pd.read_csv(RAW / "small_matrix.csv", usecols=COLS, dtype=DTYPES)

    # big_matrix contains ~968k rows that are exact copies of another row:
    # a logging glitch, not real rewatches.
    before = len(big)
    big = big.drop_duplicates()
    print(f"removed {before - len(big):,} exact duplicate rows from big_matrix")

    check_grows_on_you(big)

    train = build_train(big)
    test = build_test(small)

    # The bars come from TRAINING data only.
    edges = length_group_edges(train)
    section("LENGTH GROUPS")
    print_group_sizes(train, edges)
    bars =train.groupby(pd.cut(train["video_duration"], edges, labels=False))[
        "engagement"].quantile(1 - TOP_SHARE)
    add_liked_label(train, edges, bars)
    add_liked_label(test, edges, bars)

    section("CHECK 2: LIKED RATE BY VIDEO LENGTH (should be roughly flat)")
    check_short_video_effect(test, "test ")
    check_short_video_effect(train, "train")

    section("CHECK 3: CAN EVERY TEST USER BE SCORED?")
    likes = test.groupby("user_id")["liked"].sum()
    print(f"liked videos per test user (out of ~3,327): median {likes.median():.0f}, "
          f"min {likes.min()}, max {likes.max():,}")
    print(f"test users with 0 liked videos: {(likes == 0).sum()}, "
          f"with fewer than 10: {(likes < 10).sum()}")

    section("RESULT")
    for name, df in [("train", train), ("test ", test)]:
        print(f"{name}: {len(df):,} pairs, {df['user_id'].nunique():,} users, "
              f"{df['video_id'].nunique():,} videos, {df['liked'].mean():.1%} liked")

    OUT.mkdir(parents=True, exist_ok=True)
    train.to_parquet(OUT / "train.parquet", index=False)
    test.to_parquet(OUT / "test.parquet", index=False)
    print(f"\nsaved to {OUT}/")


if __name__ == "__main__":
    main()
