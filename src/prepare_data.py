"""Step 3: turn raw watch logs into training and test "flashcards".

Training set (from big_matrix):  one row per (user, video) pair, labelled
    liked = 1 if ANY watch of that pair reached the end (watch_ratio >= 1.0).
Test set (from small_matrix):     one row per (user, video) pair with the same
    label. These are the "answer key" the recommendations get checked against.

Also runs three checks before saving:
  1. "Grows on you": skipped on the first watch, finished on a later one.
  2. Short-video effect: does the liked rate depend on video length?
  3. Likes per test user: can every test user be scored?

Run:  uv run python -I src/prepare_data.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "KuaiRec" / "data"
OUT = ROOT / "data" / "processed"

# "Liked" = watched to the end. One number, in one place, so it's easy to change.
LIKED_THRESHOLD = 1.0

# We only need these columns; skipping the rest saves memory and time.
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
    # Only pairs that were watched more than once have a "later" watch.
    rep = pd.DataFrame({"first": first.loc[later_best.index], "later_best": later_best})

    n_pairs = len(first)
    first_skipped = rep[rep["first"] < LIKED_THRESHOLD]
    grew = first_skipped[first_skipped["later_best"] >= LIKED_THRESHOLD]

    print(f"(user, video) pairs in total:         {n_pairs:,}")
    print(f"pairs watched more than once:         {len(rep):,} "
          f"({len(rep) / n_pairs:.1%} of pairs)")
    print(f"  ...first watch already finished:    {len(rep) - len(first_skipped):,}")
    print(f"  ...first watch skipped:             {len(first_skipped):,}")
    print(f"      ...then finished on a later one: {len(grew):,} "
          f"({len(grew) / len(first_skipped):.1%} of those)")
    print(f"'grew on them' pairs as share of ALL pairs: {len(grew) / n_pairs:.2%}")


def check_short_video_effect(df, name):
    """Liked rate by video length. A strong slope = 'liked' partly means 'short'."""
    seconds = df["video_duration"] / 1000
    buckets = pd.cut(seconds, [0, 7, 15, 30, 60, float("inf")],
                     labels=["<7s", "7-15s", "15-30s", "30-60s", ">60s"])
    table = df.groupby(buckets, observed=True)["liked"].agg(["size", "mean"])
    table.columns = ["pairs", "liked_rate"]
    table["share_of_pairs"] = table["pairs"] / table["pairs"].sum()
    print(f"\n{name}:")
    print(table.to_string(formatters={"pairs": "{:,}".format,
                                      "liked_rate": "{:.1%}".format,
                                      "share_of_pairs": "{:.1%}".format}))


def build_train(big):
    """Collapse repeat watches: one row per pair, liked if ANY watch finished."""
    train = big.groupby(PAIR).agg(
        n_watches=("watch_ratio", "size"),
        max_watch_ratio=("watch_ratio", "max"),
        video_duration=("video_duration", "first"),
    ).reset_index()
    train["liked"] = (train["max_watch_ratio"] >= LIKED_THRESHOLD).astype("int8")
    return train


def build_test(small):
    """The small table already has one row per pair; just add the label."""
    test = small[PAIR + ["watch_ratio", "video_duration"]].copy()
    test["liked"] = (test["watch_ratio"] >= LIKED_THRESHOLD).astype("int8")
    return test


def main():
    big = pd.read_csv(RAW / "big_matrix.csv", usecols=COLS, dtype=DTYPES)
    small = pd.read_csv(RAW / "small_matrix.csv", usecols=COLS, dtype=DTYPES)

    # big_matrix contains ~966k rows that are exact copies of another row
    # (same user, video, timestamp, watch_ratio): a logging glitch, not real
    # rewatches. Left in, they would look like people watching twice.
    before = len(big)
    big = big.drop_duplicates()
    print(f"removed {before - len(big):,} exact duplicate rows from big_matrix "
          f"({len(big):,} remain)")

    check_grows_on_you(big)

    train = build_train(big)
    test = build_test(small)

    section("CHECK 2: DO SHORT VIDEOS GET 'LIKED' MORE EASILY?")
    check_short_video_effect(test, "TEST table (everyone saw everything: the fair view)")
    check_short_video_effect(train, "TRAIN table")

    section("CHECK 3: CAN EVERY TEST USER BE SCORED?")
    likes = test.groupby("user_id")["liked"].sum()
    print(f"test users: {len(likes):,}")
    print(f"liked videos per test user (out of ~3,327): median {likes.median():.0f}, "
          f"min {likes.min()}, max {likes.max():,}")
    print(f"test users with 0 liked videos (can't be scored): {(likes == 0).sum()}")
    print(f"test users with fewer than 10 liked videos:      {(likes < 10).sum()}")
    history = train[train["user_id"].isin(likes.index)].groupby("user_id").size()
    print(f"training pairs per test user (their 'history'):  median {history.median():.0f}, "
          f"min {history.min()}")

    section("THE FLASHCARDS")
    for name, df in [("TRAIN (homework)", train), ("TEST (exam)", test)]:
        print(f"{name}: {len(df):,} cards, {df['user_id'].nunique():,} users, "
              f"{df['video_id'].nunique():,} videos, "
              f"{df['liked'].mean():.1%} say 'liked'")

    OUT.mkdir(parents=True, exist_ok=True)
    train.to_parquet(OUT / "train.parquet", index=False)
    test.to_parquet(OUT / "test.parquet", index=False)
    print(f"\nsaved to {OUT}/train.parquet and test.parquet")


if __name__ == "__main__":
    main()
