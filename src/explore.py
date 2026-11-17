"""Step 2: load KuaiRec and print the facts we need before building anything.

Run:  uv run python -I src/explore.py
(-I = "isolated mode": Python won't pick up stray modules from the current
folder. A small safety habit when a script reads downloaded files.)
"""

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "raw" / "KuaiRec" / "data"

# Explicit column types keep memory down (the big table has 12.5M rows) and
# make sure IDs are read as integers, not floats.
DTYPES = {
    "user_id": "int32",
    "video_id": "int32",
    "play_duration": "int64",
    "video_duration": "int64",
    "date": "float64",  # float in case of missing values; checked below
    "timestamp": "float64",
    "watch_ratio": "float64",
}


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def describe_matrix(name, df):
    section(name)
    n_users = df["user_id"].nunique()
    n_videos = df["video_id"].nunique()
    print(f"rows (watch records): {len(df):,}")
    print(f"unique users:         {n_users:,}")
    print(f"unique videos:        {n_videos:,}")
    # Density = what fraction of all possible (user, video) pairs we observed.
    n_pairs = df[["user_id", "video_id"]].drop_duplicates().shape[0]
    print(f"unique (user, video) pairs: {n_pairs:,}  "
          f"(so {len(df) - n_pairs:,} rows are repeat watches of the same pair)")
    print(f"density: {n_pairs / (n_users * n_videos):.1%}")

    print("\ncolumns and missing values:")
    print(pd.DataFrame({"dtype": df.dtypes, "missing": df.isna().sum()}))

    print("\nfirst 3 rows:")
    print(df.head(3).to_string())

    print(f"\ndates: {df['time'].min()}  to  {df['time'].max()}")

    # watch_ratio is what step 3 builds "liked" from, so look at it closely.
    wr = df["watch_ratio"]
    print("\nwatch_ratio (time watched / video length):")
    print(wr.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).to_string())
    for t in [0.5, 1.0, 2.0]:
        print(f"  share of records with watch_ratio >= {t}: {(wr >= t).mean():.1%}")
    print(f"  records with video_duration == 0: {(df['video_duration'] == 0).sum():,}")

    # Popularity skew: how concentrated is watching on a few videos?
    counts = df["video_id"].value_counts()
    top1 = counts.head(max(1, len(counts) // 100)).sum() / counts.sum()
    top10 = counts.head(max(1, len(counts) // 10)).sum() / counts.sum()
    print(f"\npopularity: top 1% of videos get {top1:.1%} of records, "
          f"top 10% get {top10:.1%}")
    print(f"records per user:  median {df['user_id'].value_counts().median():.0f}")
    print(f"records per video: median {counts.median():.0f}, "
          f"min {counts.min()}, max {counts.max():,}")


def main():
    print(f"reading from {DATA}")
    big = pd.read_csv(DATA / "big_matrix.csv", dtype=DTYPES)
    small = pd.read_csv(DATA / "small_matrix.csv", dtype=DTYPES)

    describe_matrix("BIG MATRIX (big_matrix.csv)", big)
    describe_matrix("SMALL MATRIX (small_matrix.csv)", small)

    # If we train on big and evaluate on small, they must not share
    # (user, video) pairs, or we'd be testing on answers the model has seen.
    section("OVERLAP BETWEEN BIG AND SMALL")
    big_users, small_users = set(big["user_id"]), set(small["user_id"])
    big_videos, small_videos = set(big["video_id"]), set(small["video_id"])
    print(f"small users also in big:  {len(small_users & big_users):,} "
          f"of {len(small_users):,}")
    print(f"small videos also in big: {len(small_videos & big_videos):,} "
          f"of {len(small_videos):,}")
    big_pairs = big[["user_id", "video_id"]].drop_duplicates()
    small_pairs = small[["user_id", "video_id"]].drop_duplicates()
    shared = big_pairs.merge(small_pairs, on=["user_id", "video_id"])
    print(f"(user, video) pairs present in BOTH tables: {len(shared):,}")
    print(f"small-matrix time range: {small['time'].min()} to {small['time'].max()}")

    # The side tables, briefly: what they are and their shape.
    section("SIDE TABLES")
    for f in ["user_features.csv", "item_categories.csv",
              "item_daily_features.csv", "kuairec_caption_category.csv",
              "social_network.csv"]:
        try:
            t = pd.read_csv(DATA / f)
        except UnicodeDecodeError:
            # The caption file contains Chinese text; try a common fallback.
            t = pd.read_csv(DATA / f, encoding="gb18030")
        cols = list(t.columns)
        shown = cols[:8] + (["..."] if len(cols) > 8 else [])
        print(f"{f}: {t.shape[0]:,} rows x {t.shape[1]} cols  {shown}")


if __name__ == "__main__":
    main()
