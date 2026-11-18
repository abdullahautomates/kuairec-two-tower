"""Step 5: popularity baselines. Everyone gets the same top videos.

Three ways to define "popular", all counted on TRAINING data:
  1. most watched : how many training users watched the video
  2. most liked   : how many training users it counts as "liked" for
  3. like rate    : liked / watched (only videos with >= MIN_WATCHERS watchers,
                    since a rate from a handful of people is mostly luck)

Honesty rule: versions are compared on the CHOOSING group of test users; the
best one (highest recall@10, decided in advance) is then scored on the FINAL
group, which played no part in the choice. All scores are printed.

Run:  uv run python src/popularity.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from metrics import (candidate_videos, load_liked_by_user, mean_recall_at_k,
                     split_test_users)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
MIN_WATCHERS = 100
KS = [10, 50, 100]


def popularity_table(train, candidates):
    """One row per candidate video: watched, liked, like rate, length."""
    per_video = train.groupby("video_id").agg(
        watched=("user_id", "size"),
        liked=("liked", "sum"),
        seconds=("video_duration", "first"),
    )
    per_video["seconds"] /= 1000
    per_video["like_rate"] = per_video["liked"] / per_video["watched"]
    # Rates from fewer than MIN_WATCHERS people are too noisy to trust.
    per_video.loc[per_video["watched"] < MIN_WATCHERS, "like_rate"] = np.nan
    # Only videos we can grade on (the test candidates) may be recommended.
    return per_video.reindex(candidates)


def ranked_list(table, column):
    """Candidate video IDs, most popular first. Ties broken by video ID so the
    result is the same every run; videos with no value (NaN) go last."""
    t = table.reset_index().rename(columns={"index": "video_id"})
    t = t.sort_values([column, "video_id"], ascending=[False, True],
                      na_position="last")
    return t["video_id"].tolist()


def subset(liked_by_user, users):
    return {u: liked_by_user[u] for u in users}


def main():
    train = pd.read_parquet(PROCESSED / "train.parquet")
    test = pd.read_parquet(PROCESSED / "test.parquet")
    liked_by_user = load_liked_by_user(test)
    candidates = candidate_videos(test)
    choosing, final = split_test_users(test)
    print(f"test users: choosing group {len(choosing)}, final group {len(final)}")

    table = popularity_table(train, candidates)
    print(f"candidate videos: {len(table):,}; with >= {MIN_WATCHERS} watchers "
          f"(eligible for like rate): {table['like_rate'].notna().sum():,}")

    versions = {"most watched": "watched", "most liked": "liked",
                "like rate": "like_rate"}
    lists = {name: ranked_list(table, col) for name, col in versions.items()}

    # Random floor for each group, for reference (same as step 4's 0.234).
    for name, users in [("choosing", choosing), ("final", final)]:
        floor = np.mean([len(liked_by_user[u]) / len(candidates) for u in users])
        print(f"random floor ({name} group): {floor:.3f}")

    def scores(ranking, users):
        recs = {u: ranking for u in users}  # same list for every user
        return [mean_recall_at_k(recs, subset(liked_by_user, users), k) for k in KS]

    print(f"\nSCORES ON THE CHOOSING GROUP  (recall@{', @'.join(map(str, KS))})")
    choosing_scores = {}
    for name, ranking in lists.items():
        s = scores(ranking, choosing)
        choosing_scores[name] = s[0]
        print(f"  {name:13s} " + "   ".join(f"@{k}: {v:.3f}" for k, v in zip(KS, s)))

    best = max(choosing_scores, key=choosing_scores.get)
    print(f"\nchosen (highest recall@10 on choosing group): {best}")

    print("\nSCORES ON THE FINAL GROUP  (report these; chosen version marked *)")
    for name, ranking in lists.items():
        s = scores(ranking, final)
        mark = "*" if name == best else " "
        print(f" {mark}{name:13s} " + "   ".join(f"@{k}: {v:.3f}" for k, v in zip(KS, s)))

    print("\nOVERLAP: how many videos two top-10 lists share (out of 10)")
    names = list(lists)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            shared = set(lists[names[i]][:10]) & set(lists[names[j]][:10])
            print(f"  {names[i]} vs {names[j]}: {len(shared)}")

    print("\nLENGTH CHECK: are the picked videos unusually short or long?")
    print(f"  all candidates: median {table['seconds'].median():.1f}s, "
          f"{(table['seconds'] < 7).mean():.0%} under 7s")
    for name, ranking in lists.items():
        top = table.loc[ranking[:10], "seconds"]
        print(f"  {name:13s} top 10: median {top.median():.1f}s, "
              f"{(top < 7).mean():.0%} under 7s")

    print("\nTHE TOP 10 OF EACH VERSION")
    for name, ranking in lists.items():
        print(f"\n{name}:")
        print(table.loc[ranking[:10], ["watched", "liked", "like_rate", "seconds"]]
              .to_string(formatters={"like_rate": "{:.1%}".format,
                                     "seconds": "{:.1f}".format}))


if __name__ == "__main__":
    main()
