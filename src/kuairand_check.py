"""Extra check after step 8: does our "liked" label agree with real likes?

Our label is built from watch time only, because KuaiRec records nothing
else per watch. KuaiRand-Pure is a second public dataset from the same app
(2022) that also records what people DID on each watch: like, follow,
comment, share ("forward") and "see less" ("hate"). So we can apply our two
rules to its watches and see which one lines up with those real actions.

The two rules, exactly as in prepare_data.py:
  watched to the end   the user's best watch of the video reached its end
                       (the first label we tried, and rejected)
  length-aware (ours)  total watch time, relative to the video's length, is
                       in the top third among videos of similar length

Two parts of KuaiRand are checked separately:
  app-chosen   the normal logs: videos the app decided to show
               (like our training data)
  random       videos shown at random, whatever the user's taste
               (the nearest thing to our fully observed test)

What this can and cannot show. It checks the LABEL, on other users and other
videos, two years later. It cannot check our model: the model knows only
KuaiRec's users and videos. And a like tap is not the same thing as
enjoying a video; it is independent evidence, not the truth.

Data: https://kuairand.com, Gao et al., "KuaiRand: An Unbiased Sequential
Recommendation Dataset with Randomly Exposed Videos", CIKM 2022. License
CC BY-SA 4.0. Not included in this repository.

Run (the -I keeps Python from importing anything out of the data folders):
    uv run python -I src/kuairand_check.py
"""

import sys
from pathlib import Path

# With -I, Python does not add this script's folder to its search path, so we
# add it ourselves to reuse the label code from prepare_data.py.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from prepare_data import (FINISHED, MIN_TEST_GROUP_VIDEOS, TOP_SHARE,
                          length_group_edges)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "raw" / "KuaiRand" / "KuaiRand-Pure" / "data"
APP_CHOSEN = ["log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv"]
RANDOM = ["log_random_4_22_to_5_08_pure.csv"]
PAIR = ["user_id", "video_id"]
ACTIONS = {"is_like": "like", "is_follow": "follow", "is_comment": "comment",
           "is_forward": "share", "is_hate": "see less"}


def load_pairs(files):
    """One row per (user, video) pair, built the same way as our training
    table: engagement = watch time / video length, added up over the pair's
    watches. An action counts if it happened on any of those watches."""
    log = pd.concat(pd.read_csv(DATA / f) for f in files)
    log = log[log["duration_ms"] > 0]            # a few rows have no video length
    log["watch_ratio"] = log["play_time_ms"] / log["duration_ms"]
    pairs = log.groupby(PAIR).agg(
        engagement=("watch_ratio", "sum"),
        best_watch=("watch_ratio", "max"),
        video_duration=("duration_ms", "first"),
        long_view=("long_view", "max"),
        **{name: (column, "max") for column, name in ACTIONS.items()},
    ).reset_index()
    for name in list(ACTIONS.values()) + ["long_view"]:
        pairs[name] = pairs[name].astype(bool)
    pairs["any positive"] = pairs[["like", "follow", "comment", "share"]].any(axis=1)
    pairs["finished"] = pairs["best_watch"] >= FINISHED
    return pairs


def add_length_aware(pairs, min_videos, bars=None, edges=None, share=TOP_SHARE):
    """Our rule. Adds:
      rank_in_group  where this pair's engagement stands among pairs whose
                     videos have a similar length (0 = lowest, 1 = highest)
      liked          engagement reaches the bar of its length group
    If bars and edges are given (from another table), they are used as they
    are; otherwise they are worked out from this table."""
    own_edges = length_group_edges(pairs, min_videos)
    own_group = pd.cut(pairs["video_duration"], own_edges, labels=False)
    pairs["rank_in_group"] = pairs.groupby(own_group)["engagement"].rank(pct=True)
    if bars is None:
        edges = own_edges
        bars = pairs.groupby(own_group)["engagement"].quantile(1 - share)
    group = pd.cut(pairs["video_duration"], edges, labels=False)
    return pairs["engagement"] >= group.map(bars), edges, bars


def auc(score, positive):
    """Take one pair with the action and one without: how often does the pair
    with the action have the higher score? 0.5 = no relation, 1.0 = always."""
    ranks = pd.Series(score).rank().to_numpy()
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return (ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def by_length(pairs, rules):
    """Each rule's liked rate next to the real like rate, by video length.

    KuaiRand has almost no very short videos (one under 5 seconds, 42 under
    7), so it cannot say anything about the under-5-second videos that our
    KuaiRec results are sensitive to.
    """
    seconds = pairs["video_duration"] / 1000
    bucket = pd.cut(seconds, [0, 7, 10, 15, 30, 60, 120, np.inf],
                    labels=["under 7s", "7-10s", "10-15s", "15-30s", "30-60s",
                            "60-120s", "over 120s"])
    print(f"  {'video length':<14}{'videos':>8}{'pairs':>10}" +
          "".join(f"{name:>22}" for name in rules) + f"{'real like':>12}{'any positive':>14}")
    for name, rows in pairs.groupby(bucket, observed=True):
        print(f"  {name:<14}{rows['video_id'].nunique():>8,}{len(rows):>10,}" +
              "".join(f"{rows[column].mean():>22.1%}" for column in rules.values()) +
              f"{rows['like'].mean():>12.2%}{rows['any positive'].mean():>14.2%}")
    # The same thing as one number per rule. Spearman correlation: +1 = the
    # longer the video, the higher the rate; -1 = the longer, the lower;
    # 0 = no relation. Worked out per video, for videos with 50+ pairs.
    per_video = pairs.groupby("video_id").agg(
        n=("like", "size"), seconds=("video_duration", "first"),
        **{"real like": ("like", "mean")},
        **{name: (column, "mean") for name, column in rules.items()})
    per_video = per_video[per_video["n"] >= 50]
    length_rank = per_video["seconds"].rank()
    print(f"  Relation between a video's length and its rate ({len(per_video):,} videos, "
          f"Spearman): " + ",  ".join(
              f"{name} {length_rank.corr(per_video[name].rank()):+.2f}"
              for name in ["real like"] + list(rules)))


def as_labels(pairs, rules):
    """Each rule as a yes/no label: how often does a real action happen on the
    pairs it marks, compared with the pairs it does not mark?"""
    signals = ["like", "follow", "comment", "share", "any positive", "see less"]
    print(f"  {'rule':<34}{'marks':>7}" + "".join(f"{s:>15}" for s in signals))
    print(f"  {'(all pairs)':<34}{'':>7}" +
          "".join(f"{pairs[s].mean():>14.2%} " for s in signals))
    for name, column in rules.items():
        marked = pairs[column]
        cells = []
        for s in signals:
            yes, no = pairs.loc[marked, s].mean(), pairs.loc[~marked, s].mean()
            cells.append(f"{yes:.2%} ({yes / no:.1f}x)")
        print(f"  {name:<34}{marked.mean():>7.1%}" + "".join(f"{c:>15}" for c in cells))
    print("  (in brackets: how many times more often than on the pairs the rule does NOT mark)")


def same_strictness(pairs):
    """Let both rules mark the same share of pairs, then ask: of all the real
    actions, what share falls inside the marked pairs?"""
    share = pairs["finished"].mean()
    ours = pairs["rank_in_group"] >= 1 - share   # top `share` of every length group
    signals = ["like", "follow", "comment", "share", "any positive", "see less"]
    print(f"  Each rule marks {share:.1%} of pairs. Share of all real actions inside them "
          f"(pure chance would give {share:.1%}):")
    print(f"  {'rule':<34}" + "".join(f"{s:>14}" for s in signals))
    for name, marked in [("watched to the end", pairs["finished"]),
                         ("length-aware (ours), same share", ours)]:
        print(f"  {name:<34}" +
              "".join(f"{(pairs[s] & marked).sum() / pairs[s].sum():>14.1%}" for s in signals))


def auc_table(pairs):
    """No yes/no cut-off at all: compare the two underlying numbers."""
    signals = ["like", "follow", "comment", "share", "any positive", "see less"]
    print(f"  {'the number behind the rule':<34}" + "".join(f"{s:>14}" for s in signals))
    for name, column in [("share of the video watched", "best_watch"),
                         ("rank among similar-length videos", "rank_in_group")]:
        print(f"  {name:<34}" +
              "".join(f"{auc(pairs[column], pairs[s].to_numpy()):>14.3f}" for s in signals))
    print("  (for 'see less', LOWER is better: people should not have engaged with those)")


def report(title, pairs, rules):
    print(f"\n{'=' * 100}\n{title}: {len(pairs):,} (user, video) pairs\n{'=' * 100}")
    print("\nA. BY VIDEO LENGTH: how often each rule says 'liked', next to real actions")
    by_length(pairs, rules)
    print("\nB. EACH RULE AS A YES/NO LABEL: rate of each real action on the pairs it marks")
    as_labels(pairs, {**rules, "Kuaishou's own 'long view'": "long_view"})
    print("\nC. SAME STRICTNESS")
    same_strictness(pairs)
    print("\nD. WITHOUT ANY CUT-OFF (AUC)")
    auc_table(pairs)


def main():
    app = load_pairs(APP_CHOSEN)
    rnd = load_pairs(RANDOM)

    # Our rule on the app-chosen logs: bars from those logs, as in training.
    app["liked"], edges, bars = add_length_aware(app, min_videos=25)
    # On the random part, both of our markings:
    #   main marking  = bars taken from the app-chosen logs (as we do for the test)
    #   evened out    = bars set on the random part itself, same overall strictness
    rnd["liked"], _, _ = add_length_aware(rnd, MIN_TEST_GROUP_VIDEOS, bars, edges)
    rnd["liked_even"], _, _ = add_length_aware(rnd, MIN_TEST_GROUP_VIDEOS,
                                               share=rnd["liked"].mean())

    report("APP-CHOSEN VIDEOS (like our training data)", app,
           {"watched to the end": "finished", "length-aware (ours)": "liked"})
    report("RANDOMLY SHOWN VIDEOS (nearest to our test)", rnd,
           {"watched to the end": "finished", "ours, main marking": "liked",
            "ours, evened out": "liked_even"})


if __name__ == "__main__":
    main()
