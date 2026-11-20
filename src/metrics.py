"""Step 4: the scorer. Every method in steps 5-7 is graded by this file.

recall@k for one user:
    hits = how many of the top k recommended videos the user actually liked
    recall@k = hits / min(number of videos they liked, k)

Dividing by min(liked, k) means a perfect list always scores 1.0: if a user
liked only 4 videos, even a perfect top-10 can contain at most 4 hits.

The final score is the average over users, so every user counts equally
(heavy users with many likes don't dominate).

A second, steadier score (added in step 6) is per-user AUC: for one user, take
one video they liked and one they didn't; how often does the method score the
liked one higher? 0.5 = coin flip, 1.0 = always. It uses every candidate video,
not just the top 10, so it moves much less from run to run. Recall@10 stays
the headline score; AUC is for comparing versions.

A third score is NDCG@k: recall@k where a hit near the top of the list counts
for more than a hit near the bottom (see ndcg_at_k_for_user).
"""

import math

import numpy as np
import pandas as pd


def recall_at_k_for_user(ranked_videos, liked_videos, k):
    """Score one user's ranked recommendation list.

    ranked_videos: list of video IDs, best first, e.g. [812, 45, 3301, ...]
    liked_videos:  set of video IDs this user actually liked
    """
    if not liked_videos:
        # Nothing to find, so the user can't be graded (min(0, k) would be 0).
        return None
    # set(): a video recommended twice must not count as two hits.
    top_k = set(ranked_videos[:k])
    hits = len(top_k & liked_videos)
    return hits / min(len(liked_videos), k)


def mean_recall_at_k(recommendations, liked_by_user, k):
    """Average recall@k over every user who liked at least one video.

    recommendations: dict  user_id -> ranked list of video IDs
    liked_by_user:   dict  user_id -> set of liked video IDs
    A user missing from `recommendations` raises KeyError on purpose: every
    test user must get recommendations, or the score would be misleading.
    """
    scores = []
    for user, liked in liked_by_user.items():
        score = recall_at_k_for_user(recommendations[user], liked, k)
        if score is not None:
            scores.append(score)
    return sum(scores) / len(scores)


def ndcg_at_k_for_user(ranked_videos, liked_videos, k):
    """Like recall@k, but the position of each hit matters.

    A hit at position p is worth 1 / log2(p + 1):
        position 1 -> 1.00, position 2 -> 0.63, position 3 -> 0.50,
        position 10 -> 0.29.
    Add up the worth of every hit in the top k, then divide by the best total
    this user could have had (all their hits packed at the top), so a perfect
    list scores 1.0. NDCG = "normalized discounted cumulative gain".
    """
    if not liked_videos:
        return None
    counted, total = set(), 0.0
    for position, video in enumerate(ranked_videos[:k], start=1):
        if video in liked_videos and video not in counted:  # no double hits
            total += 1 / math.log2(position + 1)
            counted.add(video)
    best_hits = min(len(liked_videos), k)
    best = sum(1 / math.log2(position + 1) for position in range(1, best_hits + 1))
    return total / best


def mean_ndcg_at_k(recommendations, liked_by_user, k):
    """Average NDCG@k over every user who liked at least one video."""
    scores = []
    for user, liked in liked_by_user.items():
        score = ndcg_at_k_for_user(recommendations[user], liked, k)
        if score is not None:
            scores.append(score)
    return sum(scores) / len(scores)


def load_liked_by_user(test, label="liked"):
    """Turn the test table into {user_id: set of liked video IDs}.

    label: which answer column to grade against. "liked" is the main marking.
    "liked_even" (second table) and "finished" (the rejected first label) are
    explained in prepare_data.py.
    """
    liked = test[test[label] == 1]
    sets = liked.groupby("user_id")["video_id"].apply(set).to_dict()
    # Users who liked nothing still appear, with an empty set.
    return {user: sets.get(user, set()) for user in test["user_id"].unique()}


def candidate_videos(test):
    """The videos every method may recommend: the 3,327 in the test table."""
    return sorted(test["video_id"].unique())


def split_test_users(test, seed=0):
    """Randomly split test users into a CHOOSING group and a FINAL group.

    Choosing group (validation): compare versions / settings and pick one.
    Final group: used only to report the score of what was picked.
    The final group never influences a decision, so its score is honest.
    A fixed seed gives the same split every run, in every step.
    """
    users =np.array(sorted(test["user_id"].unique()))
    shuffled = np.random.default_rng(seed).permutation(users)
    half = len(shuffled) // 2
    return set(shuffled[:half].tolist()), set(shuffled[half:].tolist())


def liked_matrix(liked_by_user, users, candidates):
    """Grid of True/False: one row per user, one column per candidate video.
    True = that user liked that video."""
    column = {video: j for j, video in enumerate(candidates)}
    grid = np.zeros((len(users), len(candidates)), dtype=bool)
    for i, user in enumerate(users):
        for video in liked_by_user[user]:
            if video in column:
                grid[i, column[video]] = True
    return grid


def user_auc(scores, liked, counted=None):
    """Per-user AUC for each row. scores and liked are [n_users, n_videos] grids.

    Trick: rank each user's videos by score (1 = lowest). If liked videos sit
    at high ranks, the AUC is high. Tied scores share the average rank, so a
    tie between a liked and a not-liked video counts as half a win.
    Users who liked everything or nothing get NaN (no pairs to compare).

    counted (optional): True/False grid of which videos to include for each
    user. Needed for training data, where a user has an answer for only the
    ~1,400 videos they watched: the other ~9,300 must be left out, not
    treated as "not liked".
    """
    if counted is None:
        n_counted = liked.shape[1]
    else:
        scores = np.where(counted, scores, np.nan)  # NaN = left out of the ranking
        liked = liked & counted
        n_counted = counted.sum(axis=1)
    ranks = np.nan_to_num(pd.DataFrame(scores).rank(axis=1).to_numpy())
    n_liked = liked.sum(axis=1)
    n_not = n_counted - n_liked
    wins = (ranks * liked).sum(axis=1) - n_liked * (n_liked + 1) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return wins / (n_liked * n_not)


def mean_user_auc(scores, liked, counted=None):
    """Average per-user AUC, skipping users with no pairs to compare."""
    return float(np.nanmean(user_auc(scores, liked, counted)))


def grade_scores(scores, users, candidates, liked_by_user, ks=(10, 50, 100)):
    """Grade a grid of scores: one row per user, one column per candidate
    video, higher = recommend sooner. Any method that can fill in such a grid
    (the model, popularity, similar users) is graded by this one function.

    Returns ({10: recall@10, 50: ..., 100: ..., "auc": per-user AUC,
              "ndcg@10": NDCG@10}, {user: top video IDs, best first}).
    Equal scores keep the candidates' order, so a result is the same every run.
    """
    top = np.argsort(-scores, axis=1, kind="stable")[:, :max(ks)]   # best first
    cand = np.array(candidates)
    recs = {u: cand[row].tolist() for u, row in zip(users, top)}
    liked = {u: liked_by_user[u] for u in users}
    result = {k: mean_recall_at_k(recs, liked, k) for k in ks}
    result["auc"] = mean_user_auc(scores, liked_matrix(liked_by_user, users, candidates))
    result["ndcg@10"] = mean_ndcg_at_k(recs, liked, 10)
    return result, recs


if __name__ == "__main__":
    # Tiny demo:  uv run python src/metrics.py
    recs = {1: ["A", "B", "C"], 2: ["D", "E", "F"]}
    liked = {1: {"A", "C", "F", "G"}, 2: {"B"}}
    print(mean_recall_at_k(recs, liked, k=3))  # (2/3 + 0/1) / 2 = 0.333...
