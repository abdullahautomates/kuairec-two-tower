"""Step 4: the scorer. Every method in steps 5-7 is graded by this file.

recall@k for one user:
    hits = how many of the top k recommended videos the user actually liked
    recall@k = hits / min(number of videos they liked, k)

Dividing by min(liked, k) means a perfect list always scores 1.0: if a user
liked only 4 videos, even a perfect top-10 can contain at most 4 hits.

The final score is the average over users, so every user counts equally
(heavy users with many likes don't dominate).
"""


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


def load_liked_by_user(test):
    """Turn the test table into {user_id: set of liked video IDs}."""
    liked = test[test["liked"] == 1]
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
    import numpy as np  # local import keeps the scorer itself dependency-free

    users = np.array(sorted(test["user_id"].unique()))
    shuffled = np.random.default_rng(seed).permutation(users)
    half = len(shuffled) // 2
    return set(shuffled[:half].tolist()), set(shuffled[half:].tolist())


if __name__ == "__main__":
    # Tiny demo:  uv run python src/metrics.py
    recs = {1: ["A", "B", "C"], 2: ["D", "E", "F"]}
    liked = {1: {"A", "C", "F", "G"}, 2: {"B"}}
    print(mean_recall_at_k(recs, liked, k=3))  # (2/3 + 0/1) / 2 = 0.333...
