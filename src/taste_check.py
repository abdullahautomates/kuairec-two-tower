"""Step 6 check: is there any personal taste to find? Counting only, no neural network.

Everything is graded on the CHOOSING group of test users (never the final group).

PART 1: the same job the model has. Learn from the training data, then rank
the 3,327 test videos for each user.
  like rate      the step 5 popularity baseline (same list for everyone)
  my tags        does this user like a video's categories more than their own average?
  my creators    same idea, per video creator
  similar users  find training users whose likes and skips agree with this
                 user's, and recommend what those users liked
  fair like rate CONTROL for "similar users": same calculation but listening
                 to every user, so everyone gets the same list

PART 2: a deliberate cheat, to see whether taste exists in the test answers
at all. Hide half of the test videos (half B). Use each user's real test
answers on the other half (half A) to predict half B.
  others' like rate  what share of the OTHER test users liked each video
  my tags            the user's category preferences, measured on half A
  similar users      test users who agreed with this user on half A
If "similar users" clearly beats "others' like rate", people's tastes really
differ in the test answers. This is not a usable recommender (it reads test
answers); it only measures how much taste there is to find.

Run:  uv run python src/taste_check.py
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import candidate_videos, split_test_users, user_auc

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "KuaiRec" / "data"
PROCESSED = ROOT / "data" / "processed"

K = 10
MIN_WATCHERS = 100       # same rule as the step 5 like-rate baseline
TAG_SMOOTHING = 20       # as if each user had watched 20 extra "average" videos per tag
CREATOR_SMOOTHING = 5    # same idea per creator (users see few videos per creator)
N_NEIGHBOURS = 100       # how many similar users to listen to
NEIGHBOUR_SMOOTHING = 5  # as if 5 extra neighbours had no opinion on every video


def grade(name, scores, liked):
    """Print recall@10 and per-user AUC for a [users, videos] grid of scores."""
    top = np.argsort(-scores, axis=1, kind="stable")[:, :K]
    hits = np.take_along_axis(liked, top, axis=1).sum(axis=1)
    n_liked = liked.sum(axis=1)
    ok = n_liked > 0
    recall = (hits[ok] / np.minimum(n_liked[ok], K)).mean()
    auc = np.nanmean(user_auc(scores, liked))
    print(f"  {name:32s} recall@10 {recall:.3f}   AUC {auc:.3f}")


def centre(liked, watched):
    """Each user's likes minus their own like rate; 0 where they didn't watch.

    Some users like 5% of what they watch, others 80%. Subtracting a user's own
    rate removes "this person likes everything" so that what is left is which
    videos they like MORE or LESS than usual: their taste.
    Returns (centred grid, each user's like rate).
    """
    rate = liked.sum(axis=1) / np.maximum(watched.sum(axis=1), 1)
    x = liked.astype(np.float32)
    x -= rate[:, None].astype(np.float32)
    x *= watched
    return x, rate


def tag_preferences(liked, watched, tag_grid):
    """[users, tags]: how much more (+) or less (-) a user likes each tag than
    their own average. Smoothing pulls tags they barely watched towards 0."""
    liked, watched = liked.astype(np.float32), watched.astype(np.float32)
    rate = (liked.sum(axis=1) / np.maximum(watched.sum(axis=1), 1))[:, None]
    smoothed = ((liked @ tag_grid + TAG_SMOOTHING * rate)
                / (watched @ tag_grid + TAG_SMOOTHING))
    return smoothed - rate


def neighbour_scores(x_me, x_them, x_them_target, watched_them_target, same_people=False):
    """Score the target videos for each user, using their most similar users.

    x_me, x_them: centred likes on the videos used to MATCH people.
    x_them_target, watched_them_target: the other users' centred likes and
        watched flags on the videos we want to RANK.
    same_people=True when "me" and "them" are the same list (don't match a
        user with themselves).
    """
    # Similarity = do we like/skip the same videos, beyond our usual rates?
    sim = x_me @ x_them.T
    sim /= (np.linalg.norm(x_me, axis=1)[:, None]
            * np.linalg.norm(x_them, axis=1)[None, :] + 1e-9)
    if same_people:
        np.fill_diagonal(sim, 0)
    # Keep each user's N most similar users (and only if similarity is positive).
    cutoff = np.partition(sim, -N_NEIGHBOURS, axis=1)[:, -N_NEIGHBOURS][:, None]
    weight = np.where((sim >= cutoff) & (sim > 0), sim, 0)
    # Rescale so an average neighbour has weight 1.
    weight *= N_NEIGHBOURS / np.maximum(weight.sum(axis=1, keepdims=True), 1e-9)
    weight = weight.astype(np.float32)
    # Weighted average of what the neighbours thought of each target video.
    return (weight @ x_them_target) / (weight @ watched_them_target + NEIGHBOUR_SMOOTHING)


def main():
    train = pd.read_parquet(PROCESSED / "train.parquet",
                            columns=["user_id", "video_id", "liked"])
    test = pd.read_parquet(PROCESSED / "test.parquet",
                           columns=["user_id", "video_id", "liked"])
    n_users, n_videos = train["user_id"].max() + 1, train["video_id"].max() + 1
    choosing, _ = split_test_users(test)
    me = np.array(sorted(choosing))                        # the users we grade on
    cand = np.array(candidate_videos(test))                # the 3,327 test videos
    other = np.setdiff1d(np.arange(n_videos), cand)        # all the other videos
    # Training users who are not test users: the only ones who watched the
    # test videos in the training data.
    them = np.setdiff1d(np.arange(n_users), test["user_id"].unique())

    # Training data as two True/False grids: [user, video] watched? liked?
    watched = np.zeros((n_users, n_videos), dtype=bool)
    watched[train["user_id"], train["video_id"]] = True
    liked = np.zeros_like(watched)
    yes = train[train["liked"] == 1]
    liked[yes["user_id"], yes["video_id"]] = True

    # The choosing group's test answers as grids: [user, test video].
    t = test[test["user_id"].isin(choosing)]
    row = pd.Series(np.arange(len(me)), index=me)
    col = pd.Series(np.arange(len(cand)), index=cand)
    r, c = row[t["user_id"]].to_numpy(), col[t["video_id"]].to_numpy()
    seen = np.zeros((len(me), len(cand)), dtype=bool)
    seen[r, c] = True
    answers = np.zeros_like(seen)
    answers[r, c] = t["liked"].to_numpy() == 1

    print(f"choosing group: {len(me)} users, {len(cand):,} test videos")
    print(f"their training history on those test videos: "
          f"{int(watched[np.ix_(me, cand)].sum())} pairs (0 = no leakage)")

    # ------------------------------------------------------------------ PART 1
    print("\nPART 1: learn from TRAINING data, rank the test videos")
    print(f"  {'random (expected)':32s} recall@10 {answers.mean(axis=1).mean():.3f}   AUC 0.500")

    # like rate: liked / watched over all training users (step 5 baseline)
    n_watched = watched[:, cand].sum(axis=0)
    rate = liked[:, cand].sum(axis=0) / np.maximum(n_watched, 1)
    trusted = n_watched >= MIN_WATCHERS
    like_rate = np.where(trusted, rate, -1.0)              # untrusted rates rank last
    grade("like rate (popularity)", np.tile(like_rate, (len(me), 1)), answers)
    # For combining with personal scores, untrusted videos get the average rate.
    base = np.where(trusted, rate, rate[trusted].mean())[None, :]

    # my tags
    tags = pd.read_csv(RAW / "item_categories.csv")
    tag_grid = np.zeros((n_videos, 31), dtype=np.float32)
    for video, feat in zip(tags["video_id"], tags["feat"]):
        tag_grid[video, ast.literal_eval(feat)] = 1
    tag_share = tag_grid[cand] / tag_grid[cand].sum(axis=1, keepdims=True)
    my_tags = tag_preferences(liked[me], watched[me], tag_grid) @ tag_share.T
    grade("my tags", my_tags, answers)

    # my creators
    creators = (pd.read_csv(RAW / "item_daily_features.csv",
                            usecols=["video_id", "author_id"])
                .drop_duplicates("video_id").set_index("video_id")["author_id"])
    creator_of = creators.reindex(range(n_videos)).to_numpy()
    mine = train[train["user_id"].isin(choosing)].copy()
    mine["creator"] = creator_of[mine["video_id"]]
    my_rate = mine.groupby("user_id")["liked"].mean()
    per = (mine.groupby(["user_id", "creator"])["liked"].agg(["sum", "size"])
           .reset_index())
    usual = my_rate[per["user_id"]].to_numpy()
    per["pref"] = ((per["sum"] + CREATOR_SMOOTHING * usual)
                   / (per["size"] + CREATOR_SMOOTHING) - usual)
    hit = per.merge(pd.DataFrame({"creator": creator_of[cand],
                                  "col": np.arange(len(cand))}), on="creator")
    my_creators = np.zeros((len(me), len(cand)), dtype=np.float32)
    my_creators[row[hit["user_id"]].to_numpy(), hit["col"].to_numpy()] = hit["pref"]
    print(f"  (creator info exists for {len(hit) / my_creators.size:.1%} of "
          f"user-video pairs: the user saw another video by that creator)")
    grade("my creators", my_creators, answers)

    # similar users: match on the non-test videos, then ask the matches about
    # the test videos.
    x_me, _ = centre(liked[np.ix_(me, other)], watched[np.ix_(me, other)])
    x_them, _ = centre(liked[them], watched[them])
    watched_them = watched[np.ix_(them, cand)].astype(np.float32)
    similar = neighbour_scores(x_me, x_them[:, other], x_them[:, cand], watched_them)
    grade("similar users", similar, answers)

    # CONTROL: the same calculation, but listening to EVERY training user
    # instead of only the similar ones. Everyone gets the same list, so this is
    # still popularity, just measured more fairly (a like from someone who
    # likes everything counts for less). If "similar users" only matched this,
    # its gain would be better popularity, not personal taste.
    fair = x_them[:, cand].sum(axis=0) / (watched_them.sum(axis=0) + NEIGHBOUR_SMOOTHING)
    del x_them
    grade("fair like rate (same for all)", np.tile(fair, (len(me), 1)), answers)

    grade("like rate + my tags", base + my_tags, answers)
    grade("like rate + my creators", base + my_creators, answers)

    # ------------------------------------------------------------------ PART 2
    print("\nPART 2 (deliberate cheat): use test answers on half A to predict half B")
    order = np.random.default_rng(0).permutation(len(cand))
    a, b = order[:len(cand) // 2], order[len(cand) // 2:]
    truth_b = answers[:, b]
    print(f"  {'random (expected)':32s} recall@10 {truth_b.mean(axis=1).mean():.3f}   AUC 0.500")

    # others' like rate: share of the OTHER choosing users who liked each video
    others = ((truth_b.sum(axis=0)[None, :] - truth_b)
              / np.maximum(seen[:, b].sum(axis=0)[None, :] - seen[:, b], 1))
    grade("others' like rate", others, truth_b)

    tags_a = tag_preferences(answers[:, a], seen[:, a], tag_grid[cand[a]])
    my_tags_b = tags_a @ tag_share[b].T
    grade("my tags (from half A)", my_tags_b, truth_b)

    # Centre everyone by their half-A like rate, so a user's own half-B answers
    # are never used to predict their own half B.
    x_a, rate_a = centre(answers[:, a], seen[:, a])
    x_b = (answers[:, b] - rate_a[:, None]).astype(np.float32) * seen[:, b]
    similar_b = neighbour_scores(x_a, x_a, x_b, seen[:, b].astype(np.float32),
                                 same_people=True)
    grade("similar users (from half A)", similar_b, truth_b)
    # CONTROL, as in part 1: every other user instead of only the similar ones.
    seen_b = seen[:, b].astype(np.float32)
    fair_b = ((x_b.sum(axis=0)[None, :] - x_b)
              / np.maximum(seen_b.sum(axis=0)[None, :] - seen_b, 1))
    grade("others' fair like rate", fair_b, truth_b)


if __name__ == "__main__":
    main()
