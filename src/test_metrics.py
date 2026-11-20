"""Hand-checked tests for the scorer. Every expected number was worked out on paper.

Run:  uv run python src/test_metrics.py
If any check is wrong, the script stops with an error saying which one.
"""

from metrics import mean_recall_at_k, recall_at_k_for_user


def close(a, b):
    """Compare decimals safely (0.1 + 0.2 is not exactly 0.3 in a computer)."""
    return abs(a - b) < 1e-9


def check(name, got, expected):
    assert close(got, expected), f"FAILED {name}: got {got}, expected {expected}"
    print(f"ok  {name}: {got:.3f}")


# A worked example (k = 3, videos A to H).
#   user 1: recommended A,B,C; liked A,C,F,G -> hits A,C = 2; divide by min(4,3)=3 -> 0.667
#   user 2: recommended D,E,F; liked B       -> hits none = 0; divide by min(1,3)=1 -> 0.000
#   user 3: recommended G,A,H; liked H,G     -> hits G,H = 2; divide by min(2,3)=2 -> 1.000
#   average = (2/3 + 0 + 1) / 3 = 0.556
recs = {1: ["A", "B", "C"], 2: ["D", "E", "F"], 3: ["G", "A", "H"]}
liked = {1: {"A", "C", "F", "G"}, 2: {"B"}, 3: {"H", "G"}}
check("worked example, user 1", recall_at_k_for_user(recs[1], liked[1], 3), 2 / 3)
check("worked example, user 2", recall_at_k_for_user(recs[2], liked[2], 3), 0.0)
check("worked example, user 3", recall_at_k_for_user(recs[3], liked[3], 3), 1.0)
check("worked example, average", mean_recall_at_k(recs, liked, 3), (2 / 3 + 0 + 1) / 3)

# A perfect list must score 1.0; a completely wrong list must score 0.0.
check("perfect list", recall_at_k_for_user(["A", "B"], {"A", "B", "C"}, 2), 1.0)
check("all wrong", recall_at_k_for_user(["X", "Y"], {"A", "B"}, 2), 0.0)

# Only the top k count: a hit at position 4 is ignored when k = 3.
check("hit outside top k", recall_at_k_for_user(["X", "Y", "Z", "A"], {"A"}, 3), 0.0)

# Order inside the top k doesn't matter: hit at position 1 = hit at position 3.
first = recall_at_k_for_user(["A", "X", "Y"], {"A", "B"}, 3)
last = recall_at_k_for_user(["X", "Y", "A"], {"A", "B"}, 3)
check("hit at position 1", first, 0.5)
check("hit at position 3 (same score)", last, 0.5)

# Recommending the same video twice must not count as two hits.
check("duplicate recommendation", recall_at_k_for_user(["A", "A", "X"], {"A", "B"}, 3), 0.5)

# A user with no liked videos is skipped, not scored as 0.
assert recall_at_k_for_user(["A"], set(), 3) is None
check("user with no likes skipped",
      mean_recall_at_k({1: ["A"], 2: ["B"]}, {1: {"A"}, 2: set()}, 1), 1.0)

print("\nall checks passed")

# --- per-user AUC, checked by hand ---
import numpy as np

from metrics import mean_user_auc, user_auc

# One user, four videos. Scores 0.9, 0.8, 0.3, 0.1; liked = 1st and 3rd.
# Pairs (liked vs not liked): 0.9>0.8 win, 0.9>0.1 win, 0.3<0.8 lose, 0.3>0.1 win
# -> 3 wins out of 4 = 0.75
scores = np.array([[0.9, 0.8, 0.3, 0.1]])
liked_grid = np.array([[True, False, True, False]])
check("AUC worked example", user_auc(scores, liked_grid)[0], 0.75)
check("AUC perfect", user_auc(np.array([[3.0, 2.0, 1.0]]),
                              np.array([[True, False, False]]))[0], 1.0)
check("AUC all wrong", user_auc(np.array([[1.0, 2.0, 3.0]]),
                                np.array([[True, False, False]]))[0], 0.0)
# A tie between a liked and a not-liked video counts as half a win.
check("AUC tie", user_auc(np.array([[0.5, 0.5]]), np.array([[True, False]]))[0], 0.5)
# A user who liked nothing has no pairs to compare and is skipped in the average.
two_users = np.array([[0.9, 0.1], [0.9, 0.1]])
two_liked = np.array([[True, False], [False, False]])
check("AUC skips users with no likes", mean_user_auc(two_users, two_liked), 1.0)


# AUC with a "counted" mask: same worked example plus a fifth video that the
# user has no answer for. It must be ignored, so the result is still 0.75.
# (If it were wrongly counted as liked, the result would be 5/6 = 0.833.)
scores5 = np.array([[0.9, 0.8, 0.3, 0.1, 0.95]])
liked5 = np.array([[True, False, True, False, True]])
counted5 = np.array([[True, True, True, True, False]])
check("AUC ignores videos that are not counted",
      user_auc(scores5, liked5, counted5)[0], 0.75)
check("AUC without the mask (for contrast)", user_auc(scores5, liked5)[0], 5 / 6)

print("\nall AUC checks passed")

# --- NDCG@k, checked by hand ---
# Worth of a hit by position: 1 -> 1/log2(2) = 1.000, 2 -> 1/log2(3) = 0.631,
# 3 -> 1/log2(4) = 0.500.
import math

from metrics import mean_ndcg_at_k, ndcg_at_k_for_user

P1, P2, P3 = 1.0, 1 / math.log2(3), 0.5
# Same worked example as recall (k = 3):
#   user 1: recommended A,B,C; liked A,C,F,G -> hits at positions 1 and 3 = 1.5
#           best possible = 3 hits at positions 1,2,3 = 2.131 -> 1.5 / 2.131 = 0.704
#   user 2: no hits -> 0
#   user 3: recommended G,A,H; liked H,G -> hits at positions 1 and 3 = 1.5
#           best possible = 2 hits at positions 1,2 = 1.631 -> 1.5 / 1.631 = 0.920
check("NDCG worked example, user 1", ndcg_at_k_for_user(recs[1], liked[1], 3),
      (P1 + P3) / (P1 + P2 + P3))
check("NDCG worked example, user 2", ndcg_at_k_for_user(recs[2], liked[2], 3), 0.0)
check("NDCG worked example, user 3", ndcg_at_k_for_user(recs[3], liked[3], 3),
      (P1 + P3) / (P1 + P2))
check("NDCG worked example, average", mean_ndcg_at_k(recs, liked, 3),
      ((P1 + P3) / (P1 + P2 + P3) + 0 + (P1 + P3) / (P1 + P2)) / 3)
# Unlike recall, position matters: the same single hit is worth more at the top.
check("NDCG hit at position 1", ndcg_at_k_for_user(["A", "X", "Y"], {"A", "B"}, 3),
      P1 / (P1 + P2))
check("NDCG hit at position 3 (lower)", ndcg_at_k_for_user(["X", "Y", "A"], {"A", "B"}, 3),
      P3 / (P1 + P2))
check("NDCG perfect list", ndcg_at_k_for_user(["A", "B"], {"A", "B", "C"}, 2), 1.0)
check("NDCG duplicate recommendation", ndcg_at_k_for_user(["A", "A", "X"], {"A", "B"}, 3),
      P1 / (P1 + P2))
assert ndcg_at_k_for_user(["A"], set(), 3) is None

print("\nall NDCG checks passed")

# --- grade_scores: the one function every method is graded by ---
# Two users, four candidate videos with IDs 10, 20, 30, 40.
#   user 1: scores 0.9, 0.1, 0.5, 0.5; liked 10 and 40.
#     Ranking: 10 (0.9), then 30 and 40 are tied at 0.5 and keep the
#     candidates' order (30 before 40), then 20.  -> [10, 30, 40, 20]
#     recall@2: top 2 = [10, 30], one hit, divide by min(2, 2) = 2   -> 0.5
#     NDCG@10:  hits at positions 1 and 3 = 1.0 + 0.5 = 1.5;
#               best possible = positions 1 and 2 = 1.631          -> 0.920
#     AUC: 10 beats 20 and 30 (2 wins); 40 beats 20 (1 win) and ties
#          with 30 (half a win) -> 3.5 of 4                         -> 0.875
#   user 2: every score is 0.2 (all tied); liked 20 only.
#     Ranking keeps the candidates' order                -> [10, 20, 30, 40]
#     recall@2: top 2 = [10, 20], one hit, divide by min(1, 2) = 1   -> 1.0
#     NDCG@10:  one hit at position 2 = 0.631; best possible = 1.0   -> 0.631
#     AUC: three ties -> 1.5 of 3                                    -> 0.5
from metrics import grade_scores, load_liked_by_user

grid = np.array([[0.9, 0.1, 0.5, 0.5], [0.2, 0.2, 0.2, 0.2]])
graded, top = grade_scores(grid, [1, 2], [10, 20, 30, 40],
                           {1: {10, 40}, 2: {20}}, ks=(2, 10))
assert top[1] == [10, 30, 40, 20] and top[2] == [10, 20, 30, 40], top
check("grade_scores recall@2", graded[2], (0.5 + 1.0) / 2)
check("grade_scores NDCG@10", graded["ndcg@10"], (1.5 / (P1 + P2) + P2) / 2)
check("grade_scores AUC", graded["auc"], (0.875 + 0.5) / 2)

# load_liked_by_user reads the answer column it is told to.
import pandas as pd

tiny = pd.DataFrame({"user_id": [1, 1, 2], "video_id": [10, 20, 10],
                     "liked": [1, 0, 0], "finished": [0, 1, 1]})
assert load_liked_by_user(tiny) == {1: {10}, 2: set()}
assert load_liked_by_user(tiny, "finished") == {1: {20}, 2: {10}}

print("\nall grading checks passed")
