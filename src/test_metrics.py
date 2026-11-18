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
