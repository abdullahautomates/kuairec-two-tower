"""Sanity check: grade RANDOM recommendations on the real test data.

Random picks know nothing, so their score should equal the average share of
videos each user liked (about 0.234 with our label). If the scorer printed
something very different, the scorer would be wrong.

Run:  uv run python src/score_random.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from metrics import candidate_videos, load_liked_by_user, mean_recall_at_k

ROOT = Path(__file__).resolve().parent.parent

test = pd.read_parquet(ROOT / "data" / "processed" / "test.parquet")
liked_by_user = load_liked_by_user(test)
candidates = np.array(candidate_videos(test))

# A "seed" fixes the random numbers, so running this again gives the same result.
rng = np.random.default_rng(seed=0)
recommendations = {user: list(rng.permutation(candidates)[:100])
                   for user in liked_by_user}

# What random SHOULD score: each user's share of liked videos, averaged.
expected = np.mean([len(liked) / len(candidates) for liked in liked_by_user.values()])
print(f"test users: {len(liked_by_user):,}   candidate videos: {len(candidates):,}")
print(f"expected random score (average share of videos liked): {expected:.3f}")
for k in [10, 50, 100]:
    print(f"random recall@{k}: {mean_recall_at_k(recommendations, liked_by_user, k):.3f}")
