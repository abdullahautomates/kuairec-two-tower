"""Step 6 diagnostic: what did the model actually learn?

AUC = the chance that a random pair from group A scores higher than a random
pair from group B. 0.5 = can't tell the groups apart; 1.0 = perfect.

  1. EXPOSURE: watched vs never-watched pairs (training data).
     High AUC = the model learned what users get shown / choose to watch.
  2. ENGAGEMENT: liked vs watched-but-not-liked (training data, and the
     choosing test group). High AUC = the model learned what users enjoy among
     the videos they saw. This is what the test measures.

Run:  uv run python src/diagnose_model.py NAME
NAME is any saved run in outputs/, for example inbatch, hardneg or
main_seed0 (default: hardneg, the second version in the README).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import load_user_features, load_video_tags
from metrics import split_test_users
from model import TwoTower

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
SAMPLE = 200_000
rng = np.random.default_rng(0)


def auc(pos_scores, neg_scores):
    """AUC via ranks: how often a positive outranks a negative (ties count half)."""
    s = pd.Series(np.concatenate([pos_scores, neg_scores]))
    ranks = s.rank().to_numpy()
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    return (ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    user_codes, feature_sizes = load_user_features()
    video_tags, n_tag_codes = load_video_tags()
    name = sys.argv[1] if len(sys.argv) > 1 else "hardneg"
    print(f"model: outputs/{name}.pt")
    # The run's settings are saved next to the model; older runs (before the
    # "IDs only" option existed) always used the features.
    config = json.loads((ROOT / "outputs" / f"{name}.json").read_text()).get("config", {})
    model = TwoTower(len(user_codes), feature_sizes, user_codes,
                     len(video_tags), n_tag_codes, video_tags,
                     use_features=config.get("use_features", True))
    model.load_state_dict(torch.load(ROOT / "outputs" / f"{name}.pt",
                                     map_location=device))
    model.to(device).eval()

    @torch.no_grad()
    def pair_scores(users, videos):
        """Score each (user, video) pair: dot product of the two embeddings."""
        out = []
        for i in range(0, len(users), 65536):
            u = torch.tensor(users[i:i + 65536], device=device)
            v = torch.tensor(videos[i:i + 65536], device=device)
            out.append((model.user_tower(u) * model.video_tower(v)).sum(1).cpu())
        return torch.cat(out).numpy()

    train = pd.read_parquet(PROCESSED / "train.parquet",
                            columns=["user_id", "video_id", "liked"])
    n_videos = len(video_tags)

    # 1. Exposure: watched pairs vs random never-watched pairs.
    watched = train.sample(SAMPLE, random_state=0)
    seen = set(train["user_id"].to_numpy() * n_videos + train["video_id"].to_numpy())
    users = rng.choice(train["user_id"].to_numpy(), SAMPLE * 2)
    videos = rng.integers(0, n_videos, SAMPLE * 2)
    unseen = ~np.isin(users * n_videos + videos, list(seen))
    users, videos = users[unseen][:SAMPLE], videos[unseen][:SAMPLE]
    exposure = auc(pair_scores(watched["user_id"].to_numpy(), watched["video_id"].to_numpy()),
                   pair_scores(users, videos))
    print(f"1. EXPOSURE   watched vs never-watched (train):            AUC {exposure:.3f}")

    # 2a. Engagement on training data.
    liked = watched[watched["liked"] == 1]
    not_liked = watched[watched["liked"] == 0]
    eng_train = auc(pair_scores(liked["user_id"].to_numpy(), liked["video_id"].to_numpy()),
                    pair_scores(not_liked["user_id"].to_numpy(), not_liked["video_id"].to_numpy()))
    print(f"2. ENGAGEMENT liked vs watched-not-liked (train):           AUC {eng_train:.3f}")

    # 2b. Engagement on the CHOOSING test group (everyone saw everything).
    # Not the final group: that one is only for reporting results.
    test = pd.read_parquet(PROCESSED / "test.parquet")
    choosing, _ = split_test_users(test)
    t = test[test["user_id"].isin(choosing)].sample(SAMPLE, random_state=0)
    tl, tn = t[t["liked"] == 1], t[t["liked"] == 0]
    eng_test = auc(pair_scores(tl["user_id"].to_numpy(), tl["video_id"].to_numpy()),
                   pair_scores(tn["user_id"].to_numpy(), tn["video_id"].to_numpy()))
    print(f"   ENGAGEMENT liked vs not-liked (choosing test group):     AUC {eng_test:.3f}")


if __name__ == "__main__":
    main()
