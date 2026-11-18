"""Step 6 check: did we train too little, or is this the limit of the data?

Train the main model (liked vs skipped, IDs only) for many epochs. After each
epoch, ask the same question on three different sets of pairs:

    "Does the model score this user's liked videos above their skipped ones?"
    (per-user AUC: 0.5 = guessing, 1.0 = always right)

All three sets belong to the CHOOSING group's users:
  trained-on : training pairs the model learned from.
  held-out   : 10% of those users' training pairs, hidden from training.
               The same kind of data as training, but pairs the model never saw.
  test       : the fully observed test videos (the score we report).

How to read the result:
  all three still rising at the end     -> we trained too little
  trained-on rises, the other two flat  -> extra training only memorises the
                                           training pairs (called OVERFITTING);
                                           more epochs cannot help
  held-out clearly above test           -> the test asks something the
                                           training data only partly teaches

Run:  uv run python src/epoch_check.py --epochs 100
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import (load_hard_negatives, load_training_pairs, load_user_features,
                  load_video_tags, read_train_table)
from metrics import (candidate_videos, load_liked_by_user, mean_user_auc,
                     split_test_users)
from model import TwoTower
from train import (LEARNING_RATE, N_SKIPPED, SEED, evaluate, pick_device,
                   sample_skipped, skipped_loss)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
HELD_OUT_SHARE = 0.10
BATCH_SIZE = 4096


def grids(pairs, users, n_videos):
    """Two True/False grids [user, video] for a table of pairs: is there an
    answer for this pair (the user watched it), and was it liked?"""
    row = pd.Series(np.arange(len(users)), index=users)
    r, c = row[pairs["user_id"]].to_numpy(), pairs["video_id"].to_numpy()
    watched = np.zeros((len(users), n_videos), dtype=bool)
    watched[r, c] = True
    liked = np.zeros_like(watched)
    liked[r, c] = pairs["liked"].to_numpy() == 1
    return watched, liked


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    args = p.parse_args()
    torch.manual_seed(SEED)
    device = pick_device()

    # --- data: hide 10% of the choosing users' training pairs ---
    train = read_train_table()
    test = pd.read_parquet(ROOT / "data" / "processed" / "test.parquet")
    choosing, _ = split_test_users(test)
    users = sorted(choosing)
    mine = train["user_id"].isin(choosing).to_numpy()
    hide = mine & (np.random.default_rng(SEED).random(len(train)) < HELD_OUT_SHARE)
    held_out = train[hide]
    trained_on = train[mine & ~hide]
    train = train[~hide]                       # the model only ever sees this
    print(f"choosing users: {len(users)}   their training pairs: {mine.sum():,}   "
          f"hidden from training: {hide.sum():,}")

    user_codes, feature_sizes = load_user_features()
    video_tags, n_tag_codes = load_video_tags()
    n_videos = len(video_tags)
    pair_users, pair_videos = (t.to(device) for t in load_training_pairs(train))
    neg_videos, neg_start, neg_count = (
        t.to(device) for t in load_hard_negatives(len(user_codes), train))
    assert int((neg_count == 0).sum()) == 0, "a user has no skipped videos"
    n_pairs = len(pair_users)

    seen_grid, seen_liked = grids(trained_on, users, n_videos)
    held_grid, held_liked = grids(held_out, users, n_videos)
    liked_by_user = {u: {int(v) for v in s}
                     for u, s in load_liked_by_user(test).items()}
    candidates = [int(v) for v in candidate_videos(test)]

    model = TwoTower(len(user_codes), feature_sizes, user_codes,
                     n_videos, n_tag_codes, video_tags, use_features=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    user_ids = torch.tensor(users, device=device)
    all_videos = torch.arange(n_videos, device=device)

    @torch.no_grad()
    def three_scores():
        model.eval()
        scores = model.score_matrix(user_ids, all_videos).cpu().numpy()
        model.train()
        test_scores, _ = evaluate(model, choosing, candidates, liked_by_user, device)
        return {"trained_on": mean_user_auc(scores, seen_liked, seen_grid),
                "held_out": mean_user_auc(scores, held_liked, held_grid),
                "test": test_scores["auc"], "test_recall@10": test_scores[10]}

    print(f"\n{'epoch':>5}  {'loss':>6}  {'trained-on':>10}  {'held-out':>8}  "
          f"{'test':>6}  {'test recall@10':>14}")
    history = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        order = torch.randperm(n_pairs, device=device)
        total_loss, n_batches = 0.0, 0
        for i in range(0, n_pairs, BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            u, v = pair_users[idx], pair_videos[idx]
            s = sample_skipped(u, N_SKIPPED, neg_videos, neg_start, neg_count)
            skip_vecs = model.video_tower(s.reshape(-1)).reshape(len(u), N_SKIPPED, -1)
            loss = skipped_loss(model.user_tower(u), model.video_tower(v), skip_vecs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        row = {"epoch": epoch, "loss": total_loss / n_batches, **three_scores(),
               "seconds": time.time() - start}
        history.append(row)
        print(f"{epoch:5d}  {row['loss']:6.3f}  {row['trained_on']:10.3f}  "
              f"{row['held_out']:8.3f}  {row['test']:6.3f}  "
              f"{row['test_recall@10']:14.3f}   ({row['seconds']:.0f}s)")

    OUT.mkdir(exist_ok=True)
    (OUT / "epoch_check.json").write_text(json.dumps(history, indent=2))
    print("\nsaved to outputs/epoch_check.json")


if __name__ == "__main__":
    main()
