"""Step 6: train the two-tower model and score it.

Training, in short:
  - Take a BATCH of liked (user, video) pairs.
  - Score every user in the batch against every video in the batch.
  - For each user, their own liked video is the right answer; the batch's
    other videos are treated as wrong answers ("in-batch negatives").
  - LOSS (softmax cross-entropy) is high when the right video doesn't score
    clearly above the wrong ones. The optimizer (Adam) nudges every weight a
    little in the direction that lowers the loss. Repeat.
  - One EPOCH = one pass over all 3.4M liked pairs.

Honesty rule: after each epoch we score on the CHOOSING group and keep the
best epoch. The FINAL group is scored once, at the end, for reporting.

Run:  uv run python src/train.py
"""

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import load_training_pairs, load_user_features, load_video_tags
from metrics import (candidate_videos, load_liked_by_user, mean_recall_at_k,
                     split_test_users)
from model import TwoTower

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

BATCH_SIZE = 4096
EPOCHS = 10
LEARNING_RATE = 1e-3
SEED = 0
KS = [10, 50, 100]
SHORT_SECONDS = 5  # the short-video check agreed in step 5


def pick_device():
    """Use the Mac's GPU (MPS) if available, otherwise the CPU."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()  # no learning here, so skip the bookkeeping training needs
def recommend(model, users, candidates, k, device):
    """Top-k candidate video IDs for each user, best first."""
    model.eval()
    user_t = torch.tensor(sorted(users), device=device)
    cand_t = torch.tensor(candidates, device=device)
    scores = model.score_matrix(user_t, cand_t)           # [n_users, n_candidates]
    top = scores.topk(k, dim=1).indices.cpu().numpy()     # positions in candidates
    cand = np.array(candidates)
    model.train()
    return {u: cand[row].tolist() for u, row in zip(sorted(users), top)}


def evaluate(model, users, candidates, liked_by_user, device):
    recs = recommend(model, users, candidates, max(KS), device)
    liked = {u: liked_by_user[u] for u in users}
    return {k: mean_recall_at_k(recs, liked, k) for k in KS}, recs


def in_batch_loss(user_vecs, video_vecs, user_ids, video_ids):
    """Softmax cross-entropy where row i's right answer is column i."""
    logits = user_vecs @ video_vecs.T                      # [B, B] scores
    # Don't count a column as a "wrong answer" if it is really a right one:
    # the same video appearing twice in the batch, or another video liked by
    # the same user. Setting its score to -inf removes it from the softmax.
    same_video = video_ids[:, None] == video_ids[None, :]
    same_user = user_ids[:, None] == user_ids[None, :]
    not_self = ~torch.eye(len(user_ids), dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill((same_video | same_user) & not_self, float("-inf"))
    targets = torch.arange(len(user_ids), device=logits.device)
    return F.cross_entropy(logits, targets)


def main():
    torch.manual_seed(SEED)
    device = pick_device()
    print(f"device: {device}")

    # --- data ---
    pair_users, pair_videos = load_training_pairs()
    user_codes, feature_sizes = load_user_features()
    video_tags, n_tag_codes = load_video_tags()
    test = pd.read_parquet(ROOT / "data" / "processed" / "test.parquet")
    train_table = pd.read_parquet(ROOT / "data" / "processed" / "train.parquet",
                                  columns=["video_id"])
    liked_by_user = {u: {int(v) for v in s}
                     for u, s in load_liked_by_user(test).items()}
    candidates = [int(v) for v in candidate_videos(test)]
    choosing, final = split_test_users(test)
    n_pairs = len(pair_users)
    print(f"training pairs (liked): {n_pairs:,}  batches per epoch: "
          f"{(n_pairs + BATCH_SIZE - 1) // BATCH_SIZE}")

    # --- model ---
    model = TwoTower(len(user_codes), feature_sizes, user_codes,
                     len(video_tags), n_tag_codes, video_tags).to(device)
    n_weights = sum(p.numel() for p in model.parameters())
    print(f"model weights (learnable numbers): {n_weights:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    pair_users, pair_videos = pair_users.to(device), pair_videos.to(device)

    scores, _ = evaluate(model, choosing, candidates, liked_by_user, device)
    print(f"before training (random weights), choosing group recall@10: {scores[10]:.3f}")

    # --- training ---
    history, best = [], {"recall@10": -1}
    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        order = torch.randperm(n_pairs, device=device)     # shuffle each epoch
        total_loss, n_batches = 0.0, 0
        for i in range(0, n_pairs, BATCH_SIZE):
            idx = order[i:i + BATCH_SIZE]
            u, v = pair_users[idx], pair_videos[idx]
            loss = in_batch_loss(model.user_tower(u), model.video_tower(v), u, v)
            optimizer.zero_grad()   # clear the previous batch's nudges
            loss.backward()         # work out which way to nudge every weight
            optimizer.step()        # nudge them
            total_loss += loss.item()
            n_batches += 1
        scores, _ = evaluate(model, choosing, candidates, liked_by_user, device)
        row = {"epoch": epoch, "loss": total_loss / n_batches,
               "recall@10": scores[10], "recall@50": scores[50],
               "seconds": time.time() - start}
        history.append(row)
        print(f"epoch {epoch:2d}  loss {row['loss']:.3f}  choosing recall@10 "
              f"{scores[10]:.3f}  @50 {scores[50]:.3f}  ({row['seconds']:.0f}s)")
        if scores[10] > best["recall@10"]:
            best = {"epoch": epoch, "recall@10": scores[10],
                    "state": copy.deepcopy(model.state_dict())}

    # --- final report, using the best epoch (chosen on the choosing group) ---
    print(f"\nbest epoch on the choosing group: {best['epoch']} "
          f"(recall@10 {best['recall@10']:.3f})")
    model.load_state_dict(best["state"])
    OUT.mkdir(exist_ok=True)
    torch.save(best["state"], OUT / "two_tower.pt")

    final_scores, final_recs = evaluate(model, final, candidates, liked_by_user, device)
    print("\nFINAL GROUP (report these):  " +
          "   ".join(f"recall@{k}: {final_scores[k]:.3f}" for k in KS))

    # Short-video check 1: share of top-10 picks under 5 seconds.
    seconds = test.groupby("video_id")["video_duration"].first() / 1000
    short = set(seconds[seconds < SHORT_SECONDS].index.tolist())
    picks = [v for recs in final_recs.values() for v in recs[:10]]
    short_share = sum(v in short for v in picks) / len(picks)
    print(f"\nSHORT-VIDEO CHECK ({len(short)} candidate videos under {SHORT_SECONDS}s, "
          f"{len(short) / len(candidates):.1%} of candidates)")
    print(f"  share of the model's top-10 picks under {SHORT_SECONDS}s: {short_share:.1%}")

    # Short-video check 2: score with those videos removed entirely.
    kept = [v for v in candidates if v not in short]
    liked_kept = {u: s - short for u, s in liked_by_user.items()}
    no_short, _ = evaluate(model, final, kept, liked_kept, device)
    print(f"  recall@10 with them removed: {no_short[10]:.3f} "
          f"(normal: {final_scores[10]:.3f})")

    # How personal and how popular are the picks?
    watched = train_table["video_id"].value_counts()
    distinct = len(set(picks))
    print("\nWHAT THE MODEL RECOMMENDS (final group, top 10 per user)")
    print(f"  distinct videos across all users' top 10s: {distinct:,} "
          f"(a popularity list would be 10)")
    print(f"  median training watch count of picks: {watched.loc[picks].median():,.0f} "
          f"(all candidates: {watched.loc[candidates].median():,.0f})")
    print(f"  median length of picks: {seconds.loc[picks].median():.1f}s "
          f"(all candidates: {seconds.loc[candidates].median():.1f}s)")

    results = {
        "final": {f"recall@{k}": final_scores[k] for k in KS},
        "best_epoch": best["epoch"], "history": history,
        "short_share_top10": short_share, "recall@10_without_short": no_short[10],
        "distinct_videos_top10": distinct,
    }
    (OUT / "step6_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved model to {OUT / 'two_tower.pt'} and results to step6_results.json")


if __name__ == "__main__":
    main()
