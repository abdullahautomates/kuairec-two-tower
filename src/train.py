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

Two variants:
  inbatch : wrong answers = the other videos in the batch only (run 1).
            The model learns what users get shown, not what they enjoy.
  hardneg : additionally, for each liked pair, one video the SAME user
            watched but did NOT like ("hard negative"). The model must rank
            enjoyed above skipped among videos the user actually saw, which
            is what the fully observed test measures.

Honesty rule: after each epoch we score on the CHOOSING group and keep the
best epoch. The FINAL group is only scored when --report-final is passed,
which is for results that go in the README, not for experiments.

Run:  uv run python src/train.py --variant hardneg --report-final

Options for experiments (scored on the choosing group only):
  --no-features    IDs only: no user features, no video tags
  --batch-size N   pairs per batch (default 4096)
  --epochs N       passes over the data (default 10)
  --name NAME      file name for the saved model and results (default: variant)
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data import (load_hard_negatives, load_training_pairs, load_user_features,
                  load_video_tags)
from metrics import (candidate_videos, liked_matrix, load_liked_by_user,
                     mean_recall_at_k, mean_user_auc, split_test_users)
from model import TwoTower

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

LEARNING_RATE = 1e-3
SEED = 0
KS = [10, 50, 100]
SHORT_SECONDS = 5  # the short-video check agreed in step 5


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["inbatch", "hardneg"], default="hardneg")
    p.add_argument("--no-features", action="store_true",
                   help="IDs only: drop user features and video tags")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--name", default=None, help="name for saved files")
    p.add_argument("--report-final", action="store_true",
                   help="score the FINAL group (only for results we will report)")
    return p.parse_args()


def pick_device():
    """Use the Mac's GPU (MPS) if available, otherwise the CPU."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()  # no learning here, so skip the bookkeeping training needs
def evaluate(model, users, candidates, liked_by_user, device):
    """Score every user against every candidate, then grade.

    Returns ({10: recall@10, 50: ..., 100: ..., "auc": per-user AUC},
             {user: top-100 video IDs, best first}).
    """
    users = sorted(users)
    model.eval()
    scores = model.score_matrix(torch.tensor(users, device=device),
                                torch.tensor(candidates, device=device))
    model.train()
    scores = scores.cpu().numpy()                          # [n_users, n_candidates]
    top = np.argsort(-scores, axis=1)[:, :max(KS)]         # best first
    cand = np.array(candidates)
    recs = {u: cand[row].tolist() for u, row in zip(users, top)}
    liked = {u: liked_by_user[u] for u in users}
    result = {k: mean_recall_at_k(recs, liked, k) for k in KS}
    result["auc"] = mean_user_auc(scores, liked_matrix(liked_by_user, users, candidates))
    return result, recs


def sample_hard_negatives(users, neg_videos, neg_start, neg_count):
    """For each user in the batch, one random video they watched but didn't like."""
    offset = (torch.rand(len(users), device=users.device) * neg_count[users]).long()
    index = (neg_start[users] + offset).clamp(max=len(neg_videos) - 1)
    return neg_videos[index]


def batch_loss(user_vecs, pos_vecs, user_ids, pos_ids, neg_vecs=None, neg_ids=None):
    """Softmax cross-entropy where row i's right answer is column i.

    Columns 0..B-1 are the batch's liked videos (in-batch negatives).
    If hard negatives are given, columns B..2B-1 are one watched-but-not-liked
    video per row's user; each row must beat its own user's skipped video too.
    """
    logits = user_vecs @ pos_vecs.T                        # [B, B] scores
    # Don't count a column as a "wrong answer" if it is really a right one:
    # the same video appearing twice in the batch, or another video liked by
    # the same user. Setting its score to -inf removes it from the softmax.
    same_video = pos_ids[:, None] == pos_ids[None, :]
    same_user = user_ids[:, None] == user_ids[None, :]
    not_self = ~torch.eye(len(user_ids), dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill((same_video | same_user) & not_self, float("-inf"))
    if neg_vecs is not None:
        neg_logits = user_vecs @ neg_vecs.T                # [B, B] scores
        # Another user's skipped video might be this row's liked video: drop it.
        neg_logits = neg_logits.masked_fill(neg_ids[None, :] == pos_ids[:, None],
                                            float("-inf"))
        logits = torch.cat([logits, neg_logits], dim=1)    # [B, 2B]
    targets = torch.arange(len(user_ids), device=logits.device)
    return F.cross_entropy(logits, targets)


def main():
    args = parse_args()
    name = args.name or args.variant
    use_features = not args.no_features
    torch.manual_seed(SEED)
    device = pick_device()
    print(f"run: {name}   variant: {args.variant}   features: {use_features}   "
          f"batch size: {args.batch_size}   device: {device}")

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
    n_columns = args.batch_size * (2 if args.variant == "hardneg" else 1)
    print(f"training pairs (liked): {n_pairs:,}  batches per epoch: "
          f"{(n_pairs + args.batch_size - 1) // args.batch_size:,}")
    print(f"loss of pure guessing among {n_columns:,} videos: {math.log(n_columns):.2f}")

    # --- model ---
    model = TwoTower(len(user_codes), feature_sizes, user_codes,
                     len(video_tags), n_tag_codes, video_tags,
                     use_features=use_features).to(device)
    n_weights = sum(p.numel() for p in model.parameters())
    print(f"model weights (learnable numbers): {n_weights:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    pair_users, pair_videos = pair_users.to(device), pair_videos.to(device)
    if args.variant == "hardneg":
        neg_videos, neg_start, neg_count = (
            t.to(device) for t in load_hard_negatives(len(user_codes)))

    scores, _ = evaluate(model, choosing, candidates, liked_by_user, device)
    print(f"before training (random weights), choosing group: "
          f"recall@10 {scores[10]:.3f}  AUC {scores['auc']:.3f}")

    # --- training ---
    history, best = [], {"recall@10": -1}
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        order = torch.randperm(n_pairs, device=device)     # shuffle each epoch
        total_loss, n_batches = 0.0, 0
        for i in range(0, n_pairs, args.batch_size):
            idx = order[i:i + args.batch_size]
            u, v = pair_users[idx], pair_videos[idx]
            if args.variant == "hardneg":
                n = sample_hard_negatives(u, neg_videos, neg_start, neg_count)
                loss = batch_loss(model.user_tower(u), model.video_tower(v), u, v,
                                  model.video_tower(n), n)
            else:
                loss = batch_loss(model.user_tower(u), model.video_tower(v), u, v)
            optimizer.zero_grad()   # clear the previous batch's nudges
            loss.backward()         # work out which way to nudge every weight
            optimizer.step()        # nudge them
            total_loss += loss.item()
            n_batches += 1
        scores, _ = evaluate(model, choosing, candidates, liked_by_user, device)
        row = {"epoch": epoch, "loss": total_loss / n_batches,
               "recall@10": scores[10], "recall@50": scores[50],
               "auc": scores["auc"], "seconds": time.time() - start}
        history.append(row)
        print(f"epoch {epoch:2d}  loss {row['loss']:.3f}  choosing: recall@10 "
              f"{scores[10]:.3f}  @50 {scores[50]:.3f}  AUC {scores['auc']:.3f}  "
              f"({row['seconds']:.0f}s)")
        if scores[10] > best["recall@10"]:
            best = {"epoch": epoch, "recall@10": scores[10],
                    "state": copy.deepcopy(model.state_dict())}

    # --- report, using the best epoch (chosen on the choosing group) ---
    print(f"\nbest epoch on the choosing group: {best['epoch']} "
          f"(recall@10 {best['recall@10']:.3f})")
    model.load_state_dict(best["state"])
    OUT.mkdir(exist_ok=True)
    torch.save(best["state"], OUT / f"{name}.pt")

    # Experiments stay on the choosing group; only reported results use final.
    group_name, group = ("final", final) if args.report_final else ("choosing", choosing)
    group_scores, group_recs = evaluate(model, group, candidates, liked_by_user, device)
    print(f"\n{group_name.upper()} GROUP:  " +
          "   ".join(f"recall@{k}: {group_scores[k]:.3f}" for k in KS) +
          f"   AUC: {group_scores['auc']:.3f}")

    # Short-video check 1: share of top-10 picks under 5 seconds.
    seconds = test.groupby("video_id")["video_duration"].first() / 1000
    short = set(seconds[seconds < SHORT_SECONDS].index.tolist())
    picks = [v for recs in group_recs.values() for v in recs[:10]]
    short_share = sum(v in short for v in picks) / len(picks)
    print(f"\nSHORT-VIDEO CHECK ({len(short)} candidate videos under {SHORT_SECONDS}s, "
          f"{len(short) / len(candidates):.1%} of candidates)")
    print(f"  share of the model's top-10 picks under {SHORT_SECONDS}s: {short_share:.1%}")

    # Short-video check 2: score with those videos removed entirely.
    kept = [v for v in candidates if v not in short]
    liked_kept = {u: s - short for u, s in liked_by_user.items()}
    no_short, _ = evaluate(model, group, kept, liked_kept, device)
    print(f"  recall@10 with them removed: {no_short[10]:.3f} "
          f"(normal: {group_scores[10]:.3f})")

    # How personal and how popular are the picks?
    watched = train_table["video_id"].value_counts()
    distinct = len(set(picks))
    print(f"\nWHAT THE MODEL RECOMMENDS ({group_name} group, top 10 per user)")
    print(f"  distinct videos across all users' top 10s: {distinct:,} "
          f"(a popularity list would be 10)")
    print(f"  median training watch count of picks: {watched.loc[picks].median():,.0f} "
          f"(all candidates: {watched.loc[candidates].median():,.0f})")
    print(f"  median length of picks: {seconds.loc[picks].median():.1f}s "
          f"(all candidates: {seconds.loc[candidates].median():.1f}s)")

    results = {
        "name": name,
        "config": {"variant": args.variant, "use_features": use_features,
                   "batch_size": args.batch_size, "epochs": args.epochs},
        group_name: {**{f"recall@{k}": group_scores[k] for k in KS},
                     "auc": group_scores["auc"]},
        "best_epoch": best["epoch"], "history": history,
        "short_share_top10": short_share, "recall@10_without_short": no_short[10],
        "distinct_videos_top10": distinct,
    }
    (OUT / f"{name}.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved model to outputs/{name}.pt and results to outputs/{name}.json")


if __name__ == "__main__":
    main()
