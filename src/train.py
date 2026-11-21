"""Step 6: train the two-tower model and score it.

Training, in short (shown for in-batch negatives, the standard way to train a
two-tower model; the variants below change what counts as a wrong answer):
  - Take a BATCH of liked (user, video) pairs.
  - Score every user in the batch against every video in the batch.
  - For each user, their own liked video is the right answer; the batch's
    other videos are treated as wrong answers ("in-batch negatives").
  - LOSS (softmax cross-entropy) is high when the right video doesn't score
    clearly above the wrong ones. The optimizer (Adam) nudges every weight a
    little in the direction that lowers the loss. Repeat.
  - One EPOCH = one pass over all 3.4M liked pairs.

Four variants. They differ in what counts as a "wrong answer" in training:
  inbatch : the other videos in the batch only (with features: the first
            version in the README). The model learns what users get shown,
            not what they enjoy.
  hardneg : in-batch, plus ONE video the same user watched but did not like,
            mixed into the same big list of wrong answers (the second
            version). Right idea, but that one video is 1 of ~8,000, so it
            barely counts.
  mixed   : two separate questions per example, added together:
            (a) in-batch, as in the first variant;
            (b) "liked vs skipped": the liked video against 8 videos the same
                user watched but skipped, and nothing else.
  skipped : only question (b). Every wrong answer is a video the user really
            watched and skipped, which is what the fully observed test measures.
            This is the default.

The logQ correction (--logq, step 7). It only applies to the variants that
use in-batch wrong answers (inbatch, mixed):
  A batch is filled with liked pairs, so a video with 4,000 likes turns up in
  batches about 4,000 times as often as a video with 1 like. Every time it
  turns up, it is a wrong answer for the other rows and its score is pushed
  down. Over an epoch the model learns to under-rate popular videos for
  everybody, simply because they were in the room more often.
  The correction: q[v] = the chance that a batch slot holds video v (v's share
  of all liked pairs). During training only, subtract log(q[v]) from every
  score in v's column before computing the loss. log(q) is negative, and more
  negative for rare videos (about -6.7 for the most liked video, -15 for a
  video with one like), so this raises rare videos' training scores the most.
  A frequent video then looks like a weaker wrong answer than a rare one with
  the same plain score, and each appearance pushes it down less: many gentle
  pushes for frequent videos, a few hard pushes for rare ones.
  When recommending, the plain scores are used, with no correction.
  Source: Yi et al., "Sampling-Bias-Corrected Neural Modeling for Large Corpus
  Item Recommendations", RecSys 2019. Same formula as the TensorFlow
  Recommenders library: applied to every column, the row's own right answer
  included.

Honesty rule: after each epoch we score on the CHOOSING group and keep the
best epoch (by per-user AUC, which is steadier than recall@10). The FINAL
group is only scored when --report-final is passed, which is for results
that go in the README, not for experiments.

Run (one of the five runs of the "liked vs skipped" model in the README):
    uv run python src/train.py --variant skipped --no-features --epochs 30 --seed 0 --name main_seed0
The README lists the command for every model in its tables.

Options (everything is scored on the choosing group unless --report-final):
  --variant NAME   inbatch, hardneg, mixed or skipped (default), see above
  --no-features    IDs only: no user features, no video tags. Every model in
                   the README's tables uses this, except the first two versions
  --batch-size N   pairs per batch (default 4096)
  --epochs N       passes over the data (default 10)
  --name NAME      file name for the saved model and results (default: variant)
  --seed N         starting point for the random numbers (default 0). Training
                   starts from random weights and shuffles the data randomly;
                   the seed fixes both, so a different seed is a fair "rerun"
                   that shows how much a score moves by luck alone.
  --logq           apply the logQ correction (variants inbatch and mixed only)
  --label NAME     which answer column to learn from and be graded on.
                   "liked" (default) is the label we use. "finished" is the
                   rejected "watched to the end" label: it trains the model we
                   would have built had we kept it.

The numbers for the README come from src/final_report.py, which scores the
saved models on the final group.
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from data import (load_hard_negatives, load_training_pairs, load_user_features,
                  load_video_tags, read_train_table)
from metrics import (candidate_videos, grade_scores, load_liked_by_user,
                     split_test_users)
from model import TwoTower

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

LEARNING_RATE = 1e-3
SEED = 0
KS = [10, 50, 100]
SHORT_SECONDS = 5  # the short-video check introduced in step 5
N_SKIPPED = 8      # skipped videos per example in the "liked vs skipped" question


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["inbatch", "hardneg", "mixed", "skipped"],
                   default="skipped")
    p.add_argument("--no-features", action="store_true",
                   help="IDs only: drop user features and video tags")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--name", default=None, help="name for saved files")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--label", choices=["liked", "finished"], default="liked")
    p.add_argument("--logq", action="store_true",
                   help="logQ correction for the in-batch wrong answers")
    p.add_argument("--report-final", action="store_true",
                   help="score the FINAL group (only for results we will report)")
    args = p.parse_args()
    if args.logq and args.variant not in ("inbatch", "mixed"):
        p.error("--logq corrects in-batch wrong answers: use it with inbatch or mixed")
    return args


def pick_device():
    """Use the Mac's GPU (MPS) if available, otherwise the CPU."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()  # no learning here, so skip the bookkeeping training needs
def evaluate(model, users, candidates, liked_by_user, device):
    """Score every user against every candidate, then grade.

    Returns ({10: recall@10, 50: ..., 100: ..., "auc": per-user AUC,
              "ndcg@10": NDCG@10}, {user: top-100 video IDs, best first}).
    """
    users = sorted(users)
    model.eval()
    scores = model.score_matrix(torch.tensor(users, device=device),
                                torch.tensor(candidates, device=device))
    model.train()
    scores = scores.cpu().numpy()                          # [n_users, n_candidates]
    return grade_scores(scores, users, candidates, liked_by_user, KS)


def sample_skipped(users, n, neg_videos, neg_start, neg_count):
    """For each user in the batch, n random videos they watched but didn't
    like. Returns video IDs shaped [batch, n]."""
    offset = (torch.rand(len(users), n, device=users.device)
              * neg_count[users][:, None]).long()
    index = (neg_start[users][:, None] + offset).clamp(max=len(neg_videos) - 1)
    return neg_videos[index]


def skipped_loss(user_vecs, pos_vecs, skip_vecs):
    """The "liked vs skipped" question.

    For each row: one liked video and N_SKIPPED videos the same user watched
    but skipped. Softmax over just those videos; the liked one should score
    highest. Nothing the user never saw is involved.
    """
    liked = (user_vecs * pos_vecs).sum(dim=1, keepdim=True)        # [B, 1]
    skipped = (user_vecs[:, None, :] * skip_vecs).sum(dim=2)       # [B, N_SKIPPED]
    logits = torch.cat([liked, skipped], dim=1)                    # right answer = column 0
    targets = torch.zeros(len(user_vecs), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, targets)


def batch_loss(user_vecs, pos_vecs, user_ids, pos_ids, neg_vecs=None, neg_ids=None,
               log_q=None):
    """Softmax cross-entropy where row i's right answer is column i.

    Columns 0..B-1 are the batch's liked videos (in-batch negatives).
    If hard negatives are given, columns B..2B-1 are one watched-but-not-liked
    video per row's user; each row must beat its own user's skipped video too.
    log_q (optional): log of each video's chance of filling a batch slot. If
    given, the logQ correction is applied (see the top of this file).
    """
    logits = user_vecs @ pos_vecs.T                        # [B, B] scores
    if log_q is not None:
        # Column j holds video pos_ids[j]: subtract that video's log(q) from
        # the whole column. Training only; evaluate() uses the plain scores.
        logits = logits - log_q[pos_ids][None, :]
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
    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"run: {name}   variant: {args.variant}   features: {use_features}   "
          f"batch size: {args.batch_size}   seed: {args.seed}   "
          f"label: {args.label}   logQ: {args.logq}   device: {device}")

    # --- data ---
    train_pairs = read_train_table(args.label)
    pair_users, pair_videos = load_training_pairs(train_pairs)
    user_codes, feature_sizes = load_user_features()
    video_tags, n_tag_codes = load_video_tags()
    test = pd.read_parquet(ROOT / "data" / "processed" / "test.parquet")
    train_table = pd.read_parquet(ROOT / "data" / "processed" / "train.parquet",
                                  columns=["video_id"])
    liked_by_user = {u: {int(v) for v in s}
                     for u, s in load_liked_by_user(test, args.label).items()}
    candidates = [int(v) for v in candidate_videos(test)]
    choosing, final = split_test_users(test)
    n_pairs = len(pair_users)
    print(f"training pairs (liked): {n_pairs:,}  batches per epoch: "
          f"{(n_pairs + args.batch_size - 1) // args.batch_size:,}")
    # What the loss would be if the model guessed at random. Useful as a
    # yardstick: a loss close to this means the model has learned very little.
    guessing = {"inbatch": math.log(args.batch_size),
                "hardneg": math.log(2 * args.batch_size),
                "skipped": math.log(N_SKIPPED + 1),
                "mixed": math.log(args.batch_size) + math.log(N_SKIPPED + 1)}
    print(f"loss of pure guessing: {guessing[args.variant]:.2f}")

    # --- model ---
    model = TwoTower(len(user_codes), feature_sizes, user_codes,
                     len(video_tags), n_tag_codes, video_tags,
                     use_features=use_features).to(device)
    n_weights = sum(p.numel() for p in model.parameters())
    print(f"model weights (learnable numbers): {n_weights:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    pair_users, pair_videos = pair_users.to(device), pair_videos.to(device)
    log_q = None
    if args.logq:
        # q[v] = how often video v fills a batch slot = its share of the liked
        # pairs. Videos with no likes never appear in a batch, so the value
        # they get does not matter (clamp only avoids log(0)).
        likes = torch.bincount(pair_videos, minlength=len(video_tags)).float()
        log_q = torch.log(likes.clamp(min=1) / n_pairs)
        print(f"logQ: most liked video has {int(likes.max()):,} likes "
              f"(log q {log_q.max().item():.1f}); a video with one like has "
              f"log q {math.log(1 / n_pairs):.1f}")
        print("(with logQ the loss is not comparable to the 'pure guessing' number)")
    if args.variant != "inbatch":
        neg_videos, neg_start, neg_count = (
            t.to(device) for t in load_hard_negatives(len(user_codes), train_pairs))
        assert int((neg_count == 0).sum()) == 0, "a user has no skipped videos"

    scores, _ = evaluate(model, choosing, candidates, liked_by_user, device)
    print(f"before training (random weights), choosing group: "
          f"recall@10 {scores[10]:.3f}  AUC {scores['auc']:.3f}")

    # --- training ---
    history, best = [], {"auc": -1}
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        order = torch.randperm(n_pairs, device=device)     # shuffle each epoch
        total_loss, n_batches = 0.0, 0
        for i in range(0, n_pairs, args.batch_size):
            idx = order[i:i + args.batch_size]
            u, v = pair_users[idx], pair_videos[idx]
            user_vecs, pos_vecs = model.user_tower(u), model.video_tower(v)
            if args.variant == "inbatch":
                loss = batch_loss(user_vecs, pos_vecs, u, v, log_q=log_q)
            elif args.variant == "hardneg":
                n = sample_skipped(u, 1, neg_videos, neg_start, neg_count)[:, 0]
                loss = batch_loss(user_vecs, pos_vecs, u, v, model.video_tower(n), n)
            else:  # "mixed" or "skipped"
                s = sample_skipped(u, N_SKIPPED, neg_videos, neg_start, neg_count)
                skip_vecs = model.video_tower(s.reshape(-1)).reshape(
                    len(u), N_SKIPPED, -1)                          # [B, N_SKIPPED, 64]
                loss = skipped_loss(user_vecs, pos_vecs, skip_vecs)
                if args.variant == "mixed":
                    loss = loss + batch_loss(user_vecs, pos_vecs, u, v, log_q=log_q)
            optimizer.zero_grad()   # clear the previous batch's nudges
            loss.backward()         # work out which way to nudge every weight
            optimizer.step()        # nudge them
            total_loss += loss.item()
            n_batches += 1
        scores, _ = evaluate(model, choosing, candidates, liked_by_user, device)
        row = {"epoch": epoch, "loss": total_loss / n_batches,
               "recall@10": scores[10], "recall@50": scores[50],
               "auc": scores["auc"], "ndcg@10": scores["ndcg@10"],
               "seconds": time.time() - start}
        history.append(row)
        print(f"epoch {epoch:2d}  loss {row['loss']:.3f}  choosing: recall@10 "
              f"{scores[10]:.3f}  @50 {scores[50]:.3f}  AUC {scores['auc']:.3f}  "
              f"({row['seconds']:.0f}s)")
        if scores["auc"] > best["auc"]:
            best = {"epoch": epoch, "auc": scores["auc"], "recall@10": scores[10],
                    "state": copy.deepcopy(model.state_dict())}

    # --- report, using the best epoch (chosen on the choosing group) ---
    print(f"\nbest epoch on the choosing group (by AUC): {best['epoch']} "
          f"(AUC {best['auc']:.3f}, recall@10 {best['recall@10']:.3f})")
    model.load_state_dict(best["state"])
    OUT.mkdir(exist_ok=True)
    torch.save(best["state"], OUT / f"{name}.pt")

    # Experiments stay on the choosing group; only reported results use final.
    group_name, group = ("final", final) if args.report_final else ("choosing", choosing)
    group_scores, group_recs = evaluate(model, group, candidates, liked_by_user, device)
    print(f"\n{group_name.upper()} GROUP:  " +
          "   ".join(f"recall@{k}: {group_scores[k]:.3f}" for k in KS) +
          f"   AUC: {group_scores['auc']:.3f}   NDCG@10: {group_scores['ndcg@10']:.3f}")

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
                   "batch_size": args.batch_size, "epochs": args.epochs,
                   "seed": args.seed, "label": args.label, "logq": args.logq},
        group_name: {**{f"recall@{k}": group_scores[k] for k in KS},
                     "auc": group_scores["auc"], "ndcg@10": group_scores["ndcg@10"]},
        "best_epoch": best["epoch"], "history": history,
        "short_share_top10": short_share, "recall@10_without_short": no_short[10],
        "distinct_videos_top10": distinct,
    }
    (OUT / f"{name}.json").write_text(json.dumps(results, indent=2))
    print(f"\nsaved model to outputs/{name}.pt and results to outputs/{name}.json")


if __name__ == "__main__":
    main()
