"""Step 6: turn the training data and side information into tensors.

A tensor is PyTorch's array of numbers. Everything the model reads must be a
tensor of integers (IDs, category codes) or decimals.

What we build:
  - training pairs: (user_id, video_id) for every LIKED pair in train.parquet
  - user features:  one row per user, one integer code per categorical column
  - video tags:     one row per video, up to 4 tag codes (0 = "no tag here")
User and video IDs are already 0..N-1, so they double as row numbers.
"""

import ast
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "KuaiRec" / "data"
PROCESSED = ROOT / "data" / "processed"

# Categorical user columns. We use the "_range" versions of the count columns
# (e.g. follower count bucketed as "(10,50]") so everything is a category.
# onehot_feat0..17 are anonymised categories; KuaiRec doesn't say what they mean.
USER_COLUMNS = (
    ["user_active_degree", "is_lowactive_period", "is_live_streamer",
     "is_video_author", "follow_user_num_range", "fans_user_num_range",
     "friend_user_num_range", "register_days_range"]
    + [f"onehot_feat{i}" for i in range(18)]
)
MAX_TAGS = 4  # no video has more than 4 tags


def read_train_table(label="liked"):
    """The training table: one row per (user, video) pair, with liked = 1 or 0.

    label: which column supplies the 1/0 answer. "liked" is the label we use.
    "finished" is the rejected "watched to the end" label; passing it trains a
    model on that label instead (the column is renamed to "liked" here, so the
    rest of the code does not need to know).
    """
    table = pd.read_parquet(PROCESSED / "train.parquet",
                            columns=["user_id", "video_id", label])
    return table.rename(columns={label: "liked"})


def load_training_pairs(train=None):
    """All liked (user, video) pairs from the training table, as two tensors.

    train: pass your own table to train on something other than the full file
    (epoch_check.py hides some pairs this way). Default: the full table.
    """
    if train is None:
        train = read_train_table()
    liked = train[train["liked"] == 1]
    return (torch.tensor(liked["user_id"].to_numpy(), dtype=torch.long),
            torch.tensor(liked["video_id"].to_numpy(), dtype=torch.long))


def load_hard_negatives(n_users, train=None):
    """Every user's watched-but-NOT-liked videos, packed for fast sampling.

    Returns (videos, start, count): user u's not-liked videos are
    videos[start[u] : start[u] + count[u]]. Sorting by user and remembering
    where each user's block starts lets us pick a random one per user on the
    GPU without any Python loops.
    """
    if train is None:
        train = read_train_table()
    neg = train[train["liked"] == 0].sort_values("user_id")
    count = neg.groupby("user_id").size().reindex(range(n_users), fill_value=0)
    start = count.cumsum() - count  # where each user's block begins
    return (torch.tensor(neg["video_id"].to_numpy(), dtype=torch.long),
            torch.tensor(start.to_numpy(), dtype=torch.long),
            torch.tensor(count.to_numpy(), dtype=torch.long))


def load_user_features():
    """Returns (codes, sizes).

    codes: tensor [n_users, n_columns]; codes[u, c] is user u's value for
           column c, as an integer 0..sizes[c]-1. Missing values get code 0.
    sizes: how many different codes each column has (needed to size the
           embedding tables).
    """
    users = pd.read_csv(RAW / "user_features.csv").sort_values("user_id")
    assert (users["user_id"].to_numpy() == range(len(users))).all()
    codes, sizes = [], []
    for col in USER_COLUMNS:
        # factorize: map each distinct value to 0,1,2,...; missing -> -1.
        # +1 shifts so that missing becomes 0 and real values start at 1.
        values, uniques = pd.factorize(users[col])
        codes.append(values + 1)
        sizes.append(len(uniques) + 1)
    codes = torch.tensor(list(zip(*codes)), dtype=torch.long)
    return codes, sizes


def load_video_tags():
    """Returns (tags, n_tag_codes).

    tags: tensor [n_videos, MAX_TAGS]; each row holds the video's tag codes,
          padded with 0 where a video has fewer than MAX_TAGS tags.
          Real tags are stored as tag_id + 1, so 0 can mean "empty slot".
    """
    cats = pd.read_csv(RAW / "item_categories.csv").sort_values("video_id")
    assert (cats["video_id"].to_numpy() == range(len(cats))).all()
    rows = []
    for feat in cats["feat"]:
        tag_ids = [t + 1 for t in ast.literal_eval(feat)]  # "[27, 9]" -> [28, 10]
        rows.append(tag_ids + [0] * (MAX_TAGS - len(tag_ids)))
    tags = torch.tensor(rows, dtype=torch.long)
    return tags, int(tags.max()) + 1
