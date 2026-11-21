"""Step 6: the two-tower model, in plain PyTorch.

User tower:  user ID + user features  -> 64 numbers (the user's embedding)
Video tower: video ID + category tags  -> 64 numbers (the video's embedding)
Score:       dot product of the two embeddings (multiply pairwise, add up).
             High score = "this user will probably like this video".

With use_features=False each tower sees only the ID ("IDs only"). That is what
the models in the README's results use: the features and tags did not help.

nn.Embedding(n, d) is a learnable table with n rows of d numbers: row i is
the embedding for ID (or category code) i. Training adjusts these numbers.
nn.Linear is "multiply by a grid of learnable weights and add a bias".
nn.ReLU replaces negative numbers with 0, which lets stacked Linear layers
learn more than a single straight-line relationship.
"""

import torch
from torch import nn

EMBED_DIM = 64      # length of the final user / video embeddings
ID_DIM = 64         # length of the per-ID embedding inside each tower
FEATURE_DIM = 8     # length of each user-feature value's embedding
TAG_DIM = 32        # length of each tag's embedding
HIDDEN = 128        # width of the middle layer in each tower


class UserTower(nn.Module):
    def __init__(self, n_users, feature_sizes, user_codes, use_features=True):
        super().__init__()
        if not use_features:
            feature_sizes = []  # IDs only: the tower sees nothing but the user ID
        self.id_embedding = nn.Embedding(n_users, ID_DIM)
        # One small table per user-feature column, e.g. activity level.
        self.feature_embeddings = nn.ModuleList(
            nn.Embedding(size, FEATURE_DIM) for size in feature_sizes)
        # The feature codes are fixed data, not learned: register_buffer stores
        # them with the model (and moves them to the GPU with it).
        self.register_buffer("user_codes", user_codes)
        in_dim = ID_DIM + FEATURE_DIM * len(feature_sizes)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, EMBED_DIM))

    def forward(self, user_ids):
        codes = self.user_codes[user_ids]                # [batch, n_columns]
        parts = [self.id_embedding(user_ids)]
        parts += [emb(codes[:, c]) for c, emb in enumerate(self.feature_embeddings)]
        return self.mlp(torch.cat(parts, dim=1))         # [batch, EMBED_DIM]


class VideoTower(nn.Module):
    def __init__(self, n_videos, n_tag_codes, video_tags, use_features=True):
        super().__init__()
        self.use_tags = use_features
        self.id_embedding = nn.Embedding(n_videos, ID_DIM)
        if self.use_tags:
            # padding_idx=0: code 0 means "empty tag slot" and stays all zeros.
            self.tag_embedding = nn.Embedding(n_tag_codes, TAG_DIM, padding_idx=0)
        self.register_buffer("video_tags", video_tags)
        in_dim = ID_DIM + (TAG_DIM if self.use_tags else 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, EMBED_DIM))

    def forward(self, video_ids):
        if not self.use_tags:                            # IDs only
            return self.mlp(self.id_embedding(video_ids))
        tags = self.video_tags[video_ids]                # [batch, MAX_TAGS]
        # Average the embeddings of the video's real tags (ignore empty slots).
        n_real = (tags > 0).sum(dim=1, keepdim=True).clamp(min=1)
        tag_mean = self.tag_embedding(tags).sum(dim=1) / n_real
        parts = [self.id_embedding(video_ids), tag_mean]
        return self.mlp(torch.cat(parts, dim=1))         # [batch, EMBED_DIM]


class TwoTower(nn.Module):
    def __init__(self, n_users, feature_sizes, user_codes,
                 n_videos, n_tag_codes, video_tags, use_features=True):
        """use_features=False gives the "IDs only" model: no user features and
        no video tags, so each tower sees only the user's / video's ID."""
        super().__init__()
        self.user_tower = UserTower(n_users, feature_sizes, user_codes, use_features)
        self.video_tower = VideoTower(n_videos, n_tag_codes, video_tags, use_features)

    def score_matrix(self, user_ids, video_ids):
        """Scores for every (user, video) combination: [n_users, n_videos].
        One matrix multiplication computes all the dot products at once."""
        return self.user_tower(user_ids) @ self.video_tower(video_ids).T
