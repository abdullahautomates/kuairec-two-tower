"""Step 8: the demo. For a few users, show their history next to what the
models recommend, and mark every recommendation right or wrong.

    uv run python src/demo.py                    # 3 users picked at random (seed 0)
    uv run python src/demo.py --seed 7           # 3 other random users
    uv run python src/demo.py --users 14 3210    # specific test users

Why every recommendation can be marked: test users saw nearly every test
video, so for each recommended video we know whether that person liked it.

Three choices that keep the demo honest:
  - Users are picked at random (fixed seed), from the FINAL group: the users
    no decision in this project was based on. They are not hand-picked.
  - Each model was trained five times. The demo uses the MIDDLE run of the
    five (by its score on the choosing group): not the luckiest, not the
    unluckiest.
  - After the examples, the same counts are printed for all users in the
    final group, so three examples are not mistaken for the whole picture.

Models shown:
  step 7: "mixed + logQ"      best top-10 score of the project
  step 6: "liked vs skipped"  best ranking of the whole list, more varied picks
  popularity (step 5):        the same 10 videos for everyone

Video categories come from kuairec_caption_category.csv. The names there are
in Chinese; the English names below are our own translations. Captions are
not shown: they are the dataset's content and we do not republish it.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_user_features, load_video_tags
from final_report import LOGQ_RUNS, MAIN_RUNS, model_scores
from metrics import (candidate_videos, liked_matrix, load_liked_by_user,
                     split_test_users)
from popularity import popularity_table, ranked_list

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "KuaiRec" / "data"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
K = 10
N_FAVOURITES = 3  # "favourite categories" = the 3 a user liked most often
WIDTH = 52        # characters per model column in the side-by-side lists

CATEGORY_ENGLISH = {
    "民生资讯": "Everyday news", "时尚": "Fashion", "其他": "Other",
    "颜值": "Good looks", "情感": "Feelings & relationships", "美食": "Food",
    "明星娱乐": "Celebrities", "舞蹈": "Dance", "生活": "Daily life",
    "自拍": "Selfies", "喜剧": "Comedy", "亲子": "Parents & children",
    "运动": "Sports", "艺术": "Art", "二次元": "Anime", "三农": "Farming & rural life",
    "宠物": "Pets", "摄影": "Photography", "音乐": "Music", "健康": "Health",
    "健身": "Fitness", "美妆": "Beauty & make-up", "短剧": "Short dramas",
    "旅游": "Travel", "汽车": "Cars", "游戏": "Games",
    "科学与法律": "Science & law", "军事": "Military", "读书": "Books",
    "影视综": "Film, TV & shows", "教育": "Education", "人文": "Culture & history",
    "时政资讯": "Politics news", "房产家居": "Homes & property",
    "高新数码": "Tech & gadgets", "星座命理": "Astrology", "财经": "Finance",
    "奇人异象": "Oddities", "娱乐八卦": "Celebrity gossip", "生活泛拍": "Everyday clips",
}


def load_categories(n_videos):
    """English category name for every video ID (an array indexed by ID)."""
    table = pd.read_csv(RAW / "kuairec_caption_category.csv", engine="python",
                        usecols=["video_id", "first_level_category_name"])
    # A few rows of this file are broken (text where the ID should be): skip them.
    table["video_id"] = pd.to_numeric(table["video_id"], errors="coerce")
    table = table.dropna(subset=["video_id"]).drop_duplicates("video_id")
    names = table.set_index(table["video_id"].astype(int))["first_level_category_name"]
    names = names.reindex(range(n_videos))
    return names.map(lambda name: CATEGORY_ENGLISH.get(name, "Unknown")).to_numpy()


def typical_run(runs):
    """Of several runs of one recipe, the middle one by recall@10 on the
    choosing group."""
    def choosing_score(run):
        return json.loads((OUT / f"{run}.json").read_text())["choosing"]["recall@10"]
    return sorted(runs, key=choosing_score)[len(runs) // 2]


def top_k(scores):
    """Column numbers of each user's K best-scoring candidates, best first."""
    return np.argsort(-scores, axis=1, kind="stable")[:, :K]


def show_user(user, row, train, categories, candidates, seconds, liked_share,
              truth, lists, popular):
    """Print one user's history and both models' top 10, marked right/wrong."""
    history = train[train["user_id"] == user]
    n_watched, n_liked = len(history), int(history["liked"].sum())
    print("=" * 100)
    print(f"USER {user}")
    print(f"  Training history: watched {n_watched:,} videos, liked {n_liked:,} "
          f"({n_liked / n_watched:.0%}).")
    profile = (pd.DataFrame({"category": categories[history["video_id"].to_numpy()],
                             "liked": history["liked"].to_numpy()})
               .groupby("category")["liked"].agg(["sum", "size"])
               .sort_values("sum", ascending=False).head(5))
    print("  The 5 kinds of video they liked most often (liked / watched, and the rate):")
    for name, r in profile.iterrows():
        print(f"    {name:<28}{int(r['sum']):>4} / {int(r['size']):<5}({r['sum'] / r['size']:.0%})")
    favourites = set(profile.index[:N_FAVOURITES])
    share = truth[row].mean()
    print(f"  In the test they liked {int(truth[row].sum()):,} of the {len(candidates):,} "
          f"test videos ({share:.0%}),\n  so 10 random picks would contain about "
          f"{K * share:.1f} liked videos.\n")

    names = list(lists)
    print("  " + "".join(f"{name:<{WIDTH}}" for name in names))
    column = f"    {'kind of video':<25}{'length':>6}  {'liked?':<6}{'all':>4}"
    print("  " + f"{column:<{WIDTH}}" * len(names))
    for rank in range(K):
        cells = []
        for name in names:
            c = lists[name][row, rank]
            kind = categories[candidates[c]]
            star = "*" if kind in favourites else " "   # one of their 3 favourites
            mark = "LIKED" if truth[row, c] else "no"
            cells.append(f"{rank + 1:>2}.{star}{kind:<25}{seconds[c]:>5.0f}s  {mark:<6}"
                         f"{liked_share[c]:>4.0%}")
        print("  " + "".join(f"{cell:<{WIDTH}}" for cell in cells))
    hits = {name: int(truth[row, lists[name][row]].sum()) for name in names}
    print("  " + "".join(f"{'    liked: ' + str(hits[name]) + ' of 10':<{WIDTH}}"
                         for name in names))
    both = len(set(lists[names[0]][row]) & set(lists[names[1]][row]))
    print(f"\n  Videos on both lists: {both}.   Popularity's list (the same 10 for "
          f"everyone): {int(truth[row, popular].sum())} of 10 liked.")
    print(f"  * = one of this user's {N_FAVOURITES} most-liked kinds of video.   "
          f"liked? = did THIS user like it in the test.\n"
          f"  all = share of all test users who liked the video.\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--users", type=int, nargs="*", help="test user IDs to show")
    p.add_argument("--seed", type=int, default=0, help="which random users to show")
    p.add_argument("--n", type=int, default=3, help="how many random users")
    args = p.parse_args()

    train = pd.read_parquet(PROCESSED / "train.parquet",
                            columns=["user_id", "video_id", "video_duration", "liked"])
    test = pd.read_parquet(PROCESSED / "test.parquet")
    _, final = split_test_users(test)
    users = sorted(test["user_id"].unique().tolist())          # every test user
    row_of = {u: i for i, u in enumerate(users)}
    candidates = [int(v) for v in candidate_videos(test)]
    liked_by_user = {u: {int(v) for v in s}
                     for u, s in load_liked_by_user(test).items()}
    truth = liked_matrix(liked_by_user, users, candidates)     # [users, candidates]
    liked_share = test.groupby("video_id")["liked"].mean().reindex(candidates).to_numpy()

    user_codes, feature_sizes = load_user_features()
    video_tags, n_tag_codes = load_video_tags()
    categories = load_categories(len(video_tags))
    table = popularity_table(train, candidates)
    seconds = table["seconds"].to_numpy()
    column_of = {v: j for j, v in enumerate(candidates)}
    popular = [column_of[v] for v in ranked_list(table, "like_rate")[:K]]

    recipes = {"STEP 7 MODEL (mixed + logQ)": LOGQ_RUNS,
               "STEP 6 MODEL (liked vs skipped)": MAIN_RUNS}
    lists, every_run = {}, {}
    for name, runs in recipes.items():
        every_run[name] = []
        for run in runs:
            scores, _ = model_scores(run, users, candidates, user_codes,
                                     feature_sizes, video_tags, n_tag_codes)
            every_run[name].append(top_k(scores))
        shown = typical_run(runs)
        lists[name] = every_run[name][runs.index(shown)]
        print(f"{name}: showing run {shown} (the middle one of five)")

    if args.users:
        chosen = args.users
        missing = [u for u in chosen if u not in row_of]
        if missing:
            p.error(f"not test users: {missing}")
    else:
        rng = np.random.default_rng(args.seed)
        chosen = sorted(rng.choice(sorted(final), args.n, replace=False).tolist())
        print(f"{args.n} users picked at random from the final group (seed {args.seed}): {chosen}")
    print()
    for user in chosen:
        show_user(user, row_of[user], train, categories, candidates, seconds,
                  liked_share, truth, lists, popular)

    # ---- the same counts for every user in the final group ----
    rows = np.array([row_of[u] for u in sorted(final)])
    t = truth[rows]
    hits = {name: np.take_along_axis(t, top[rows], axis=1).sum(axis=1)
            for name, top in lists.items()}
    hits["POPULARITY (same 10 for everyone)"] = t[:, popular].sum(axis=1)
    print("=" * 100)
    print(f"ALL {len(rows)} USERS IN THE FINAL GROUP: liked videos among the 10 recommended")
    print(f"  {'':<36}{'average':>8}{'0-1':>8}{'2-3':>8}{'4-5':>8}{'6+':>8}"
          f"   <- share of users with that many hits")
    for name, h in hits.items():
        line = (f"  {name:<36}{h.mean():>8.2f}{(h <= 1).mean():>8.0%}"
                f"{((h >= 2) & (h <= 3)).mean():>8.0%}{((h >= 4) & (h <= 5)).mean():>8.0%}"
                f"{(h >= 6).mean():>8.0%}")
        if name in every_run:   # the shown run is one of five: also give their average
            five = np.mean([np.take_along_axis(t, top[rows], axis=1).sum(axis=1).mean()
                            for top in every_run[name]])
            line += f"   (average of all five runs: {five:.2f})"
        print(line)
    print(f"  {'random picks (expected)':<36}{K * t.mean(axis=1).mean():>8.2f}")

    names = list(lists)
    pop_hits = hits["POPULARITY (same 10 for everyone)"]
    print("\n  User by user, against popularity:")
    for name in names:
        print(f"    {name:<34} more hits for {(hits[name] > pop_hits).mean():.0%} of users, "
              f"the same for {(hits[name] == pop_hits).mean():.0%}, "
              f"fewer for {(hits[name] < pop_hits).mean():.0%}")
    a, b = hits[names[0]], hits[names[1]]
    print(f"  Step 7 model against step 6 model: more hits for {(a > b).mean():.0%} "
          f"of users, the same for {(a == b).mean():.0%}, fewer for {(a < b).mean():.0%}")

    # Do the recommendations follow each user's favourite categories?
    mine = train[train["user_id"].isin(final) & (train["liked"] == 1)]
    codes, kinds = pd.factorize(categories)
    counts = np.zeros((len(users), len(kinds)), dtype=int)
    np.add.at(counts, (mine["user_id"].map(row_of).to_numpy(),
                       codes[mine["video_id"].to_numpy()]), 1)
    favourite = np.zeros_like(counts, dtype=bool)
    best = np.argsort(-counts, axis=1, kind="stable")[:, :N_FAVOURITES]
    np.put_along_axis(favourite, best, True, axis=1)
    in_favourites = favourite[rows][:, codes[candidates]]      # [final users, candidates]
    # CONTROL: hand every user the list made for a different, random user. If
    # that hits their favourite kinds just as often, the lists are not following
    # personal favourites; the model is simply recommending the big categories
    # to everybody.
    other = np.random.default_rng(0).permutation(len(rows))
    print(f"\n  Share of recommended videos that are one of the user's {N_FAVOURITES} "
          f"most-liked kinds of video:")
    print(f"    {'':<34} {'own list':>9} {'someone else' + chr(39) + 's list':>21}")
    for name, top in lists.items():
        own = np.take_along_axis(in_favourites, top[rows], axis=1).mean()
        swapped = np.take_along_axis(in_favourites, top[rows][other], axis=1).mean()
        print(f"    {name:<34} {own:>9.0%} {swapped:>21.0%}")
    print(f"    {'any test video (what chance gives)':<34} {in_favourites.mean():>9.0%}")
    distinct = {name: len(np.unique(top[rows])) for name, top in lists.items()}
    print("\n  Different videos recommended across all these users: " +
          ",  ".join(f"{name.split(' (')[0].lower()} {n:,}" for name, n in distinct.items()))


if __name__ == "__main__":
    main()
