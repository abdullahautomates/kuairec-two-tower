# Two-tower video recommender on KuaiRec

A small, carefully tested learning project: a **two-tower retrieval model**
trained on [KuaiRec](https://kuairec.com), a public dataset from the
short-video app Kuaishou.

One neural network turns a user into a list of 64 numbers, a second network
does the same for a video, and the videos whose numbers best match a user's are
recommended. Meta has described the same general idea as one of the retrieval
sources behind Instagram Explore
([Scaling the Instagram Explore recommendations system](https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/), 2023).

**What this is not.** It is only the *retrieval* stage of a recommender: the
step that picks a short list of candidates out of thousands. Real systems add
several ranking stages on top. It is not Instagram's algorithm, it uses no
Instagram data, and it is not new research. The value of the project is in how
the model was tested, and in what went wrong along the way.

## Results at a glance

Each method recommends 10 of 3,327 videos to each of 706 test users. These
users were held back for reporting: every choice between versions was made on
the other half of the test users (see [How methods are tested](#how-methods-are-tested)).

| Method | recall@10 | NDCG@10 | AUC |
|---|---|---|---|
| Random picks | 0.233 | 0.233 | 0.500 |
| Popularity (highest like rate) | 0.248 | 0.249 | 0.525 |
| Similar users (counting, no neural network) | 0.316 | 0.313 | 0.559 |
| Two-tower, trained on "liked vs skipped" | 0.295 ± 0.006 | 0.294 ± 0.006 | 0.558 ± 0.001 |
| **Two-tower, mixed training + logQ correction** | **0.317** ± 0.007 | **0.318** ± 0.007 | 0.551 ± 0.003 |

- **recall@10**: the share of the 10 recommended videos that the user liked.
  (Nearly every test user likes more than 10 videos, so here it is the same
  as precision@10.)
- **NDCG@10**: the same, with extra credit for hits near the top of the list.
- **AUC**: over the whole list of 3,327 videos, how often a liked video is
  ranked above a not-liked one. 0.5 is guessing.
- **±**: the spread (standard deviation) over five training runs that differ
  only in their random starting point.

How to read it:

- The best two-tower model gets about **3.2 of 10** recommendations right,
  against 2.5 for popularity and 2.3 for random picks.
- It **does not beat a simple counting method** ("people whose likes agree
  with yours liked this"). The two are level. Under a second, stricter way of
  marking the test (explained below) the model is ahead, 0.306 against 0.290.
- The scores are modest because the test is hard. "Liked" marks about 23% of
  watches, and even a method that cheats by reading each user's real test
  answers on half of the videos only reaches 0.396 (measured on the other
half of the test users).

## What the project found

### 1. A label can give an excellent score that means almost nothing

KuaiRec has no like button in its logs, only watch time. The obvious label is
"the user watched the video to the end".

A model trained on that label scores **recall@10 = 0.887**. It looks like a
great result. Three checks show what it really is:

- 85% of its recommendations are under 5 seconds long. Only 1.1% of the
  videos are that short.
- A rule with no learning at all, "recommend the shortest videos to
  everyone", scores 0.897 on the same label and beats the trained model.
- Videos under 7 seconds are watched to the end 70% of the time; videos over
  60 seconds, under 1% of the time. The label mostly measures length.

The label used instead is length-aware: a video counts as liked if the user's
watch time is in the top third *among videos of similar length*.

### 2. Train the model on the question the test asks

The first version was trained the standard way for two-tower models, with
**in-batch negatives**: for each user, the video they liked is the right
answer and the other videos in the same batch of training examples are
treated as wrong answers. It tied with popularity (0.251 against 0.248).

The reason: that training teaches the model *which videos a user gets shown*,
because the "wrong answers" are mostly videos the user never saw. The test
asks something different. Test users saw nearly every video, so the question
is *which of the videos you saw did you like?*

Changing the training question to exactly that (each liked video against 8
videos the same user watched and skipped) raised recall@10 to 0.295 and AUC
from 0.507 to 0.558. Training longer, a larger model, and adding user
features and video tags did not help.

### 3. Always report the simple baseline

A counting method with no neural network (find the training users whose
likes agree with yours, recommend what they liked) matches the two-tower
model. This is a known pattern: a 2019 study re-ran 18 published neural
recommenders, could reproduce 7, and found that 6 of those were often beaten
by simple methods ([Ferrari Dacrema et al., 2019](https://arxiv.org/abs/1907.06902)).
Leaving the counting method out would have made the model look better than
it is.

### 4. The logQ correction helps the top of the list

With in-batch negatives, popular videos turn up in batches far more often,
so they are used as "wrong answers" far more often and the model learns to
under-rate them. The **logQ correction**
([Yi et al., 2019](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/))
cancels this.

It did what it is designed to do: the model's picks moved from less popular
videos to more popular ones. Added to a model trained on both questions at
once ("mixed"), it gave the best top-10 of the project (0.317). It did not
give the best ranking of the whole list; the "liked vs skipped" model keeps
that (AUC 0.558 against 0.551), and recommends about three times as many
different videos across users.

## How it works

### Data

KuaiRec has two tables of watch logs from the same app:

| | Training | Test |
|---|---|---|
| Source table | `big_matrix` | `small_matrix` |
| Users × videos | 7,176 × 10,728 | 1,411 × 3,327 |
| (user, video) pairs | 10,300,969 | 4,676,570 |
| How it was collected | videos the app chose to show | nearly every user saw every video (99.6%) |

No (user, video) pair appears in both tables. 968,004 exact duplicate log
rows were removed from the training table first.

The dataset is not included in this repository. See [Run it yourself](#run-it-yourself).

### What counts as "liked"

1. **Engagement** of a (user, video) pair = watch time divided by video
   length, added up over all of that user's watches of the video.
2. Videos are split into 107 groups of similar length.
3. In each group, the **bar** is the engagement that only the top third of
   training pairs reach.
4. **liked** = engagement reaches the bar of the video's length group.

The bars come from the training table only and are then applied to the test
table. That gives 33.3% liked in training and 23.4% in the test.

Why length-aware: platform-wide counts in KuaiRec show that completion
depends strongly on video length (Spearman correlation −0.47) while likes,
saves, shares and follows do not favour short videos (+0.14 to +0.34). The
same problem, for watch time, is described by Kuaishou researchers as
*duration bias* ([Zhan et al., 2022](https://arxiv.org/abs/2206.06003)).

### How methods are tested

- **recall@10** = hits in the top 10 ÷ min(videos the user liked, 10). The
  scorer is in [`src/metrics.py`](src/metrics.py), with hand-worked test
  cases in [`src/test_metrics.py`](src/test_metrics.py).
- The 1,411 test users are split at random into a **choosing group** (705)
  and a **final group** (706). Every choice (which version, how many passes
  over the data) was made on the choosing group. The final group is only
  used to report. Two early results on the final group were seen before
  later changes: the first popularity score, which exposed a flaw in the
  label, and the first two model versions. After that the final group was
  not scored again until every choice was fixed.
- Models that are compared closely were trained five times; the tables show
  the average and the spread.

**Two markings of the test.** The bars come from training watches, which are
videos the app chose to show, with rewatches added up. The test is one watch
of nearly everything. One side effect remains: test videos under 5 seconds
count as liked about 30% of the time, against 23% for other lengths. So
results are also reported under a second, **evened-out marking**, where the
bars are set on the test side so that every length has the same liked share.
The main marking stays the main result, because it was fixed before any
two-tower model was trained.

### The models

Each tower is an ID embedding (64 numbers per user or video) followed by a
small two-layer network that outputs the final 64 numbers. The score of a
(user, video) pair is the dot product of the two outputs. About 1.2 million
learnable numbers in total. Plain PyTorch, no recommender library, no vector
index: with 3,327 candidates, every video is scored directly.

| Model | What it is trained on |
|---|---|
| "liked vs skipped" | each liked video against 8 videos the same user watched but did not like |
| mixed + logQ | the question above, plus in-batch negatives with the logQ correction |

The logQ correction subtracts log(*q*) from each video's score during
training, where *q* is the video's share of all liked pairs (its chance of
filling a slot in a batch). Scores are uncorrected when recommending. A
two-video example worked by hand is in [`src/test_logq.py`](src/test_logq.py).
Unlike the paper, this implementation does not normalise the embeddings or
use a temperature, and it counts *q* exactly instead of estimating it from a
stream.

**Similar users**, the counting baseline: for each test user, find the 100
training users whose likes and skips agree most with theirs on the videos
outside the test set, then score each test video by what those users thought
of it. Code in [`src/taste_check.py`](src/taste_check.py).

## Full results

Final group, 706 users. recall@10 / NDCG@10 / AUC.

| Method | Main marking | Evened-out marking |
|---|---|---|
| Random picks | 0.233 / 0.233 / 0.500 | 0.233 / 0.233 / 0.500 |
| Shortest videos first | 0.291 / 0.288 / 0.510 | 0.242 / 0.251 / 0.489 |
| Longest videos first | 0.158 / 0.126 / 0.490 | 0.226 / 0.177 / 0.511 |
| Popularity (like rate) | 0.248 / 0.249 / 0.525 | 0.250 / 0.261 / 0.514 |
| Similar users | 0.316 / 0.313 / 0.559 | 0.290 / 0.298 / 0.545 |
| Two-tower, in-batch, with features (first version) | 0.251 / 0.246 / 0.507 | 0.238 / 0.231 / 0.500 |
| Two-tower, in-batch + 1 skipped video, with features (second version) | 0.253 / 0.252 / 0.513 | 0.234 / 0.232 / 0.502 |
| Two-tower, in-batch | 0.284 / 0.288 / 0.542 | 0.267 / 0.269 / 0.528 |
| Two-tower, in-batch + logQ | 0.299 / 0.299 / 0.529 | 0.297 / 0.298 / 0.526 |
| Two-tower, mixed | 0.295 / 0.293 / 0.551 | 0.274 / 0.272 / 0.531 |
| Two-tower, mixed + logQ (5 runs) | 0.317 / 0.318 / 0.551 | 0.306 / 0.308 / 0.542 |
| Two-tower, liked vs skipped (5 runs) | 0.295 / 0.294 / 0.558 | 0.287 / 0.292 / 0.547 |
| Same model, trained on "watched to the end" | 0.326 / 0.326 / 0.523 | 0.271 / 0.270 / 0.504 |

Unless marked, the two-tower rows use IDs only (no user features or video
tags) and are single runs.

**The last row is the reason for the second column.** Under the main
marking, the model trained on "watched to the end" has the highest recall@10
in the table, because it recommends almost only very short videos and those
are marked liked more often. Its AUC is at the level of popularity, and under
the evened-out marking it falls below both real models. So recall@10 under
the main marking still rewards very short videos to some degree. Whether
that reward is an error is not settled: see the label check below.

What each method puts in the top 10:

| Method | Picks under 5s | Typical pick was watched (in training) | Different videos across all users |
|---|---|---|---|
| All test videos, for reference | 1.1% | 2,404 times | 3,327 |
| Similar users | 12.8% | 3,521 | 721 |
| Two-tower, in-batch | 1.6% | 1,279 | 227 |
| Two-tower, in-batch + logQ | 2.6% | 3,345 | 173 |
| Two-tower, mixed | 6.0% | 1,550 | 319 |
| Two-tower, mixed + logQ | 0.8% | 3,337 | 380 |
| Two-tower, liked vs skipped | 4.6% | 3,494 | 1,053 |
| Trained on "watched to the end" | 84.7% | 2,441 | 236 |

Other checks, all on the choosing group:

- **More training does not help.** Over 100 passes, the score on pairs the
  model trained on kept rising (AUC 0.61 to 0.88) while the test score
  stopped at 0.559 after about 15 passes. The extra passes only memorise.
- **The test is harder than the training logs.** On training pairs hidden
  from the model, AUC is 0.656; on the test, 0.559. Predicting likes among
  videos the app chose to show is easier than among all videos.
- **A larger model does not help** (embedding size 16, 64 and 256 give AUC
  0.554, 0.559 and 0.554).
- **User features and video tags add nothing** once the training question is
  right (AUC 0.559 with and without them).

## Demo

`src/demo.py` picks users at random from the final group and marks each
recommendation with what that user actually did. One of them:

```
USER 4660
  Training history: watched 249 videos, liked 87 (35%).
  The 5 kinds of video they liked most often (liked / watched, and the rate):
    Fashion                       15 / 47   (32%)
    Everyday news                 11 / 26   (42%)
    Good looks                    11 / 32   (34%)
    Celebrities                    6 / 12   (50%)
    Other                          5 / 17   (29%)
  In the test they liked 1,169 of the 3,327 test videos (35%),
  so 10 random picks would contain about 3.5 liked videos.

  TWO-TOWER, MIXED + LOGQ                             TWO-TOWER, LIKED VS SKIPPED
      kind of video            length  liked? all         kind of video            length  liked? all
   1.*Everyday news                8s  LIKED  22%      1.*Fashion                     31s  no     27%
   2.*Everyday news               10s  LIKED  23%      2.*Fashion                     27s  LIKED  24%
   3.*Everyday news                6s  no     19%      3. Other                       12s  no     22%
   4. Comedy                      31s  no     20%      4.*Fashion                     10s  no     22%
   5.*Everyday news               10s  no     23%      5.*Everyday news                7s  LIKED  23%
   6.*Everyday news                6s  no     25%      6.*Everyday news                7s  no     24%
   7.*Fashion                     31s  no     27%      7.*Fashion                     29s  no     21%
   8. Fitness                     10s  no     23%      8.*Everyday news               10s  LIKED  21%
   9. Pets                        10s  LIKED  21%      9.*Good looks                   8s  no     24%
  10.*Fashion                     27s  LIKED  24%     10. Sports                      11s  no     19%
      liked: 4 of 10                                      liked: 3 of 10

  Videos on both lists: 2.   Popularity's list (the same 10 for everyone): 3 of 10 liked.
  * = one of this user's 3 most-liked kinds of video.   liked? = did THIS user like it in the test.
  all = share of all test users who liked the video.
```

Category names are translated from the dataset's Chinese labels. The two
models are each shown by the middle run of their five training runs.

Over all 706 users:

- The mixed + logQ model gets more hits than popularity for 47% of users, the
  same number for 26%, and fewer for 27%.
- One user in five gets 6 or more hits out of 10 (popularity: one in eleven).
  One in three still gets 0 or 1.
- The two models recommend almost entirely different videos to the same
  person and score about the same.
- 43% of the model's picks fall in the user's three favourite categories,
  against 27% by chance. But handing each user *someone else's* list still
  gives 39%. Most of that effect is the model recommending the largest
  categories to everybody; the personal part is about 4 points.

## Checking the label against real likes

The label is built from watch time alone. To check that it means something,
both label rules were applied to [KuaiRand](https://kuairand.com), a second
public dataset from the same app that also records real likes, follows,
comments and shares for each watch. Script: [`src/kuairand_check.py`](src/kuairand_check.py).

| | "Watched to the end" | Length-aware (this project) |
|---|---|---|
| Real like rate on watches the rule marks, compared with the rest (videos the app chose) | 3.4 times | 4.2 times |
| The same, on videos shown at random | 6.8 times | 7.4 times |
| AUC for predicting a real like (random part) | 0.747 | 0.766 |
| AUC for a follow / comment / share (random part) | 0.799 / 0.900 / 0.761 | 0.822 / 0.925 / 0.825 |
| Share of watches marked, shortest videos compared with longest (random part) | 22 times higher | 1.6 times higher |

What it shows:

- **Watch time is a real signal.** Watches marked as liked carry several
  times more real likes than the rest.
- **The length-aware rule agrees with real actions somewhat better**,
  clearly on randomly shown videos and about equally on the app's own choices.
- **It behaves almost exactly like Kuaishou's own "long view" signal**: that
  one marks 35.1% of watches, this project's rule 33.3%, with nearly the same
  real-like rates (3.69% and 3.79%).
- **"Watched to the end" is a fair sign of liking but a poor training
  target.** On randomly shown videos, real likes are 2.4 times as common
  on the shortest videos as on the longest; "watched to the end" is 22 times
  as common. A model can raise it simply by recommending short videos.

Whether short videos are truly liked a little more is not settled by this
data. KuaiRand's logs say slightly yes (correlation between length and real
like rate −0.13); KuaiRec's platform-wide counts say slightly no (+0.14).
Both agree the relation is weak, and that is why both markings of the test
are reported.

## Other findings from the data

**Some videos grow on people.** In the training table, 10.2% of
(user, video) pairs were watched more than once. Of the rewatched pairs where
the first watch was *not* finished, 23.7% were watched to the end on a later
viewing (174,099 pairs). Two caveats: people who come back to a video were
probably already curious about it, and the data does not say whether the
second viewing was recommended by the app or found by the user. The models
here do not use the order of watches.

**Popularity is nearly useless on a fully observed test.** Recommending the
most-watched videos scores 0.226, below random. In ordinary logs, view counts
partly reflect what the app pushed; once everyone has seen everything, they
carry little information.

## How the project went

The step numbers match the comments at the top of each script.

1. **Set up** the project and its Python environment.
2. **Download and explore the data.** Checked the dataset's published numbers
   against the files, and found 968,004 duplicate log rows.
3. **Define "liked", split training and test.** With "watched to the end",
   recommending the 10 shortest videos scored 0.90. Replaced by the
   length-aware label.
4. **Write the scorer.** recall@k, checked by hand on small examples before
   any method was scored.
5. **Popularity baseline.** A first result of 0.434 looked too good. It was a
   flaw in the label (one length group ran from 77 seconds to 5 minutes).
   After the fix: 0.248.
6. **Two-tower model.** The first version tied with popularity. A diagnostic
   showed it had learned what users are shown, not what they like. A counting
   method reached 0.31, so there was taste to learn. The training question
   was changed to "liked vs skipped": 0.295.
7. **logQ correction.** No gain for the plain in-batch model's ranking; a
   clear gain at the top of the list for the mixed model: 0.317.
8. **Demo.**
9. **Label check on KuaiRand**, and this write-up.

## Limitations

- **The neural model does not clearly beat a counting method.**
- **"Liked" is a constructed label.** The KuaiRand check supports it, on
  other users and videos, but a like tap is not the same as enjoyment either.
- **Very short videos.** The main marking favours videos under 5 seconds
  somewhat, and KuaiRand has almost no videos that short, so the check above
  cannot say how much of that is real.
- **IDs only.** The model knows each user and video by its ID, so it cannot
  recommend to a new user or recommend a new video.
- **One dataset, one app, 2020.** Retrieval only, measured offline.
- **Small differences are noise.** recall@10 moves by about 0.01 between
  identical training runs.

## Run it yourself

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 and the
dependencies). Developed on an Apple-silicon laptop; training everything takes
about 3 to 4 hours there. Numbers vary slightly from run to run.

```bash
uv sync

# Data: download KuaiRec.zip from https://kuairec.com and unzip it so that
# data/raw/KuaiRec/data/big_matrix.csv exists. Then:
uv run python src/prepare_data.py     # builds the training and test tables

# Checks and baselines
uv run python src/test_metrics.py
uv run python src/test_logq.py
uv run python src/popularity.py
uv run python src/taste_check.py      # the "similar users" method

# The two main models, five runs each
for s in 0 1 2 3 4; do
  uv run python src/train.py --variant skipped --no-features --epochs 30 --seed $s --name main_seed$s
done
uv run python src/train.py --variant mixed --logq --no-features --epochs 40 --name s7_mixed_logq
for s in 1 2 3 4; do
  uv run python src/train.py --variant mixed --logq --no-features --epochs 12 --seed $s --name s7_mixed_logq_seed$s
done

# The comparison rows
uv run python src/train.py --variant inbatch --name inbatch
uv run python src/train.py --variant hardneg --name hardneg
uv run python src/train.py --variant inbatch --no-features --epochs 30 --name s7_inbatch
uv run python src/train.py --variant inbatch --logq --no-features --epochs 30 --name s7_inbatch_logq
uv run python src/train.py --variant mixed --no-features --epochs 40 --name s7_mixed
uv run python src/train.py --variant skipped --no-features --epochs 30 --label finished --name oldlabel

# Tables and demo
uv run python src/final_report.py --group final
uv run python src/demo.py

# Optional: the label check. Download KuaiRand-Pure from https://kuairand.com
# and extract it into data/raw/KuaiRand/
uv run python -I src/kuairand_check.py
```

## Files

| File | What it does |
|---|---|
| `src/check_torch.py` | checks that PyTorch can use the Apple GPU |
| `src/explore.py` | first look at the dataset; checks its published numbers |
| `src/prepare_data.py` | builds the training and test tables and the labels |
| `src/compare_labels.py`, `src/length_bias.py` | the evidence behind the label choice |
| `src/metrics.py`, `src/test_metrics.py` | the scorer and its hand-checked tests |
| `src/score_random.py`, `src/popularity.py` | random and popularity baselines |
| `src/taste_check.py` | counting methods, including "similar users" |
| `src/data.py`, `src/model.py`, `src/train.py` | the two-tower model and its training |
| `src/test_logq.py` | the logQ correction on a two-video example |
| `src/diagnose_model.py`, `src/epoch_check.py` | what a model learned; training too little or too much |
| `src/final_report.py` | the results tables |
| `src/demo.py` | the demo |
| `src/kuairand_check.py` | the label check on KuaiRand |

## Sources and licences

- **KuaiRec.** Gao et al., "KuaiRec: A Fully-Observed Dataset and Insights for
  Evaluating Recommender Systems", CIKM 2022,
  [doi:10.1145/3511808.3557220](https://doi.org/10.1145/3511808.3557220).
  Data licence CC BY-SA 4.0. Not redistributed here.
- **KuaiRand.** Gao et al., "KuaiRand: An Unbiased Sequential Recommendation
  Dataset with Randomly Exposed Videos", CIKM 2022,
  [doi:10.1145/3511808.3557624](https://doi.org/10.1145/3511808.3557624).
  Data licence CC BY-SA 4.0. Not redistributed here.
- **Two-tower retrieval at Instagram.** Vorotilov and Shugaepov,
  ["Scaling the Instagram Explore recommendations system"](https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/),
  Engineering at Meta, 9 August 2023.
- **logQ correction.** Yi et al.,
  ["Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations"](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/),
  RecSys 2019.
- **Duration bias.** Zhan et al.,
  ["Deconfounding Duration Bias in Watch-time Prediction for Video Recommendation"](https://arxiv.org/abs/2206.06003),
  arXiv:2206.06003, 2022.
- **Neural models and simple baselines.** Ferrari Dacrema, Cremonesi and
  Jannach, ["Are We Really Making Much Progress? A Worrying Analysis of Recent
  Neural Recommendation Approaches"](https://arxiv.org/abs/1907.06902),
  RecSys 2019.

The code in this repository is released under the [MIT licence](LICENSE).
