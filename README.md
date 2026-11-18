# Two-tower video recommender on KuaiRec

A small learning project: a two-tower **retrieval** model trained on the public
KuaiRec short-video dataset, compared against a popularity baseline.

This is only the retrieval stage of a recommender (picking a few hundred
candidates out of thousands), not a full production recommendation system.

## Findings from the data

**Some videos grow on people.** In KuaiRec's big interaction table (after
removing 968,004 exact duplicate log rows), 10.2% of user–video pairs were
watched more than once. Of the rewatched pairs where the first watch was *not*
finished, **23.7% were watched to the end on a later viewing** (174,099 pairs,
1.7% of all pairs).

Two caveats: people who come back to a video were probably already curious about
it, and the data doesn't say whether the second viewing was recommended by the
app or found by the user. This model doesn't use the order of watches; "show a
skipped video again" would be a separate re-ranking idea for future work.

**"Watched to the end" is mostly about video length.** On the fully observed
test table, videos under 7 seconds are watched to the end 70.4% of the time;
videos over 60 seconds, 0.3% of the time.

**That gap is mostly mechanical, not preference.** KuaiRec's per-video daily
statistics include platform-wide counts of deliberate actions (likes, saves,
shares, follows, "see less"). Across the 10,728 videos, none of the positive
actions favours short videos: the Spearman correlation between video length
and the share of players who acted is +0.14 for likes, +0.30 for saves, +0.34
for shares and +0.31 for follows, against −0.47 for completion. Short videos
are finished more because they are short, not because they are liked more.
(The platform's own `complete_play` counts are not used: their definition is
undocumented and inconsistent with the platform's average play progress.)

This is a known problem, usually called *duration bias*. Kuaishou researchers
describe it for watch-time prediction in [Zhan et al., 2022,
"Deconfounding Duration Bias in Watch-time Prediction for Video
Recommendation"](https://arxiv.org/abs/2206.06003). There, raw watch time
favours *long* videos; here, the watch *ratio* favours *short* ones. Both are
the same underlying issue: length changes the signal regardless of interest.

**The fix used here:** a video counts as "liked" by a user if their total
engagement (sum of watch ratios over all their views) is in the top third
among videos of similar length (100 length groups, thresholds set on training
data only). With "watched to the end" as the label, recommending the 10
shortest videos to everyone scored recall@10 = 0.90 against 0.32 for random
picks. With the length-aware label, the same shortcut scores 0.29 against 0.23.

**Known remaining imperfection:** videos under 5 seconds (38 of the 3,327 test
videos) are still slightly easier to count as liked (about 30% of users vs 23%
for the rest). Their length groups have the least training data behind them.
The model's results are also reported with those videos removed, to show
whether it benefits from this.

_Work in progress. Results, setup instructions, and sources will be added as
the project is built._
