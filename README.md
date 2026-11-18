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

_Work in progress. Results, setup instructions, and sources will be added as
the project is built._
