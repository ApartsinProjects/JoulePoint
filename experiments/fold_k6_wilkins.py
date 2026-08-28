# -*- coding: utf-8 -*-
"""
Two integrations.

K6. The 1.9 per cent gap between the two simulator variants is traced and closed. It came
from one line: Recipe 1 draws arrivals with rng.integers, B8 with rng.choice(p=w). For a
uniform mix these are distributionally identical but consume different generator state, so
the same seed yields different arrival sequences. Over 60 seeds the two agree (z = +0.79,
means 0.21 per cent apart) and the seed-to-seed standard deviation is 1.39 per cent, the
same order as the single-seed gap. A positive control on a skewed mix, where rng.integers
genuinely cannot see the weights, separates them at z = +46.9, so the test can detect a
real difference when one exists.

The consequence for the paper is not a corrected number but a missing interval: Section 8's
figures were single-seed point estimates. They are now reported with their seed spread, and
the optimal fleet is confirmed identical in 12 of 12 seeds.

Wilkins. A fifth corpus for Table 2, at 2.66 per cent interaction on 32 workloads by 4
platforms. Raw data is NOT redistributable (the repository carries no licence), so only
derived statistics appear here.
"""
import io, json

P = "paper/greenmatch-paper.html"
s = io.open(P, encoding="utf-8").read()
K = json.load(io.open("experiments/results/k6_seed_variance.json", encoding="utf-8"))
W = json.load(io.open("experiments/results/wilkins_corpus.json", encoding="utf-8"))
applied, failed = [], []

def sub(tag, old, new):
    global s
    if old in s:
        s = s.replace(old, new, 1); applied.append(tag)
    else:
        failed.append(tag)

# ---------------------------------------------------------------- Table 2, fifth corpus
sub("table2-wilkins",
"""<tr><td>Price of Prompting [67]</td><td>prompt corpus, fixed across machines</td><td>2.5%</td></tr>""",
"""<tr><td>Price of Prompting [67]</td><td>prompt corpus, fixed across machines</td><td>2.5%</td></tr>
<tr><td>Hybrid heterogeneous clusters [72]</td><td>token counts, fixed across machines</td><td>2.7%</td></tr>""")

# ---------------------------------------------------------------- Section 6, fifth corpus
sub("sec6-wilkins",
"""<p>
Correlation across four corpora would still leave the direction of causation open""",
"""<p>
A fifth campaign, again by a different group with a different instrumentation stack,
reproduces the same magnitude. Wilkins et al. [72] measure LLM inference energy across an
Apple M1 Pro laptop part and three datacenter hosts carrying V100 and A100 accelerators,
integrating RAPL package counters and the vendor power series rather than reading a single
counter as we do. On the balanced sub-matrix of 32 workloads by 4 platforms the interaction
is 2.66 per cent, and Proposition 1 again holds to every digit, with a fixed ranking and an
additive model both scoring 0.71875. Four independent measurement campaigns, on hardware
ranging from a laptop to a datacenter accelerator, now put the interaction between 2.5 and
2.7 per cent wherever configuration is held fixed across machines.
</p>
<p>
That corpus repeats the scoping lesson of the previous one, and sharpens it. Nine of its 32
workloads reverse, and every one is the M1 Pro beating the A100 on short generations, where
a laptop part running a small job is genuinely the cheaper machine. Restricted to the three
<i>datacenter</i> platforms, reversals fall to <b>zero of 32</b>: the A100 host wins
everywhere. Reversals appear when devices are close enough in capability for configuration
to decide between them and disappear when one device simply dominates, which is the
condition on the fleet that Section 8 measures directly.
</p>
<p>
Correlation across five corpora would still leave the direction of causation open""")

# ---------------------------------------------------------------- precision on the Wilkins routing claim
sub("wilkins-precision",
"""Most directly
comparable, Wilkins et al. [72] route LLM inference across a hybrid heterogeneous cluster
using workload information and measure roughly 7.5 per cent energy reduction against a
workload-unaware allocation, which falls inside the 6.9 to 9.5 per cent we measure for
placement in Section 8 and is an independent confirmation of that magnitude.""",
"""Most directly
comparable, Wilkins et al. [72] route LLM inference across a hybrid heterogeneous cluster
using workload information and report roughly 7.5 per cent energy reduction against a
workload-unaware allocation, of the same order as the 6.9 to 9.5 per cent we measure for
placement in Section 8. The two figures are not directly comparable: theirs comes from a
threshold policy simulated over a request-length distribution, and re-deriving an oracle
routing gain from their released matrix alone gives 2.14 per cent over all four platforms
and 0.00 per cent within the datacenter subset, where one host dominates. The agreement to
be drawn is that heterogeneous routing is worth single-digit percentages, not that the two
experiments measure the same quantity.""")

# ---------------------------------------------------------------- seed intervals
sub("seed-8.1",
"""<i>heterogeneous</i>: four L4 plus six L40S at 4,476 J per job, 34.3 per cent below the all-A100 fleet
a throughput-first buyer would purchase.""",
"""<i>heterogeneous</i>: four L4 plus six L40S at 4,476 J per job, 34.3 per cent below the all-A100 fleet
a throughput-first buyer would purchase. Because that figure is one realisation of a
stochastic arrival stream, it is reported here with its spread: over twelve independent
seeds the saving is 34.5 per cent with a standard deviation of 0.4 and a range of 34.0 to
35.5, and the selected fleet is <b>identical in all twelve</b>. The composition result is
therefore a statement about the fleet, not about a seed.""")

# ---------------------------------------------------------------- Section 9 note on replication
sub("sec9-seeds",
"""<p>
Three limitations bear on how far these results travel.""",
"""<p>
On reproducibility, facility figures come from a stochastic arrival stream and carry
Monte Carlo variation of roughly 1.4 per cent in standard deviation between seeds, which is
the scale of disagreement to expect when re-running these simulations. Quantities reported
as differences between policies on the same replayed stream are unaffected, since the stream
is shared; the absolute energies per job are not, and are given with their seed spread where
they carry weight.
</p>
<p>
Three limitations bear on how far these results travel.""")

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("applied: {}".format(", ".join(applied)))
print("FAILED : {}".format(", ".join(failed) if failed else "none"))
print("\nWilkins interaction: {}".format(W.get("shares", W.get("decomposition", "see json"))))
