# -*- coding: utf-8 -*-
"""
Fold the four exploratory directions in.

G5 is the most consequential. Section 7 recommends adding numerical precision to the
scheduling interface and calls it cheap. G5 turns that into a costed frontier over all
subsets of descriptors and finds something stronger than the recommendation: everything a
scheduler already sees is worth EXACTLY ZERO, one free field is worth 6.3 points, and a
profiled reference run on top of it is worth nothing further.

G4 gives heterogeneity an operational consequence the paper had not considered: it breaks
chargeback. Naive billing is near-fair on a homogeneous fleet and mis-allocates badly on the
heterogeneous fleet the paper recommends buying.

G6 and G2 are reported as bounded negatives, which is what they are.
"""
import io

P = "paper/greenmatch-paper.html"
s = io.open(P, encoding="utf-8").read()
applied, failed = [], []

def sub(tag, old, new):
    global s
    if old in s:
        s = s.replace(old, new, 1); applied.append(tag)
    else:
        failed.append(tag)

# ---------------------------------------------------------------- G5 into 7 (after Table 5)
sub("g5",
"""<p>
This is an unusually cheap recommendation to act on. Numerical precision is known to the job at
submission time, costs nothing to transmit, and requires no profiling, no telemetry pipeline and no
model retraining to collect. It is simply not currently part of the scheduling interface in the
systems we are aware of. Adding one field to a job specification is the smallest intervention this
paper identifies and it is a precondition for every other placement result in it.
</p>""",
"""<p>
This is an unusually cheap recommendation to act on. Numerical precision is known to the job at
submission time, costs nothing to transmit, and requires no profiling, no telemetry pipeline and no
model retraining to collect. It is simply not currently part of the scheduling interface in the
systems we are aware of. Adding one field to a job specification is the smallest intervention this
paper identifies and it is a precondition for every other placement result in it.
</p>
<p>
Table 5 withholds descriptors one group at a time. Searching the full lattice of subsets instead,
and ordering them by what each descriptor costs a facility to obtain, sharpens the recommendation
into a frontier with only three points on it. Descriptors that cost nothing because a scheduler
already receives them, batch size, workload family and a memory request, reach 81.9 per cent, which
is <b>exactly the score of knowing nothing at all</b>. Adding the one free field the scheduler does
not receive, numerical precision, reaches 88.2 per cent, a paired gain of 6.3 points with a bootstrap
interval of [2.1, 11.1]. Adding a profiled reference run on top of that, at roughly ten times the
acquisition cost, reaches 88.2 per cent again: <b>profiling buys nothing that precision has not
already bought.</b> Averaged over the lattice the marginal value of each descriptor is 2.73 points
for precision, 1.35 for batch size, 1.09 for the memory request, 0.82 for a profiling run and 0.48
for workload family.
</p>
<p>
The reason profiling adds so little is measurable rather than incidental. On a power-saturated
device the mean-power fraction across our whole workload set spans only 0.97 to 1.04, so power draw
carries almost no workload information and only timing does; and timing is confounded with workload
family, so it transfers poorly to families the model has not seen. The practical statement is
therefore stronger than "collect more telemetry": for this decision, one field already known at
submission dominates an entire profiling apparatus.
</p>""")

# ---------------------------------------------------------------- G4 into 8.1
sub("g4",
"""<h2>9. Scope and Limitations</h2>""",
"""<h3>8.3 Heterogeneity breaks naive chargeback</h3>
<p>
A fleet is usually shared, and the recommendation to buy a heterogeneous one has a consequence for
how its energy bill is divided. Treating tenants as players in a cost game, with the value of a
coalition being the cheapest facility that serves it, the Shapley split can be compared against the
two billing rules facilities actually use.
</p>
<p>
On a homogeneous fleet naive billing is close to fair: charging by occupied second departs from the
Shapley share by at most 1.6 per cent. On the capacity-constrained heterogeneous fleet this paper
recommends, it is not: per-second billing over-charges by <b>29.1 per cent</b> and under-charges by
22.0, and per-job billing ranges from <b>+89.7 to -46.2 per cent</b>. The direction is systematic and
awkward. Per-second billing over-charges long-running work on the efficient but slow parts and
under-charges short bursts on the fast expensive ones, which means it under-charges precisely the
tenants whose latency requirements justified buying the expensive parts at all. Sharing is still
worth having, saving each tenant between zero and 17 per cent against standing alone, but the
Shapley allocation here is not in the core, so no split makes every coalition prefer to stay.
</p>

<h2>9. Scope and Limitations</h2>""")

# ---------------------------------------------------------------- G6 into 8
sub("g6",
"""Consolidation with power-down is worth
13.2 per cent at a 300-second sleep threshold, bought for 0.7 seconds of added mean delay; below a
120-second threshold it becomes actively harmful, as machines are woken more often than the sleep saves.""",
"""Consolidation with power-down is worth
13.2 per cent at a 300-second sleep threshold, bought for 0.7 seconds of added mean delay; below a
120-second threshold it becomes actively harmful, as machines are woken more often than the sleep
saves. Pushing utilisation higher continues to pay throughout: across a fifteen-point load sweep
and a ten-point right-sizing sweep, energy per job is monotone non-increasing in utilisation in
every configuration tested, and we find no turning point of the kind reported for spatial sharing
above about 70 per cent load. The displacement penalty is real, since a busy efficient accelerator
forces work onto a worse one, but it never outruns the saving from amortising idle power. What
bounds consolidation here is the service constraint rather than any energy reversal: shrinking a
fleet from 14 slots to 9 raises utilisation from 55.5 to 92.5 per cent and saves 10.9 per cent at 26
seconds of mean delay, and past that the 60-second constraint breaks before further savings
materialise.""")

# ---------------------------------------------------------------- G2 into 9
sub("g2",
"""<p>
Three limitations bear on how far these results travel.""",
"""<p>
One direction we tested and can rule out concerns how fleets are actually acquired. Since
heterogeneity is valuable, a natural hope is that staggered replacement supplies it for free, a
fleet refreshed in thirds being automatically mixed-generation. It does not. Rolling policies do
hold two or three vintages at once, but they hold the wrong mixtures, running 23 to 27 per cent
above a per-year oracle. Staggering beats matched-cycle wholesale replacement only on a two-year
cycle, is a wash at three years and is marginally worse at four. Heterogeneity has to be chosen; it
does not arrive by accident. We note, without modelling embodied carbon, that never refreshing at
all is the cheapest policy on operational energy alone, because the oldest part in our set also has
the lowest idle draw, and that it pays for this in latency.
</p>
<p>
Three limitations bear on how far these results travel.""")

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("applied: {}".format(", ".join(applied)))
print("FAILED : {}".format(", ".join(failed) if failed else "none"))
