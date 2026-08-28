# -*- coding: utf-8 -*-
"""
Correct the four claims the 19 August audit showed to be wrong or unsupported.

None of these is a rewording. Each replaces a number or withdraws a conclusion.

1. Section 7.1 cold start on our own grid. The reported 87.0 and 88.9 per cent were scored over
   all ten machine pairs in each row, of which only four involve the held-out accelerator; the
   other six are in-sample pairs among machines the model was fitted on. That is not the
   construct the MLPerf 91.2 per cent uses, and the phrase "the same construction" was the
   error. Construct-matched values are 74.4 and 79.3.

2. Section 7.7, joint assignment. "0.0 J per job attributable to policy" is an artefact.
   Instrumenting the scheduler shows min(queue, free) equals one at 100 per cent of epochs at
   every arrival rate, so the assignment problem is always 1x1 and the Hungarian solve is
   identically greedy. The invariant "joint never loses to greedy" passed vacuously. The claim
   is withdrawn, not restated, because the experiment never posed the question.

3. Section 8.1 and Figure 5, the Pareto frontier. Idle power was charged over a 3,600 s horizon
   while jobs ran to 16,074 s, so fleets that overrun are under-counted, one by 29 per cent. The
   frontier is 19 non-dominated fleets, not 23. The SLA-feasible optimum survives: same fleet,
   4,476.3 -> 4,484.2 J/job, +0.18 per cent, because it runs at 97 per cent utilisation and its
   overhang is 342 s of 34,919 s.

4. Table 9 and the Grid'5000 cold-start gain. Accuracy was restricted to held-out pairs but
   regret was not. Instrumented, the model selects the held-out platform in 2 of 90 rows, so the
   regret column is not measuring a cold-start decision at all. Construct-matched, the
   bilinear-over-additive gap collapses from 0.0433 to 0.0013.
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

# ---------------------------------------------------------------- 1. cold start
sub("coldstart",
"""<p>
On our own grid the same construction reaches 87.0 per cent with <i>no</i> calibration run on the new
accelerator, rising to 88.9 per cent with two, which is 99 per cent of what eight calibration runs
deliver.""",
"""<p>
On our own grid the corresponding figures are lower and must be read carefully, because the two
corpora do not pose the same problem. Restricted to the pairs that actually involve the held-out
accelerator, which is the construct the MLPerf figure uses, accuracy is 74.4 per cent with no
calibration run and 79.3 per cent with two. Scored over all pairs in the row, including the six of
ten that compare machines the model was fitted on, the same run reports 87.0 and 88.9; that is a
diluted measure and it is not comparable to the 91.2 above. The five-machine grid is also a harder
instance than MLPerf's thirteen, since a held-out column is a larger share of the decision.""")

# ---------------------------------------------------------------- 2. joint assignment
sub("joint",
"""A natural objection is that the remaining energy could be recovered by a better scheduler rather than a
different fleet. It cannot. Replacing greedy placement with an exact joint assignment, solved as a linear assignment problem by the
Hungarian algorithm, over the identical set of waiting jobs and free machines at every scheduling epoch,
yields a gain of <b>0.00 per cent at every arrival rate tested</b>, from lightly loaded to saturated.
Decomposing the shortfall of the deployed system against the true optimum at the operating point gives
<b>0.0 J per job attributable to policy and 13.1 J per job to prediction</b>, together 0.27 per cent of
the facility bill.""",
"""A natural objection is that the remaining energy could be recovered by a better scheduler rather than a
different fleet. We attempted to settle this by replacing greedy placement with an exact joint
assignment solved by the Hungarian algorithm, and the attempt failed for an instructive reason that we
report rather than suppress. Instrumenting the scheduler shows that the number of jobs waiting when a
machine frees is <b>exactly one at every scheduling epoch, at every arrival rate tested</b>: at low load
a job meets an idle fleet, and at high load the simulation advances to the next completion, which frees
exactly one slot. The assignment problem is therefore always one job by one machine, the Hungarian
solve is identically greedy, and the resulting "0.00 per cent gain" measures nothing. The question of
whether a batching scheduler that deliberately delays placement to assemble a choice could recover
energy is <b>open</b>, and answering it needs an arrival model in which such a scheduler has something
to choose between.""")

# ---------------------------------------------------------------- 3. Pareto
sub("pareto",
"""<b>Recipe 1, the frontier.</b> Enumerating all 1001 compositions of ten slots gives a Pareto frontier
of 23 non-dominated fleets.""",
"""<b>Recipe 1, the frontier.</b> Enumerating all 1001 compositions of ten slots gives a Pareto frontier
of 19 non-dominated fleets.""")

sub("pareto-note",
"""The unconstrained optimum is a trap: all-L4 reaches 2,492 J
per job at 2,142 seconds of mean delay, which is unusable.""",
"""The unconstrained optimum is a trap, and doubly so. All-L4 appears to reach 2,492 J per job at 2,142
seconds of mean delay, which is unusable on its face; and that figure is itself understated by 29 per
cent, because a fleet whose jobs run long past the measurement horizon accrues idle power that a
fixed-horizon accounting never charges. Correctly charged it is 3,219 J per job. Fleets that breach
the service constraint therefore look cheaper than they are, which strengthens rather than weakens
the argument for the constraint. The feasible optimum is unaffected at 4,484 J per job, since it runs
at 97 per cent utilisation with an overhang of 342 seconds in 34,919.""")

# ---------------------------------------------------------------- 4. Table 9
sub("table9",
"""A platform never measured on anything is placed correctly in 82.8 per cent of its pairings from
published figures alone, against a hindsight ceiling of 91.6, and 89.6 per cent once a single
idle-power characterisation run is permitted. Energy regret is 1.004, and <b>16 of the 18 held-out
platforms give a regret of exactly 1.000</b>: for most platforms the cold prediction picks the same
machine an oracle would.""",
"""A platform never measured on anything is placed correctly in 82.8 per cent of its pairings from
published figures alone, against a hindsight ceiling of 91.6, and 89.6 per cent once a single
idle-power characterisation run is permitted. The accompanying energy regret needs a caveat that the
accuracy figure does not: regret is computed over the whole row rather than over the cold-start
decision, and the model selects the held-out platform in only 2 of 90 rows, so the regret column
mostly scores placements among platforms already observed. Measured on the cold-start decision alone
the advantage of the interaction model over an additive one is 0.0013 rather than the 0.0433 an
unrestricted scoring suggests. The accuracy result stands; the regret result should be read as a
row-level rather than a cold-start quantity.""")

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("applied: {}".format(", ".join(applied)))
print("FAILED : {}".format(", ".join(failed) if failed else "none"))
