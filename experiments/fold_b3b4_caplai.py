# -*- coding: utf-8 -*-
"""
Two folds that do not depend on the running agents.

1. B3 and B4 into Section 8.1. Recipes 2 and 3 are implemented and measured, so the standing
   sentence "Only the first is implemented here" is now false and is replaced by what they
   found. The interesting result is not the regret, which stays near 1.00, but the failure
   MODE: a short log does not cost energy, it buys an under-provisioned fleet that misses the
   service constraint.

2. CAPLAI into Section 2.8. Nie et al. (e-Energy '25) is the closest work in AI systems to our
   formulation and must be cited. Read in full; the review is in references/CAPLAI_review.md.
   Its demand constraint gives each GPU type one scalar performance figure and prices energy as
   rated power times hours, so by Proposition 1 it is exactly a fixed accelerator ranking. It
   illustrates the gap rather than closing it, and saying so is both fair and useful.
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

# ---------------------------------------------------------------- B3 / B4 into 8.1
sub("recipes",
"""Each recipe consumes a workload mix, and a facility can obtain one three ways: state an expected
composition at design time, infer it from the composition of current production allocations, or probe
actively by sampling workloads across configurations. Only the first is implemented here; the second
requires a production trace and the third requires spare capacity, and both are natural extensions
rather than obstacles.
</p>""",
"""Each recipe consumes a workload mix, and a facility can obtain one three ways: state an expected
composition at design time, infer it from the composition of current production allocations, or probe
actively by sampling workloads across configurations. All three are implemented and measured.
</p>
<p>
<b>Inferring the mix from a scheduler log.</b> A facility that has not stated a mix still has a log.
We give an estimator only what a log records at admission, a memory request, a batch size, the
measured runtime and the machine the job happened to land on, and deliberately withhold numerical
precision, which Section 7 identifies as the descriptor that decides placement and which schedulers
do not record. Fleets chosen from the inferred mix are then scored by simulation against the true
mix.
</p>
<p>
The energy penalty is small throughout, within about two per cent of the oracle fleet, but that is
not where the risk lies. The dominant failure of a short log is <i>infeasibility</i>: it selects an
under-provisioned fleet that then misses the service constraint. At 25 log lines under a uniform mix,
60 per cent of selected fleets violate the constraint when scored against the true mix; that falls to
20 per cent at 250 lines and to zero by 500. A few hundred observations is therefore the practical
threshold, and it is a threshold on constraint satisfaction rather than on energy. Ignoring the log
entirely and assuming a uniform mix costs 11.1 per cent on a vision workload, 33.4 per cent under an
fp16-only mix, and on a transformer-heavy mix produces a fleet that misses a 60-second constraint by
running at 193 seconds.
</p>
<p>
<b>Active probing, and when the missing descriptor bites.</b> A facility can instead fully
characterise a few arrivals, learning their exact configuration including the precision the log
omits. Under a clean log this buys nothing: runtime measured on a known machine already separates
half from single precision by a median factor of 2.2, so the blind spot costs nothing and the same
budget spent on ordinary log lines is worth more, because extra lines sample the mix while probes
merely re-explain jobs already identified. The picture inverts once logging noise approaches that
runtime separation. At a log-noise level of 0.3 in log runtime, probing wins in most conditions and
uncertainty-directed selection beats random probing, reaching an inferred-mix error of 0.054 against
random probing's 0.116 at a budget of 40; on an fp16 mix at 200 probes the regret is 1.001 against
1.092 for the same budget spent passively. At 0.75 the passive estimator degrades to a regret of
1.238 and probing is clearly worth its cost.
</p>
<p>
The practical rule that follows is narrower than "profile your workloads" and easier to act on. A
facility should spend its observation budget on ordinary logging until its runtime measurements
become noisy relative to the gap between precisions, and only then pay to characterise jobs
properly, choosing which to characterise by predictive uncertainty rather than at random.
</p>""")

# ---------------------------------------------------------------- CAPLAI into 2.8
sub("caplai",
"""First-stage capacity with second-stage allocation is the normal shape of capacity planning
under demand uncertainty [80, 81], and the closest structural precedent is not in computing
at all: Swaminathan [82] places semiconductor tool orders before demand is realised and
assigns wafer types to tools afterwards.""",
"""First-stage capacity with second-stage allocation is the normal shape of capacity planning
under demand uncertainty [80, 81], and the closest structural precedent is not in computing
at all: Swaminathan [82] places semiconductor tool orders before demand is realised and
assigns wafer types to tools afterwards. Within computing, Nie et al. [85] come closest: they
pose GPU lifecycle planning as a stochastic program over purchase, retirement and a one-time
cooling retrofit under uncertain demand, hardware degradation and resale value, with scenarios
drafted by a language model, and report 27 to 40 per cent lower lifecycle cost than a
threshold heuristic. Their treatment of <i>when</i> to buy is complementary to our question of
<i>what</i> to buy, and the two are separable for a reason worth stating: their demand
constraint gives each GPU type a single scalar performance figure and prices energy as rated
power multiplied by hours, so by Proposition 1 their model is exactly a fixed accelerator
ranking and cannot prefer different hardware for different workloads. The quantity it
abstracts away is the one measured here.""")

sub("caplai-ref", "</ol>",
"""<li>Nie, C., Xing, A., Latif, I., Liu, Z. AI-assisted stochastic optimization for GPU data
centers lifecycle planning. In Proceedings of the 16th ACM International Conference on Future
and Sustainable Energy Systems (e-Energy '25), Rotterdam, pp. 870-873, 2025.
doi:10.1145/3679240.3735099</li>
</ol>""")

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("applied: {}".format(", ".join(applied)))
print("FAILED : {}".format(", ".join(failed) if failed else "none"))
