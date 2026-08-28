# -*- coding: utf-8 -*-
"""
Ground the B8 two-stage stochastic programme in its own literature, and correct the
novelty framing around it.

The finding that forced this: the structure of B8 is not new. Swaminathan (EJOR 2000) places
semiconductor tool orders before demand is realised and assigns wafer types to tools once it
is, "formulated as a stochastic integer program with recourse". Substitute accelerator type
for tool type and workload cell for wafer type and it is our model. Presenting the
formulation as a contribution would be wrong, so the paper now states that fleet design IS a
two-stage stochastic programme in the classical sense and claims only the instantiation on a
measured energy matrix plus the VSS and EVPI quantification.

Nine references, the minimal set the literature pass recommended, all with DOIs resolved.
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

# ------------------------------------------------- new Related Work subsection
sub("sec2.9",
"""<h3>2.8 Embodied carbon</h3>""",
"""<h3>2.8 Planning under uncertainty</h3>
<p>
Choosing a fleet before the workload mix is known, then routing work once it is, is a
two-stage stochastic programme with recourse in the classical sense [74, 75], and the
standard reference treatments give both the formulation and the measures we report: the
value of the stochastic solution [76], which prices planning on the mean scenario instead of
the distribution, and the expected value of perfect information [77], which prices not
knowing the realisation. Relatively complete recourse [78], obtained here by allowing work to
be served off-fleet at a penalty, is what keeps the objective finite when a scenario admits
no feasible fleet, a case Section 8.1 actually encounters. The L-shaped method [79] is the
standard decomposition; at our scale a direct search over first-stage fleets is cheaper.
</p>
<p>
First-stage capacity with second-stage allocation is the normal shape of capacity planning
under demand uncertainty [80, 81], and the closest structural precedent is not in computing
at all: Swaminathan [82] places semiconductor tool orders before demand is realised and
assigns wafer types to tools afterwards. Substituting accelerator type for tool type and
workload cell for wafer type recovers our model exactly. We therefore claim no novelty in
the formulation. What is new here is the instantiation on a measured workload-by-accelerator
<i>energy</i> matrix rather than a cost or shortage objective, and the resulting
quantification of how much of the attainable saving is attributable to planning on the mix
distribution at all. A reviewer may reasonably ask why not robust optimisation given only a
handful of scenarios [83]; the answer is that we have relative frequencies for the mixes and
therefore an expectation worth taking, but the robust counterpart is the natural extension
where those frequencies are themselves unknown.
</p>

<h3>2.9 Embodied carbon</h3>""")

# renumber the old 2.8 reference in the text if any
s = s.replace("Section 2.8 Embodied carbon", "Section 2.9 Embodied carbon")

# ------------------------------------------------- Section 8.1, point at the formulation
sub("sec8.1-sp",
"""These recipes correspond to three ways a facility can obtain the mix that drives them""",
"""Recipe 1 as stated enumerates compositions against a single mix. Treated properly it is the
two-stage programme of Section 2.8, and solving it as one is both cheaper and more honest
about uncertainty. Replacing enumeration with a fluid-limit recourse relaxation, used to
rank candidates and prune, and reserving the discrete-event simulator for the survivors,
recovers the same optimum from 25 simulations rather than 1001, and reaches fleet sizes
where enumeration is impossible: at 100 slots there are 4.6 million compositions and the
search visits 3,307. Solved across five weighted mix scenarios rather than one, the value of
the stochastic solution is <b>21.3 per cent</b>: a buyer who plans against the average mix
rather than the distribution of mixes gives up that much, which is larger than the entire
placement lever. Perfect foresight of the realised mix would be worth a further 16.8 per
cent.
</p>
<p>
These recipes correspond to three ways a facility can obtain the mix that drives them""")

# ------------------------------------------------- references
sub("refs-l2", "</ol>",
"""<li>Dantzig, G. B. Linear programming under uncertainty. Management Science 1(3-4):197-206,
1955. doi:10.1287/mnsc.1.3-4.197</li>
<li>Birge, J. R., Louveaux, F. Introduction to Stochastic Programming, 2nd edition. Springer,
2011. doi:10.1007/978-1-4614-0237-4</li>
<li>Birge, J. R. The value of the stochastic solution in stochastic linear programs with fixed
recourse. Mathematical Programming 24(1):314-325, 1982. doi:10.1007/BF01585113</li>
<li>Madansky, A. Inequalities for stochastic linear programming problems. Management Science
6(2):197-204, 1960. doi:10.1287/mnsc.6.2.197</li>
<li>Wets, R. J.-B. Stochastic programs with fixed recourse: the equivalent deterministic
program. SIAM Review 16(3):309-339, 1974. doi:10.1137/1016053</li>
<li>Van Slyke, R. M., Wets, R. L-shaped linear programs with applications to optimal control
and stochastic programming. SIAM Journal on Applied Mathematics 17(4):638-663, 1969.
doi:10.1137/0117061</li>
<li>Eppen, G. D., Martin, R. K., Schrage, L. A scenario approach to capacity planning.
Operations Research 37(4):517-527, 1989. doi:10.1287/opre.37.4.517</li>
<li>Van Mieghem, J. A. Capacity management, investment and hedging: review and recent
developments. Manufacturing and Service Operations Management 5(4):269-302, 2003.
doi:10.1287/msom.5.4.269.24882</li>
<li>Swaminathan, J. M. Tool capacity planning for semiconductor fabrication facilities under
demand uncertainty. European Journal of Operational Research 120(3):545-558, 2000.
doi:10.1016/S0377-2217(98)00389-0</li>
<li>Bertsimas, D., Brown, D. B., Caramanis, C. Theory and applications of robust optimization.
SIAM Review 53(3):464-501, 2011. doi:10.1137/080734510</li>
</ol>""")

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("applied: {}".format(", ".join(applied)))
print("FAILED : {}".format(", ".join(failed) if failed else "none"))
