# -*- coding: utf-8 -*-
"""
Reposition the novelty claims against prior work surfaced by an external literature
consult and verified individually against the arXiv API.

Three claims in the paper were too strong and are corrected here:

1. del Rey et al. (arXiv:2307.05520; Comput. Stand. Interfaces 2026) already report that
   interaction effects account for under about 3 per cent of energy variance, close to our
   2.5 per cent. Our contribution is not the magnitude but its decision consequence, and
   the paper must say so plainly rather than present the small residual as a discovery.

2. EcoServe (arXiv:2502.05043) already performs carbon-aware right-sizing over
   heterogeneous GPU types under service objectives. The Section 2.5 claim that no work
   makes accelerator purchase the decision variable under an energy objective and an SLA
   is therefore withdrawn and replaced by the narrower, defensible claim: the
   decomposition of attainable saving into composition versus placement on a common
   measured matrix.

3. Wilkins et al. (arXiv:2407.00010) measure roughly 7.5 per cent energy reduction from
   workload-aware routing across heterogeneous systems, which is inside our own 6.9 to
   9.5 per cent placement range. This corroborates our number and removes any claim that
   the value of heterogeneous placement was previously unmeasured.

Also added: Zine et al. on configuration interaction effects, DynamoLLM and FleetOpt on
provisioning under non-energy objectives.
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

# ---------------------------------------------------------------- 2.5 novelty claim
sub("sec2.5-claim",
"""We found no work making accelerator purchase the decision variable,
energy the objective, an SLA the constraint, and workload mix the input.""",
"""Carbon-aware provisioning does reach this cell: EcoServe [54] right-sizes heterogeneous
GPU resources under service objectives and reports up to 47 per cent carbon reduction, and
FleetOpt [69] optimises GPU types and counts for an inference fleet under tail-latency
constraints, though its objective is cost rather than energy. DynamoLLM [70] optimises
energy under latency service objectives but varies frequency, parallelism and instance
count on homogeneous hardware rather than the accelerator mix. What we did not find, and
what this paper contributes, is a <i>decomposition</i>: how much of the attainable saving
belongs to composition and how much to placement, measured on one common workload by
accelerator energy matrix. We claim that decomposition, not the idea of treating
composition as a decision variable.""")

# ---------------------------------------------------------------- 2.2, interaction magnitude precedent
sub("sec2.2-delrey",
"""Roofline-inspired models state explicitly that neither operation counts nor
runtime is a sufficient energy proxy [11]""",
"""Closest to our own decomposition, del Rey et al. [71] model training energy from
architecture and execution environment, question FLOP- and TDP-based estimation across
environments, and report that main effects explain nearly all variance while interactions
account for under roughly 3 per cent, a figure close to the 2.5 per cent we measure. They
do not draw the decision-theoretic consequence that Section 5 develops, which is where our
contribution lies rather than in the magnitude itself. Roofline-inspired models state
explicitly that neither operation counts nor runtime is a sufficient energy proxy [11]""")

# ---------------------------------------------------------------- 2.4, placement precedent
sub("sec2.4-wilkins",
"""Energy-efficient multi-instance scheduling [32] and agentic CPU-GPU
assignment [33] both report performance-first placement is suboptimal.""",
"""Energy-efficient multi-instance scheduling [32] and agentic CPU-GPU
assignment [33] both report performance-first placement is suboptimal. Most directly
comparable, Wilkins et al. [72] route LLM inference across a hybrid heterogeneous cluster
using workload information and measure roughly 7.5 per cent energy reduction against a
workload-unaware allocation, which falls inside the 6.9 to 9.5 per cent we measure for
placement in Section 8 and is an independent confirmation of that magnitude.""")

# ---------------------------------------------------------------- 5, position the residual honestly
sub("sec5-position",
"""<p>
The practical reading is that variance share and decision relevance are close to unrelated
in this problem, and optimising the former actively discards the latter.""",
"""<p>
The magnitude itself is not the contribution, and it is worth being precise about that.
Working on training energy across execution environments, del Rey et al. [71] independently
report interaction effects below roughly 3 per cent of variance, so a small interaction
term is an established observation rather than a new one. What Proposition 1 adds is the
consequence: that the 97.5 per cent everyone fits is precisely the part that cannot choose
a device, and the 2.5 per cent everyone discards is precisely the part that can.
</p>
<p>
The practical reading is that variance share and decision relevance are close to unrelated
in this problem, and optimising the former actively discards the latter.""")

# ---------------------------------------------------------------- 6, configuration precedent
sub("sec6-zine",
"""This has a direct consequence for how schedulers are built, and it is uncomfortable.""",
"""That execution configuration matters for energy is itself established: Zine et al. [73]
evaluate energy, performance and accuracy across a large controlled set of serving
configurations and analyse task and configuration effects directly. The narrower claim
here is about <i>which device</i> the configuration selects, and that the resulting
reversals are what a placement policy consumes.
</p>
<p>
This has a direct consequence for how schedulers are built, and it is uncomfortable.""")

# ---------------------------------------------------------------- 9, scope the zero-placement result
sub("sec9-zero",
"""<p>
Three limitations bear on how far these results travel.""",
"""<p>
One clarification precedes the limitations, because it is easy to over-read. The
statement that placement is worth nothing on a uniform fleet is conditional on machines of
one type being exchangeable for the energy of a job. Where they are not, because of
thermal state, network locality, memory-bandwidth contention from co-tenants or per-device
frequency policy, physical placement can still matter on nominally identical hardware. The
claim here is specifically that <i>accelerator-type selection</i> has no remaining value
once every accelerator is the same type, which is what our two-type-pool measurement shows
and what Proposition 1 predicts.
</p>
<p>
Three limitations bear on how far these results travel.""")

# ---------------------------------------------------------------- references
sub("newrefs2", "</ol>",
"""<li>Chen, H., Liu, X., Liu, Y., et al. FleetOpt: analytical fleet provisioning for LLM
inference with compress-and-route. arXiv:2603.16514, 2026.</li>
<li>Stojkovic, J., Zhang, C., Goiri, I., Torrellas, J., Choukse, E. DynamoLLM: designing
LLM inference clusters for performance and energy efficiency. arXiv:2408.00741, 2024.</li>
<li>del Rey, S., Cruz, L., Franch, X. Estimating deep learning energy consumption based on
model architecture and training environment. arXiv:2307.05520, 2023; Computer Standards and
Interfaces, 2026. doi:10.1016/j.csi.2026.104170</li>
<li>Wilkins, G., Keshav, S., Mortier, R. Hybrid heterogeneous clusters can lower the energy
consumption of LLM inference workloads. arXiv:2407.00010, 2024.</li>
<li>Zine, N., Coignion, T., Stoico, V., et al. Attention to detail: evaluating energy,
performance and accuracy trade-offs across vLLM configurations. arXiv:2607.09172, 2026.</li>
</ol>""")

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("applied: {}".format(", ".join(applied)))
print("FAILED : {}".format(", ".join(failed) if failed else "none"))
