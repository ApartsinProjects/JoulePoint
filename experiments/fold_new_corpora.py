# -*- coding: utf-8 -*-
"""
Fold three pieces of new evidence into the paper, plus the bibliography fixes bibtest
surfaced.

1. Husom et al. as a fourth corpus in Table 2. Its 2.49 per cent interaction lands on
   top of our 2.5 per cent, and its zero reversals over two widely separated machines
   independently reproduce the two-type-pool result of Section 8.
2. Watt Counts (Fadel Argerich et al., 2026) as independent corroboration at 50 models
   by 10 GPUs, where the energy-optimal GPU changes with deployment scenario, and as a
   citable justification for our GPU-scoped instrumentation.
3. B5: the Shapley sweep showing marginal value depends on the service constraint.

Bibliography: Notomista et al. upgraded from preprint to the IEEE T-RO version; live
URLs added to the two web resources bibtest could not resolve.
"""
import io, json

P = "paper/greenmatch-paper.html"
s = io.open(P, encoding="utf-8").read()
applied, failed = [], []

def sub(tag, old, new):
    global s
    if old in s:
        s = s.replace(old, new, 1); applied.append(tag)
    else:
        failed.append(tag)

E = json.load(io.open("experiments/results/ejhusom_corpus.json", encoding="utf-8"))
B5 = json.load(io.open("experiments/results/b5_shapley_sla_sweep.json", encoding="utf-8"))
inter = E["shares"]["interaction"]

# ------------------------------------------------------------------ Table 2 fourth row
sub("table2-row",
"""<tr><td>llm-perf-leaderboard [8]</td><td>quantisation scheme and dtype</td><td>9.0%</td></tr>
</tbody>""",
"""<tr><td>Price of Prompting [67]</td><td>prompt corpus, fixed across machines</td><td>{:.1f}%</td></tr>
<tr><td>llm-perf-leaderboard [8]</td><td>quantisation scheme and dtype</td><td>9.0%</td></tr>
</tbody>""".format(inter))

sub("table2-caption",
"<caption><b>Table 2.</b> Interaction residual across three independently collected corpora.</caption>",
"<caption><b>Table 2.</b> Interaction residual across four independently collected corpora, ordered by "
"how much execution configuration each was allowed to vary.</caption>")

# ------------------------------------------------------------------ Section 6 discussion
sub("sec6-fourth",
"""<p>
Correlation across three corpora would still leave the direction of causation open""",
"""<p>
A fourth corpus tests the same structure well outside the datacenter regime. Husom et al. [67] measure
LLM inference energy for Llama, CodeLlama and Gemma across a workstation, two laptops and a server
under two prompt corpora, releasing per-request records rather than aggregates. On the balanced
sub-matrix of five workloads by two machines, built from between 3,109 and 11,828 requests per cell,
the interaction is {:.2f} per cent, within 0.01 percentage points of our own, and Proposition 1 again
holds exactly, with a fixed ranking and an additive model tying to the digit. That the same figure
appears on consumer and workstation hardware measured by a different group with different instruments
suggests the magnitude is a property of the problem rather than of our grid.
</p>
<p>
That corpus also supplies a negative result worth stating plainly, because it sharpens the scope of the
reversal claim. Its energy-optimal machine is the workstation for all five workloads: <b>no reversals
at all</b>. The two machines are far enough apart that the additive machine effect dominates the
interaction everywhere, so nothing is left for a placement policy to exploit. This is precisely the
two-type-pool regime in which Section 8 measures a 0.0 per cent placement gain, arrived at
independently. Reversals require devices close enough in capability that configuration can decide
between them, which is a condition on the <i>fleet</i>, not on the workload, and it is the paper's
central claim seen from the other side. (Verified against sampling noise: the between-workload spread
in the log energy ratio is 0.0296, against a bootstrap null of 0.0014, p &lt; 0.0001.)
</p>
<p>
Correlation across four corpora would still leave the direction of causation open""".format(inter))

# ------------------------------------------------------------------ Watt Counts corroboration
sub("wattcounts",
"""<h2>7. Predicting Unmeasured Pairs and Unowned Hardware</h2>""",
"""<h3>6.1 Corroboration at larger scale</h3>
<p>
Contemporaneous work reaches the same qualitative conclusion on a hardware axis twice the width of
ours. Fadel Argerich et al. [68] measure over 5,000 experiments across 50 open-weight language models
and 10 NVIDIA GPUs spanning five architectures, in both batch and server deployment scenarios. Their
finding is that the energy-optimal GPU is not a fixed property: in batch execution the H100 attains the
lowest energy per token for 90 per cent of models, while under a high-load server scenario the L4 is
lowest-power for 19 models and the T4 for 14, and under low load the T4 for 14 and the A30 for 11. The
hardware set and the models are identical across those scenarios; only the deployment configuration
changes, and the optimum moves with it. They report savings up to 70 per cent in server scenarios from
GPU selection alone, of the same order as the 34.3 per cent composition effect measured here.
</p>
<p>
Their modelling choice is instructive for the argument of Section 5. They fit mixed-effects models for
energy per token from model architecture, finding that a tenfold increase in active parameters raises
energy roughly 1.7x, with key-value head count the strongest independent architectural predictor. These
are excellent models of the energy <i>level</i>, and by Proposition 1 the level is exactly the component
that cannot select a device. The two results are complementary rather than competing: their models say
how much energy a model will draw, ours says which device should draw it, and a facility needs both.
</p>

<h2>7. Predicting Unmeasured Pairs and Unowned Hardware</h2>""")

# ------------------------------------------------------------------ Section 9 instrumentation scope
sub("sec9-scope",
"""Energy
comes from vendor counters covering the accelerator on a single vendor's hardware and a single cloud
provider, excluding host processor, memory and cooling; including them would raise the static share of
facility energy and therefore strengthen rather than weaken the consolidation result of Section 8.""",
"""Energy
comes from vendor counters covering the accelerator on a single vendor's hardware and a single cloud
provider, excluding host processor, memory and cooling. Independent instrumentation bounds what that
omits: measuring both accelerator and wall power on the same runs, Fadel Argerich et al. [68] report
the GPU accounts for 91.2 to 92.5 per cent of total energy in batch and high-load server scenarios, so
the excluded terms are a modest and roughly proportional addition. Including them would in any case
raise the static share of facility energy and therefore strengthen rather than weaken the consolidation
result of Section 8.""")

# ------------------------------------------------------------------ B5 into Recipe 2
sub("b5",
"""Because the value depends on how many of each type rather than merely which are
present, this is a multi-unit variant of a coalitional game.
</p>""",
"""Because the value depends on how many of each type rather than merely which are
present, this is a multi-unit variant of a coalitional game.
</p>
<p>
Marginal value is also a function of the contract, not only of the mix. Sweeping the service constraint
from 15 to 900 seconds, the total saving available rises monotonically, from 31.9 to 49.5 per cent under
a uniform mix as more compositions become feasible, and the <i>ranking</i> of accelerator types changes
with it: four distinct orderings appear across six constraint levels under a uniform mix and four under
an fp16-only mix, against two under a vision-only mix. Under a uniform mix the A100 ranks last at a
15-second constraint and third at 60 seconds and beyond, because a tight deadline rewards devices that
can be held idle cheaply while a loose one rewards raw efficiency. A buyer must therefore price the
service agreement into the purchase, and a marginal-value table computed at one deadline does not
transfer to another.
</p>""")

# ------------------------------------------------------------------ bibliography
sub("ref49",
"Notomista, G., Mayya, S., Emam, Y., Kroninger, C., Bohannon, A., Hutchinson, S., Egerstedt, M. A resilient and energy-aware task allocation framework for heterogeneous multi-robot systems. arXiv:2105.05586, 2021.",
"Notomista, G., Mayya, S., Emam, Y., Kroninger, C., Bohannon, A., Hutchinson, S., Egerstedt, M. A "
"resilient and energy-aware task allocation framework for heterogeneous multi-robot systems. IEEE "
"Transactions on Robotics 38(1):159-175, 2022. doi:10.1109/TRO.2021.3102379")
sub("ref8",
"Optimum-Benchmark. llm-perf-leaderboard. Hugging Face Datasets.",
"Optimum-Benchmark. llm-perf-leaderboard. Hugging Face Datasets. "
"https://huggingface.co/datasets/optimum-benchmark/llm-perf-leaderboard")
sub("ref53",
"NVIDIA. HGX H100 product carbon footprint summary. ISO 14067, third-party reviewed.",
"NVIDIA. HGX H100 product carbon footprint summary. ISO 14067, third-party reviewed. "
"https://www.nvidia.com/en-us/sustainability/")

# new references 67 and 68
sub("newrefs",
"</ol>",
"""<li>Husom, E. J., Goknil, A., Shar, L. K., Sen, S. The price of prompting: profiling energy use in
large language model inference. arXiv:2407.16893, 2024. Dataset: llm-inference-energy-consumption,
Hugging Face, CC BY-SA 4.0.</li>
<li>Fadel Argerich, M., Furst, J., Patino-Martinez, M. Watt Counts: energy-aware benchmark for
sustainable LLM inference on heterogeneous GPU architectures. arXiv:2604.09048, 2026.</li>
</ol>""")

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("applied: {}".format(", ".join(applied)))
print("FAILED : {}".format(", ".join(failed) if failed else "none"))
