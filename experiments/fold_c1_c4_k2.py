# -*- coding: utf-8 -*-
"""
Fold the measured C1 and C4 runs and the Grid'5000 corpus into the paper, including one
correction to a claim that turned out to be overstated.

The correction. Section 6 says that MLPerf's 0.3 per cent interaction is explained by taking
a maximum over configurations, "precisely the operation that removes the variation generating
the interaction". Applying exactly that operation to our own grid (30 configurations by 7
accelerators, collapsed to 5 workloads by 7 accelerators at each cell's best configuration)
removes 34 per cent of the interaction, from 1.20 to 0.80 per cent, not nearly all of it. The
claim is directionally right and quantitatively too strong, and is now stated as measured.
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

# ---------------------------------------------------------------- Table 2, Grid'5000
sub("table2-grid5000",
"""<tr><td>Hybrid heterogeneous clusters [72]</td><td>token counts, fixed across machines</td><td>2.7%</td></tr>""",
"""<tr><td>Grid'5000 NPB campaign [84]</td><td>none; fixed problem size per benchmark</td><td>2.2%</td></tr>
<tr><td>Hybrid heterogeneous clusters [72]</td><td>token counts, fixed across machines</td><td>2.7%</td></tr>""")
sub("table2-cap5to6",
"Interaction residual across five independently collected corpora",
"Interaction residual across six independently collected corpora")

# ---------------------------------------------------------------- Section 6, Grid'5000
sub("sec6-grid5000",
"""<p>
Correlation across five corpora would still leave the direction of causation open""",
"""<p>
The widest test available is not on accelerators at all. The Grid'5000 campaign [84] runs
fixed-problem-size numerical benchmarks across <b>eighteen</b> hardware platforms with
full-node wattmeters reporting energy directly in joules, giving a completely dense 5 by 18
matrix. Its interaction is 2.2 per cent, inside the same band, Proposition 1 holds exactly,
and the energy-optimal platform differs for three of the five benchmarks. Since these are CPU
nodes running numerical kernels with no precision or batch axis at all, the structure is
evidently not a property of AI accelerators or of machine-learning software; it is a property
of heterogeneous hardware executing heterogeneous work.
</p>
<p>
Correlation across six corpora would still leave the direction of causation open""")

# ---------------------------------------------------------------- Section 6, the correction
sub("sec6-correction",
"""The ordering is exact and it is the ordering the hypothesis predicts. MLPerf submissions are
tuned per cell by the submitting vendor and then reported at their best, so taking a maximum over
configurations is precisely the operation that removes the variation generating the interaction; the
residual there is 0.3 per cent, effectively nothing.""",
"""The ordering is exact and it is the ordering the hypothesis predicts. MLPerf submissions are tuned
per cell by the submitting vendor and then reported at their best, and the residual there is 0.3 per
cent, effectively nothing. How much of that is the tuning itself can be measured rather than assumed,
by applying the same maximum-over-configurations operation to our own grid: collapsing 30
configurations to 5 workloads at each cell's best setting reduces the interaction from 1.20 to 0.80
per cent. So the collapse accounts for about a third of the gap, not all of it, and the remainder
must come from MLPerf's different hardware and workload set and from submission-level optimisation
we cannot observe. The tuning is a real and measurable cause; it is not the only one.""")

# ---------------------------------------------------------------- Section 9, transfer now partial
sub("sec9-transfer",
"""Only
two accelerators appear in both our grid and MLPerf's datacenter set, so cross-corpus
transfer of a fitted model cannot be tested here, and the three-corpus comparison of Section 6
is consequently a comparison of interaction magnitudes rather than of transferred predictions.""",
"""Cross-corpus transfer was previously untestable, since our grid and MLPerf's datacenter set shared
two accelerators and no workloads at all. Adding H100 and H200 to our grid, together with BERT-Large,
an MLPerf benchmark, makes a partial test possible and it now runs: on the shared sub-matrix the
accelerator ordering agrees between corpora, and the interaction is negligible on both sides
(0.08 and 0.12 per cent) once each is read at its per-cell best configuration, as MLPerf necessarily
is. The test remains partial, and the reason is a fact about MLPerf worth stating: the H200 has no
power-measured submission for either shared workload, and the 40 GB A100 has BERT but no ResNet, so
an exact-part bridge covers 3 of 6 cells and the analysis has to pool accelerator families, which
confounds the 40 GB and 80 GB A100 variants. The comparison of interaction magnitudes across corpora
in Section 6 therefore remains the stronger evidence.""")

# ---------------------------------------------------------------- Section 9, F3 update
sub("sec9-f3",
"""The extension to training workloads
in Section 7.2 breaks this, producing three distinct winners, and the reversal result of
Section 5 does not depend on it at all, since it concerns pairs rather than a global optimum.""",
"""Two extensions break this. Section 7.2's training grid produces three distinct winners, and a
further measured grid of five workloads in training mode across seven accelerators, spanning a
tenfold power-limit range from a 70 W T4 to a 700 W H200, gives two: the L4 wins 13 of 18 balanced
configurations and the H200 wins 5, with the fastest machine also the cheapest in only 5 of 18. The
reversal result of Section 5 does not depend on the point at all, since it concerns pairs rather than
a global optimum.""")

# ---------------------------------------------------------------- Section 9, reproducibility
sub("sec9-repro",
"""<p>
On reproducibility, facility figures come from a stochastic arrival stream and carry
Monte Carlo variation of roughly 1.4 per cent in standard deviation between seeds, which is
the scale of disagreement to expect when re-running these simulations.""",
"""<p>
On reproducibility of the measurements themselves, the grid was re-executed months later on a
different cloud account, and the 120 overlapping cells give a median absolute difference of 2.4 per
cent with machine ordering preserved on 96.7 per cent of within-row accelerator pairs. The
disagreement is not uniform: it is concentrated almost entirely on the T4, at a median of +10.5 per
cent, while the other four machines lie within 2.4 per cent, which is consistent with a different
physical card of the oldest and most variably binned part in the set. The reversal of Figure 2 was
re-tested directly on the new measurements and survives in all six configurations. Since every claim
here concerns orderings rather than absolute energies, the 96.7 per cent figure is the material one,
but absolute energies should be read with a several-per-cent tolerance rather than the 1.6 per cent
within-run replicate figure.
</p>
<p>
On reproducibility of the simulations, facility figures come from a stochastic arrival stream and
carry Monte Carlo variation of roughly 1.4 per cent in standard deviation between seeds, which is
the scale of disagreement to expect when re-running them.""")

# ---------------------------------------------------------------- reference
sub("ref-grid5000", "</ol>",
"""<li>Da Costa, G., et al. Multi-hardware energy and performance measurements of the NAS Parallel
Benchmarks across the Grid'5000 testbed. Zenodo, CC BY 4.0. doi:10.5281/zenodo.10982238</li>
</ol>""")

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("applied: {}".format(", ".join(applied)))
print("FAILED : {}".format(", ".join(failed) if failed else "none"))
