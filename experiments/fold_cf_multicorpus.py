# -*- coding: utf-8 -*-
"""
Fold the multi-corpus evaluation into Section 7.

This is the first time the predictive model has been fitted on anything but our own grid and
MLPerf, and it changes what the paper can honestly claim. The single "87.9 against a ceiling of
81.7" figure is replaced by a condition: the interaction is learnable where there are enough
workload rows and informative descriptors, and not otherwise. Two corpora give a significant
NEGATIVE gain and are reported as such.

The device that makes the negatives interpretable is an oracle-loading bound: the same rank-1
form with the held-out row's TRUE loading substituted. It separates "no interaction exists
here" from "the loading regression could not be fitted". It clears the ceiling on every corpus,
so the model form is never what fails.

Grid'5000's eighteen platforms give the machine cold start the paper needed and could not
previously run outside MLPerf. It succeeds, but for a reason worth stating plainly: the
interaction contributes almost nothing to it. Cold start rides the machine level.
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

# ---------------------------------------------------------------- new 7.4, before 7.3
NEW = """<h3>7.3 When the interaction is learnable, and when it is not</h3>
<p>
The comparison above is on one grid. Five further corpora are now available, and fitting the same
model on each, under leave-one-workload-family-out throughout, gives a more useful answer than a
single number: the interaction is worth capturing under stated conditions, and outside them it is
not.
</p>
<div class="tablewrap">
<table>
<caption><b>Table 8.</b> The bilinear model across every available corpus, against a ceiling
estimated on training rows only. Intervals are a paired cluster bootstrap over rows. Proposition 1
held exactly on all eight panels, to ten decimal places.</caption>
<thead><tr><th>Corpus</th><th>Shape</th><th>Train rows</th><th>Ceiling</th><th>Bilinear</th>
<th>Gain</th><th>Regret</th></tr></thead>
<tbody>
<tr><td>Wilkins et al. [72]</td><td>32 x 4</td><td>24</td><td>89.6</td><td>99.0</td>
<td>+9.4 [+4.7, +14.6]</td><td>1.000</td></tr>
<tr><td>This work</td><td>24 x 5</td><td>18</td><td>81.7</td><td>87.9</td>
<td>+6.2 [+2.5, +10.0]</td><td>1.000</td></tr>
<tr><td>Extended grid</td><td>30 x 7</td><td>24</td><td>82.5</td><td>84.9</td>
<td>+2.4 [-0.5, +5.2]</td><td>1.030</td></tr>
<tr><td>llm-perf [8]</td><td>74 x 3</td><td>72</td><td>95.5</td><td>95.5</td>
<td>+0.0</td><td>1.007</td></tr>
<tr><td>Training grid</td><td>18 x 7</td><td>14</td><td>87.6</td><td>85.2</td>
<td>-2.4 [-4.2, -0.3]</td><td>1.021</td></tr>
<tr><td>Grid'5000 [84]</td><td>5 x 17</td><td>4</td><td>88.2</td><td>85.0</td>
<td>-3.2 [-8.8, 0.0]</td><td>1.203</td></tr>
</tbody>
</table>
</div>
<p>
The negatives are the informative rows, and they have a single diagnosable cause. To separate "no
interaction exists here" from "the loading could not be estimated", we compute an oracle-loading
bound: the same rank-1 form with the held-out row's <i>true</i> loading substituted. That bound
clears the ceiling on every corpus, so the model form is never what fails. What fails is the
regression from descriptors to loading, and it fails when rows are scarce. On Grid'5000 the
leave-one-workload-out fit estimates five descriptor coefficients from four points; the ridge
penalty saturates in three of five folds, shrinking the loading to zero, and pooled sign agreement
with the true loading is 20 per cent. The training grid shows the same mechanism more mildly at 14
rows. The controlled comparison is the pair in between: the extended grid is <i>the same seven
accelerators</i> with 30 rows instead of 18, and it gains where the training grid loses, which is
what a sample-size cause predicts and a hardware cause does not.
</p>
<p>
The llm-perf result is a different limit and equally worth stating. Its gain is exactly zero, not
because the interaction is absent, since the residual is 93 per cent rank-one and the oracle-loading
bound reaches 96.8 per cent, but because the only workload descriptors the corpus records are the
quantisation scheme and a parameter count. The fitted loading therefore takes about three distinct
values and shifts a whole scheme at once, flipping no pair. The missing ingredient is a descriptor
that varies <i>within</i> a quantisation scheme.
</p>
<p>
The practical condition is therefore twofold and can be checked before any model is built: roughly
twenty or more measured workloads, and descriptors that vary within the groups they define. Below
that, a fixed ranking is the better estimator and this paper recommends it, which is a narrower
claim than the literature on learned energy models generally makes.
</p>

<h3>7.4 Placing hardware that has never been run, on eighteen platforms</h3>
<p>
Section 7.1 established machine cold start on the MLPerf matrix. Grid'5000 tests it far harder: 18
platforms, each held out in turn, with its terms predicted from published specifications alone,
giving 1,530 cold-start comparisons against platforms the model has seen.
</p>
<div class="tablewrap">
<table>
<caption><b>Table 9.</b> Cold start over 18 Grid'5000 platforms, each held out in turn and predicted
from published descriptors. Chance is 50 per cent.</caption>
<thead><tr><th>Descriptors available</th><th>Pairwise accuracy</th><th>Energy regret</th></tr></thead>
<tbody>
<tr><td>Published specifications only</td><td>82.8</td><td>1.0040</td></tr>
<tr><td>Specifications, permuted (negative control)</td><td>73.8</td><td>1.0090</td></tr>
<tr><td>Specifications plus one idle-power probe</td><td>89.6</td><td>1.0053</td></tr>
</tbody>
</table>
</div>
<p>
A platform never measured on anything is placed correctly in 82.8 per cent of its pairings from
published figures alone, against a hindsight ceiling of 91.6, and 89.6 per cent once a single
idle-power characterisation run is permitted. Energy regret is 1.004, and <b>16 of the 18 held-out
platforms give a regret of exactly 1.000</b>: for most platforms the cold prediction picks the same
machine an oracle would. Permuting the descriptors costs 9.0 points [6.1, 11.9], so the signal is in
the descriptors and not in the fold structure.
</p>
<p>
One qualification matters and cuts against the framing a reader might expect. The interaction term
contributes almost nothing to this result, 0.1 and 0.4 points respectively; the machine-effect
regression carries it, at a leave-one-machine-out coefficient of determination of 0.59 on
specifications and 0.91 with the idle probe, while the interaction sensitivity is only weakly
identified. <b>Cold start on hardware works through the level, not through the interaction.</b>
Those are separate claims resting on separate evidence, and the procurement result of Section 8
depends only on the first.
</p>

"""
sub("sec7.3-7.4", "<h3>7.3 Placement is algorithmically saturated</h3>",
    NEW + "<h3>7.5 Placement is algorithmically saturated</h3>")

# renumber the later subsection reference
s = s.replace("Section 7.3's", "Section 7.5's")
s = s.replace("in Section 7.3", "in Section 7.5")

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("applied: {}".format(", ".join(applied)))
print("FAILED : {}".format(", ".join(failed) if failed else "none"))
