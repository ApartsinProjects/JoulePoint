# -*- coding: utf-8 -*-
import io, json
V = json.load(io.open("experiments/report_values.json", encoding="utf-8"))

HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Configuration-Dependent Energy Rankings Across Heterogeneous AI Accelerators</title>
<style>
  @page {{ size: A4; margin: 2.2cm; }}
  :root {{ --ink:#15181c; --muted:#5b636b; --rule:#d5dae0; --accent:#0b4f6c; --hl:#e8f2f6; --bg:#fff; --card:#f7f9fb; }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
    --ink:#e6e9ed; --muted:#9aa4b0; --rule:#2c333b; --accent:#6fb3cc; --hl:#1d3038; --bg:#14171b; --card:#1b1f25; }} }}
  :root[data-theme="dark"] {{ --ink:#e6e9ed; --muted:#9aa4b0; --rule:#2c333b; --accent:#6fb3cc; --hl:#1d3038; --bg:#14171b; --card:#1b1f25; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:"Charter","Georgia","Times New Roman",serif; font-size:16px; line-height:1.62;
         color:var(--ink); background:var(--bg); max-width:52rem; margin:0 auto; padding:2.5rem 1.25rem 4rem; }}
  h1 {{ font-size:1.6rem; line-height:1.25; margin:0 0 .4rem; }}
  h2 {{ font-size:1.16rem; margin:2.1rem 0 .5rem; border-bottom:1px solid var(--rule); padding-bottom:.25rem; }}
  h3 {{ font-size:1rem; margin:1.4rem 0 .35rem; color:var(--accent); }}
  .byline {{ color:var(--muted); font-size:.95rem; margin-bottom:1.4rem; }}
  .abstract {{ background:var(--card); border-left:3px solid var(--accent); padding:.9rem 1.1rem; border-radius:0 6px 6px 0; margin:1.2rem 0 1.8rem; font-size:.96rem; }}
  .abstract h3 {{ margin:0 0 .35rem; }}
  p {{ text-align:justify; }}
  .wrap {{ overflow-x:auto; margin:.9rem 0; }}
  table {{ border-collapse:collapse; width:100%; font-size:.85rem; font-family:-apple-system,"Segoe UI",sans-serif; }}
  caption {{ caption-side:top; text-align:left; font-size:.85rem; color:var(--muted); padding-bottom:.3rem; font-style:italic; }}
  th,td {{ border:1px solid var(--rule); padding:.32rem .5rem; text-align:right; }}
  th:first-child, td:first-child {{ text-align:left; }}
  th {{ background:var(--card); font-weight:600; }}
  td.best {{ background:var(--hl); font-weight:700; }}
  .key {{ border:1px solid var(--accent); border-radius:6px; padding:.7rem 1rem; margin:1rem 0; background:var(--card); }}
  .key strong {{ color:var(--accent); }}
  code {{ font-family:"SF Mono",Consolas,monospace; font-size:.85em; background:var(--card); padding:.1em .3em; border-radius:3px; }}
  .small {{ font-size:.88rem; color:var(--muted); }}
  ol.refs {{ font-size:.9rem; }}
  ol.refs li {{ margin-bottom:.35rem; }}
  figure {{ margin:1.2rem 0; }}
  @media print {{ body {{ max-width:none; padding:0; font-size:10.5pt; }} h2 {{ page-break-after:avoid; }} table {{ page-break-inside:avoid; }} }}
</style>
</head>
<body>

<h1>The Fastest Accelerator Is Never the Greenest: Configuration-Dependent Energy Rankings Across Heterogeneous AI Hardware</h1>
<div class="byline">
Alexander Apartsin · Department of Computer Science, Holon Institute of Technology<br>
Technical Report · GreenMatch-AI Pilot Study · 18 August 2026
</div>

<div class="abstract">
<h3>Abstract</h3>
<p>
Energy-aware placement of AI workloads presumes that the energy cost of running a given workload on a given accelerator
can be anticipated. We report a complete, fully-measured pilot sweep testing that presumption: {n_cells} cells covering
four inference workloads on five accelerator families spanning three architecture generations, at two precisions and
three batch sizes, with energy read from the NVML total-energy counter alongside runtime and peak power. Three results
follow. First, selecting the highest-throughput accelerator, the default of performance-oriented schedulers, chose a
sub-optimal machine in <strong>all {n_cfg} of {n_cfg} configurations</strong>, at a median energy penalty of
<strong>{pen_med}%</strong> and up to <strong>{pen_max}%</strong>. Second, the energy ordering of accelerators is not
fixed: <strong>{n_rev} of {n_pairs} machine pairs ({pct_rev}%)</strong> reverse their ranking across configurations of
the same workload, with batch size and numerical precision each sufficient on their own to flip the order. Third, the
log-energy matrix is approximately low-rank, a rank-2 factorisation capturing {r2}% of its variance, and an additive
workload-plus-machine model explains {add_pct}% of that variance while leaving a residual interaction term worth up to
{max_res_x}x in energy. Energy efficiency is therefore not a property of hardware that can be tabulated once; it is a
property of the workload-hardware-configuration triple, with a small but operationally decisive interaction component.
The measurement harness and all {n_cells} cells are released.
</p>
</div>

<h2>1. Introduction</h2>
<p>
Data-center electricity demand driven by AI compute is growing faster than overall demand, and modern AI fleets are
heterogeneous: several accelerator generations coexist, with different compute throughput, memory bandwidth, and power
envelopes. Schedulers must therefore choose where to run each workload. In practice they choose on availability or
throughput, on the implicit assumption that the faster device is also the more efficient one, since it finishes sooner.
</p>
<p>
That assumption is testable, and it is the subject of this report. If the highest-throughput accelerator were reliably
the lowest-energy one, energy-aware placement would reduce to performance-aware placement and would need no new
machinery. If instead the energy-optimal choice depends on the workload and on the execution configuration, then
placement requires knowing an interaction that cannot be read off a specification sheet, and the size of that
interaction determines whether the problem is worth solving.
</p>
<p>
We measure the interaction directly. The contribution is not a new method but a clean empirical characterisation: a
complete, unpruned grid with no missing cells, measured under one protocol, on rented hardware anyone can rent.
</p>

<h2>2. Method</h2>

<h3>2.1 Workloads, machines, configurations</h3>
<p>
Four inference workloads were selected to span distinct computational profiles: <strong>ResNet-50</strong>
(convolutional, compute-bound), <strong>ViT-B/16</strong> (vision transformer, attention-dominated),
<strong>ConvNeXt-Tiny</strong> (modernised convolutional), and a <strong>six-layer Transformer encoder</strong>
(d_model 1024, 16 heads, feed-forward 4096, sequence length 256). All were constructed with randomly initialised
weights; the study concerns execution cost, which does not depend on trained parameter values.
</p>
<p>
Five accelerator families were used, spanning three NVIDIA architecture generations and a 5.7-fold range of enforced
power caps. Configurations crossed two precisions (fp32, and fp16 via autocast) with three batch sizes (8, 32, 128),
giving 4 x 5 x 2 x 3 = {n_cells} cells. Every cell returned a valid measurement; no cell was memory-infeasible at these
model sizes.
</p>
<div class="wrap">
<table><caption>Table 1. Machines under test.</caption>
<tr><th>Label</th><th>Device</th><th>Enforced cap (W)</th><th>Observed peak, min (W)</th><th>Observed peak, max (W)</th><th>Dynamic range</th></tr>
{pw_rows}
</table>
</div>
<p class="small">
Observed peaks exceed the enforced cap by 1 to 8%, consistent with short transients above the sustained limit. The
dynamic range column is the ratio of the highest to the lowest peak power observed across configurations on that
machine: the A100-40GB varies by 2.77x depending only on what it is running, which is why peak power must be predicted
per workload rather than assumed at the cap.
</p>

<h3>2.2 Measurement protocol</h3>
<p>
Each cell ran three warmup iterations, then a measured window repeated until at least four seconds of work had elapsed,
synchronised at both ends. Energy was taken as the difference of NVML's <code>nvmlDeviceGetTotalEnergyConsumption</code>
counter (millijoule resolution), with mean-power integration as fallback; the counter was available on every machine, so
all {n_cells} cells use the counter. Peak and mean power were sampled during the window, and peak device memory recorded.
The reported quantity throughout is <strong>energy per sample</strong>, total joules divided by the number of inputs
processed, which normalises across the differing iteration counts that a fixed time window produces.
</p>
<p>
All measurements ran on serverless GPU containers (Modal), one container per machine, dispatched in a single run. This
matters for reproducibility: the hardware is rentable on demand rather than institution-specific, so the grid can be
re-measured by anyone at a cost of a few dollars.
</p>

<h2>3. Results</h2>

<h3>3.1 Choosing the fastest machine is consistently the wrong energy decision</h3>
<div class="key">
<strong>Result 1.</strong> Selecting the highest-throughput accelerator chose a machine other than the lowest-energy one
in <strong>{n_cfg} of {n_cfg} configurations (100%)</strong>. The median energy penalty was <strong>{pen_med}%</strong>
and the maximum <strong>{pen_max}%</strong>.
</div>
<p>
The two selection rules never agreed. The highest-throughput machine was in every case either the L40S or the
A100-40GB, the two largest devices, while the lowest-energy machine was in every case the L4, a 72 W inference-oriented
part. Finishing sooner does not compensate for drawing four to five times the power.
</p>
<div class="wrap">
<table><caption>Table 2. Performance-first selection versus energy-optimal selection, all {n_cfg} configurations.</caption>
<tr><th>Workload</th><th>Precision</th><th>Batch</th><th>Fastest</th><th>Lowest energy</th><th>Energy penalty</th></tr>
{pen_rows}
</table>
</div>
<p class="small">
<strong>Scope of this result.</strong> That the penalty is universal here rather than merely common is anchored by one
device: the L4 was the energy winner in every cell, so no configuration existed in which the two rules could coincide.
A hardware set without an efficiency-optimised part would show a lower agreement rate. The magnitude of the penalty,
median {pen_med}%, does not depend on that anchoring, and it is the operationally relevant quantity.
</p>

<h3>3.2 The energy ranking of machines is not fixed</h3>
<div class="key">
<strong>Result 2.</strong> <strong>{n_rev} of {n_pairs} workload-machine pairs ({pct_rev}%)</strong> reverse their energy
ordering across configurations of the same workload. Batch size drove {n_bs} of the reversals and precision drove
{n_prec}.
</div>
<p>
A reversal means that for one workload, machine A uses less energy per sample than machine B at one configuration and
more at another, with no change of workload. The clearest instance is the Transformer encoder on the A100-40GB versus
the T4: at fp16 the A100 is 1.41x more efficient, and at fp32 it is 1.35x less efficient. Precision alone inverts the
choice. The same pattern appears on ViT-B/16 across the same pair.
</p>
<div class="wrap">
{m_fp16_32}
</div>
<div class="wrap">
{m_fp32_32}
</div>
<p>
Comparing the two tables above: moving from fp16 to fp32 leaves the L4 in front but rearranges everything behind it, and
the A100-40GB moves from third place to last on three of the four workloads. Any system that stores a single preference
order over accelerators is wrong for some configuration it will encounter.
</p>
<div class="wrap">
<table><caption>Table 5. All {n_rev} reversing pairs and the configuration dimension that flips them.</caption>
<tr><th>Workload</th><th>Machine A</th><th>Machine B</th><th>Flip driver</th></tr>
{rev_rows}
</table>
</div>

<h3>3.3 How much is at stake between machines</h3>
<p>
For a fixed workload and configuration, energy per sample between the best and worst feasible machine differed by a
median of <strong>{sp_med}x</strong>, with a range of {sp_min}x to {sp_max}x across the {n_cfg} configurations. The
spread is widest at fp32 and small batch, where the largest devices are least well utilised.
</p>
<div class="wrap">
{m_fp32_128}
</div>

<h2>4. Structure of the energy matrix</h2>
<p>
The practical question behind these results is whether the energy of unmeasured workload-machine pairs could be
predicted from a sparse sample, which requires the matrix to have exploitable structure. We examine the {n_cfg} x 5
matrix of log energy per sample, rows indexed by (workload, precision, batch) and columns by machine.
</p>
<h3>4.1 The matrix is approximately low-rank</h3>
<p>
A rank-1 factorisation captures <strong>{r1}%</strong> of the variance of the centred log-energy matrix, rank-2
<strong>{r2}%</strong>, and rank-3 <strong>{r3}%</strong>. The effective rank is therefore about two, against a matrix
of five columns. This is the structural precondition that matrix-completion approaches require, and it is what makes
prediction from a sparse sample plausible rather than hopeful.
</p>
<h3>4.2 Most of the variance is additive, but the interaction is what decides placement</h3>
<p>
Decomposing log energy into a workload effect plus a machine effect, the best additive model explains
<strong>{add_pct}%</strong> of the variance, leaving <strong>{res_pct}%</strong> in the interaction residual. Read
naively this suggests the interaction hardly matters. Read in the units that matter it says the opposite: the largest
residual corresponds to a <strong>{max_res_x}x</strong> difference in energy, and the reversals of Section 3.2 are
precisely the cases where the residual exceeds the gap between two machines' additive effects.
</p>
<p>
This is the central quantitative finding of the study. A "workload difficulty times machine efficiency" model is a good
variance-explainer and a poor decision-maker, because placement decisions are made in the tail of the residual, not in
its mean. Any method that fits only the additive part will produce accurate-looking predictions and systematically wrong
choices for exactly the pairs where the choice is non-obvious.
</p>

<h2>5. Threats to validity</h2>
<p>
<strong>Inference only.</strong> All workloads are forward-pass inference. Training changes the arithmetic intensity,
adds optimiser state and gradient traffic, and would plausibly shift the balance toward higher-bandwidth devices. The
generality of Result 1 to training is untested here.
</p>
<p>
<strong>Moderate model sizes.</strong> No configuration exhausted device memory, so memory capacity never became a
binding constraint. On larger models the smaller devices would become infeasible rather than merely slower, which
removes them from the comparison and would reduce the observed penalty.
</p>
<p>
<strong>One efficiency-optimised part anchors Result 1.</strong> As noted in Section 3.1, the L4's consistent win is
what makes the disagreement universal rather than frequent.
</p>
<p>
<strong>Single vendor, single provider, one measurement session.</strong> All devices are NVIDIA, rented from one
provider, measured once per cell without repetition across days. Run-to-run and host-to-host variance is therefore not
characterised, and NVML counters are vendor-reported rather than wall-socket measurements; they exclude host CPU,
memory and cooling.
</p>
<p>
<strong>Default software configuration.</strong> Results reflect stock PyTorch 2.4 with autocast and no vendor-specific
tuning; TF32 behaviour, kernel selection and compilation would all shift absolute numbers.
</p>

<h2>6. Reproducibility</h2>
<p>
The measurement harness is a single self-contained file that defines one function per machine and dispatches them
concurrently; the full sweep completes in one run. The harness, the raw {n_cells}-cell result set, and the analysis
script that produced every number in this report are released together. Re-running the study requires a Modal account
and costs on the order of one dollar of compute.
</p>

<h2>7. Conclusion</h2>
<p>
Across a complete grid of {n_cells} measured cells, the accelerator that finishes an AI inference workload fastest was
never the one that finishes it using the least electricity, at a median cost of {pen_med}% additional energy. The
ordering of accelerators by energy is not a fixed property that can be tabulated: {pct_rev}% of machine pairs invert
their ranking under a change of batch size or numerical precision alone. Beneath that variability the log-energy matrix
is approximately rank-2, so the structure needed to predict unmeasured pairs is present, but the component that decides
which machine to pick is the small interaction residual rather than the large additive part.
</p>
<p>
For practitioners the immediate implication is that a scheduler cannot become energy-aware by preferring the biggest
available device, nor by consulting a static efficiency ranking. For researchers the implication is that the modelling
target should be the interaction term explicitly, and that evaluation should be reported in placement regret rather
than in prediction error, since a model can be accurate on the additive structure and wrong on every decision that
matters.
</p>

<h2>References</h2>
<ol class="refs">
  <li>Apartsin, A., Meshulam, Y., and Aperstein, Y. (2026). <em>Acting on the Unseen: Communication-Free Collaborative Filtering for Decentralized Multi-Robot Task Allocation.</em> arXiv:2605.25584.</li>
  <li>Tripp, C. E., Perr-Sauer, J., Gafur, J., et al. (2024). <em>Measuring the Energy Consumption and Efficiency of Deep Neural Networks: An Empirical Analysis and Design Recommendations.</em> arXiv:2403.08151.</li>
  <li>Chung, J.-W., Ma, J. J., Wu, R., et al. (2025). <em>The ML.ENERGY Benchmark: Toward Automated Inference Energy Measurement and Optimization.</em> arXiv:2505.06371.</li>
  <li>Chung, J.-W., Wu, R., Ma, J. J., et al. (2026). <em>Where Do the Joules Go? Diagnosing Inference Energy Consumption.</em> arXiv:2601.22076.</li>
  <li>You, J., Chung, J.-W., and Chowdhury, M. (2022). <em>Zeus: Understanding and Optimizing GPU Energy Consumption of DNN Training.</em> arXiv:2208.06102.</li>
  <li>Fadel Argerich, M., Fürst, J., and Patiño-Martínez, M. (2026). <em>WattGPU: Predicting Inference Power and Latency on Unseen GPUs and LLMs.</em> arXiv:2607.02391.</li>
  <li>Lee, S., Phanishayee, A., and Mahajan, D. (2024). <em>Forecasting GPU Performance for Deep Learning Training and Inference.</em> arXiv:2407.13853.</li>
  <li>Zhang, M. and Chen, Y. (2019). <em>Inductive Matrix Completion Based on Graph Neural Networks.</em> arXiv:1904.12058.</li>
</ol>

<h2>Appendix A. Complete measurement set</h2>
<p class="small">All {n_cells} cells. Energy per sample in mJ; throughput in samples/s; power in W; peak memory in GB.</p>
<div class="wrap">
<table>
<tr><th>Workload</th><th>Prec.</th><th>Batch</th><th>Machine</th><th>Energy/sample</th><th>Throughput</th><th>Peak W</th><th>Mean W</th><th>Peak mem</th><th>Iters</th></tr>
{full}
</table>
</div>

</body>
</html>
"""

io.open("E:/Projects/Grants/energey/reports/pilot-technical-report.html", "w",
        encoding="utf-8", newline="\n").write(HTML.format(**V))
print("report written")
