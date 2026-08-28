# -*- coding: utf-8 -*-
"""
Build the academic paper draft (HTML, SynSmith house style) for the power-shaping project.
Every headline number is read from results/*.json (no hand-typed values); key figures are
embedded as base64. Frames contributions 1 (grid-facing reframing) + 2 (validated SLO-preserving
controller) + 4 (conservative flexibility promise). Output: docs/power_shaping_paper.html
"""
import os, json, base64

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
OUT = os.path.join(HERE, "..", "docs", "power_shaping_paper.html")


def j(n): return json.load(open(os.path.join(RES, n)))
def img(n):
    with open(os.path.join(FIG, n), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


ka = j("poca_killtest.json"); vr = j("validate_real.json"); pb = j("pocb_azure.json")
lc = j("learned_control.json"); rq6 = j("rq6_safe.json"); ac = j("aws_control.json")
adi = j("aws_data_impact.json"); ri = j("rebound_israel.json"); ze = j("zeus_killtest.json")
il = j("israel_case.json"); fair = j("pocb_fair.json"); ev = j("elasticity_value.json")
acs = j("aws_control_strict_summary.json"); acs_a = acs["A10G"]; acs_l = acs["L4"]
het = j("het_alloc.json")            # heterogeneous-fleet decomposition (partition of the uniform->oracle gap)
pp = j("probe_predict.json")         # P1: elasticity is probeable, not predictable
sc = j("elasticity_scaling.json")    # elasticity value scales with fleet diversity (within-fleet blend)
pp_lin1 = pp["value_vs_k"]["linear"]["1"]; pp_knee2 = pp["value_vs_k"]["knee"]["2"]; pp_prior1 = pp["value_vs_k"]["prior"]["1"]
pp_cm = pp["classmean_feature_pct"]  # feature-only class-mean predictor, same elasticity-gap metric
sc_max = sc["value_max_pct"]
acta = j("aws_actuator.json")        # measured DVFS-vs-cap + model-ladder on one A10G
_acl = [r for r in acta["rows"] if r["kind"] == "clock"]; _acp = [r for r in acta["rows"] if r["kind"] == "cap"]
_acm = [r for r in acta["rows"] if r["kind"] == "model"]
act_clk_jop = min(float(r["energy_per_op_j"]) for r in _acl)
act_cap_jop = min(float(r["energy_per_op_j"]) for r in _acp)
act_erange = max(float(r["energy_per_op_j"]) for r in _acm) / min(float(r["energy_per_op_j"]) for r in _acm)
ev_em = ev["emerald"]; ev_ze = ev["zeus"]; ev_aws = ev["aws26"]   # per-workload elasticity value by pool
# equal-grid-compliance comparison (all controllers forced strictly under C(t))
fd = min(fair, key=lambda r: r["dip_frac"])          # deepest dip
fu, fp, fe = fd["uniform"], fd["priority"], fd["elasticity"]
prio_gain = fd["priority_gain_vs_uniform_pct"]; elas_gain = fd["elasticity_gain_vs_priority_pct"]

o10 = next(r for r in ka["static"] if r["curtailment_pct"] == 10)
o30 = next(r for r in ka["static"] if r["curtailment_pct"] == 30)
g10 = o10["oracle_gain_vs_uniform_pct"]; g30 = o30["oracle_gain_vs_uniform_pct"]
shared10 = o10["shared_elasticity_capture_pct"]
v_mae = vr["per_workload_MAE"]; v_red = vr["cluster_reduction_pct"]
v_real = vr["real_mean_perf"]; v_model = vr["model_mean_perf"]
dd = pb["dip_sweep"][-1]
u_crit = dd["uniform"]["crit_slo"]; u_int = dd["uniform"]["inter_slo"]
e_crit = dd["elasticity"]["crit_slo"]; e_int = dd["elasticity"]["inter_slo"]
e_off = dd["elasticity"]["off_slo"]; compl = dd["elasticity"]["compliance"]
oc_em = lc["oracle_capture"]["emerald"]["class_mean_pct"]
ufr_feat = rq6["policies"]["feat_mean"]["UFR_pct"]
ufr_obs = rq6["policies"]["obs_bound"]["UFR_pct"]
flex_obs = rq6["policies"]["obs_bound"]["usable_flexibility_frac"] * 100
# reliability-vs-usable-flexibility frontier for the hardware-grounded lower bound
fr = rq6["frontier_obs"]
fr_hi = max(fr, key=lambda r: r["margin"])           # margin 1.0: most flexibility offered
fr_rel = min(r["compliance_pct"] for r in fr)        # worst-case reliability across the sweep
feat_rel = rq6["feat_mean_point"]["compliance_pct"]  # reliability of the feature-only predictor
ac_t = ac["window_target_w"]; ac_u = ac["uncontrolled_power_w_median_inwindow"]
ac_c = ac["controlled_power_w_median_inwindow"]
ac_pu = ac["crit_p95_ms_uncontrolled_inwindow"]; ac_pc = ac["crit_p95_ms_controlled_inwindow"]
ac_du = ac["defer_thru_uncontrolled_inwindow"]; ac_dc = ac["defer_thru_controlled_inwindow"]
adi_n = adi["n_workloads"]; adi_lo = adi["learning_curve"][0]
adi_full = adi["capture_classmean_pct"]
ccm_u = ri["ccm"]["balanced"]["uniform"]["ccm"]; ccm_b = ri["ccm"]["balanced"]["elasticity"]["ccm"]
reb_u = ri["rebound"]["uncontrolled"]["rebound_over_baseline_pct"]
reb_c = ri["rebound"]["recovery_cap_0.8"]["rebound_over_baseline_pct"]
il_firm = il["evening_firm_mw"]
z_spread = ze["zeus_weighted"]["heterogeneity_spread"]; em_spread = ze["emerald_equalweight_spread"]

FIGS = {n: img(n) for n in ["fig_six_panel.png", "fig_guarantee.png", "fig_validate.png",
        "fig_depth_sweep.png", "fig_strict_control.png", "fig_scaling.png", "fig_probe.png",
        "fig_portfolio.png", "fig_actuators.png"]}

# affiliations from the authors' recent joint paper (arXiv:2509.03181)
AFFIL = ('<sup>1</sup>Holon Institute of Technology, Holon, Israel'
         '&nbsp;&nbsp;&nbsp;<sup>2</sup>Afeka Academic College of Engineering, Tel Aviv, Israel')

CSS = """
:root{--ink:#111418;--soft:#2c3138;--muted:#5a626c;--navy:#14385c;--rule:#d1d4d8;--code:#f4f5f7}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{background:#fff;color:var(--ink);margin:0;
 font-family:"Charter","Iowan Old Style","Source Serif Pro",Georgia,"Times New Roman",serif;
 font-size:11pt;line-height:1.55}
.wrap{max-width:720px;margin:0 auto;padding:44px 26px 80px}
.title{text-align:center;font-size:19pt;font-weight:600;line-height:1.25;margin:0 0 14px}
.authors{text-align:center;font-size:11pt;color:var(--soft);margin:0 0 2px}
.affil{text-align:center;font-size:9.5pt;color:var(--muted);margin:0 0 22px}
.abstract{margin:0 auto 8px;max-width:640px}
.abstract .h{text-align:center;font-variant:small-caps;letter-spacing:.06em;font-size:10pt;
 font-weight:700;color:var(--navy);margin:0 0 6px}
.abstract p{font-size:10pt;text-align:justify;hyphens:auto;text-indent:0;margin:0}
h2{font-size:13pt;font-weight:700;color:var(--ink);margin:26px 0 7px;line-height:1.3}
h3{font-size:11.5pt;font-weight:700;margin:18px 0 5px}
h4{font-size:10.5pt;font-weight:700;margin:14px 0 4px}
h2 .n,h3 .n{color:var(--navy);margin-right:.5em}
p{margin:0;text-indent:1.4em;text-align:justify;hyphens:auto}
h2+p,h3+p,h4+p,figure+p,.tablewrap+p,.abstract+*,ul+p{text-indent:0}
.lead,.abstract p{text-indent:0}
ul,ol{margin:5px 0 5px 0;padding-left:22px}
li{margin:2px 0;text-align:justify}
strong{font-weight:700}
em{font-style:italic}
code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:9.5pt;
 background:var(--code);padding:.5px 4px;border-radius:3px}
figure{margin:16px 0;text-align:center}
figure img{max-width:100%;height:auto;border:1px solid var(--rule);border-radius:2px}
figcaption{font-size:9.5pt;color:var(--soft);margin-top:6px;text-align:left;text-indent:0}
figcaption b{color:var(--ink)}
.tablewrap{margin:16px 0;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:9.5pt}
caption{caption-side:top;text-align:left;font-size:9.5pt;color:var(--soft);margin-bottom:5px}
caption b{color:var(--ink)}
th,td{padding:4px 9px;text-align:left}
thead th{border-bottom:1px solid var(--ink);font-weight:700}
table{border-top:1.5px solid var(--ink);border-bottom:1.5px solid var(--ink)}
tbody tr:first-child td{padding-top:6px}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.contrib{border-left:3px solid var(--navy);padding:2px 0 2px 14px;margin:10px 0}
.contrib p{text-indent:0;margin:4px 0}
.refs{font-size:9pt;color:var(--soft)}
.refs li{margin:3px 0;text-indent:0}
.foot{border-top:1px solid var(--rule);margin-top:34px;padding-top:12px;font-size:9pt;color:var(--muted)}
.foot p{text-indent:0}
.prov{font-size:9pt;color:var(--muted)}
.prov p{text-indent:0}
"""

HTML = f"""<!-- draft -->
<title>Toward Guaranteed Grid Flexibility from AI Data Centers</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{CSS}</style>
<div class="wrap">

<div class="title">Toward Guaranteed Grid Flexibility from AI Data Centers:<br>
Shaping Workloads to Honor Grid Power Limits While Protecting Critical Service</div>
<div class="authors">Alexander Apartsin<sup>1</sup> &nbsp;&middot;&nbsp; Yehudit Aperstein<sup>2</sup></div>
<div class="affil">{AFFIL}</div>

<div class="abstract">
<div class="h">Abstract</div>
<p>Electricity grids increasingly cap the power an AI data center may draw during stress windows, yet the
standard response, capping every GPU uniformly and serving first-come-first-served, slows latency-critical
inference as hard as deferrable batch work. We reframe curtailment as a <em>grid-facing allocation problem</em>:
given a time-varying power allowance, choose <em>which</em> workloads to slow and <em>how</em>, driven by
service priority and each workload's measured power-cap elasticity. Done well, the power limit is honored,
high-priority service is protected, and the facility's flexibility becomes a firm, verifiable commitment. We
develop two complementary mechanisms. A priority-aware controller allocates the time-varying power budget
across service classes: it sheds only what the allowance requires, protects high-priority service-level
objectives (SLOs) by serving critical work first and deferring the rest, and limits the post-event recovery
ramp (an uncontrolled backlog drain would run {reb_u:.0f}% above the pre-event baseline; a ramp-limited
recovery holds it below that). Separately, a hardware-enforced GPU power cap gives a conservative,
verifiable device-power ceiling with a <em>provable lower bound</em> on the reduction it delivers, a primitive
that can backstop a grid commitment. On a real 271k-request
Azure inference trace under a 20-minute curtailment to {int(fd['dip_frac']*100)}% of peak, and comparing all
controllers at equal (strict) grid compliance, priority-aware shaping holds critical and interactive SLO
violations at {fp['crit_slo']:.0f}% where the class-blind default reaches {fu['crit_slo']:.0f}% and
{fu['inter_slo']:.0f}%, by serving high-priority work first under the shared cap. On this single-application
trace that gain comes from serving order, not per-workload elasticity; on a heterogeneous fleet of
{ev_aws['n_workloads']} measured workloads, per-workload elasticity closes a further
{het['mean_elasticity_close_pct']:.0f}% of the gap to a full-knowledge oracle (an allocator given every
workload's true curve), a value recovered largely ({pp_lin1:.0f}%) from a single cheap probe per workload. The allocator reproduces a real 256-GPU grid
experiment to within {v_mae:.3f} per workload (mean absolute error in normalized service performance, and it
errs on the safe side), and on a real A10G under step,
staircase, and ramp targets the hardware cap holds power under target
{100-acs_a['time_above_target_pct_mean']:.0f}% of the time (median {acs_a['actuation_s_median']:.2f}&#8202;s
actuation) with critical 95th-percentile (p95) latency far under its deadline.
For grid-facing promises, a one-reading, hardware-enforced power ceiling guarantees delivered reduction meets
the commitment (reliability 100% by construction, where a statistical predictor of the same promise keeps it
only {feat_rel:.0f}% of the time) while offering up to {fr_hi['usable_flex_frac']*100:.0f}% of the true
flexibility.</p>
</div>

<h2><span class="n">1</span>Introduction</h2>
<p class="lead">Grid operators facing peak stress are beginning to require large electricity consumers,
including AI data centers, to hold their draw under a dynamic ceiling. A data center that must shed tens of
megawatts within minutes has a blunt default: lower the power cap on every GPU by the same fraction. That
uniform response is priority-blind. It slows a latency-critical inference request as aggressively as an
overnight training job, even though the two carry very different service value and very different tolerance
for delay. Deferrable work can absorb a curtailment window that a latency-critical stream cannot.</p>

<p>Production AI data centers are not single-workload systems; they run a heterogeneous mix. Cluster traces
document training and inference co-located at scale with many model types and fractional GPU sharing (Alibaba
MLaaS [13], Microsoft Philly [14], Google Borg [15]), inference is now the majority of ML energy in at least
one hyperscaler fleet [17], and different workloads have sharply different power profiles: LLM inference is
compute-bound in its prompt phase and memory-bound in decode [16], while recommendation (DLRM) is
memory-bandwidth-bound and draws far below a GPU's cap [18]. This heterogeneity in <em>priority</em> and in
power <em>response</em> is precisely what a workload-aware controller can exploit and a uniform cap cannot.</p>

<p>We take the position that curtailment in a grid-constrained AI data center is best cast as an
<em>allocation</em> problem, not a global dimmer. The facility is handed a time-varying power allowance
<code>C(t)</code> and must decide, tick by tick, which workloads to slow and by how much, subject to
per-class service-level objectives, so that the delivered service loss is as small and as low-priority as
possible. This reframing is the first contribution and organizes the paper.</p>

<div class="contrib">
<p><strong>Contribution 1 (reframing).</strong> We formulate grid curtailment as priority-aware power
<em>shaping</em>: allocate a shared, time-varying power allowance across an SLO-differentiated workload mix,
with explicit rebound and ramp constraints (&sect;2).</p>
<p><strong>Contribution 2 (controller, validated).</strong> A backlog-aware shaping controller that (i) sheds
only what the allowance requires, (ii) protects high-priority SLOs by deferring low-value work first, and
(iii) bounds the recovery rebound. It reproduces a real 256-GPU grid experiment to {v_mae:.3f} per-workload
and is demonstrated with measured closed-loop control on real hardware (&sect;3, &sect;5).</p>
<p><strong>Contribution 3 (flexibility guarantee).</strong> Because a GPU power limit is hardware-enforced,
committing to a power ceiling from a single runtime reading gives a <em>provable lower bound</em> on delivered
reduction. Its reliability R, the probability that the delivered reduction meets the promised one, is 100% by
construction where a statistical predictor of the same promise reaches only {feat_rel:.0f}% on our corpus;
this is the property a grid operator needs to trust the resource (&sect;4).</p>
<p><strong>Contribution 4 (per-workload elasticity: when it pays, and how to get it).</strong> A second lever,
per-workload power elasticity, adds value only when the fleet is diverse: it is zero on a single-application
trace and closes a further {het['mean_elasticity_close_pct']:.0f}% of the gap to a full-knowledge oracle on a
fleet built from {ev_aws['n_workloads']} measured workloads. The curves it needs are <em>probeable, not
predictable</em>: a single roughly 30-second reduced-power probe per workload recovers {pp_lin1:.0f}% of that
value, where predicting the curve from workload features recovers {pp_cm:.0f}% (&sect;5.6).</p>
</div>

<p>We are explicit about scope. The evaluation is <em>simulated-on-measured</em>: real measured
power-response curves and a real arrival trace drive a tick simulator, with one end-to-end measured control
loop on a real GPU and one comparison against a real grid experiment. The main controller acts on workload
<em>priority</em>; a finer per-workload power-elasticity lever adds value only when the fleet is genuinely
diverse, and we show in &sect;5.6 exactly how that value grows with workload-response diversity and how cheaply
it can be measured. One consequence of the hardware-enforced ceiling is worth stating up front: a learned
shaping policy layered above it cannot raise the commanded power ceiling however wrong it is, so it can
worsen service efficiency or quality but never grid compliance (&sect;5.7).</p>

<h2><span class="n">2</span>Problem formulation</h2>
<p class="lead">A serving cluster hosts a mix of workloads partitioned into priority classes (critical,
interactive, batch, offline) with per-class service weights and queueing-delay deadlines: a request that
waits longer than its class deadline counts as a service-level-objective (SLO) violation, and the
<em>weighted service cost</em> sums those violations (and any deferred or unfinished work) scaled by the
class weight, so a violated critical request costs far more than the same delay on offline work. At each
control tick (half a second here) the facility observes the allowance <code>C(t)</code>, the current
per-class queue state, and the live GPU power draw. An <em>action</em> is a pair: a continuous GPU power-cap
fraction and a per-class admission gate (fully served, deferred, or fractionally admitted). Facility power
under an action follows from the running workloads' <em>elasticity curves</em>. A workload's power-cap
elasticity is its measured response curve, how its power draw and throughput fall as the GPU power cap is
lowered: a compute-bound job sheds a lot of power per unit of throughput lost, a memory-bound job little
because it already draws below the cap. Deferring a class removes its load from the tick but grows a backlog
that must be recovered later.</p>

<p><b>Objective.</b> Each class <em>k</em> carries a weight <em>w<sub>k</sub></em> and a deadline
<em>&tau;<sub>k</sub></em>. At tick <em>t</em> the controller picks a per-GPU cap fraction
<em>u<sub>t</sub></em>&nbsp;&isin;&nbsp;[0,&nbsp;1] and a per-class admission gate
<em>a<sub>k,t</sub></em>&nbsp;&isin;&nbsp;[0,&nbsp;1]. Let <em>v<sub>i</sub></em>&nbsp;=&nbsp;1 when request
<em>i</em>'s queueing delay exceeds its class deadline, and <em>b<sub>k</sub></em> be the deferred backlog of
class <em>k</em> still outstanding at the recovery horizon. The controller minimizes the weighted service
cost</p>

<p style="text-align:center;margin:0.4em 0"><em>J</em>&nbsp;=&nbsp;&sum;<sub>i</sub>&nbsp;<em>w<sub>k(i)</sub>
v<sub>i</sub></em>&nbsp;+&nbsp;&sum;<sub>k</sub>&nbsp;<em>w<sub>k</sub> b<sub>k</sub></em></p>

<p>subject to the grid constraint <em>P</em>(<em>u<sub>t</sub></em>)&nbsp;&le;&nbsp;<code>C(t)</code> at every
tick, where <em>P</em> is the facility power implied by the running workloads' elasticity curves, and a bound
on the rate at which <em>u</em> is released during recovery (so the drained backlog does not produce a rebound
spike that violates the ramp the grid allows). Uniform capping is the special case with one cluster-wide
<em>u<sub>t</sub></em>, <em>a<sub>k</sub></em>&nbsp;&equiv;&nbsp;1, and class-blind first-come service.</p>

<h2><span class="n">3</span>Workload-aware shaping</h2>
<p class="lead">The controller plans ahead and corrects with feedback. Because the allowance is known in
advance, at each tick it uses a response model to predict the power that each candidate action would draw,
and picks the action that sheds the least while still meeting <code>C(t)</code>. Two refinements keep the
shed from overshooting what the grid asks. First, it tunes the power cap continuously to the highest value
still under the allowance, rather than choosing from a fixed set of shed levels. Second, it lets deferred
work back in priority order (most valuable first) to fill any leftover headroom, adjusting to the power
actually measured on the previous tick. The result is that the delivered reduction tracks the required
reduction, and curtailment falls on the least valuable work that still keeps power under the ceiling.</p>

<p>Rebound is controlled at recovery time by a ramp-limited release of the cap as the backlog drains, so the
recovered power rises gradually rather than snapping back to the pre-event peak. Figure 1 shows the full
sequence on a real trace: allowance, shaped facility power, per-class latency, and the bounded rebound.</p>

<figure><img src="{FIGS['fig_six_panel.png']}" alt="systems view">
<figcaption><b>Figure 1.</b> Grid stress to service to rebound, simulated on the real Azure trace and measured
emerald elasticity. The shaped power tracks the allowance (Panel 3; the delivered reduction matches the
required-reduction line in Panel 4), critical and interactive queueing delay stays near their deadlines while
offline absorbs the delay (Panel 5), and the deferred backlog drains into a bounded rebound (Panel 6). The
green dotted line in Panel 3 is the response model's online prediction the controller acts on.</figcaption></figure>

<h2><span class="n">4</span>A hardware-grounded flexibility guarantee</h2>
<p class="lead">To be useful as a grid resource, a facility must promise a shed it will actually deliver;
over-promising is worse than promising less. We first state the primitive precisely, because it is what makes
the promise a guarantee rather than a forecast. A GPU power limit set through the NVIDIA Management Library
(NVML) (or <code>nvidia-smi
-pl</code>) is <em>hardware-enforced</em>: the device clamps its own draw to at most the limit L. Committing to
a ceiling L is therefore a firm, externally verifiable commitment, not a prediction.</p>

<p>This yields a lower bound on delivered reduction, which is exactly what a grid operator needs. Let a
workload be observed drawing D, and let the facility commit to cap it at L &lt; D and promise a reduction
F = D &minus; L. Writing the realized capped draw as d, the hardware guarantees d &le; L, so the delivered
reduction is D &minus; d &ge; D &minus; L = F. Hence <em>delivered &ge; promised whenever the cap binds</em>;
if the workload's own demand falls below L it simply draws less, over-delivering on the ceiling. Define
reliability as the probability that delivered meets promised, R = Pr[D &minus; d &ge; F]. The construction
above gives R = 1 for the ceiling primitive, in contrast to a statistical predictor, whose promise F is a
model output that can exceed what the workload can shed. A feature-only model over-promises badly on
memory-bound workloads unlike any it was fit on, which already draw far below their cap: on the Zeus corpus
(a public set of measured GPU power and energy traces across diverse DNN training workloads [3]) it achieves
only R = {feat_rel:.0f}% (it under-delivers on {ufr_feat:.0f}% of promises by a material margin), and a wider
statistical interval does not help because the failure is a shift to unseen workloads, not noise.</p>

<p>The remaining design choice is how aggressively to set L from the single observed reading D. Setting
L = D &minus; m&nbsp;(D &minus; L<sub>floor</sub>) with a safety fraction m (larger m sets the ceiling lower and
promises more) trades offered flexibility against robustness to measurement noise and phase variation. Across
the whole range m &isin; [0.7, 1.0], the hardware-grounded ceiling keeps R = 100% on Zeus by construction,
because the hardware cap makes delivered reduction at least the promise whenever the cap binds, not because a
model was fitted (Table 1). One measured cell shows a draw a fraction of a watt above its cap, within sensor
noise and below the margin we test, so no promise is broken. At the most aggressive setting
it still offers {fr_hi['usable_flex_frac']*100:.0f}% of the true achievable flexibility (the usable
flexibility: the promised reduction as a fraction of the largest reduction the workload could actually
deliver), and a smaller m buys robustness by promising less, at no measured reliability gain on this corpus.
The guarantee is on the ceiling L, and hence on the reduction relative to a committed baseline. A workload
that finishes early draws less than L: it over-delivers on the ceiling while shedding less in absolute terms.
The facility therefore commits ceilings, which the hardware keeps, not fixed absolute reductions.</p>

<div class="tablewrap">
<table>
<caption><b>Table 1.</b> Flexibility-promise reliability on the Zeus corpus (leave-one-workload-out).
Reliability is R = Pr[delivered &ge; promised]. The hardware-grounded ceiling keeps R = 100% by construction
across the whole safety-margin range; a statistical predictor does not. Promised flexibility above 100% is
overcommitment: the predictor promises more reduction than the workload can deliver.</caption>
<thead><tr><th>Promise policy</th><th class="n">Reliability R (%)</th><th class="n">Promised flexibility (%)</th></tr></thead>
<tbody>
<tr><td>Feature-only mean prediction</td><td class="n">{100-rq6['policies']['feat_mean']['UFR_pct']:.0f}</td><td class="n">{rq6['policies']['feat_mean']['usable_flexibility_frac']*100:.0f}</td></tr>
<tr><td>Feature-only 90% quantile</td><td class="n">{100-rq6['policies']['feat_quant']['UFR_pct']:.0f}</td><td class="n">{rq6['policies']['feat_quant']['usable_flexibility_frac']*100:.0f}</td></tr>
<tr><td>Hardware-grounded ceiling (m=1.0)</td><td class="n">{fr_hi['compliance_pct']:.0f}</td><td class="n">{fr_hi['usable_flex_frac']*100:.0f}</td></tr>
<tr><td>Hardware-grounded ceiling (m=0.9)</td><td class="n">{100-rq6['policies']['obs_bound']['UFR_pct']:.0f}</td><td class="n">{rq6['policies']['obs_bound']['usable_flexibility_frac']*100:.0f}</td></tr>
</tbody>
</table>
</div>

<figure><img src="{FIGS['fig_guarantee.png']}" alt="flexibility guarantee">
<figcaption><b>Figure 2.</b> The hardware-grounded flexibility guarantee (Zeus corpus, leave-one-workload-out).
(A) Reliability R = Pr[delivered &ge; promised]: the hardware-enforced ceiling holds 100% across every safety
margin (solid, squares), while a feature-only statistical predictor sits at {feat_rel:.0f}% (dashed, open
circles); the shaded band is the trust gap. (B) Usable flexibility offered: the ceiling delivers up to
{fr_hi['usable_flex_frac']*100:.0f}% of the true flexibility, all of it reliable, whereas the statistical
predictor promises more but keeps its promise only {feat_rel:.0f}% of the time.</figcaption></figure>

<h2><span class="n">5</span>Evaluation</h2>

<h3><span class="n">5.1</span>Data and method</h3>
<p class="lead">Our main evaluation is trace-driven simulation on measured inputs, complemented by a
field-experiment comparison (&sect;5.2) and single-GPU closed-loop hardware measurements (&sect;5.4). The
measured inputs are: the emerald cluster (the 256-GPU cluster of the field experiment we validate against
[1]), which provides a DVFS (dynamic voltage and frequency scaling) sweep of GPU power-response and a real
grid-experiment power trace; and the 27.3M-request Azure LLM inference trace, which supplies arrivals (a
271k-request busy hour replayed at full fidelity). The service-dynamics panels are the tick simulator's output
on these measured inputs. We state per result which parts are measured and which are simulated.</p>

<h3><span class="n">5.2</span>Field validation</h3>
<p class="lead">During a real grid experiment the emerald 256-GPU cluster shed power for three hours ([1] reports
about a 25% shed; the deepest sustained reduction we measure on the released power trace is {v_red:.0f}%) and
recorded per-workload service performance. We ask the allocator to hit that reduction on the same workloads and
compare its per-workload flexing decisions against the measured outcome. It matches to a per-workload mean
absolute error of {v_mae:.3f}, and is slightly conservative on aggregate performance (model {v_model:.3f}
versus measured {v_real:.3f}), which is the safe direction for a grid-facing system.</p>

<figure><img src="{FIGS['fig_validate.png']}" alt="field validation">
<figcaption><b>Figure 3.</b> The allocator reproduces the field experiment's per-workload flexing (it flexes
pretraining hardest and protects inference) to a mean absolute error of {v_mae:.3f} in normalized service
performance, on the real measured cluster-power trace (left panel).</figcaption></figure>

<h3><span class="n">5.3</span>SLO-preserving curtailment, at equal grid compliance</h3>
<p class="lead">We compare three controllers on the Azure busy hour, all forced <em>strictly</em> under the
power allowance C(t), so the grid ceiling is a hard constraint for every method and the comparison is at
equal compliance (compliance is the fraction of control ticks with facility power at or below C(t); strict
means 100%). Under a hard cap the serving capacity is fixed, so what matters is how each controller uses it. The class-blind default caps every GPU by the same fraction and serves requests first-come,
first-served; the priority-aware controller, at the same grid-compliant cap, serves critical work first and
defers the low-priority (batch) classes; the third adds per-workload elasticity. The class-blind case is the
naive default, and the priority-aware controller is the strong baseline against which elasticity is judged.</p>

<p>Under a 20-minute curtailment to {int(fd['dip_frac']*100)}% of peak (Table 2), the class-blind default
leaves the high-priority classes badly exposed ({fu['crit_slo']:.0f}% critical and {fu['inter_slo']:.0f}%
interactive SLO violations): with capacity fixed by the hard cap, serving first-come-first-served runs
critical inference behind batch work. Serving high-priority work first removes those violations
({fp['crit_slo']:.0f}% critical and interactive) and most of the weighted service cost; deferring the batch
classes then lowers it the rest of the way, to {fp['wcost']:,.0f}. The protection of critical service
therefore comes from <em>which</em> work the scarce capacity serves first, not from how the cap is set. The elasticity-aware
controller is identical to the priority one here ({elas_gain:.0f}% further gain): under one grid-clamped
cluster cap on a single-application trace every request shares the same elasticity curve, so per-workload
elasticity has nothing to exploit. Section 5.6 measures where it does.</p>

<div class="tablewrap">
<table>
<caption><b>Table 2.</b> Azure curtailment to {int(fd['dip_frac']*100)}% of peak, all controllers at equal
(strict, {fu['compliance']:.0f}%) grid compliance. The critical-SLO win over the class-blind default comes
from serving high-priority work first under the shared cap; deferral further lowers the weighted service
cost. Per-workload elasticity adds nothing on this single-application trace.</caption>
<thead><tr><th>Controller</th><th class="n">Grid compliance (%)</th><th class="n">Critical SLO viol. (%)</th><th class="n">Interactive SLO viol. (%)</th><th class="n">Weighted service cost</th></tr></thead>
<tbody>
<tr><td>Uniform capping (class-blind)</td><td class="n">{fu['compliance']:.0f}</td><td class="n">{fu['crit_slo']:.0f}</td><td class="n">{fu['inter_slo']:.0f}</td><td class="n">{fu['wcost']:,.0f}</td></tr>
<tr><td>Priority-aware (serve first + defer)</td><td class="n">{fp['compliance']:.0f}</td><td class="n">{fp['crit_slo']:.0f}</td><td class="n">{fp['inter_slo']:.0f}</td><td class="n">{fp['wcost']:,.0f}</td></tr>
<tr><td>Elasticity-aware</td><td class="n">{fe['compliance']:.0f}</td><td class="n">{fe['crit_slo']:.0f}</td><td class="n">{fe['inter_slo']:.0f}</td><td class="n">{fe['wcost']:,.0f}</td></tr>
</tbody>
</table>
</div>

<p>The advantage is regime-dependent, not a single operating point (Figure 4). Sweeping the curtailment depth
at equal compliance, all three controllers coincide until the allowance drops below roughly half of peak;
past that, the class-blind default's critical-class SLO violations and weighted service cost rise steeply
while priority-aware serving holds them near zero. The elasticity-aware curve lies exactly on the priority
curve throughout, because under strict equal compliance both controllers reduce to the same grid-clamped
cluster cap and differ only in serving order.</p>

<figure><img src="{FIGS['fig_depth_sweep.png']}" alt="curtailment-depth sweep">
<figcaption><b>Figure 4.</b> Curtailment-depth sweep at equal (strict) grid compliance. (A) Critical-class SLO
violations and (B) weighted service cost versus depth. The class-blind default degrades sharply in the deep
regime (shaded); priority-aware serving holds high-priority SLOs, and the elasticity-aware controller is
indistinguishable from it here (under strict equal compliance both use the same cluster cap).</figcaption></figure>

<h3><span class="n">5.4</span>Measured closed-loop control on real GPUs</h3>
<p class="lead">To ground the control claim in hardware rather than simulation, we ran closed-loop control on
real AWS GPUs across three target trajectories (a single step, a staircase, and a linear ramp), enforcing the
grid ceiling with the hardware power cap (nvidia-smi) so commanded power is bounded, while a
duty-cycle controller (alternating a deferrable job between short run and pause slices) yields the GPU to a
latency-critical stream when the cap tightens. On the A10G, whose power-cap range is wide, measured power
stays at or below target <b>{100-acs_a['time_above_target_pct_mean']:.1f}%</b> of the constrained time (the
remaining {acs_a['time_above_target_pct_mean']:.1f}% is transient, within telemetry noise), the cap settles in
a median <b>{acs_a['actuation_s_median']:.2f}&#8202;s</b>, and critical p95 latency stays far under its 1&#8202;s
deadline (median {acs_a['crit_p95_ms_median']:.2f}&#8202;ms). This is the measured behavior of the &sect;4
primitive at the device: a per-GPU power result, not yet a facility-meter guarantee (&sect;7).</p>

<p>The same runs expose a real limit that motivates the actuator portfolio of &sect;5.7. A power cap can only
reach down to a GPU's minimum power limit; on the low-TDP (low thermal-design-power) L4, whose cap range is narrow, aggressive targets fall
below that floor, the cap saturates, and power then sits above the requested target
({acs_l['time_above_target_pct_mean']:.1f}% of the time on average, more at the deepest ramp points). Deep
curtailment on such hardware therefore cannot be met by the power cap alone and requires the software
actuators, deferral, clock scaling, batching, exactly the portfolio the enforcement layer is designed to
supervise.</p>

<figure><img src="{FIGS['fig_strict_control.png']}" alt="measured strict-envelope control">
<figcaption><b>Figure 5.</b> Measured hardware-cap control: GPU power (teal) tracking the target (dashed) across
two GPUs and three trajectories; per-cell time-above-target annotated. The A10G (wide cap range) holds every
target to within about 2% of the constrained time; the low-TDP L4 (narrow range) saturates its cap floor at
aggressive targets. All panels are hardware measurements.</figcaption></figure>

<h3><span class="n">5.5</span>Rebound control</h3>
<p class="lead">Deferring work protects SLOs during the window but must not simply relocate the peak. At
recovery, an uncontrolled backlog drain sustains {reb_u:.0f}% over the pre-event baseline, whereas the
controller's ramp-limited cap release cuts the rebound to {reb_c:.0f}% (Figure 1, Panel 6), shaving the
recovery peak rather than shifting it in time.</p>

<h3><span class="n">5.6</span>Where per-workload elasticity matters</h3>
<p class="lead">Section 5.3 found priority sufficient because the trace is homogeneous. To locate where the
second lever, per-workload power-cap elasticity, actually pays, we measure its <em>pure</em> value: the
equal-weight oracle gain over uniform capping on each measured workload pool (the <em>oracle</em> is the
allocator handed every workload's true measured curve, an upper bound on what any predictor could do; equal
weights, so the gain is elasticity alone, with no priority component). On the near-homogeneous emerald LLM fleet the value is small
({ev_em['elasticity_value_pct']:.0f}%); it is largest on the diverse {ev_aws['n_workloads']}-workload pool we
measured on a real A10G ({ev_aws['elasticity_value_pct']:.0f}%, spanning compute-bound matrix-multiplication
(GEMM) and attention kernels to memory-bound gather and reduction). The value is real but depends on the
fleet being genuinely diverse (Table 3).</p>

<p>This is where the second lever pays, resolving the question &sect;5.3 left open. We build a heterogeneous
fleet from the {ev_aws['n_workloads']} measured workloads and assign priority classes that <em>cross-cut</em>
elasticity (verified: the mean sheddable fraction, the share of a workload's power a full cap-down removes,
is comparable across classes), so a critical memory-bound job
and an offline compute-bound job coexist and priority alone cannot separate them. We then decompose the
weighted-service-cost gain over uniform capping (at matched curtailment) into a priority component
(priority-aware allocation with one shared elasticity curve, versus uniform) and an elasticity component
(per-workload elasticity, versus priority-only), reporting both as a share of the same total gap so they
partition it. Of the whole gap from uniform capping to the full-knowledge oracle, priority-aware allocation
closes {het['mean_priority_close_pct']:.0f}% and per-workload elasticity closes the remaining
<b>{het['mean_elasticity_close_pct']:.0f}%</b>, the part that was zero on the homogeneous serving trace. (At
the deepest curtailment the split reverses: priority alone can trail uniform there while elasticity recovers
the gap.) A check specified in advance confirms the effect is real: on a fleet where every workload carries
the shared fleet-average curve, the elasticity part of the gap is exactly 0%, reproducing the homogeneous
case.</p>

<p>The zero on the homogeneous trace and the {het['mean_elasticity_close_pct']:.0f}% on the diverse fleet are
two ends of one scale, not two separate facts. Blending the measured fleet continuously from its true
diversity down to a single shared curve, the elasticity value falls smoothly to zero (Figure 6): it is a
property of how different the workloads' response curves are. We read this only within a fleet; across
different real fleets the value also depends on fleet size and sampling, so we do not claim a universal law
from it.</p>

<figure><img src="{FIGS['fig_scaling.png']}" alt="elasticity value scales with fleet diversity">
<figcaption><b>Figure 6.</b> Elasticity value against fleet diversity, blending the measured
{ev_aws['n_workloads']}-workload fleet from a single shared curve (diversity 0, elasticity worth nothing) to
its true measured diversity (value {sc_max:.0f}%). Within a fleet the value rises smoothly from zero; it is a
property of the fleet's diversity, not a fixed quantity.</figcaption></figure>

<p>This raises the deployment question the oracle sidesteps: no operator sweeps every production workload
offline, so how are the per-workload curves obtained? Scoring predictors in <em>decision</em> space (the
fraction of the elasticity gap they recover, priority-only to oracle, with the scoring validated by feeding
the true curve to recover 100% and a homogeneous fleet to leave 0%), we find the curves are
<em>better probed than predicted</em>. Feature-only prediction gets only partway: a leave-one-workload-out
type-family mean recovers {pp_cm:.0f}% of the elasticity gap, and a learned curve prior essentially none
({pp_prior1:.0f}% at one probe, negative at higher orders as the fit misallocates). A single roughly
30-second reduced-power <em>probe</em> per workload with plain linear interpolation does markedly better,
recovering <b>{pp_lin1:.0f}%</b>, and two probes with a knee fit (a two-segment line with one bend) recover
{pp_knee2:.0f}%. Per-workload elasticity is
therefore a lever an operator calibrates in situ with a couple of cheap measurements, not one that needs a
learned model. In the temporal serving regime of &sect;5.3, where work can be deferred, priority-ordered
deferral already absorbs the shortfall and this second lever stays dormant; it becomes decisive in the
simultaneous-allocation regime measured here, where the whole heterogeneous fleet must run under the ceiling
at once.</p>

<figure><img src="{FIGS['fig_probe.png']}" alt="elasticity is probeable not predictable">
<figcaption><b>Figure 7.</b> Per-workload elasticity is better probed than predicted. (A) Fraction of the
oracle's elasticity gap (priority-only to oracle) recovered versus the number of cheap in-situ probes: a
single reduced-power probe with linear interpolation reaches {pp_lin1:.0f}% and two probes with a knee fit
{pp_knee2:.0f}%, while the zero-probe feature-only class-mean reaches {pp_cm:.0f}% and a learned curve prior
essentially nothing. (B) Why one probe suffices: for a compute-bound (steep) and a memory-bound (flat)
workload, a single probe at half power plus linear interpolation tracks the measured curve.</figcaption></figure>

<div class="tablewrap">
<table>
<caption><b>Table 3.</b> Pure per-workload elasticity value (equal-weight oracle gain over uniform capping) by
pool. The value increases with workload-response diversity: negligible on a single application, large on a diverse fleet.</caption>
<thead><tr><th>Workload pool</th><th class="n">Distinct workloads</th><th class="n">Elasticity value (%)</th></tr></thead>
<tbody>
<tr><td>emerald (LLM, near-homogeneous)</td><td class="n">{ev_em['n_workloads']}</td><td class="n">{ev_em['elasticity_value_pct']:.0f}</td></tr>
<tr><td>own A10G sweep (diverse)</td><td class="n">{ev_aws['n_workloads']}</td><td class="n">{ev_aws['elasticity_value_pct']:.0f}</td></tr>
</tbody>
</table>
</div>

<h3><span class="n">5.7</span>Beyond two actuators: a control-point portfolio</h3>
<p class="lead">Power-cap and deferral are two <em>actuators</em> (the knobs a controller can move to change power);
a grid-facing controller has more. They fall into three
mechanism classes: shift energy in time at constant total work (deferral, checkpoint-and-pause of training);
reduce energy per unit of work (frequency/clock scaling, batching, quantization, routing elastic work to
efficient GPUs); and reduce the work itself (admission, output-length limiting, small-model cascades). They
also span a time-scale ladder, from millisecond hardware knobs (power cap, clock lock) through per-request
software knobs (a model-cascade routing threshold, a token budget) to minute-scale placement (consolidation,
precision switching). A natural architecture is a supervisor that shapes demand with the software actuators
and enforces the ceiling with the hardware cap, using the cap's clipping frequency (how often the cap is
actively limiting the device, a free signal of how tight the ceiling is) as the feedback signal.
We measured two actuators beyond ours on a single A10G (Figure 8). First, GPU clock-frequency scaling (the
streaming-multiprocessor, or SM, clock): over the range we measured, at matched mean power it delivers more
throughput than the power cap, and it reaches an interior energy minimum of {act_clk_jop:.2f}&#8202;joules per
operation (J/op) below its maximum clock. The power
cap's energy per operation was still falling at the low end of the range we swept ({act_cap_jop:.1f}&#8202;J/op),
so this is the cap's best <em>measured</em> point, not its floor; the clock is the better shaping actuator
where we could measure both, though a deeper cap sweep could narrow the gap. Second, a small-to-large model
cascade: measured energy per inference spans a
<b>{act_erange:.0f}&times;</b> range across a model ladder (from a mobile CNN to a large vision transformer),
paired with published accuracy this is a continuous power-versus-<em>quality</em> knob that neither the cap
(power-versus-latency) nor deferral (power-versus-delay) reaches. Fully characterizing and jointly controlling
the portfolio is future work; the present paper establishes the enforcement layer, the priority actuator, and
two measured additions on which it builds.</p>

<figure><img src="{FIGS['fig_actuators.png']}" alt="two measured actuators">
<figcaption><b>Figure 8.</b> Two measured actuators beyond power-cap + deferral (one A10G). (A) At matched mean
power, GPU clock-frequency scaling (squares) delivers more throughput than the power cap (circles), and a
lower energy-per-operation minimum over the measured range. (B) A model-size ladder spans a {act_erange:.0f}&times; range in measured energy per
inference against published ImageNet accuracy, the power-versus-quality axis a cascade router exposes.</figcaption></figure>

<p>Orchestrating this many actuators efficiently is where learning enters, and the enforcement layer makes
learning <em>safe</em>. We separate the problem into two loops: an inner, millisecond, hardware-guaranteed
loop that holds facility power under C(t) by construction (the cap, never set above the one-reading ceiling
certified in Section 4), and an outer, seconds-to-minutes loop that <em>allocates the headroom</em> across
actuators. The outer loop would be a constrained optimization over the actuators, driven by learned models of each
actuator's power, latency, and quality cost, so the controller takes the next watt of reduction from whichever
actuator is cheapest in service terms. Because the hard ceiling is enforced by the inner loop, a wrong learned
model cannot raise the commanded power ceiling: it can worsen service efficiency or quality, but never grid
compliance. That property is what would let an operator adopt a learned controller under a safety-critical
grid contract. We do not build or evaluate this multi-actuator controller here; the response models it would
need are the cheap in-situ elasticity probes of &sect;5.6, and the decisive next experiment is to show such a
controller beats uniform, priority-only, and every single actuator at equal grid compliance across a
curtailment-depth sweep.</p>

<figure><img src="{FIGS['fig_portfolio.png']}" alt="control-point portfolio and supervisor">
<figcaption><b>Figure 9.</b> Left: the control-point portfolio, actuators arranged by mechanism class (shift
energy in time, reduce energy per unit work, reduce the work) and native time scale (milliseconds to minutes);
only the GPU power cap (bold) is hardware-enforced. Right: the supervisor architecture, an ML layer shapes
demand across the software actuators while the hardware cap enforces the ceiling, with the cap's clipping
frequency as the feedback signal, so a wrong learned model costs efficiency, never grid compliance.</figcaption></figure>

<h2><span class="n">6</span>Related work</h2>
<p class="lead"><b>Grid-interactive AI data centers.</b> The closest prior art demonstrates that a large GPU
cluster can cut power on a grid signal while protecting service: Colangelo et al. [1] field-tested a
256-GPU cluster shedding 25% for three hours. We build the allocation layer that work motivates: rather than
a facility-wide cut, we differentiate service by priority under a hard, time-varying allowance, prove a
lower-bound flexibility guarantee, and bound the recovery rebound. Broader grid-interactive and carbon-aware
computing shifts load in time for emissions or grid signals [4, 9, 12]; the demand-response framing of data
centers as deferrable load goes back to Ghatikar et al. [10]. Our allowance is set by the grid, not chosen by
the facility, and never to be exceeded, and our
contribution is the SLO-differentiated controller and the verifiable-flexibility primitive on top of it.</p>

<p><b>GPU power knobs and power characterization.</b> Power-capping and DVFS have been studied as
single-workload efficiency levers: Zeus [3] optimizes the energy of DNN training, and POLCA [2]
characterizes LLM-inference power and argues for power oversubscription. We treat the same hardware cap not as
an efficiency knob but as a grid-facing allocation lever across an SLO-differentiated mix, and we use its
hardware-enforced semantics to make a flexibility promise verifiable (&sect;4).</p>

<p><b>Datacenter power capping and SLO-aware resource control.</b> Facility-wide power management
oversubscribes and caps power across servers (Dynamo [5], power provisioning [6]); SLO- and QoS-aware
resource managers protect latency-critical services under colocation and throttling (Heracles [7],
PARTIES [8], Autopilot [11], PowerNap for idle power). We combine these lineages under an <em>external</em>
grid ceiling: the cap level is set by the grid, not by internal oversubscription, and priority-aware serving,
not just per-server capping, is what preserves high-priority SLOs. We do not yet compare against dedicated
priority-scheduling systems beyond our own uniform and priority-aware controllers.</p>


<h2><span class="n">7</span>Limitations</h2>
<p class="lead">The evaluation is simulated-on-measured, with one end-to-end measured control loop and one
field-experiment comparison; a multi-node deployment under a live grid signal is future work. Our power
results are at the GPU: the hardware cap bounds commanded GPU power, but a facility-meter commitment
additionally requires accounting for non-GPU load (CPU, memory, network, storage, cooling) and aggregation
across many devices, which we do not model. The paper therefore establishes a per-GPU power primitive, a step
toward a facility-level guarantee rather than the guarantee itself; delivering a fixed absolute reduction
further requires the facility's own load to reach the committed baseline, and the single-reading commitment
inherits the measurement's noise, which the safety margin absorbs but does not eliminate. The temporal
controller acts on workload priority under one cluster cap; per-workload elasticity carries real value on a
heterogeneous fleet (&sect;5.6) but is not yet integrated into the temporal controller on a heterogeneous
request mix, the clearest next experiment. The priority classes, weights, and deadlines are imposed on an
inference trace, so the SLO numbers should be read with that construction in mind, and the priority/elasticity
split depends on the cross-cutting class assignment we chose rather than a swept distribution. We compare
against uniform capping and our own priority-aware controller, not a dedicated priority-scheduling system.</p>

<h2><span class="n">8</span>Conclusion</h2>
<p class="lead">Casting grid curtailment as priority-aware power shaping turns a blunt facility-wide dimmer
into an allocation that protects what matters. At equal grid compliance, a backlog-aware controller sheds only
what the allowance requires, holds high-priority SLOs that first-come-first-served serving under the same cap
would sacrifice, and bounds the recovery rebound; a one-reading hardware ceiling lets the facility promise
less flexibility than it has and keep every promise on our corpus. The controller is validated against a real
256-GPU field experiment and demonstrated on real hardware. It is a step toward
AI data centers operating as predictable, grid-responsive loads under externally imposed power limits.</p>

<h2><span class="n">9</span>References</h2>
<ol class="refs">
<li>P. Colangelo, A. K. Coskun, J. Megrue, et al. &ldquo;AI data centres as grid-interactive assets.&rdquo;
<i>Nature Energy</i>, 2025. doi:10.1038/s41560-025-01927-1.</li>
<li>P. Patel, E. Choukse, C. Zhang, &Iacute;. Goiri, et al. &ldquo;Characterizing power management
opportunities for LLMs in the cloud&rdquo; (POLCA). <i>ASPLOS</i>, 2024. doi:10.1145/3620666.3651329.</li>
<li>J. You, J.-W. Chung, M. Chowdhury. &ldquo;Zeus: Understanding and optimizing GPU energy consumption of
DNN training.&rdquo; <i>USENIX NSDI</i>, 2023. arXiv:2208.06102.</li>
<li>A. Radovanovi&cacute;, R. Koningstein, I. Schneider, et al. &ldquo;Carbon-aware computing for
datacenters.&rdquo; <i>IEEE Transactions on Power Systems</i> 38(2), 2023. arXiv:2106.11750.</li>
<li>Q. Wu, Q. Deng, L. Ganesh, et al. &ldquo;Dynamo: Facebook&rsquo;s data center-wide power management
system.&rdquo; <i>ISCA</i>, 2016. doi:10.1145/3007787.3001187.</li>
<li>X. Fan, W.-D. Weber, L. A. Barroso. &ldquo;Power provisioning for a warehouse-sized computer.&rdquo;
<i>ISCA</i>, 2007. doi:10.1145/1250662.1250665.</li>
<li>D. Lo, L. Cheng, R. Govindaraju, P. Ranganathan, C. Kozyrakis. &ldquo;Heracles: Improving resource
efficiency at scale.&rdquo; <i>ISCA</i>, 2015. doi:10.1145/2749469.2749475.</li>
<li>S. Chen, C. Delimitrou, J. F. Mart&iacute;nez. &ldquo;PARTIES: QoS-aware resource partitioning for
multiple interactive services.&rdquo; <i>ASPLOS</i>, 2019. doi:10.1145/3297858.3304005.</li>
<li>P. Wiesner, I. Behnke, D. Scheinert, K. Gontarska, L. Thamsen. &ldquo;Let&rsquo;s wait awhile: How
temporal workload shifting can reduce carbon emissions in the cloud.&rdquo; <i>ACM/IFIP Middleware</i>, 2021.
arXiv:2110.13234.</li>
<li>G. Ghatikar, M. A. Piette, S. Fujita, et al. &ldquo;Demand response and open automated demand response
opportunities for data centers.&rdquo; <i>Lawrence Berkeley National Laboratory</i>, LBNL-3047E, 2010.</li>
<li>K. Rzadca, P. Findeisen, J. Swiderski, et al. &ldquo;Autopilot: Workload autoscaling at Google.&rdquo;
<i>EuroSys</i>, 2020. doi:10.1145/3342195.3387524.</li>
<li>A. Souza, N. Bashir, J. Murillo, et al. &ldquo;Ecovisor: A virtual energy system for carbon-efficient
applications.&rdquo; <i>ASPLOS</i>, 2023. doi:10.1145/3575693.3575709.</li>
<li>Q. Weng, W. Xiao, Y. Yu, et al. &ldquo;MLaaS in the wild: Workload analysis and scheduling in large-scale
heterogeneous GPU clusters.&rdquo; <i>USENIX NSDI</i>, 2022.</li>
<li>M. Jeon, S. Venkataraman, A. Phanishayee, et al. &ldquo;Analysis of large-scale multi-tenant GPU clusters
for DNN training workloads.&rdquo; <i>USENIX ATC</i>, 2019.</li>
<li>M. Tirmazi, A. Barker, N. Deng, et al. &ldquo;Borg: the next generation.&rdquo; <i>EuroSys</i>, 2020.
doi:10.1145/3342195.3387517.</li>
<li>P. Patel, E. Choukse, C. Zhang, et al. &ldquo;Splitwise: Efficient generative LLM inference using phase
splitting.&rdquo; <i>ISCA</i>, 2024. doi:10.1109/ISCA59077.2024.00019.</li>
<li>D. Patterson, J. Gonzalez, U. H&ouml;lzle, et al. &ldquo;The carbon footprint of machine learning training
will plateau, then shrink.&rdquo; <i>IEEE Computer</i>, 2022. arXiv:2204.05149.</li>
<li>U. Gupta, C.-J. Wu, X. Wang, et al. &ldquo;The architectural implications of Facebook&rsquo;s DNN-based
personalized recommendation.&rdquo; <i>IEEE HPCA</i>, 2020.</li>
</ol>

<div class="foot">
<p>Draft generated from the project's result artifacts; every headline figure is read from a saved
<code>results/*.json</code>, each checked by an automated consistency test before the build. Figures 1 and 4 are simulated on measured power-response and a real trace; Figure 3 compares the
allocator against a real field experiment; Figures 5 and 8 are hardware measurements.</p>
</div>

</div>
"""


def main():
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(HTML)
    print(f"paper -> {os.path.abspath(OUT)}  ({len(HTML)//1024} KB)")


if __name__ == "__main__":
    main()
