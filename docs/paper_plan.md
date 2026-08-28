# Detailed Paper Plan
## Predicting and Exploiting AI Workload Power Flexibility Without Per-Job Profiling

## 0. Working Paper Positioning

### Recommended working title
**Predicting AI Workload Power Flexibility for Grid-Constrained Data Centers**

Alternative titles:
1. **From Unprofiled AI Jobs to Flexible MW: Learning Workload Power Response for Data-Center Power Shaping**
2. **Learning Computational Flexibility: Workload-Aware Power Shaping for AI Data Centers**
3. **A World Model of AI Workload Flexibility for Data-Center Power Control**
4. **Predicting Counterfactual Power–Performance Response of AI Workloads for Grid-Aware Scheduling**
5. **Toward Flexible AI Data Centers: Predicting Which Workloads to Shape Under Power Constraints**

### Central scientific claim
> **The power flexibility of an unseen AI workload can be predicted from workload attributes, hardware information, and runtime state well enough to make better power-shaping decisions than workload-agnostic or priority-only policies, without exhaustive per-job profiling.**

### Central systems claim
> **A controller using predicted workload flexibility can track a facility power envelope with lower service degradation than simple uniform power capping or static priority heuristics.**

### Application claim
> **Predictive workload flexibility can translate part of installed AI computing demand from apparently firm electrical load into measurable, controllable load under explicit service guarantees.**

### What the paper is NOT
The paper should not claim:
- first AI workload-shaping system;
- first digital twin for AI power;
- first grid-responsive AI data center;
- first GPU power-capping controller;
- first data-center demand-response system;
- guaranteed reduction in total energy;
- direct proof that a real 100 MW Israeli facility can operate behind a smaller connection.

The paper should instead focus on:
\[
\boxed{
\text{Can flexibility of previously unseen AI workloads be predicted accurately enough to improve power-control decisions?}
}
\]

---

# 1. Core Story Arc

## Step 1 — Establish the physical/computational phenomenon
Different AI workloads must react differently to the same intervention.

Question:
> **Is workload power flexibility heterogeneous enough to matter?**

If not, there is no need for workload-aware prediction.

## Step 2 — Establish decision relevance
Even if workloads differ, a simple heuristic may capture nearly all useful flexibility.

Question:
> **Does knowing workload-specific response improve the actual control decision?**

This is the first major kill test.

## Step 3 — Establish the practical gap
Existing profiled controllers can exploit known workload flexibility, but profiling every new workload/configuration is costly or impractical.

Question:
> **Can response be predicted from attributes and state instead?**

## Step 4 — Evaluate prediction on truly unseen workloads
Random row-level train/test splits are insufficient.

Question:
> **Can the model generalize to jobs it has never seen or profiled?**

## Step 5 — Put prediction inside the controller
Prediction accuracy alone is not enough.

Question:
> **Does prediction improve power-envelope control?**

## Step 6 — Add uncertainty and temporal effects
A grid-facing system must not promise flexibility it cannot deliver.

Questions:
> **Can uncertainty calibration reduce under-delivery?**

> **Does rebound-aware planning prevent shifted peaks?**

## Step 7 — Demonstrate realistic workload-scale behavior
Replay production traces.

Question:
> **Does the advantage survive burstiness, workload mixtures, and priorities?**

## Step 8 — Translate to the electricity-system application
Use Israeli grid traces as an application case study.

Question:
> **What amount of firm electrical capacity would be required for a given installed AI workload under explicit assumptions?**

---

# 2. Recommended Contributions

## C1 — Empirical characterization
A measurement and simulation study showing that AI workloads exhibit heterogeneous, state-dependent power–performance response under common control interventions.

Possible wording:
> We characterize the power elasticity of representative AI workloads and show when identical power-management actions produce substantially different service costs.

## C2 — Generalizable intervention-response model
A model predicting:
\[
(	ext{workload},	ext{state},	ext{hardware},	ext{action})
\rightarrow
(\Delta P,\Delta QoS,\Delta E,\Delta T)
\]
for workloads withheld from profiling.

This is the main scientific contribution.

## C3 — Workload-aware power-shaping controller
A controller that uses predicted response to select the least damaging set of interventions required to satisfy:
\[
P(t)\le C(t).
\]

## C4 — Safety/reliability layer
An uncertainty-aware version that uses conservative predicted flexibility rather than mean response, and optionally models deferred work/rebound.

## C5 — Trace-driven grid application
Evaluation under production AI workload traces and Israeli electricity-system scenarios, translating workload flexibility into a firm-capacity / flexible-capacity interpretation.

---

# 3. Research Questions

## RQ1 — Is AI workload power flexibility heterogeneous?
> How much do power reduction and service degradation vary across workloads, states, hardware, and actions?

### H1
\[
rac{-\Delta P}{\Delta C_{	ext{service}}}
\]
varies materially across workloads and workload states.

### Evidence
- intervention sweeps;
- distributions;
- variance decomposition;
- pairwise comparison of response curves.

## RQ2 — Is the heterogeneity decision-relevant?
> How much better can a controller perform if it knows the true flexibility of every workload?

### H2
A profiled/oracle workload-aware policy significantly outperforms:
- uniform power capping;
- priority-first;
- largest-power-first;
- static workload-class rules.

### Evidence
Power-reduction vs service-cost curves.

### Kill test
If simple heuristics capture nearly all oracle value, predictive ML is not justified.

## RQ3 — Can flexibility be predicted for unseen workloads?
> Can a model trained on other workloads predict \(\Delta P\), \(\Delta QoS\), and \(\Delta E\) for a held-out workload without profiling that workload under the candidate intervention?

### H3
Workload attributes + runtime state provide predictive value beyond:
- global average;
- job class;
- current power/utilization only.

## RQ4 — Does learned flexibility improve control?
> Does a learned controller reduce service cost for a required power reduction compared with simple policies?

### H4
The learned controller captures a substantial fraction of the value achieved by a controller with true profiles.

\[
	ext{Oracle Capture} =
rac{
C_{	ext{simple}}-C_{	ext{learned}}
}{
C_{	ext{simple}}-C_{	ext{oracle}}
}
\]

## RQ5 — Does the approach survive realistic workload traces?
> Does the benefit persist under bursty online inference and mixed production AI workloads?

Use:
- Azure LLM traces;
- Alibaba GPU traces.

## RQ6 — Can flexibility be promised safely?
> Does uncertainty calibration reduce the rate at which actual delivered power reduction is less than promised?

\[
UFR =
\Pr(
\Delta P_{	ext{actual}}
<
\Delta P_{	ext{promised}}
)
\]

## RQ7 — Does rebound matter?
> Does ignoring deferred work create secondary power peaks or deadline failures?

## RQ8 — What is the application value for grid-constrained AI infrastructure?
> Under realistic workload mixes, how does workload flexibility affect the firm electrical capacity required for a virtual 10–100 MW AI facility?

---

# 4. Section-by-Section Manuscript Plan

## 4.1 Abstract
Target: 180–250 words.

Structure:
1. Electrical capacity is becoming a constraint for AI data centers.
2. Software workload shaping can provide flexibility, but existing methods often assume known flexibility/profiles.
3. New workloads/configurations continuously arrive; exhaustive profiling is impractical.
4. Learn action-conditioned power/performance response from workload attributes and runtime state.
5. Use model inside a controller to select least-cost interventions under a power envelope.
6. Evaluate with physical measurements, public measured data, production traces, and simulation.
7. Insert quantitative results only after experiments.
8. End with implications for flexible data-center demand.

## 4.2 Introduction
Target: 1–1.5 pages.

### Paragraph 1 — Power is becoming a compute constraint
Explain:
- electrical capacity can constrain AI deployment;
- local interconnection limits matter independently of annual MWh;
- computation is programmable and therefore potentially flexible.

Key framing:
> The key question is no longer only how efficiently AI consumes energy, but how flexibly computational demand can be shaped under a time-varying power constraint.

### Paragraph 2 — Workload shaping exists, but requires knowledge
Briefly position:
- Google demand response;
- Phoenix/Emerald;
- GPU power capping;
- scheduler priorities;
- grid-interactive computing.

Then:
> These systems demonstrate that computational load can be shaped. They do not eliminate the need to know which workload is safe and efficient to modify.

### Paragraph 3 — Scalability problem
A modern AI cluster continuously receives new:
- models;
- hardware combinations;
- sequence lengths;
- batch sizes;
- parallelism settings;
- training configurations;
- serving loads.

Therefore exhaustive workload-specific profiling is difficult to scale.

### Paragraph 4 — Core idea
\[
f_	heta(S,a)
\rightarrow
(\Delta P,\Delta QoS,\Delta E,\Delta T)
\]

### Paragraph 5 — Controller
\[
D(t)=P(t)-C(t)
\]
and select actions minimizing service cost while covering the deficit.

### Paragraph 6 — Contributions
List C1–C5.

---

## 4.3 Background and Related Work
Target: 1.5–2 pages.

### 4.3.1 Grid-responsive data centers
Cover hyperscaler demand response, Phoenix/Emerald, flexible loads, and interconnection constraints.

Conclusion:
> The feasibility of compute-driven grid response is established.

### 4.3.2 GPU and cluster power control
Cover GPU power capping, DVFS, cooperative caps, NVIDIA power reservation, training stabilization, and oversubscription.

Conclusion:
> Existing systems provide actuators and control methods but typically assume known performance/power behavior or restricted action spaces.

### 4.3.3 AI workload energy/performance prediction
Cover Watt Counts, MLPerf Power, Zeus, BEAM, and inference energy modeling.

Conclusion:
> Static power or energy prediction differs from predicting response to an intervention.

### 4.3.4 Digital twins / simulation
Cover OpenDT/OpenDC, HPC digital twins, SchedTwin, SimGrid/Batsim, LLMServingSim, Opal/APEX+.

Conclusion:
> Simulators enable counterfactual branching but require calibration and independent physical validation.

### 4.3.5 Gap statement
> **Gap:** We study generalization of action-conditioned power–service response to workloads whose intervention profiles are not observed during training, and quantify whether that prediction creates decision value under facility-level power constraints.

---

## 4.4 Problem Formulation
Target: 1 page.

### Workload state
\[
x_i =
[
	ext{job type},
	ext{model features},
	ext{hardware},
	ext{batch},
	ext{tokens},
	ext{priority},
	ext{deadline},
\dots
]
\]

\[
s_i(t)=
[
P_i,
U_i,
Q_i,
	ext{phase},
	ext{remaining work},
	ext{queue state}
]
\]

### Action space
\[
a_i\in
\{
	ext{do nothing},
	ext{power cap},
	ext{delay},
	ext{pause},
	ext{batch adjustment},
	ext{queue adjustment}
\}
\]

### Response model
\[
f_	heta(x_i,s_i,a_i)
=
(\hat{\Delta P_i},
\hat{\Delta E_i},
\hat{\Delta Q_i},
\hat{\Delta T_i})
\]

### Service cost
\[
C_i(a_i)=
w_i
[
lpha L_i
+
eta S_i
+
\gamma D_i
]
\]

### Power constraint
\[
P_{	ext{facility}}(t)\le C(t)
\]

Optionally:
\[
|\dot P(t)|\le R_{\max}
\]

### Safe flexibility
\[
\Delta P_i^{safe}
=
LCB_{1-lpha}(\Delta P_i)
\]

---

## 4.5 Data and Experimental Platform
Target: 1.5–2 pages.

### Physical measurements
Describe:
- GPUs;
- server;
- power monitoring;
- workload set;
- intervention sweeps;
- repetitions;
- sample rate.

### Phoenix/Emerald
Use for:
- public measured response;
- external validation;
- initial kill test.

### Production workload traces
Azure:
- inference arrival rate;
- input/output tokens;
- burstiness.

Alibaba:
- mixed workload population;
- priority;
- GPU requests;
- training/inference mix.

### Synthetic counterfactual data
Use:
- OpenDT/OpenDC;
- LLMServingSim;
- Batsim/SimGrid;
- optionally Opal/APEX+.

State explicitly:
> Synthetic data expand action coverage but are not treated as independent physical ground truth.

---

# 5. Results Plan

## 5.1 RQ1 — Characterizing Workload Power Flexibility

### E1 — Power-cap response
For each workload:
\[
u\in\{100\%,90\%,80\%,70\%,60\%\}
\]

Measure:
- power;
- throughput;
- latency;
- runtime;
- energy.

### E2 — Alternative actions
For selected jobs:
- delay;
- batching;
- pause/resume;
- queue slack.

### Metric
\[
FE =
rac{-\Delta P/P}
{\Delta C_{	ext{service}}+\epsilon}
\]

### Figure 2
Power reduction vs service degradation by workload.

### Figure 3
Flexibility heatmap: workloads × actions.

### Table 2
| Workload | Action | ΔPower | ΔEnergy | ΔThroughput | ΔLatency | Flexibility efficiency |
|---|---|---:|---:|---:|---:|---:|

### Required conclusion
Only if supported:
> Workload flexibility is heterogeneous enough that intervention choice can matter.

---

## 5.2 RQ2 — Decision Headroom

### E3
Impose:
\[
P_{\max}=
\{95,90,85,80,75,70\}\%
\]
of unconstrained power.

Compare:
- no control;
- uniform cap;
- priority-first;
- largest-power-first;
- static workload-class heuristic;
- profiled response-aware controller;
- oracle.

### Main metric
Weighted service cost at a required power reduction.

### Figure 4 — Kill Figure
X-axis: required power reduction.  
Y-axis: service cost.

### Decision headroom
\[
H =
rac{
C_{	ext{simple}}-C_{	ext{oracle}}
}{
C_{	ext{simple}}
}
\]

### Kill conditions
- \(H < 5\%-10\%\) across meaningful operating points;
- priority-only captures >90–95% of oracle value.

These are project-management thresholds.

---

## 5.3 RQ3 — Predicting Flexibility of Unseen Workloads

### Models
- M0 global mean;
- M1 workload-class lookup;
- M2 linear model;
- M3 gradient-boosted trees;
- M4 neural model/embedding if justified;
- M5 uncertainty-aware model.

### Inputs
Static:
- workload class;
- model size;
- batch;
- sequence length;
- precision;
- hardware;
- GPU count.

Dynamic:
- utilization;
- current power;
- queue length;
- throughput;
- execution phase;
- remaining work;
- deadline slack.

Action:
- type;
- magnitude.

### Targets
At minimum:
\[
\Delta P,\quad \Delta C_{	ext{service}}
\]

Optional:
\[
\Delta E,\Delta T,	ext{rebound}
\]

### Splits
- S1 random interpolation;
- S2 leave-model-out;
- S3 leave-workload-family-out;
- S4 leave-hardware-out;
- S5 leave-action-level-out.

### Metrics
- MAE;
- normalized MAE;
- RMSE;
- calibration;
- ranking accuracy;
- top-k action recall;
- control regret.

### Figure 5
Measured vs predicted \(\Delta P\) for seen and unseen cases.

### Figure 6
Predicted action ranking vs true ranking.

### Table 3
| Model | Seen MAE | Unseen-model MAE | Unseen-hardware MAE | Service-cost MAE | Top-1 action accuracy |
|---|---:|---:|---:|---:|---:|

---

## 5.4 RQ4 — Learned Power-Shaping Control

### Candidate score
\[
V_{i,a}
=
rac{
\Delta P^{safe}_{i,a}
}{
C_{i,a}+\epsilon
}
\]

### Greedy controller
1. calculate deficit:
\[
D=P_{	ext{current}}-C(t)
\]
2. generate candidate actions;
3. rank by value;
4. apply until safe predicted reduction covers deficit;
5. observe actual power;
6. correct at next tick.

### Optional MPC
Use only if temporal effects justify it.

### Compare
- uniform;
- priority;
- static;
- learned;
- profiled;
- oracle.

### Figure 7
Service cost vs requested curtailment.

### Oracle Capture
\[
OC =
rac{
C_{	ext{simple}}-C_{	ext{learned}}
}{
C_{	ext{simple}}-C_{	ext{oracle}}
}
\]

### Table 4
| Controller | Envelope violation | Service cost | HP SLA violations | Energy change | Oracle capture |
|---|---:|---:|---:|---:|---:|

---

## 5.5 RQ5 — Production Trace Replay

### Azure inference track
Replay:
- timestamps;
- input tokens;
- output tokens.

Assign service classes with explicit synthetic assumptions.

Measure:
- TTFT;
- p95/p99 latency;
- throughput;
- envelope violations;
- delayed requests.

### Alibaba mixed-workload track
Replay:
- training;
- online inference;
- offline inference;
- development;
- priority;
- GPU requirements.

### Figure 8
Dynamic trace with arrivals, envelope, controlled power, and service state.

### Figure 9
Stacked power by workload class, showing which classes provide curtailment.

---

## 5.6 RQ6 — Uncertainty and Safe Flexibility

### Method
\[
\Delta P^{safe}
=
Q_lpha(\Delta P)
\]

Compare:
- mean-prediction controller;
- safe controller.

### Metrics
\[
UFR =
rac{
\#(\Delta P_{	ext{actual}} < \Delta P_{	ext{promised}})
}{
N
}
\]

Also quantify over-conservatism.

### Figure 10
Confidence level vs:
- delivered compliance;
- usable promised MW.

---

## 5.7 RQ7 — Rebound-Aware Planning

### Computational debt
\[
D_i(t)
=
W_i^{expected}(t)-W_i^{completed}(t)
\]

Compare:
- controller ignoring debt;
- rebound-aware controller.

Metrics:
- maximum rebound power;
- energy debt;
- recovery duration;
- deadline misses;
- secondary peak.

### Figure 11
Power time series before, during, and after event.

---

## 5.8 RQ8 — Israeli Grid Application Case Study

### Input
Use public Israeli:
- national demand;
- PV generation;
- representative high-demand days.

### Virtual facility sizes
\[
10,\ 25,\ 50,\ 100	ext{ MW}
\]

### Workload mixes
- inference-heavy;
- balanced;
- training-heavy.

### Scenarios
- IL1 fixed connection cap;
- IL2 evening constrained period;
- IL3 PV-aware shifting.

### Metrics
Firm Capacity Ratio:
\[
FCR=
rac{
P_{	ext{firm}}
}{
P_{	ext{installed}}
}
\]

Compute Capacity Multiplier:
\[
CCM=
rac{
P_{	ext{installed}}
}{
P_{	ext{firm}}
}
\]

### Figure 12
Israeli demand + PV + virtual AI facility load.

### Figure 13
Firm grid capacity vs useful AI work completed within SLO.

### Claim limit
Do not say:
> Israel can connect X% more data centers.

Say:
> Under the modeled workload and service assumptions, the controller changes the firm-capacity requirement by X relative to the unconstrained baseline.

---

# 6. Experimental Platform

## Physical testbed
Minimum:
- 1–4 NVIDIA GPUs;
- Linux;
- NVML;
- optional PDU/wall meter.

Preferred:
- heterogeneous GPU types;
- server-level AC measurement.

## Workloads
Inference:
- small LLM;
- medium LLM;
- larger model if feasible.

Vary:
- input length;
- output length;
- batch;
- concurrency.

Training:
- transformer fine-tuning;
- vision training;
- embedding/encoder workload.

## Initial actions
1. GPU power cap;
2. bounded delay;
3. pause/resume;
4. batching / queue slack.

Keep action space modest in the first paper.

---

# 7. Synthetic Data Role

Synthetic data should support, not replace, physical evidence.

Training:
\[
D_{	ext{sim}}
+
D_{	ext{public measured}}
\]

Calibration:
\[
D_{	ext{own measured,cal}}
\]

Final test:
\[
D_{	ext{own measured,test}}
\]

### Sim-to-real experiment
Train with:
- 0% real data;
- 1%;
- 5%;
- 20%;
- real-only.

Test on unseen physical workloads.

### Figure 14
Real calibration fraction vs physical-test prediction/control error.

---

# 8. Baselines

## Prediction
- global mean;
- workload class;
- current power only;
- linear regression;
- GBDT;
- profiled lookup.

## Control
- no shaping;
- uniform power capping;
- priority-first;
- largest-power-first;
- static class rule;
- profiled controller;
- oracle.

## Temporal
- greedy;
- rebound-aware;
- optional MPC.

---

# 9. Ablations

## A1 — Remove workload semantic features
Keep only current power/utilization.

## A2 — Remove runtime state
Use static attributes only.

## A3 — Remove priority
Test whether gains merely sacrifice low-priority jobs.

## A4 — Remove action identity
Test whether explicit intervention type matters.

## A5 — Remove uncertainty
Use mean response prediction.

## A6 — Remove rebound
Test whether control only moves the peak.

## A7 — Remove synthetic training data
Train only on physical data.

## A8 — Remove physical calibration
Simulation-only model.

---

# 10. Statistical Analysis

For key metrics report:
- mean;
- median;
- 95% confidence interval;
- distribution across workload families;
- per-hardware breakdown.

Use paired comparisons where identical traces are replayed under multiple policies.

For stochastic serving:
- fixed request traces;
- common seeds;
- identical arrival sequences.

For physical runs:
- randomized intervention order;
- at least 3 repetitions;
- controlled warm-up;
- thermal stabilization.

---

# 11. Kill Criteria

## K1 — No elasticity heterogeneity
Pivot away from workload-aware prediction.

## K2 — No decision headroom
If oracle/profiled controller barely beats uniform/priority, do not pursue complex ML.

## K3 — Poor unseen-workload generalization
Pivot to:
> few-shot calibration or active profiling.

## K4 — Good prediction but no control gain
Move from regression to decision-aware ranking.

## K5 — Priority dominates
Pivot toward open policy/control architecture.

## K6 — Rebound removes the benefit
Shift toward longer-horizon scheduling / flexible interconnection.

---

# 12. Possible Pivots

## P1 — Few-shot flexibility profiling
Question:
> Can 1–5 short intervention probes characterize a new workload?

## P2 — Active profiling
Use model uncertainty to select the next physical measurement.

## P3 — Ranking instead of exact regression
Predict:
> which job/action is cheapest to shape?

## P4 — Open control standard
If predictive novelty is weak, focus on:
- PowerEnvelope;
- FlexibilityOffer;
- DeliveryRecord;
- scheduler adapters;
- measurement and verification.

---

# 13. Expected Tables

## Table 1 — Related systems
Columns:
- system;
- grid-responsive;
- workload-aware;
- requires profiling;
- action space;
- open source;
- real deployment;
- predicts unseen workload response.

## Table 2 — Workload measurement set
Columns:
- workload;
- model;
- task;
- hardware;
- batch;
- sequence;
- baseline power;
- baseline throughput.

## Table 3 — Intervention response
Columns:
- workload;
- action;
- power reduction;
- energy change;
- service impact;
- response time.

## Table 4 — Prediction
Columns:
- model;
- seen MAE;
- unseen-workload MAE;
- unseen-hardware MAE;
- ranking accuracy;
- calibration.

## Table 5 — Control
Columns:
- policy;
- envelope violations;
- service cost;
- SLA violations;
- energy;
- rebound;
- oracle capture.

## Table 6 — Production traces
Columns:
- trace;
- workload mix;
- duration;
- jobs/requests;
- peak power;
- flexibility;
- service cost.

## Table 7 — Israeli scenarios
Columns:
- day/scenario;
- virtual facility size;
- firm cap;
- delivered flexibility;
- workload completion;
- rebound;
- CCM.

---

# 14. Figure Plan

1. **Architecture** — power envelope → predictor/world model → planner → scheduler/runtime → workloads → telemetry.
2. **Elasticity curves** — heterogeneous workload response.
3. **Flexibility heatmap** — workloads × actions.
4. **Oracle headroom / kill figure** — simple vs profiled vs oracle.
5. **Prediction on unseen workloads** — measured vs predicted \(\Delta P\).
6. **Action ranking** — predicted vs optimal.
7. **Control performance** — service cost vs required curtailment.
8. **Production trace replay** — dynamic envelope tracking.
9. **Workload composition during shaping** — which jobs are modified.
10. **Safe flexibility** — promised vs delivered MW.
11. **Rebound** — naive vs rebound-aware.
12. **Israeli system day** — national demand + PV + virtual AI facility.
13. **Firm capacity vs useful work** — main application result.
14. **Sim-to-real** — optional calibration curve.

---

# 15. Main Metrics

## Electrical
Envelope violation energy:
\[
V_E=
\int \max(0,P(t)-C(t))dt
\]

Maximum violation:
\[
V_P=
\max_t(P(t)-C(t))_+
\]

Peak reduction:
\[
PR=
1-rac{P_{\max}^{control}}{P_{\max}^{baseline}}
\]

Ramp:
\[
R_{\max}=
\max_t |\Delta P/\Delta t|
\]

Energy change:
\[
\Delta E=
rac{E_{control}-E_{baseline}}{E_{baseline}}
\]

## Service
Inference:
- throughput;
- TTFT;
- TPOT;
- p95/p99 latency;
- SLO violation.

Training:
- completion time;
- deadline miss;
- slowdown;
- pause overhead.

## Decision
Oracle Capture:
\[
OC=
rac{C_{simple}-C_{learned}}
{C_{simple}-C_{oracle}}
\]

Decision regret:
\[
Regret=
C(a_{	ext{predicted}})
-
C(a_{	ext{optimal}})
\]

## Reliability
Under-delivery frequency:
\[
UFR=
\Pr(
\Delta P_{	ext{actual}}<
\Delta P_{	ext{promised}}
)
\]

---

# 16. Strongest Possible Headline Result

The ideal result is not:
> Prediction RMSE = X%.

The strongest result is:
> **On workloads completely withheld from power profiling, the learned controller recovers most of the decision benefit of workload-specific profiling while substantially reducing service degradation relative to uniform and priority-only control.**

The main plot:
\[
x=	ext{required power reduction}
\]
\[
y=	ext{service cost}
\]

Curves:
- Oracle;
- Profiled;
- Learned-unprofiled;
- Priority;
- Uniform.

The paper is compelling if:
- learned-unprofiled is close to profiled;
- profiled materially beats priority/uniform;
- the result survives realistic traces.

---

# 17. Discussion

Discuss:
1. what features the model actually relies on;
2. where generalization fails;
3. deployment integration:
   \[
   	ext{grid/DCIM}
   ightarrow
   	ext{PowerEnvelope}
   ightarrow
   	ext{controller}
   ightarrow
   	ext{Slurm/Kueue}
   ightarrow
   	ext{GPU/runtime}
   \]
4. policy relevance:
   - transparent flexibility measurement;
   - firm vs conditional capacity;
   - vendor-neutral external interface.

Do not overclaim regulation.

---

# 18. Limitations

Must include:
1. small physical testbed vs hyperscale extrapolation;
2. simulator bias;
3. synthetic SLO assumptions in production traces;
4. incomplete cooling/facility modeling;
5. GPU-centric actuation;
6. limited hardware diversity;
7. mismatch between academic and proprietary workloads;
8. no direct validation on an Israeli production data center;
9. market/tariff rules not modeled;
10. service-cost weights are operator-dependent.

---

# 19. Reproducibility Package

Recommended structure:

```text
data/
  public-source manifests
  processed traces
  measured intervention summaries

simulator/
  OpenDT_OpenDC adapters
  LLMServingSim adapter
  workload replay

models/
  baselines
  response predictor
  uncertainty calibration

controllers/
  uniform
  priority
  learned
  profiled
  oracle

experiments/
  rq1_heterogeneity
  rq2_headroom
  rq3_prediction
  rq4_control
  rq5_traces
  rq6_uncertainty
  rq7_rebound
  rq8_israel

figures/
tables/
```

---

# 20. Target Venues

## Systems-oriented
- ACM e-Energy
- IEEE/ACM CCGrid
- IEEE Cluster
- HPDC
- MLSys if the ML-systems contribution is strong
- SC workshops / main SC if scale and systems evidence are strong

## Energy/computing intersection
- ACM e-Energy
- Sustainable Computing: Informatics and Systems
- Future Generation Computer Systems
- IEEE Transactions on Sustainable Computing

## Power/grid-oriented
Only if the grid model becomes substantially deeper:
- Applied Energy
- IEEE Transactions on Smart Grid
- Energy and AI

The first manuscript should likely remain a **computer-systems + energy** paper.

---

# 21. Recommended Paper Identity

The strongest identity is:

> **An empirical and systems paper about generalizing power-flexibility knowledge across AI workloads.**

Not:
- a digital-twin paper;
- an Israeli energy-policy paper;
- a pure regression paper;
- a scheduler paper alone.

The digital twin, synthetic data, predictor, controller, and Israeli case study should all serve the central question:

\[
oxed{
	extbf{Can we avoid exhaustive workload profiling while still making high-quality power-shaping decisions?}
}
\]

---

# 22. One-Sentence Paper Arc

> **We first show that AI workloads differ substantially in the electrical flexibility they provide, then establish that this heterogeneity creates real scheduling headroom, learn to predict the intervention response of workloads withheld from profiling, use those predictions to control a time-varying data-center power envelope, and finally evaluate the resulting flexible-load capability under realistic production traces and Israeli grid conditions.**

---

# 23. Minimal Result Set Required for a Strong Paper

The paper should ideally have all five:

1. **Heterogeneity** — response curves differ materially.
2. **Decision headroom** — oracle/profiled control beats simple policies.
3. **Generalization** — unseen workload response can be predicted.
4. **Control value** — learned controller captures substantial oracle/profiled benefit.
5. **Realistic application** — result survives production workload replay.

Uncertainty, rebound, sim-to-real, and Israeli grid evaluation strengthen the paper but cannot compensate if items 1–4 fail.

---

# 24. Final Go/No-Go Logic

```text
Do workloads respond differently?
        |
       NO --> stop ML framing
        |
       YES
        |
Does response knowledge improve control?
        |
       NO --> use simple policy / pivot
        |
       YES
        |
Can unseen workload response be predicted?
        |
       NO --> pivot to few-shot profiling
        |
       YES
        |
Does learned controller recover decision value?
        |
       NO --> prediction is not operationally useful
        |
       YES
        |
Does it survive realistic traces?
        |
       NO --> narrow scope / revise assumptions
        |
       YES
        |
Add uncertainty + rebound + Israeli case study
        |
        v
Strong paper
```

This decision tree should guide the research itself, not merely the manuscript.
