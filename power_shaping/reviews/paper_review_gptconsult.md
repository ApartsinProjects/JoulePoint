## Overall assessment

**Major revision.** The manuscript has a coherent systems problem, but the current paper makes the integration look substantially more complete than the evidence supports. The central issue is that three different results are being presented as one system:

1. **Priority scheduling under a fixed power shortage**, which produces the headline 44%→0% SLO result.
2. **A hardware power-cap primitive**, which can enforce a GPU-level ceiling under stated assumptions.
3. **Per-workload elasticity allocation**, evaluated separately on a constructed 26-workload simultaneous-allocation setting, not integrated into the temporal Azure controller.

The paper itself eventually admits this separation: the temporal controller uses priority under one cluster cap, and heterogeneous per-workload elasticity has not yet been integrated into that controller.  The abstract and contributions need to reflect this much more clearly.

The most serious scientific issue is the word **“guaranteed.”** Section 4 mathematically assumes `d ≤ L`, but the hardware experiment later reports 1.1% of samples above target on A10G and 5.8% on L4.  This does not necessarily kill the underlying idea, but it means the paper must distinguish a **commanded device-cap property** from **measured instantaneous power compliance**, and both from a **facility-level grid commitment**.

---

# (A) Bad writing

### A1. Abstract: overloaded central sentence

**Section: Abstract**

> “We present a shaping controller that sheds only what the allowance requires, protects high-priority service-level objectives (SLOs) by serving critical work first and deferring the rest, and bounds the post-event rebound ...; and a hardware-grounded mechanism that lets the facility commit a firm power ceiling with a provable lower bound on the flexibility it delivers.”

This tries to introduce the controller, SLO mechanism, rebound, and guarantee in one sentence. The semicolon does not rescue the structure. It also prematurely conflates the workload controller and the hardware guarantee. 

**Rewrite:**

> “We develop two complementary mechanisms. A priority-aware controller allocates a time-varying power budget across service classes and limits the post-event recovery ramp. Separately, a hardware-enforced GPU power cap provides a conservative device-power ceiling that can backstop a grid commitment.”

That wording also avoids claiming facility-level enforcement before it is proved.

---

### A2. Introduction: three arguments compressed into a comma splice

**Section 1**

> “Cluster traces document training and inference co-located at scale with many model types and fractional GPU sharing ..., inference is now the majority of ML energy in at least one hyperscaler fleet ..., and different workloads have sharply different power profiles...”

This is structurally poor. There are three independent arguments: workload diversity exists, inference matters energetically, and workloads differ in elasticity. 

**Rewrite:**

> “Production AI clusters host heterogeneous workloads. Existing traces show training and inference co-located at scale, often with multiple model types and fractional GPU sharing. Their power responses also differ substantially: prompt processing can be compute-bound, whereas decode and recommendation workloads may be limited by memory behavior.”

Then explain why that matters to your controller.

---

### A3. Introduction: awkward objective language

**Section 1**

> “...so that the delivered service loss is as small and as low-priority as possible.”

“Small and low-priority” does not define an optimization objective. 

**Rewrite:**

> “...to minimize priority-weighted service loss while satisfying the power ceiling.”

That directly connects to the objective introduced later.

---

### A4. Problem formulation: definition overload

**Section 2**

> “A serving cluster hosts a mix of workloads partitioned into priority classes ... with per-class service weights and queueing-delay deadlines: a request that waits longer than its class deadline counts as a service-level-objective (SLO) violation, and the weighted service cost sums those violations ...”

This is too much formal content in prose. Worse, the paper calls this a “problem formulation” but gives no actual mathematical formulation. 

**Rewrite as definitions plus equations**, for example:

> “Each class \(k\) has an SLO deadline \(\tau_k\) and penalty weight \(w_k\). Request \(i\) violates its SLO when its queueing delay exceeds \(\tau_{k(i)}\). We minimize the weighted number of violations plus a penalty for unfinished deferred work.”

Then give the actual objective mathematically.

---

### A5. Controller description: “plans ahead” is unsupported by what follows

**Section 3**

> “The controller plans ahead and corrects with feedback. Because the allowance is known in advance, at each tick it uses a response model to predict the power that each candidate action would draw...”

Nothing following “plans ahead” specifies a horizon, look-ahead optimization, or future-state model. The described controller sounds essentially myopic with one-tick feedback. 

**Rewrite unless there really is MPC/look-ahead:**

> “At each 0.5-s control tick, the controller selects the highest-service action predicted to satisfy the current power allowance and corrects model error using measured power from the preceding tick.”

If there is future knowledge, state the horizon explicitly.

---

### A6. Controller description: wrong optimization language

**Section 3**

> “...picks the action that sheds the least while still meeting \(C(t)\).”

The point is not to minimize shed; the ceiling determines how much power must be removed. The controller should minimize **service cost subject to the ceiling**. 

**Rewrite:**

> “Among actions predicted to satisfy \(C(t)\), the controller selects the one with the lowest priority-weighted service cost.”

---

### A7. Guarantee section: colloquial and promotional

**Section 4**

> “To be useful as a grid resource, a facility must promise a shed it will actually deliver; over-promising is worse than promising less.”

“Promise a shed” is awkward, and the next paragraph says the result is “exactly what a grid operator needs,” which reads like sales language rather than analysis. 

**Rewrite:**

> “A grid commitment requires a conservative bound on the power reduction that the resource can reliably provide. We therefore distinguish an enforceable power ceiling from a statistical forecast of available flexibility.”

---

### A8. Evaluation opening is misleading

**Section 5.1**

> “The controller runs on real measured data.”

This strongly suggests a measured controller experiment. The following sentences explain that the service dynamics are generated by a tick simulator using measured inputs. 

**Rewrite:**

> “Our main evaluation is trace-driven simulation using measured GPU response curves and a real request-arrival trace. We complement it with a field-experiment comparison and single-GPU closed-loop hardware measurements.”

This should be the first sentence of §5.

---

### A9. “Serving capacity is fixed” is much too broad

**Section 5.3**

> “Under a hard cap the serving capacity is fixed, so what matters is how each controller uses it.”

That statement conflicts with the paper's later argument that workload elasticity affects capacity under the same power constraint. It is only defensible for the particular homogeneous replay. 

**Rewrite:**

> “For this single-application replay, all requests share the same power-response curve; therefore, at a given cluster cap, the controllers have effectively the same aggregate serving capacity and differ primarily in which classes receive it.”

---

### A10. Hardware paragraph is too dense and “preserved” is undefined

**Section 5.4**

> “On the A10G ... the cap holds power under the target essentially strictly: across all constrained segments the fraction of time above target is 1.1%, cap actuation settles in a median 0.02 s, and critical p95 latency is preserved (median 0.12 ms).”

Three unrelated metrics are jammed together. More importantly, “preserved” requires a baseline or an SLO threshold. An absolute 0.12 ms does not establish preservation. 

**Rewrite:**

> “On A10G, 98.9% of constrained samples satisfy the target, and the median cap-settling time is 20 ms. The latency-critical stream has a median p95 latency of 0.12 ms; compared with the unconstrained baseline of X ms, this corresponds to Y% change.”

If X/Y are unavailable, remove “preserved.”

---

### A11. Rebound language is unnecessarily confusing

**Section 5.5**

> “...cuts the rebound to -13%...”

A negative rebound is not intuitive. 

**Rewrite:**

> “An uncontrolled backlog drain raises recovery power to 35% above the pre-event baseline. Ramp-limited recovery instead keeps the recovery peak 13% below that baseline.”

Also specify whether 35% is peak, mean, or sustained power.

---

### A12. The “probeable, not predictable” paragraph is overpacked

**Section 5.6**

> “Scoring predictors in decision space (the fraction of the elasticity gap they recover, priority-only to oracle, with the scoring validated by feeding the true curve to recover 100% and a homogeneous fleet to leave 0%), we find the curves are probeable, not predictable.”

This requires the reader to decode the metric, its endpoints, two baselines, and the conclusion before seeing the evidence. 

**Rewrite:**

> “We evaluate curve estimates by their downstream allocation value rather than curve-fitting error. The metric is the fraction of the service-cost gap between priority-only allocation and a full-curve oracle that the estimate recovers. On our 26-workload corpus, one reduced-power probe recovers 88% of this gap, compared with 60% for our best feature-only baseline.”

Then conclude conservatively.

---

### A13. Section 5.7 reads as a proposal for the next paper

**Section 5.7**

> “Orchestrating this many actuators efficiently is where learning enters, and the enforcement layer makes learning safe.”

and later:

> “...the decisive next experiment is to show a learned multi-actuator controller beats uniform, priority-only, and every single actuator...”

The paragraph explicitly describes an unevaluated future controller, yet speaks as if its safety and adoption properties have already been established. 

**Rewrite:**

> “The hardware cap suggests a two-loop architecture in which an outer optimizer selects software actuators while an inner loop enforces a device-power envelope. We do not evaluate this multi-actuator controller here.”

Move most of §5.7 to **Discussion/Future Work**.

---

### A14. Conclusion ends informally

**Section 8**

> “...AI data centers behaving as well-mannered grid resources...”

This is journalistic, not scientific. 

**Rewrite:**

> “...AI data centers operating as predictable, grid-responsive loads under externally imposed power limits.”

---

# (B) Inconsistency

### B1. The same 256-GPU experiment is described as 31% and 25%

§5.2:

> “During a real grid experiment the emerald 256-GPU cluster cut power 31%...”



§6:

> “Colangelo et al. [1] field-tested a 256-GPU cluster shedding 25% for three hours.”



If these are different operating points within the same experiment, say exactly that. Otherwise this looks like a factual inconsistency.

---

### B2. “Strict” means two incompatible things

§5.3 explicitly defines:

> “strict means 100%.”



Figure 5 says:

> “The A10G ... holds every target strictly (<2%).”



And the body says 1.1% of samples are **above** target. 

Do not use “strict” for 98–99% compliance. Reserve it for exactly the definition in §5.3.

---

### B3. The mathematical guarantee and hardware measurements do not use the same semantics

§4 assumes:

> “the hardware guarantees \(d \le L\)”

and therefore derives \(R=1\). 

But the same section acknowledges:

> “One measured cell shows a draw a fraction of a watt above its cap...”



and §5.4 reports target exceedances.

This must be resolved formally. Is \(L\):

- a driver-enforced time-averaged power limit,
- an instantaneous measured-power upper bound,
- or a contractual envelope after adding a tolerance?

Those are different guarantees.

---

### B4. GPU-level guarantee is repeatedly stated as a facility-level guarantee

The proof is for one workload/device with observed draw \(D\), cap \(L\), and realized draw \(d\).  Yet the paper repeatedly says “the facility” can make a firm commitment.

The missing bridge is:

\[
P_{\text{facility}}
=
P_{\text{GPU}}
+
P_{\text{CPU/memory/network}}
+
P_{\text{cooling/other}}.
\]

The grid contracts against the facility meter, not an NVML device counter. Until aggregation and non-GPU load are modeled, the paper has a **GPU-power guarantee**, not yet a full facility-power guarantee.

This is the biggest conceptual gap in the title.

---

### B5. Abstract says the allocation is driven by priority *and elasticity*; the main temporal controller does not demonstrate that combination

The contributions say per-workload elasticity is a second lever that closes 47% of the oracle gap.  But the limitations explicitly admit that per-workload elasticity has **not yet been integrated into the temporal controller on a heterogeneous request mix**. 

The abstract should not make the heterogeneous elasticity experiment sound like part of the 271k-request temporal result.

---

### B6. “31% elasticity value” and “47% of the gap” appear to describe the same benefit but are different metrics

Table 3 reports 31% “elasticity value” on the A10G pool.  Elsewhere the paper says elasticity closes 47% of the uniform-to-oracle weighted-service-cost gap.

These can both be correct, but the reader should not have to infer that:

- **31%** = equal-weight oracle gain over uniform capping;
- **47%** = elasticity's share of a different priority-weighted total gap.

Name these metrics differently throughout.

---

### B7. “Scales with fleet diversity” contradicts the manuscript's own caveat

Table 3:

> “The value scales with fleet diversity...”



But the text later says:

> “...we do not claim a universal law...”



The latter is the scientifically defensible wording. Replace “scales with” by something like:

> “increases under a controlled within-pool interpolation from shared to heterogeneous response curves.”

---

### B8. The predictor baseline changes names

The same family of baseline is called:

- “feature-only statistical predictor,”
- “leave-one-workload-out type-family mean,”
- “zero-probe feature-only class-mean.”

Standardize the terminology and specify exactly what features and groups it uses. 

---

### B9. “Worst case degrades exactly to today's uniform cap” is not generally true

The paper says:

> “a bad prediction costs efficiency, and the worst case degrades exactly to today's uniform cap.”



The hard cap may preserve **power compliance**, but a bad outer controller could defer the wrong requests, select the wrong model cascade, sacrifice quality unnecessarily, or create excessive backlog. Its **service outcome** can be worse than uniform capping.

Rewrite as:

> “Errors in the outer optimizer cannot raise the commanded hardware ceiling; they can, however, worsen service efficiency or quality.”

---

# (C) Dense jargon / undefined terms

Several terms are either introduced too late or never made operationally precise.

| Term | Problem | Concrete fix |
|---|---|---|
| **“grid-facing allocation problem”** | Sounds attractive but does not identify the contractual interface. | Define the input as a power ceiling at a specific measurement point, with notice time, sampling interval, duration, ramp and tolerance. |
| **\(C(t)\), “facility power allowance”** | Is this GPU sum, IT power, UPS output, or utility meter power? | State the physical measurement boundary explicitly in §2. |
| **“weighted service cost”** | Central metric, but no formula, units, weights, or actual values. | Give the equation and a table of class weights/deadlines.  |
| **“elasticity curve”** | Defined qualitatively, but axes and normalization remain loose. | Define \(P_w(L)\), throughput \(T_w(L)\), cap range, normalization and interpolation. |
| **“emerald”** | Used as though readers know what it denotes. | First use: “the Emerald 256-GPU cluster reported in [1]” or whatever the exact provenance is. |
| **“normalized service performance”** | The 0.057 headline MAE is meaningless without the scale. | Define the normalization and interpretation before reporting the number. |
| **“usable flexibility”** | Table 1 gives statistical predictors values of 123% and 113%; something called “usable” should not exceed physically achievable flexibility. | Rename to “promised / achievable flexibility (%)” or explicitly label >100% as overcommitment.  |
| **“fleet diversity” / “true measured diversity”** | No diversity statistic is defined. A 26-workload pool is also not what many systems readers will call a hardware “fleet.” | Use “workload-response diversity” and define the interpolation coefficient or actual diversity metric. |
| **“elasticity gap”** | Appears in the probe experiment without a compact mathematical definition. | Give one equation for the oracle-normalized decision metric. |
| **“type-family mean”, “class-mean”, “learned curve prior”** | Baselines are impossible to reproduce from these labels. | State features, grouping, estimator, fitting data and LOO protocol. |
| **“knee fit”** | Eventually parenthetically explained, but still underspecified. | Define its fitting rule and probe locations. |
| **“clipping frequency”** | Introduced as a “free signal” but not evaluated. | Either define and validate it or move it into future work. |
| **“marginal-cost water-filling”** | Introduces optimization jargon for a controller you do not actually implement. | Remove it from the current paper unless you formalize and evaluate that algorithm. |
| **“firm-capacity framing” / “flexible-work fraction”** | These appear in Limitations without having been established clearly beforehand. | Either introduce them formally earlier or delete the terminology. |
| **LLM, DLRM, CNN, QoS** | Some acronyms are used without expansion for a general systems/energy audience. | Expand once at first use. |

The current §2 packs most of the important system terminology into one paragraph rather than providing a clean model. 

---

# (D) Logical arc

The intended arc is sensible, but the implementation currently has two major gaps.

### 1. The foundational guarantee comes too late

Current ordering is:

**problem → workload controller → hardware guarantee → evaluation.**

But the argument in §5.7 is that the hardware cap is the safety substrate underneath every higher-level controller. If that is the architecture, the paper should establish the enforcement boundary **before** describing the allocator.

A stronger ordering would be:

**1. Problem and operational contract**  
**2. System model and objectives**  
**3. Enforceable power-envelope primitive and its exact guarantee**  
**4. Priority-aware allocation and recovery controller**  
**5. Evaluation**  
**6. When per-workload elasticity helps + probing**  
**7. Discussion: additional actuators**  
**8. Related work / limitations / conclusion**

### 2. The main controller and elasticity characterization never meet

The main Azure experiment proves that, for a homogeneous trace under a fixed cap, **priority order matters enormously**. The paper itself states that the protection comes from “which work the scarce capacity serves first, not from how the cap is set.” 

Then §5.6 switches to a constructed heterogeneous, simultaneous-allocation regime where elasticity matters. That is scientifically legitimate as a characterization study, but it is not validation of a single integrated controller.

This distinction must become explicit in the paper architecture.

### Abstract/body match

The **numbers** in the abstract mostly have corresponding body results. The **mechanistic story** does not match as well. The abstract sounds like one controller jointly exploits priority and workload elasticity. The limitations say this joint heterogeneous temporal controller has not been built. That is the main abstract mismatch.

### The four contributions are not optimally chosen

I would reduce them to **three**:

1. **Grid-constrained allocation system:** formulation + priority-aware temporal controller + rebound control.
2. **Hardware-backed ceiling primitive:** precise guarantee scope plus measured enforcement characterization.
3. **Elasticity characterization:** when heterogeneous workload response adds allocation value, and the empirical finding that cheap probes outperform the tested feature-only baselines.

“Reframing” alone is too weak for Contribution 1. Merge it with the controller.

The actuator portfolio should not be Contribution 4 in disguise. It is currently exploratory future work.

---

# (E) Practical value

The operational value is present, but it is not yet expressed in the language of an actual grid/data-center contract.

The paper repeatedly says the grid gives the facility \(C(t)\), but never makes the reader answer the most basic deployment question:

**What exactly can the data center promise, at what meter, for how long, with what response time, and what service consequence?**

### The single most valuable change

Add an early **“operational contract” figure/table immediately after the opening problem statement**:

**Grid input**

`Facility ceiling C_grid(t)`  
+ event start/end  
+ required ramp  
+ sampling/compliance interval

↓  

**Facility translation**

`GPU budget = C_grid(t) − uncontrollable/non-GPU load − safety margin`

↓  

**Enforcement**

hardware cap prevents the controlled GPU budget from exceeding its envelope

↓  

**Allocation**

priority scheduler decides who receives the remaining capacity

↓  

**Recovery**

backlog release respects a recovery envelope

↓  

**Outputs**

measured facility power + per-class SLO impact + remaining backlog

That one figure would clarify what the grid operator buys and what the data-center operator controls.

### What is currently missing operationally

The paper should explicitly specify:

- measurement point for the grid ceiling;
- treatment of CPU, memory, networking, storage and cooling power;
- baseline from which “delivered reduction” is measured;
- event notice/look-ahead assumption;
- response time and measurement window;
- permitted transient exceedance;
- GPU minimum-cap floors;
- available flexible-work fraction;
- recovery deadline as well as recovery peak;
- how requests acquire critical/interactive/batch/offline priority;
- cost and disruption of a 30-s workload probe;
- what happens when available controllable load is insufficient to meet \(C(t)\).

Without these, a grid operator cannot tell what is actually guaranteed, and a data-center operator cannot tell what must be instrumented.

---

# (F) Over-claiming / rigor

### F1. The title currently exceeds the proof

**“Guaranteed Grid Flexibility from AI Data Centers”** suggests a facility-level guaranteed demand-response resource.

The paper proves a much narrower primitive: conditional on the semantics of the GPU cap and a baseline \(D\), a commanded device ceiling implies a lower bound relative to that baseline. The limitations themselves acknowledge that absolute delivered reduction requires the facility's load to reach the committed baseline. 

Either extend the system proof to the facility meter or narrow the title/claim to **hardware-backed power-envelope enforcement**.

---

### F2. “100% by construction” needs a theorem with assumptions, not an empirical percentage

The paper simultaneously says:

- hardware guarantees \(d \le L\);
- R = 100% “by construction”;
- one measured cell is above the cap;
- A10G spends 1.1% of constrained samples above target.

The correct scientific presentation is:

**Proposition:** under assumptions A1–A4, commanded capped power satisfies X.

Then separately:

**Measurement:** with the actual telemetry sampling and hardware implementation, observed compliance is Y%.

Do not blend a mathematical implication and noisy hardware measurement into one “100% reliability” number.

---

### F3. The 53% statistical-predictor result is much too weak a foil for the guarantee

Table 1 makes the hardware method look dramatically superior to a feature predictor, but the comparison is not the central comparison needed to establish the hardware primitive. 

At minimum report:

- number of workloads;
- exact features;
- predictor;
- training protocol;
- uncertainty;
- distribution of overcommitment;
- stronger conservative statistical baselines.

Otherwise “53%” risks looking selected to make a deterministic cap look good.

---

### F4. “Probeable, not predictable” is not supported

Your own feature-only baseline recovers **60%** of the relevant gap. One probe reaches 88%. 

That does not establish “not predictable.” It establishes:

> **“On this 26-workload corpus, one in-situ probe substantially outperforms the feature-only predictors we tested.”**

That is strong enough and defensible.

---

### F5. “Scales with fleet diversity” is too strong

There are effectively:

- two measured workload pools in Table 3; and
- a controlled interpolation inside one pool.

That establishes a useful **within-pool sensitivity experiment**, not a general scaling law across real fleets. Your own caveat acknowledges this.

Use “increases with induced response diversity in this pool,” not “scales with fleet diversity.”

---

### F6. The 47% result needs uncertainty and construction sensitivity

The heterogeneous study explicitly assigns priority classes so they “cross-cut” elasticity. That is a sensible experimental construction, but the resulting 53%/47% partition depends on those assignments.

You need repeated assignments or sensitivity over:

- class mix;
- class weights;
- SLO deadlines;
- curtailment depth;
- elasticity/priority correlation.

A single constructed partition should not become a general headline percentage.

---

### F7. The 44%→0% result may mainly demonstrate priority queueing

The paper itself says that serving high-priority requests first eliminates the violations and that per-workload elasticity contributes zero on this trace. 

This is therefore not strong evidence of a novel power-allocation algorithm unless you compare against serious priority-aware schedulers.

The manuscript even acknowledges:

> “We do not yet compare against dedicated priority-scheduling systems beyond our own uniform and priority-aware controllers.”



For a top systems venue, that is a blocking baseline deficiency.

---

### F8. Priority labels, weights and SLOs are insufficiently documented

The main result depends completely on what counts as critical, interactive and batch work. Yet the manuscript does not give the reader a reproducible table showing class proportions, deadlines, weights and their provenance.

If those labels are synthetically imposed on an inference trace, say so prominently and provide sensitivity analysis. Otherwise the 44%→0% result cannot be interpreted.

---

### F9. Field validation is narrower than “controller validated”

The 0.057 experiment compares per-workload flexing against a previously measured cluster outcome. It does not validate the complete temporal backlog-aware controller, its priority mechanism, rebound controller, and heterogeneous elasticity mechanism simultaneously.

Contribution 2 should therefore not be called simply:

> “controller, validated.”

Say exactly what is validated.

---

### F10. Single-GPU actuator measurements should stay characterization results

The A10G/L4 experiments are valuable implementation evidence, but they do not establish facility-scale control. Likewise the two extra actuators in Figure 8 are measured on one A10G and are not jointly controlled. Figure 8 itself labels them as one-A10G measurements. 

Keep them as **actuator characterization**, not evidence for a demonstrated multi-actuator system.

---

### F11. Rebound control needs a cost tradeoff

Showing 35% → −13% power is insufficient. A slower recovery necessarily leaves backlog outstanding longer.

Report at least:

- peak rebound;
- recovery duration;
- backlog completion time;
- offline/batch completion penalty.

Otherwise one can trivially eliminate rebound by never draining the backlog.

---

# (G) Anything else needed for a strong submission

### 1. Turn §2 into a real systems formulation

You need equations for:

- facility/controlled power;
- action vector;
- queue evolution;
- SLO violation;
- weighted service cost;
- power constraint;
- recovery-ramp constraint.

Right now §2 is descriptive prose.

### 2. Add an algorithm or pseudocode

A reviewer must be able to determine whether this is:

- greedy scheduling,
- MPC,
- exhaustive action search,
- continuous optimization,
- rule-based feedback,
- or some combination.

“The controller picks the action” is not enough.

### 3. Separate three evidence types visually

Every result should be marked consistently as:

- **trace-driven simulation**,
- **replay against measured field data**,
- **new hardware measurement**.

The footer actually gives this provenance explicitly.  Move that information into the evaluation methodology and figure captions. Remove the “Draft generated from...” footer from the submitted paper.

### 4. Fix Table 1

A column called **“Usable flexibility (%)”** containing 123% is semantically broken. It is actually evidence of infeasible overcommitment. Rename the metric.

### 5. Make the homogeneous-vs-heterogeneous split explicit

Use subsection titles such as:

- **Temporal priority allocation on a homogeneous application trace**
- **Separate heterogeneous-workload elasticity study**

This would eliminate a large amount of reviewer confusion.

### 6. Either integrate elasticity into the temporal controller or reduce its prominence

The strongest version of this paper needs the experiment your limitations already identify: a heterogeneous temporal mix where both priority and elasticity operate simultaneously.

Without it, elasticity should be framed as a characterization result that motivates future integration.

### 7. Add a serious scheduler baseline

Uniform capping + FCFS is a useful lower baseline, but it cannot carry the main novelty claim. Compare against a competent priority-aware scheduler/resource manager under exactly the same hard power envelope.

### 8. Clarify the physical control boundary

You must distinguish:

**GPU power → IT power → facility power → utility-meter power.**

That single distinction affects the title, guarantee, formulation and practical contribution.

---

# Prioritized TOP 10 edits

1. **Title / Abstract / §4 — Narrow or extend “guaranteed grid flexibility”:** distinguish GPU-cap enforcement, measured power compliance, and facility-meter commitment.

2. **§2 — Replace prose-only formulation with equations and an assumption table:** define \(C(t)\), measurement boundary, queue dynamics, class weights/deadlines, objective and recovery constraint.

3. **§5.3 — Document the Azure class construction:** give exact priority assignments, class fractions, SLO deadlines, weights and sensitivity; disclose explicitly if they are synthetic.

4. **§5.3 / Related Work — Add a strong priority-aware scheduling baseline:** the current 44%→0% headline otherwise mostly demonstrates priority scheduling versus FCFS.

5. **Abstract / Contributions / §5.6 — Stop implying an integrated priority+elasticity temporal controller:** label the 26-workload experiment as a separate heterogeneous allocation study unless you add the missing integrated experiment.

6. **§4 / §5.4 / Figure 5 — Reconcile “100% by construction,” “strict=100%,” 1.1% exceedance and the above-cap measurement:** state the exact time-averaged/instantaneous/tolerance semantics of the guarantee.

7. **§5.6 — Replace “probeable, not predictable” and “scales with fleet diversity”:** use corpus-specific claims and define workload-response diversity quantitatively.

8. **Introduction — Add an operational-contract figure:** show grid meter \(C(t)\) → non-GPU/base-load subtraction → GPU envelope → priority allocation → recovery, including response time, ramp and compliance metric.

9. **§5.7 — Move the learned multi-actuator architecture to Discussion/Future Work:** retain Figure 8 as actuator characterization; remove claims that the unevaluated outer controller is already “safe” or degrades exactly to uniform.

10. **Throughout — Tighten terminology and provenance:** replace “controller runs on real measured data,” “promise a shed,” “rebound −13%,” “well-mannered grid resource,” undefined “normalized service performance,” and inconsistent predictor names with precise technical language.