1. Summary of the claimed contributions

The paper argues that grid-driven curtailment of AI data centers should be treated as a workload-allocation problem rather than a uniform GPU power-cap reduction. The proposed controller combines per-workload measured power elasticity with workload priority, selectively power-caps or defers lower-value work, and rate-limits backlog recovery to reduce rebound. A second component attempts to make grid-facing flexibility commitments safer by grounding the promised shed in a current physical power observation. The evaluation is explicitly PoC-level and “simulated-on-measured”: measured elasticity curves plus a real Azure arrival trace drive a simulator, supplemented by a single-GPU closed-loop hardware experiment and comparison with a real 256-GPU field experiment. The paper additionally argues that most benefits disappear on homogeneous fleets and that learning workload elasticity is data-limited. 

power_shaping_paper

My assessment is that there is a potentially good HotCarbon-style systems idea here, but several headline claims currently outrun the evidence. In particular, the 0%-versus-42% SLO result is not yet a fair or grid-compliant comparison, the field-validation MAE is weakly defined, and the “verifiable flexibility” construction appears to conflate an upper bound on possible shed with a guarantee of deliverable shed.

2. Strengths
A. The paper is unusually explicit about what is measured and what is simulated

This is a major positive. Section 1 explicitly says that this is simulated-on-measured, with one measured control loop and one comparison to a real grid experiment rather than implying a deployed multi-node system. 

power_shaping_paper

 Section 5.1 similarly distinguishes the measured elasticity data, Azure trace, simulator results, field comparison, and hardware experiment. 

power_shaping_paper

For a PoC paper, I would not reject it merely because most of the end-to-end evaluation is simulation. The issue is whether the simulation and hardware experiments support the particular claims being made.

B. The problem formulation is practical and systems-oriented

The controller has credible operational inputs: a facility allowance C(t), queue state, measured GPU draw, admission control, and power-cap actuation. The use of measured feedback rather than pure offline prediction is a sensible design choice. 

power_shaping_paper

The recovery/rebound issue is also valuable. Many papers demonstrate that load can be delayed without asking whether all deferred work then produces another peak. Explicitly representing backlog and recovery makes the paper more complete than a simple power-capping study.

C. The paper reports an important negative/boundary result

Section 5.5 is arguably one of the strongest parts conceptually. On the near-homogeneous emerald fleet, the paper says that 98% of the advantage over uniform control comes from priority-aware spreading rather than per-workload elasticity knowledge. That is exactly the kind of result a systems paper should report rather than suppress. 

power_shaping_paper

This result may ultimately become more scientifically interesting than the learned elasticity model itself: it identifies when a sophisticated workload model is unnecessary.

D. The single-GPU hardware loop is useful grounding

Section 5.4 demonstrates actual control rather than merely replaying profiles: a latency-sensitive stream and deferrable job share an A10G, and the controller regulates deferrable duty cycle in response to a stepped power target. 

power_shaping_paper

For an early-stage paper this is meaningful evidence that the basic mechanism is implementable.

E. The authors are not hiding several important limitations

Section 7 acknowledges the lack of a live multi-node deployment, that the firm-capacity figures are scenario bounds, and that the learned-generalization evidence is still small. 

power_shaping_paper

That substantially improves credibility.

3. Weaknesses and concerns
3.1 Soundness: the central SLO result currently has a serious problem

The biggest issue in the paper is Section 5.3.

The paper reports:

uniform capping: critical SLO violations 42%, interactive 42%;

shaping: both 0%;

offline absorbs 30% violations;

but shaping stays under the power allowance during only 83% of the binding window. 

power_shaping_paper

That last number materially changes the interpretation.

A controller whose primary constraint is P(t)≤C(t) cannot claim a clean SLO victory while violating that constraint for 17% of the supposedly binding period. Section 2 explicitly defines the objective as keeping facility power under C(t). 

power_shaping_paper

At minimum, the 0%-versus-42% comparison must be made at equal grid compliance.

I would currently treat the 0%-versus-42% number as suggestive but not a valid headline result.

There is a second issue: the baseline is too weak. “Uniform capping” intentionally treats every class identically. But the paper itself discovers that priority handling provides almost all the gain on a homogeneous fleet. A competent production scheduler would not necessarily combine uniform GPU capping with workload-blind scheduling.

The critical missing baseline is:

uniform power capping + priority-aware admission/deferral, but no workload-specific elasticity knowledge.

Without it, the experiment combines two benefits:

knowing that offline work is less valuable;

knowing that different workloads have different power elasticity.

Section 5.5 indicates that benefit (1) may dominate. 

power_shaping_paper

That substantially weakens the claim that workload-aware power elasticity is producing the SLO advantage.

The SLO definition is also insufficiently specified

Section 2 calls these “per-class queueing-delay SLOs.” 

power_shaping_paper

 Figure 2 likewise discusses “in-queue wait.” This is not obviously equivalent to user-visible inference latency.

The paper needs to state:

actual numerical SLO thresholds;

how Azure requests are assigned into critical/interactive/elastic/offline classes;

whether service-time inflation caused by capping is included;

whether end-to-end latency or only waiting time determines a violation;

whether deferred work has a completion deadline;

what percentage of the trace is flexible enough to absorb the event.

Without these details, 0% critical violations can become nearly tautological: give enough flexible offline work to the controller and allow it to absorb nearly all curtailment.

3.2 The measured control loop is useful, but the paper describes a miss as successful tracking

The hardware target is 165 W and the reported controlled power is 171 W. 

power_shaping_paper

That is approximately a 3.6% overshoot.

For ordinary application control, this may be quite good. For a paper framed around compliance with a hard grid ceiling, it is a violation.

Similarly, critical p95 changes from 13.3 to 14.5 ms—about 9%. Calling this “essentially flat” is too casual unless the relevant SLO has substantial slack. 

power_shaping_paper

Report instead:

mean power error;

p95/p99 error;

maximum target exceedance;

fraction of constrained seconds above target;

energy above the requested envelope;

controller sampling period;

NVML measurement window/delay;

actuation latency.

Then let the reader judge whether this is acceptable control.

3.3 Field “validation” is weaker than the wording implies

Section 5.2 takes the same workloads and target reduction from a real 256-GPU experiment and reports MAE 0.062 in per-workload flexing. Figure 3 says the allocator reproduces the observed behavior, flexing pretraining hardest while protecting inference. 

power_shaping_paper

This is useful sanity checking, but I would not call it strong validation without answers to several questions:

What exactly is 0.062? The quantity is apparently dimensionless, but its normalization and operational significance are not defined.

How many workload points produce this MAE? MAE over four workloads and MAE over hundreds of independent decisions are very different evidence.

Were those workload elasticity curves available to the allocator? If the model is effectively using measurements of those same workloads, reproducing their relative response is partly in-sample replay.

Was the measured field experiment implementing a comparable objective? If not, agreement in allocation pattern does not establish optimality.

Finally, the statement that aggregate modeled performance of 0.824 versus measured 0.886 is “the safe direction” is not obviously meaningful for grid safety. Underpredicting service performance is conservative for SLO planning; it says little about whether power delivery is conservative.

I would rename this “field-experiment consistency check” unless stronger out-of-sample evidence is added.

3.4 The “verifiable flexibility” mechanism currently has a conceptual problem

Section 4 states:

a workload can shed at most its currently observed draw minus the cap

and uses this observation-grounded quantity to support the claim that the facility can promise a shed it can actually deliver. 

power_shaping_paper

But “can shed at most” describes an upper bound on achievable reduction.

A safe flexibility commitment requires a lower bound on reduction that will actually be delivered.

For example, if the GPU currently draws 200 W and one sets a 150 W power limit, 200−150=50 W may bound the nominal opportunity. But a workload that does not actually drive the GPU to the new power limit could shed substantially less than 50 W. The one-reading argument by itself does not physically guarantee the reduction.

The empirical result is still interesting: Table 1 reports 0% material under-delivery and 70% usable flexibility on the Zeus leave-one-workload-out experiment. 

power_shaping_paper

 But that supports:

“this conservative heuristic did not under-deliver on this corpus”

rather than:

“the promised flexibility is physically verifiable.”

The latter is substantially stronger.

Also problematic:

“material under-delivery” is not defined;

“usable flexibility” reaches 123% for one baseline, making the metric difficult to interpret intuitively;

one instantaneous power reading is subject to noise and workload phase variation;

zero observed failures on one corpus does not imply a reliability guarantee.

This section needs a formal definition of commitment F, actual delivered reduction D, and reliability P(D≥F).

3.5 Novelty versus existing power-management/demand-response work is not established

Section 6 is currently only one paragraph. It acknowledges GPU DVFS/power-capping and the long literature on datacenter demand response, then states that this paper is “orthogonal.” 

power_shaping_paper

There is no References section in the supplied draft.

That is nowhere near enough for either e-Energy or a serious HotCarbon submission.

The paper must differentiate itself against at least these prior-work categories:

GPU power capping / DVFS and workload-specific power-performance curves;

SLO/deadline-aware resource allocation;

priority-aware load shedding;

datacenter demand response using deferrable workloads;

load shifting and rebound-aware scheduling;

capacity/flexibility bidding under uncertain available load;

power-budget allocation across heterogeneous devices.

The claim that “grid curtailment is an allocation problem” is unlikely to be sufficient novelty by itself. Allocation of limited power among heterogeneous workloads is a natural formulation with substantial precedent.

The potentially defensible narrower contribution is closer to:

measured per-workload elasticity + priority-aware admission under an exogenous grid envelope + rebound control + conservative flexibility estimation, evaluated specifically for AI workloads.

That is a systems integration/construction contribution, not a fundamental reframing of demand response.

3.6 “When shaping pays” is presently over-generalized

Section 5.5 compares elasticity spread 0.09 on emerald with 0.14 on Zeus and concludes that elasticity awareness matters when heterogeneity is genuine. 

power_shaping_paper

The qualitative intuition is plausible. The evidence is too thin to support the abstract's broad statement that value “grows with fleet heterogeneity and curtailment depth.”

You effectively have two fleet heterogeneity points from different datasets. Dataset, workload type, hardware, and heterogeneity all change simultaneously.

A convincing “when it pays” result requires a controlled 2-D experiment:

benefit=f(elasticity heterogeneity, curtailment depth)

preferably also varying flexible-work fraction.

There is also an internal inconsistency in Table 2. The text claims Oracle Capture “rises monotonically” with workload count, but the table goes:

6: −133
10: −59
14: 37
18: 47
22: 55
26: 48

So it does not rise monotonically. 

power_shaping_paper

This should be fixed before submission. It is exactly the sort of inconsistency that makes reviewers distrust other generated headline summaries.

Likewise, “the bottleneck is data, not model capacity” is stronger than this experiment establishes. You show that more data helps one simple model; that does not isolate model capacity as irrelevant.

3.7 Firm-capacity claims are too large for the supporting evidence

Section 5.6 reports a jump in compute-capacity multiplier from 1.8× to 5.0×, plus roughly 75% firm capacity. 

power_shaping_paper

Those are enormous practical claims. Section 7 appropriately admits that they depend on an assumed flexible fraction and are scenario bounds. 

power_shaping_paper

I would therefore not feature these numbers prominently until the assumptions are shown directly beside the result and sensitivity analyses demonstrate how quickly they collapse.

Also, “rebound cuts to −13%” is confusing terminology. A negative rebound sounds like the recovered load remains below baseline, not that a rebound was reduced. Define the metric explicitly.

4. Specific actionable improvements

I would prioritize these in roughly this order.

1. Redo the main evaluation at equal grid compliance. The power envelope must be a constraint, not merely another metric. If occasional violations are permitted, define the tolerance and apply it identically to every method.

2. Add the missing strong baseline: priority-aware deferral/admission + uniform power capping, with no workload-elasticity model. Also include an oracle-elasticity upper bound. This will cleanly decompose:

total gain=priority gain+elasticity-awareness gain.

Given the existing 98% result, this decomposition is essential.

3. Turn Section 5.3 into curves rather than one dramatic operating point. Sweep perhaps 10%, 20%, 30%, 40%, 50%, 60% required reduction and report high-priority SLO violations, total weighted service loss, offline backlog, and power-envelope violation. “To 40% of peak” is an unusually severe event and currently makes the 42% number look cherry-picked.

4. Strengthen the hardware experiment modestly rather than attempting a huge deployment. For a PoC, even 2–4 GPUs, several workload combinations, several target trajectories, and repeated trials would materially improve the paper. Measure strict-envelope compliance and telemetry/actuation latency.

5. Make the field comparison genuinely interpretable. Define the MAE variable and units, number of observations, train/test relationship, and uncertainty. Ideally hold out workloads or workload phases rather than replaying known elasticity.

6. Reformulate flexibility promises as a reliability problem. Construct a conservative lower confidence bound on deliverable shed, not merely an upper bound on possible shed. Evaluate promise utilization versus under-delivery probability across workload shifts. Report severity as well as frequency of failures.

7. Build a controlled “when it pays” experiment from the measured curves. Construct fleets with increasing elasticity diversity while holding other factors fixed, then sweep curtailment depth and flexible-work fraction. This could become an excellent figure and a real contribution.

8. Add serious related work and narrow the novelty claim. Do not sell “allocation instead of global dimmer” as the principal novelty. Sell the particular AI-workload/grid interface and empirically identify where elasticity knowledge adds value beyond priority scheduling.

5. Writing and clarity issues

The prose is generally concise and much better than the evaluation maturity, but several issues require correction.

First, the contributions are numbered 1, 2, 4; Contribution 3 is missing. 

power_shaping_paper

Second, a systems paper needs more formal definitions. “Service cost,” “material under-delivery,” “usable flexibility,” “elasticity spread,” “Oracle Capture,” “firm capacity,” and MAE 0.062 all appear without enough definition for independent reproduction.

Third, Table 2 directly contradicts the word “monotonically,” as noted above. 

power_shaping_paper

Fourth, distinguish “curtailment by 40%” from “curtailment to 40% of peak.” The latter means a 60% reduction. The draft uses the latter. 

power_shaping_paper

Fifth, phrases such as “verifiable,” “offers a shed amount it can actually deliver,” “preserves high-priority SLOs,” and “AI data centers can be a … trustworthy grid resource” are stronger than the current PoC demonstrates. The final conclusion particularly jumps from one GPU and one field consistency check to a broad operational claim. 

power_shaping_paper

Finally, the paper needs a real bibliography and substantially expanded Related Work.

6. Venue fit

Best fit in the current stage: HotCarbon.

The paper has the right profile for HotCarbon: a concrete systems problem, measured evidence underneath a larger simulation, a provocative cross-layer framing, useful negative results, and a clear path toward a stronger system. I would consider e-Energy only after substantially strengthening the evaluation and positioning, because e-Energy reviewers are likely to demand much more careful treatment of grid compliance, demand-response baselines, flexibility guarantees, and operational constraints.

Single most important change

Replace the 0%-vs-42% uniform-capping headline with an equal-grid-compliance comparison against a priority-aware non-elasticity baseline.

That experiment will determine whether the central workload-elasticity idea actually contributes something beyond “defer low-priority work first.”

If the gain remains substantial, the paper becomes much stronger. If it mostly disappears, Section 5.5 is already telling you what the more honest paper is about.

7. Overall recommendation

Weak reject.

One-line justification: The PoC is technically plausible and unusually transparent about simulation versus measurement, but the headline SLO comparison is not yet fair or strictly grid-compliant, the flexibility guarantee is overclaimed, and the novelty relative to priority-aware demand response is not established.