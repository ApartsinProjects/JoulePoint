I read the complete 523-line HTML from start to finish. The core paper is strong, but there are several submission-level problems: one serious notation collision, one dimensional error in the energy definition, a mismatch between the Joule-Point objective and the Section 7 scheduler, several stale numerical/methodological remnants, and a limitations section that is materially more defensive than the rest of the paper.

## Title and Abstract

**A1 — [Logical arc / terminology]. Location: title. Quote: “Scheduling AI Inference in the Space of Energy and its Derivatives.”** The paper develops an operating-point space, a power–rate response law, and one derivative used to locate an energy minimum. It does not really develop “energy derivatives” as a general scheduling formalism. The title promises more calculus than the paper delivers.

**Fix:** Either make derivatives an explicit part of the framework—e.g., define marginal energy/power slopes and use them in scheduling—or simplify the title to “Scheduling AI Inference in Energy Space” / “...by Energy Operating Point.”

---

**A2 — [Unsupported generalization]. Location: Abstract, opening. Quote: “Data centers serving AI inference strive to maximize GPU utilization, running their cards at full power by default.”** This is a broad statement about production practice, but the manuscript supplies no evidence for the “by default” claim. The same problem recurs in the Introduction. 

**Fix:** Either cite supporting evidence or narrow it: “A common performance-oriented operating point is to leave GPUs uncapped...” That is sufficient for the argument.

---

**A3 — [Internal conceptual inconsistency]. Location: Abstract. Quote: “a GPU's board power rises superlinearly with how fast it runs a job.”** Section 5 later insists that power is *not* a function of rate alone. The abstract formulation sounds exactly like a rate-only law.

**Fix:** Say: “within a fixed workload state, board power rises superlinearly along the GPU’s power-performance curve.” That preserves the conditional nature of \(P(\theta)\).

---

**A4 — [Stale number / inconsistency]. Location: Abstract. Quote: “each request runs about 1.2 to 1.4 times slower.”** Table 1 gives 1.24× for A100 and 1.22× for A10G. Nothing in the quantitative large-GPU result supports 1.4×. The same stale range reappears in the Conclusion.  

**Fix:** Use “about 1.22–1.24×” if reporting these two measured medians, or simply “about 1.2×.”

---

**A5 — [GPU-count wording]. Location: Abstract. Quote: “sweeps 20 inference models across four GPUs, three of them finely enough ... to locate the Joule Point and fit the law.”** The eventual scope is more complicated: four cards are released; T4 is excluded from quantitative law analysis; L4 requires a clock sweep to expose its energy minimum. The sentence compresses these facts enough to be misleading. 

**Fix:** “ELF measures four GPUs; the quantitative response-law analysis uses A100, A10G, and L4, while T4 is retained in the release but excluded because its cap range is too narrow.”

---

**A6 — [Major logical inconsistency]. Location: Abstract. Quote: “capping each job to the least power meeting its deadline...”** This policy is not generally the energy-optimal policy implied by the Joule Point. Once the deadline is loose enough that powers below the Joule Point are feasible, choosing the *least* feasible power moves left of the energy minimum and increases energy again. This becomes explicit—and problematic—in Section 7. 

**Fix:** If the scheduler's objective is minimum energy subject to the SLO, its operating point should be approximately

\[
P_{\rm run}=\max(P_J,P_{\rm SLO}),
\]

subject to the measured actuator grid. If the intended scheduler instead minimizes *instantaneous power*, say so and stop presenting it as the scheduler implied by the Joule Point.

---

**A7 — [Metric inconsistency]. Location: Abstract. Quote: “serves the same work for 18 to 45 per cent less energy.”** Section 7 actually reports “18 to 45 per cent less energy **per served job**” while the controllers can admit different numbers and potentially different sets of jobs. Equal goodput or more admitted jobs is not automatically “the same work.”  

**Fix:** Either evaluate energy on a matched set of jobs and retain “same work,” or change the abstract to the exact metric: “18–45% lower energy per SLO-compliant served job.”

---

## 1. Introduction

**1.1 — [Unsupported motivation]. Quote: “Inference now dominates the lifetime compute of a deployed AI model” and “electricity ... has become a leading constraint on how fast data centers can grow.”** Both are important motivation claims and both appear uncited. 

**Fix:** Cite them or narrow them. The paper does not need universal dominance: “For heavily used deployed models, inference can dominate lifetime compute...” is enough.

---

**1.2 — [Overclaim]. Quote: “the party who benefits from running at full power is not the party who pays the electricity bill.”** That is true for a particular tenant/operator split, not first-party inference fleets. Section 8 itself distinguishes first-party operators.

**Fix:** “In rented-GPU settings, the tenant who benefits from lower runtime and the operator who pays for electricity can have opposing incentives.” 

---

**1.3 — [Used-before-defined notation]. Location: contribution 2. Quote: “\(P(\theta)=P_0+a\theta^\beta\).”** At first appearance, \(\theta\), \(P_0\), \(a\), and \(\beta\) have not been defined. Their definitions are delayed until Sections 3 and 5. 

**Fix:** Do not put the formula in the contribution list unless you define it immediately: “where \(\theta\) is normalized throughput, \(P_0\) the fitted floor, \(a>0\) a scale coefficient, and \(\beta\) the response exponent.”

---

**1.4 — [Overgeneralization]. Quote: “one fixed cap per card serves every job.”** The experiment covers 20 measured workloads under the loaded regime, not “every job.”

**Fix:** “one fixed cap per card serves all 20 measured workloads with low energy penalty.”

---

**1.5 — [Jargon]. Quote: “The exponent \(\beta\) lives in the measured run.”** “Lives in” is imprecise and sounds generated rather than scientific.

**Fix:** “\(\beta\) must be estimated from measured power–performance behavior rather than from the model specification.”

---

## 2. Related Work

**2.1 — [Notation order]. Quote: “The fixed floor \(P_0\) in our law...”** This partially explains \(P_0\), but only after the undefined formula has already appeared in Section 1. 

**Fix:** Define notation in Section 1 or remove formulas until Section 3/5.

---

**2.2 — [Conceptual overstatement]. Quote: “The fixed floor \(P_0\) in our law is exactly that share measured for modern GPUs.”** A fitted intercept in a workload-conditioned GPU board-power curve is not necessarily “exactly” the work-independent server power considered in energy-proportional computing. It includes whatever fixed board-level contribution the fit absorbs.

**Fix:** “\(P_0\) plays the analogous role of a workload-independent board-power floor in our fitted curves.”

---

**2.3 — [Unsupported blanket characterization]. Quote: “These controllers treat the energy-optimal operating point as an online, per-job control target, searched for at runtime.”** This is asserted collectively about several distinct systems. The paper should be precise about which systems search a power optimum, which tune frequency, and which jointly optimize batch/placement/power. 

**Fix:** Split the claim by system or write “Several of these systems...” rather than “These controllers.”

---

**2.4 — [Logical overclaim]. Quote: “The two axes are orthogonal and compose.”** Adaptive inference changes computation, utilization, phase mix, and potentially \(P_0\), \(\beta\), and therefore the Joule Point. Section 10 correctly proposes *measuring their joint surface*, which implicitly concedes that interaction is unknown. 

**Fix:** “The mechanisms are conceptually distinct and can potentially be composed; their interaction requires joint measurement.”

---

**2.5 — [Jargon / arc drift]. Quote: “the thermodynamics of prediction ... and Landauer's limit setting the ultimate floor for that energy-per-information view.”** This has essentially no role in the paper's measured systems argument and introduces an “energy-per-information” framing that is never developed.

**Fix:** Remove it. It dilutes the concrete contribution.

---

**2.6 — [Undefined term]. Quote: “the compute-power elasticity.”** This quantity is invoked as something ELF makes measurable, but it is never formally defined.

**Fix:** Either define the derivative/elasticity explicitly, e.g. \(d\log R/d\log P\), and actually use it, or replace it with “local power–throughput slopes.” 

---

**2.7 — [Unsupported economic wording]. Quote: “at today's rental-to-electricity price ratios...”** Section 8 does not actually provide a cloud rental-price calculation; it compares a $12,000 purchase price with electricity at $80/MWh.

**Fix:** Delete “today’s rental-to-electricity price ratios” or provide the actual rental-price comparison.

---

## 3. Energy Space and Its Calculus

This section contains the most important correctness problem.

### Symbol audit

| Symbol / term | First use | Status |
|---|---|---|
| \(\theta\) | Sec. 1 contribution formula | **Used before definition**; defined in Sec. 3 |
| \(R\) | Sec. 4 measurement paragraph | Defined in same sentence, but should be introduced earlier because \(R_{\max}\) appears first |
| \(R_{\max}\) | Sec. 3 | Called the normalization ceiling, but the defining relation \(\theta=R/R_{\max}\) is never explicitly written |
| \(P_0\) | Sec. 1 formula | **Used before definition** |
| \(a\) | Sec. 1 formula | **Used before definition and never explicitly described semantically** |
| \(\beta\) | Sec. 1 formula | Used before definition; later described as exponent |
| \(\theta^*\) | Eq. 2 | Defined correctly at first use |
| \(P^*\) / \(P^\star\) | Sec. 3 | **Critical collision:** SLO power in Sec. 3, Joule-point power in Eq. 3 |
| \(W_0\) | Sec. 6 | Defined parenthetically, but then never used; likely stale notation |
| \(E\) | Sec. 3 | Defined as energy/inference, but formula is dimensionally incomplete |
| \(e(\theta)\) | Eq. 1 | Introduced as \(P/\theta\), but distinction from physical \(E\) is not made cleanly |
| TDP | Sec. 1 | Correctly expanded at first use |
| SLO | Sec. 1 contributions | Correctly expanded at first use |
| ESG | — | **Does not occur anywhere in the manuscript** |

The relevant definitions are concentrated in the Section 3 opening. 

---

**3.1 — [Major correctness issue]. Quote: “Energy per inference \(E=P/\theta\).”** Because \(\theta\) is dimensionless normalized rate, \(P/\theta\) still has units of power, not energy. The manuscript later acknowledges missing constants, but this should never be called literal energy.

If

\[
\theta=\frac{R}{R_{\max}},
\]

then energy per batch step is

\[
E_{\rm step}=\frac{P}{R}
=\frac{P}{\theta R_{\max}}.
\]

For energy per individual sample in a batch of size \(B\),

\[
E_{\rm sample}=\frac{P}{B\theta R_{\max}}.
\]

The omitted constant does not move the optimum, but it matters for dimensional correctness. 

**Fix:** Define physical \(E\) correctly, then introduce \(e(\theta)=P/\theta\) explicitly as an energy-proportional objective whose multiplicative constant does not affect the optimum.

---

**3.2 — [Derivative inconsistency]. Quote: “its derivative vanishes, \(dE/dp=0\).”** Section 6 defines the Joule Point using \(dE/d\theta=0\). Because \(p\) is the commanded cap and \(\theta\) is achieved normalized rate, these are not interchangeable without a monotonic mapping argument.

**Fix:** Use \(dE/d\theta=0\) consistently for the analytical derivation. If discussing the measured cap domain, say the measured optimum is \(\arg\min_p E(p)\).

---

**3.3 — [Critical symbol collision]. Quote: “The ... SLO power \(P^\star\) is the least power...” versus Eq. (3), “\(P^{*}=P_0\beta/(\beta-1)\).”** The same symbol denotes two different powers: deadline-feasible power and Joule-point power.  

**Fix:** Use, for example,

\[
P_{\rm SLO}
\]

for the minimum SLO-feasible power and

\[
P_J
\]

for the Joule Point. This also makes the Section 7 decision rule immediately clear.

---

**3.4 — [Missing basic relation]. Quote: “the absolute ceiling used to normalize is \(R_{\max}\).”** You never write the actual definition.

**Fix:** Add:

\[
\theta \equiv R/R_{\max}.
\]

Then define \(R\) once here, rather than waiting until Section 4.

---

**3.5 — [Arc / overstatement]. Quote: “Hardware-space scheduling discards all of them.”** Related Work has just described schedulers that explicitly manage power and SLOs, so “discards” is too absolute.

**Fix:** “A representation containing only hardware placement does not expose these quantities explicitly.”

---

## 4. The ELF Dataset

**4.1 — [Acronym used before expansion].** “ELF” appears in the Introduction, but “Energy-Latency Frontier” is not expanded until Section 4. 

**Fix:** Expand ELF at first use in Section 1.

Several other acronyms are also left unexplained: **DVFS**, **NVML**, **AMI**, and later **NIC**, **PUE**, and **PSU**. For a systems audience some are familiar, but the paper's stated house style is “precise but plain”; expanding them once costs almost nothing.

---

**4.2 — [Repetition-count inconsistency]. Quote: “The batch-32 collection ... (two repetitions)” and later “every operating point is measured three times.”** The latter applies to the second collection, not all ELF operating points. Section 9 incorrectly turns this into the global claim “Each operating point is a three-repetition sweep.” 

**Fix:** Keep the distinction throughout: batch-32 = two repetitions; load-regime collection = three.

---

**4.3 — [Dataset-schema gap / possible editing residue]. Quote: “Two sweep collections and one trace make up ELF” versus the following paragraph's graphics-clock sweep.** The schema enumerates cap sweeps and an actuation trace but never explains where the clock-sweep data supporting Figures 1, 3, 4 and the L4 result reside.

**Fix:** Explicitly add the clock-sweep collection to the schema, its cards/models/batches/repetitions, and whether its rows are included in the “about 5500” count. 

---

**4.4 — [Internal wording mismatch]. Quote: “add a clock sweep on the cards whose cap floor is too high to reach the energy minimum.”** Yet Figure 1's equivalence experiment is on the **A10G**, whose Joule Point is reachable by power cap.

**Fix:** Distinguish the purposes: “We use an A10G clock sweep to verify actuator equivalence, and an L4 clock sweep to reach operating points below the L4's cap floor.”

---

**4.5 — [Overclaim]. Quote: “the two are interchangeable.”** Figure 1 reports a mean energy difference of about 3% where board-draw ranges overlap. That demonstrates approximate agreement in this experiment, not general interchangeability across clocks, workloads, performance metrics, or thermal behavior.

**Fix:** “They produce closely matching energy–draw curves in the measured overlap region.”

---

**4.6 — [Measurement-method gap].** A two-second measurement at 20 Hz is short, particularly when Section 7 later reports ~191 ms actuation settling and Section 9 discusses transient bias. The paper does not state warm-up time, cap sweep ordering, whether points were randomized, or how thermal state was controlled.

**Fix:** Add these details to Section 4 rather than defending the consequence in Limitations.

---

**4.7 — [Reproducibility ambiguity]. Quote: “a p4d whose eight A100s shared the sweep.”** It is unclear whether all eight A100s execute concurrently, whether values are averaged across devices, or whether one device is used at a time.

**Fix:** State precisely how the eight A100s participated and how board power/throughput are aggregated.

---

## 5. The Response Surface and Its Law

**5.1 — [Undefined parameter]. Quote: “\(P(\theta)=P_0+a\theta^\beta\).”** \(P_0\) and \(\beta\) are explained, but \(a\) never gets a direct definition.

**Fix:** “\(a>0\) is the workload-dependent dynamic-power scale coefficient.”

---

**5.2 — [Model-assumption gap]. Quote: “For a compute-bound inference kernel the achieved rate rises with the clock...”** ELF includes heterogeneous workloads including LLM decoding; the paper does not establish that they are all compute-bound.

**Fix:** Present the CMOS discussion as motivation for the empirical functional form, not a derivation that necessarily applies workload-by-workload: “For compute-bound kernels this gives...; we then test whether the resulting form fits the heterogeneous ELF workloads.” 

---

**5.3 — [Impossible configuration description / likely stale wording]. Quote: “173 configurations of GPU, workload, and sweep collection.”** With three analyzed GPUs × 20 workloads × two named sweep collections, that tuple can produce at most 120 combinations. With four GPUs it gives 160. Therefore “173 configurations of GPU, workload, and sweep collection” cannot literally be correct.

Most likely the count includes **load mode/batch regime** as another dimension.

**Fix:** State exactly what constitutes one fit, e.g. “173 (GPU, workload, load-regime) response curves.” This is a high-priority stale-content fix. 

---

**5.4 — [Unsupported inference from censored fits]. Quote: “41 of 173 fits reach the \(\beta=8\) ceiling ... so ... the true superlinearity there is stronger still.”** Reaching an imposed optimizer bound shows that the estimate is censored; it does **not** establish that the true exponent is >8. It could also indicate poor identifiability or model mismatch.

**Fix:** “41 fits hit the imposed \(\beta=8\) bound, so the upper tail of \(\beta\) is not identifiable from these constrained fits.”

---

**5.5 — [R² presentation ambiguity].** The manuscript reports at least three materially different fit statistics:

- per-response-curve median \(R^2=0.986\);
- saturating/light medians 0.98/0.99;
- pooled per-card normalized fits of 0.92/0.88/0.95;
- pooled-across-batch rate-only fit “near zero.”

All can simultaneously be true, but they are currently presented closely enough to look contradictory. 

**Fix:** Name each experiment: **within-curve fit**, **pooled normalized card fit**, and **pooled absolute-rate counterexample**.

---

**5.6 — [Unsupported quantitative example]. Quote: “two workloads delivering the same throughput differ by 2.7× in board power.”** No workloads, rate, operating points, or figure/table are identified.

**Fix:** Give the concrete pair and operating points, or put the example in Figure 2/inset/table.

---

**5.7 — [Overclaim in Figure 2]. Quote: “All workloads collapse onto a single fitted law per card.”** An A10G pooled \(R^2=0.88\), combined with the acknowledged wide within-card \(\beta\) spread, supports “approximately align,” not “all collapse.”

**Fix:** Caption: “The normalized loaded-regime curves show a strong card-level trend...” 

---

## 6. The Joule Point

**6.1 — [Major mathematical omission]. Quote: “A minimum exists only for \(\beta>1\).”** For an **interior minimum on the stated domain** \(\theta\in(0,1]\), \(\beta>1\) is not sufficient. The stationary point

\[
\theta^*=
\left[\frac{P_0}{a(\beta-1)}\right]^{1/\beta}
\]

must also lie at or below 1 and within the actuator's reachable range. If \(\theta^*>1\), the constrained optimum is the full-power boundary. If the hardware has a lower reachable \(\theta_{\min}\), the deployed optimum can instead hit that boundary. 

**Fix:** State the constrained result explicitly, preferably as a clipped optimum over the measured feasible interval.

---

**6.2 — [3-vs-4-card wording]. Quote: “an energy minimum exists on every card we measure.”** T4 is measured but explicitly cannot be used to characterize the law.

**Fix:** “The three analyzed cards exhibit an energy minimum over the combined cap/clock operating range.”

---

**6.3 — [Stale variable]. Quote: “fixed job constants \(W_0\) ... and \(R_{\max}\).”** \(W_0\) appears exactly once and never participates in a displayed expression. It looks like residue from an earlier formulation.

**Fix:** Remove \(W_0\), unless you explicitly define an energy-per-unit-work derivation that needs it. 

---

**6.4 — [Overclaim of exactness]. Quote: “The price of reaching it is exact.”** The identity is exact only in the simplified service model where per-card rate scales as \(\theta\), latency is its reciprocal, workloads are continuously divisible across cards, and system/queueing overhead is ignored.

**Fix:** “Under the per-card service model, the throughput-preserving card multiplier equals the service-time ratio, \(1/\theta^*\).”

---

**6.5 — [Figure/text actuator mismatch]. Location: Figure 4. Quote: “what capping costs ... lowering the graphics clock.”** The caption calls the phenomenon “capping” while the actual actuator is a clock lock.

**Fix:** Replace “what capping costs” with “what lowering the operating point costs,” or make Figure 4 a power-cap experiment. 

---

**6.6 — [AI-tell / misleading heading]. Quote: “One fixed cap per card places it for free.”** It is not free: the paper has just quantified a latency/card penalty. What is nearly free is the *energy-optimality penalty relative to per-model tuning*.

**Fix:** “One fixed cap per card is nearly as efficient as per-model tuning.”

---

**6.7 — [Numerical logical error]. Key box quote: “within about one per cent of every job's own energy optimum.”** The underlying statistic is **mean penalty** 0.9% on A100 and 0.4% on A10G. A mean below 1% does not imply every workload is within 1%. Indeed the preceding text says only 17/20 A100 workloads are within **2%**. 

**Fix:** “one static cap has mean energy penalty below 1% relative to per-model optima.”

---

**6.8 — [Overgeneralization]. Quote: “the regime real inference serving keeps GPUs in.”** Production inference can be bursty, phase-varying and imperfectly utilized; Section 9 explicitly identifies production serving validation as future work.

**Fix:** “a regime representative of throughput-oriented, well-batched serving.”

---

**6.9 — [Unsupported sensitivity numbers]. Quote: “folding a representative 100 to 200 W of per-GPU node overhead...”** The source or justification for 100–200 W is absent, and the resulting 57%, 52%, 12–18%, and 10–16% values are not shown in a figure/table. 

**Fix:** Label it an explicit sensitivity analysis, show its equation/table, and state that 100–200 W is an assumed range rather than measured ELF data.

---

**6.10 — [Technical simplification presented as fact]. Quote: “Facility multipliers (PUE, PSU efficiency) are constant factors that cancel.”** PUE can be approximated as a multiplier for a narrow analysis, but PSU efficiency is generally load-dependent, precisely when changing power draw.

**Fix:** “If PUE and conversion efficiency are treated as constant over the operating range, they cancel in relative savings.”

---

## 7. Fleet Simulation

This section is currently the weakest link in the paper's narrative.

**7.1 — [Terminology / reproducibility gap]. Quote: “trace-driven simulation ... over a 200-tick trace whose job mix drifts...”** The only trace described in ELF is the ~50 Hz cap-actuation trace. The 200-tick job-mix trace appears to be synthetic, but its generation is never described.

**Fix:** Call it a “seeded synthetic workload trace” if that is what it is, define one tick, arrival/job generation, model probabilities, drift process, and publish the generated trace. 

---

**7.2 — [Major objective inconsistency]. Quote: “cap each admitted job to the least power meeting its SLO” and “it coincides with the Joule Point when the SLO is loose enough ... and sits below it when the SLO is looser still.”** This explicitly shows that the scheduler intentionally drives below the energy minimum when allowed.

That breaks the central argument: Section 6 identifies the Joule Point as the energy optimum; Section 7 then uses minimum feasible **power**, not minimum feasible **energy**.

**Fix:** For an energy-minimizing scheduler:

\[
P_{\rm chosen}
=
\max(P_J,P_{\rm SLO})
\]

on a monotone performance curve. If there is a fleet-power feasibility problem that requires going below \(P_J\), describe that as a separate power-budget allocation mode and quantify its extra energy cost.

---

**7.3 — [Unsupported “ground truth” wording]. Quote: “measured curves ... supply ground-truth power and latency at every (GPU, model, cap).”** ELF is measured on a discrete cap grid. Also Section 4 says the load-regime collection records throughput, utilization and board power, while explicit phase latency is reported for the batch-32 collection.

**Fix:** Say exactly whether service time is inferred as \(1/R\), whether interpolation occurs, and use “measured lookup values” rather than “ground truth at every cap.”

---

**7.4 — [Missing simulation detail]. Quote: “competent placement.”** This is not an algorithm.

**Fix:** Specify the exact shared placement rule, tie-breaking, admission criterion, cap-grid selection, and energy accumulation. A systems reviewer must be able to reproduce the baseline.

---

**7.5 — [Headline result lacks sufficient evidence]. Quote: “18 to 45 per cent less energy per served job.”** This is an abstract-level contribution but receives one paragraph and no dedicated result table/plot showing the sweep over +10% to +100% SLO slack.

**Fix:** Add a table/figure with budget × SLO slack, showing energy, admitted/SLO-met jobs, and matched-work energy. 

---

**7.6 — [Same-work problem].** Because the capped scheduler frees power and admits more jobs, “energy per served job” can change partly because the served-job population changes.

**Fix:** Report both:
1. energy for the same matched jobs; and
2. total SLO goodput / total input energy.

That will substantiate the abstract's “same work” claim.

---

**7.7 — [Unexplained fleet size / likely leftover]. Quote: “a static fleet of 18 identical A100s, one measured model per GPU.”** ELF has 20 workloads. Why exactly 18? Which two are absent and why? 

**Fix:** Explain the exclusion explicitly. If 18 is residue from an earlier dataset version, regenerate the experiment on all eligible 20 workloads.

The 15-GPU dynamic fleet versus 18-GPU static fleet is not itself inconsistent because the experiments are separately described, but the unexplained **18-of-20** is conspicuous.

---

**7.8 — [Unsupported isolated result]. Quote: “whenever the latency budget is about 158 ms or looser...”** This exact routing threshold appears without a supporting table/figure or enough information to reproduce the comparison.

**Fix:** Give the workload/batch and measured energies/latencies of all three competing GPU points, or remove this isolated number. 

---

**7.9 — [Unsupported isolated result]. Quote: “uncapped A10G draw drifts by about 17 per cent as the card warms.”** Again, no quantitative support is shown.

**Fix:** Include it in the actuation figure/results or remove it.

---

**7.10 — [Plain-language issue]. Quote: “the greenest option.”** The study measures board energy, not lifecycle environmental impact.

**Fix:** “the lowest-board-energy measured operating point.”

---

## 8. Implications for Operators

**8.1 — [Apologetic / over-hedged opening]. Quote: “they are prompts for evaluation, not deployment prescriptions” and “each point below is a hypothesis to test.”** This reads as pre-emptive reviewer defense rather than a limitations statement. 

**Fix:** Replace the whole paragraph with something like: “These implications follow from the measured GPUs and loaded workloads in ELF. Production deployments should re-characterize the same curves under their serving stack and workload mix.”

That is confident and properly scoped.

---

**8.2 — [Overgeneralization beyond measured regime]. Quote: “The same holds under light or bursty demand...”** The central fixed-cap result is specifically justified under *loaded* operation, while bursty production behavior is deferred to future work.

**Fix:** Present this as an analytical regime argument, not an empirical finding: “If demand is sufficiently below installed capacity, the additional-card penalty can disappear, although request latency still increases.”

---

**8.3 — [Important missing caveat]. Quote: “capping removes energy at no capital cost.”** Even in an under-utilized fleet it is not necessarily free: request latency can still increase and SLOs can bind.

**Fix:** “capping can require no additional installed cards when sufficient capacity slack exists, although service latency still changes.”

---

**8.4 — [Likely stale experiment]. Quote: “keeping 95 per cent of a throughput utility on the A100 saved a median 13 per cent of energy, but 28 per cent under a deadline utility.”** Neither “throughput utility” nor “deadline utility” is defined anywhere, and these 13%/28% results are not developed in Sections 5–7. This reads exactly like residue from a removed table or earlier experiment. 

**Fix:** Either restore the experiment/method/table that defines these utilities or delete this sentence.

---

**8.5 — [Economic logic mismatch]. Quote: “A per-joule surcharge is too weak ... at $80 per MWh a $12,000 A100 must run about 43 years for its electricity to equal its purchase price.”** The arithmetic is internally correct for a continuously drawing 400 W A100:

- 0.4 kW × 8760 h ≈ 3.504 MWh/year;
- × $80/MWh ≈ $280/year;
- $12,000 / $280 ≈ 42.8 years.

But it does **not** directly demonstrate that a *tenant-facing per-joule surcharge* is too weak relative to hourly rental. That requires comparing energy cost/hour with rental price/hour. Purchase capex is a different economic comparison.

**Fix:** Separate the arguments:
- operator capex versus electricity: use the 43-year calculation;
- tenant incentive: compare $/GPU-hour with $/kWh attributable to that GPU.

---

**8.6 — [Overclaim]. Quote: “realigning it is the main lever on adoption.”** The paper has not studied adoption barriers. Section 9 itself identifies production stack, actuation privileges, workload variation, hardware generation, etc.

**Fix:** “Per-hour pricing creates a material split incentive that can discourage energy-optimal operation.”

---

**8.7 — [Figure 6 terminology]. Quote: “static idle floor (grey, taken as the least measured board draw).”** The least measured board draw under a running workload is not necessarily an **idle** floor and is not necessarily the fitted \(P_0\). This conflates three quantities.

**Fix:** Call it “minimum measured board draw in this sweep,” unless it is actually an independent idle measurement. 

---

**8.8 — [Capacity-model assumption hidden]. Quote: “installing more GPUs forces each to a deeper cap.”** Installing extra cards does not force anything if cards may be left idle. The figure appears to assume all installed GPUs are concurrently active behind the fixed feed.

**Fix:** State that assumption explicitly: “If all installed GPUs are active and share the 1 MW envelope...” Otherwise “strictly dominated” is too broad.

---

**8.9 — [Generic vs workload-specific result].** The text calls the 5,500-card point “the measured A100 optimum,” but Figure 6 is specifically **ViT-B/32, batch 64**.

The **5,500-card count itself is internally sensible**: 1 MW / (0.46 × 400 W) ≈ 5,435 cards, so it is consistent with the ~46% A100 Joule Point. The issue is scope, not arithmetic.

**Fix:** Name the workload in the prose as the caption already does.

---

## 9. Limitations

This section should be rewritten almost completely. Its factual scope is useful; its rhetoric is defensive and several statements are incorrect or stale.

**9.1 — [Apologetic tone]. Quote: “What this paper establishes, it establishes by direct measurement.”**

**Fix:** “Our measurements cover 20 workloads on three quantitatively analyzed NVIDIA GPUs under fixed-batch regimes.”

---

**9.2 — [Overclaim]. Quote: “the deployed cap is set from a one-time measurement, which is exact by construction.”** Measurement-derived optima are not “exact”: there is finite cap resolution, measurement noise, workload drift and actuator variability.

**Fix:** “the deployed cap is selected directly from the measured sweep rather than inferred from the analytical fit.”

---

**9.3 — [Unsupported new number]. Quote: “the analytic optimum lands within about ten points of TDP of it on the high-exponent cards.”** This result appears nowhere else and has no supporting table.

**Fix:** Add it to Section 6 with quantitative error statistics or remove it.

---

**9.4 — [Stale economic content]. Quote: “The economic analysis uses ... a four-year life.”** The Section 8 economic argument does not actually specify or use a four-year lifetime in the visible calculation.

**Fix:** Either introduce the four-year horizon explicitly in Section 8 or delete it here. 

---

**9.5 — [Defensive prose]. Quote: “the physical measurements beneath them do not move.”**

**Fix:** Delete. It adds no scientific content.

---

**9.6 — [Direct repetition inconsistency]. Quote: “Each operating point is a three-repetition sweep.”** False according to Section 4, where the batch-32 collection has two repetitions.

**Fix:** “The principal load-regime sweep uses three repetitions per point; the batch-32 collection uses two.”

---

**9.7 — [Likely sign error]. Quote: “under-reads board power ... which biases the reported savings downward; the numbers are a lower bound.”** If the capped operating point's power is **under-read**, capped energy is underestimated, making the calculated saving *larger*, not smaller. Moreover, the sign of transient bias depends on sweep direction.

**Fix:** Determine the actual sweep direction and transient behavior from the data. Do not claim a lower bound unless demonstrated. This is a correctness issue.

---

**9.8 — [Unsupported deployment claim]. Quote: “bare-metal and VM tenants hold [nvidia-smi privileges], the tenancy tier where large fleets run.”** This is a broad infrastructure assertion and unnecessary.

**Fix:** “Power-cap control requires sufficient NVIDIA management privileges; availability depends on deployment environment.”

---

**9.9 — [Overclaim / contradiction with future work]. Quote: “a live deployment adds arrival realism, not new physics.”** A production stack can alter batching, concurrency, prefill/decode balance, memory behavior and utilization—and the next section explicitly proposes validating those effects.

**Fix:** “A live deployment would test whether the measured response law remains stable under production batching, arrival variation, and phase mixing.”

---

**9.10 — [Buried result]. Quote: “L4 ... median 8 per cent saving.”** This is the first aggregate quantitative L4 saving result and it appears in Limitations.

**Fix:** Put the L4 result in Section 6/Table 1 or omit it here.

---

## 10. Conclusion and Future Work

**10.1 — [Stale latency range]. Quote: “1.2 to 1.4 times slower.”** Again, Table 1 supports 1.22–1.24× for the two large GPUs. 

**Fix:** Correct both Abstract and Conclusion simultaneously.

---

**10.2 — [GPU-count wording]. Quote: “Measuring that surface for 20 models on three GPUs...”** This is defensible only if explicitly understood as “three quantitatively analyzed GPUs,” because the dataset contains four.

**Fix:** Use exactly that wording.

---

**10.3 — [Overclaim / rhetoric]. Quote: “The physics is favorable and the actuation is a one-line command.”** L4 already requires a different actuator, privilege availability varies, and production validation remains future work.

**Fix:** Delete the sentence or say: “The hardware controls are already exposed through standard NVIDIA management interfaces.”

---

**10.4 — [Unsupported causal conclusion]. Quote: “That obstacle is economic, not technical.”** The paper establishes a split incentive; it does not establish that economics is the sole or dominant adoption obstacle.

**Fix:** “Per-hour pricing therefore creates an economic obstacle even when the energy-efficient operating point is technically accessible.”

---

**10.5 — [Future-work contradiction]. Quote: “under a production serving engine ... only the load generator differs.”** That is exactly what is *not* known: vLLM/TensorRT-LLM batching, co-location and phase scheduling may change the response curve.

**Fix:** “Production serving changes workload state through dynamic batching, phase mixing, and concurrency; measuring whether the law persists under those conditions is the key next validation.”

---

## Data and Code Availability / Structural Audit

**D1 — [Overbroad provenance claim]. Quote: “Every figure and headline number in this paper is computed from that dataset.”** The manuscript also contains economic assumptions such as $80/MWh and $12,000/card that are not measurement outputs.

**Fix:** “Every empirical figure and measurement-derived headline result is generated from ELF by the accompanying build script.” 

The mechanical numbering is otherwise clean:

- Sections **1–10** are consecutive with none skipped.
- Figures **1–6** are consecutive.
- Equations **(1)–(3)** are consecutive.
- There is one **Table 1**.
- Every Figure 1–6 is referenced from surrounding text.
- Table 1 is referenced.
- I found no unresolved numbered section/figure/table forward reference.
- The Section 4 wording “see **Data availability**” should ideally match the actual heading “**Data and code availability**.”
- References are numbered **1–60**, all 60 entries are present, and every reference number 1–60 is cited somewhere in the manuscript. 

## Overall logical arc

The arc is strong through Sections 1–6:

**operator problem → operating point → ELF → response law → energy minimum → static per-card characterization.**

The thread breaks in Section 7. The natural consequence of Section 6 is: *for every job, select the lowest-energy operating point that satisfies its SLO*. Instead, Section 7 selects the **lowest-power** point satisfying the SLO and explicitly allows operation below the Joule Point. That makes the central construct less important exactly where the paper should demonstrate its scheduling value.

Section 8 then expands into three different arguments—energy efficiency, power-limited capacity, and pricing incentives—without cleanly separating their operating regimes. This is why some statements appear contradictory: capping costs 1.24× cards in one paragraph, “no capital cost” in another, and increases capacity in another. All three can be true, but only in different regimes.

A much cleaner operator framing is:

**capital-limited:** full power may be rational;  
**energy-cost-limited:** operate at \(P_J\);  
**latency-constrained:** operate at \(\max(P_J,P_{\rm SLO})\);  
**hard-power-limited:** allocate operating points jointly, possibly below \(P_J\) if capacity/goodput rather than energy is the primary objective.

That would make the “so what” considerably stronger.

## Five highest-priority fixes

1. **Fix the mathematics and notation in Sections 3/6:** correct the dimensional definition of energy, explicitly write \(\theta=R/R_{\max}\), remove stale \(W_0\), and separate \(P_J\) from \(P_{\rm SLO}\).

2. **Repair Section 7's objective.** The current “least SLO-feasible power” scheduler is not the scheduler implied by the Joule Point. Use the energy-minimizing feasible operating point or clearly define a separate power-minimization objective.

3. **Resolve stale/internal inconsistencies:** 1.2–1.4× versus 1.22–1.24×; “173 configurations” with an impossible stated tuple; two-versus-three repetitions; unexplained 18-of-20 workloads; four-year life; 95%-utility 13%/28% result; and the transient-bias/lower-bound statement.

4. **Give the 18–45% fleet result first-class experimental support.** Show budget × SLO results, define the synthetic trace completely, compare matched work, and make the abstract metric exactly match the evaluated metric.

5. **Rewrite Sections 8–9 for scope rather than defense.** Remove “prompts, not prescriptions,” “exact by construction,” “not new physics,” “the physical measurements do not move,” and “economic, not technical.” State measured scope plainly and separate capital-, energy-, latency-, and power-limited operating regimes.