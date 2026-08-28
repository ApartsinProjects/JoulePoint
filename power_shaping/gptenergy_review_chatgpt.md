# Peer Review and Science-Editing Report
# Peer Review: *From GPUs to Joules: Scheduling AI Inference in the Space of Energy and its Derivatives*

## 1. Overall assessment

The paper has a strong publishable core: **GPU operating power should be treated as a first-class scheduling variable, not a fixed property of the selected hardware**, and a dense response surface can expose an energy-minimizing operating point that hardware-only placement misses. ELF is potentially a valuable dataset, the A100/A10G result is memorable (median Joule caps near 46%/43% of TDP with 29%/31% lower GPU-board energy per measured work unit), and the idea of using local power–performance derivatives to turn a cap from a configuration knob into a scheduling quantity is compelling. However, the current draft pushes the framing beyond what its measurements and mathematics presently support. Several issues are substantive: the coordinate system omits workload state such as batch size even though batch is used to demonstrate non-uniqueness; CPE is defined with respect to the cap rather than actual board power, making the “CPE = 1” identity conditional rather than universal; the physical derivation overstates what the fitted power law establishes, especially with 58/232 fits hitting the imposed beta ceiling; the small-GPU interpretation conflicts with the direction of Figures 3–4; the fleet policy minimizes power subject to an SLO rather than energy subject to an SLO; the simulation’s “same work” claim is not established by the described admission procedure; Table 3 contradicts its own caption for A10G; and one routing example has an explicit latency inconsistency (158 ms cannot satisfy a 45–120 ms budget). The economic section also conflates the energy optimum with a provider’s total-cost optimum despite the 22–24% capacity/card-hour penalty. Fixing these points would make the paper both more rigorous and more eye-opening: the strongest version is not “we discovered a universal law that makes existing scheduling obsolete,” but “dense measurement reveals a stable, actionable energy operating point on the measured large GPUs, and treating actual power as a scheduling coordinate exposes decisions that hardware-only placement cannot express.”

---

## 2. Prioritized highest-impact edits

### 1. Fix the definition of the coordinate system before anything else
**Priority: blocking.** The paper defines an operating point as `(GPU, power cap)`, yet its central counterexample uses different `(batch, cap)` combinations and Section 10 introduces a second actuator (graphics clock). As written, `(GPU, cap)` is not sufficient to determine the response surface.

Define a workload state explicitly, for example

\[
w=(\text{model},\text{batch},\text{precision},\text{serving stack},\ldots),
\]

and an actuator setting `u` (power limit, clock limit, etc.). Then, conditional on `(w,g)`, the actuator induces measured `(P,R,L)`.

A clean definition is:

> For a fixed workload state \(w\) on GPU \(g\), actuator setting \(u\) induces observed board power \(P(u)\), rate \(R(u)\), and latency \(L(u)\). We call \((g,u)\), conditional on \(w\), an operating point. The scheduler chooses among these operating points subject to service constraints.

If batch size itself is a scheduler decision, include it in the chosen operating point rather than hiding it inside the workload state.

This also resolves the current tension between “power is not a function of rate” and the notation \(P(R)\): the power-law fit is **a per-workload-state curve**, not a universal function of rate.

### 2. Rewrite the calculus in terms of *actual board power* rather than the configured cap
**Priority: blocking.** The current definition

> “\(\mathrm{CPE}=d\ln R/d\ln p\)”

and the claim

> “the minimum sits exactly where CPE falls through one”

are exact only while the cap binds so that actual board power \(P=p\). The paper itself shows inert caps in Figure 6.

Use

\[
\epsilon_P=\frac{d\ln R}{d\ln P}
\]

as the primary elasticity. Then

\[
E=\frac{P}{R},\qquad
\frac{d\ln E}{d\ln P}=1-\epsilon_P,
\]

so an interior energy minimum occurs at \(\epsilon_P=1\). This is the clean physical identity the paper wants. If you retain cap elasticity, state explicitly that \(d\ln R/d\ln p=d\ln R/d\ln P\) only in the binding region.

### 3. Separate the Joule point from the minimum-power SLO point
**Priority: blocking.** Section 8 repeatedly says the energy-space controller

> “cap[s] each admitted job to the least power meeting its SLO.”

That is a **power-minimizing** feasible point, not generally the **energy-minimizing** feasible point. Because \(E(P)\) is U-shaped, reducing power below the Joule point makes a job slower enough that energy per completion rises again.

Define two quantities:

\[
p_{\mathrm{SLO}}=\min\{p:L(p)\le L_{\mathrm{SLO}}\},
\]

and

\[
p_{E\mid\mathrm{SLO}}
=\arg\min_{p:L(p)\le L_{\mathrm{SLO}}} E(p).
\]

Under a monotone latency curve and a single U-shaped energy minimum,

\[
p_{E\mid\mathrm{SLO}}=\max(p_J,p_{\mathrm{SLO}}).
\]

If the controller deliberately chooses below \(p_J\) to free instantaneous fleet power and admit more jobs, that is legitimate—but it is a **power/headroom trade**, not energy minimization, and the paper should say so.

### 4. Recast \(P(R)=P_0+aR^\beta\) as an empirical response model, not a governing physical law
**Priority: blocking for tone and scientific validity.** The CMOS relation motivates convexity, but it does not derive a pure power law in achieved inference rate for mixed compute/memory workloads. The sentence

> “The functional form is not assumed; it follows from how a GPU turns power into rate.”

is too strong. In addition, 58 of 232 fits hit the imposed \(\beta=8\) boundary. That does **not** justify

> “the true superlinearity there is stronger still.”

A boundary hit can mean weak identifiability, near-saturation, too few points, or model mismatch.

Use language such as:

> “CMOS switching motivates a convex power–rate relationship. We therefore test the compact empirical model \(P=P_0+aR^\beta\) on each fixed workload/card sweep. It fits the measured range well (median \(R^2=0.986\)), but \(\beta\) should be interpreted as a curvature parameter, not a direct device-level voltage–frequency exponent.”

Report leave-one-cap-out error or another complexity-aware fit statistic in addition to \(R^2\), because the nonlinear model has an extra parameter and only about eight cap points per curve.

### 5. Reconcile the small-GPU story; the current interpretation is mathematically inconsistent with Figures 3–4
**Priority: blocking.** The manuscript states:

> “on the small cards [the optimum] lies below the reachable power-cap floor.”

But Figures 3–4 show the L4 energy decreasing as cap/power increases toward TDP, and the T4 roughly flat with its best measured point near the high end. If the same smooth U-shaped curve had its optimum **below** the minimum reachable cap, moving downward toward that floor should reduce energy, not increase it. The observed derivative points the other way.

Section 10 then says a graphics-clock sweep reveals a lower-clock interior optimum. That may be a valid *different actuator path*, but then it cannot be presented as the unobserved continuation of the power-cap curve without evidence.

Required fix:

- Describe the power-cap result exactly as measured: **on T4/L4 the cap sweep does not expose a useful sub-TDP energy minimum**.
- Introduce the clock sweep in Section 4 with ranges, step size, repetitions, stabilization protocol, and dataset schema.
- Plot the clock-sweep result or place it in a table.
- If cap and clock trace the same \((P,R,E)\) curve on A10G, show that overlap explicitly; do not assume the same equivalence on T4/L4.
- Stop calling the high-end cap optimum in Figures 3/5 the “Joule point” if Section 10 claims the true actuator-independent Joule point is elsewhere.

### 6. Fix the measurement protocol around the 191 ms actuation transient
**Priority: blocking for measurement validity.** Section 4 says each operating point is measured over a two-second window. Section 8 says cap actuation takes a median 191 ms and up to 242 ms. Section 10 acknowledges that the first two-second window after a cap change is affected by settling, then asserts that this makes the energy savings a lower bound.

That lower-bound claim is not established: energy is \(P/R\), so the sign depends on how both power and throughput evolve during the transient and on sweep direction.

The clean solution is to recompute all headline values from a steady-state window after discarding at least the measured settling interval (preferably a conservative 300–500 ms), or rerun with a longer post-settle measurement. Report uncertainty across repetitions. A systems reviewer should not have to accept a directional bias argument when the transient can simply be removed from the analysis.

### 7. Fix three explicit internal contradictions before submission
**Priority: blocking and easy to fix.**

1. **Table 3 / A10G:** draw CV falls `0.15 -> 0.08`, but dispersion gain rises `4 -> 11%`. This contradicts both the surrounding statement that saturation collapses dispersion *and the gain* and the caption’s statement that this happens “on the large cards.” Recompute the row, revise the metric, or revise the claim.
2. **Routing example:** the text says a latency budget is “roughly 45 to 120 ms” and then calls an A100 point at “about 158 ms” fast enough. One of these numbers is wrong.
3. **Figure 3 caption:** it says the small cards’ energy-optimal cap is “pinned by the driver power floor,” yet the plotted minimum is at/near the *high-power* end. This is the wrong boundary/direction.

### 8. Scope the headline claims to what ELF actually measures
**Priority: major.** The manuscript measures **GPU board power**, not total server or facility energy. It measures standalone model passes, not a production serving stack. It includes one LLM decode micro-workload and one UNet forward, not end-to-end LLM or diffusion requests.

Therefore replace broad claims such as

> “AI data centers run their GPUs flat out”

and

> “wastes roughly a quarter to a third of the energy behind every inference”

with measured-scope claims: loaded A100/A10G operating points, GPU-board energy per defined work unit, 20 standalone workloads, and the specific batch/stack configuration.

Also distinguish “power limit left at TDP” from “GPU actually draws TDP.” Figure 5 itself shows many workloads whose natural uncapped draw is below TDP.

### 9. Verify and standardize the definition of “inference rate”
**Priority: major and potentially blocking.** Section 4 says

> “throughput is iterations completed over the same window,”

while the paper consistently labels \(R\) as “inferences per second.” At batch 32, an iteration is not necessarily one inference/sample. This matters especially because the manuscript pools or compares different batch sizes.

Check the build script and state explicitly whether \(R\) is:

- forward passes / second,
- samples / second (`batch × iterations/s`),
- tokens / second for LLM decode,
- or a workload-specific work unit / second.

If different batch sizes are compared, use a constant useful-work unit (e.g., samples/s or tokens/s). Otherwise the pooled-batch “same rate, different power” result can partly reflect a changing amount of work per iteration rather than a new energy coordinate.

### 10. Strengthen the fleet experiment so “same work for less energy” is literally true
**Priority: major.** The two controllers use the same admission order, but the capped controller can admit more jobs. Therefore “18–36% less energy per served job” is not automatically the same as “the same work for 18–36% less energy.” Different admitted/completed sets can change the workload mix and average energy.

Report at least two paired metrics:

- **equal-work energy:** energy to complete the same matched set of jobs under both controllers; and
- **equal-budget goodput:** completed SLO-compliant jobs under the same power budget.

Then the paper can cleanly separate energy efficiency from admission headroom.

### 11. Rewrite the economics around total cost, not a binary renter/operator story
**Priority: major.** The per-hour tenant result is correct in a narrow sense: with a fixed GPU-hour price and no energy charge, cost per completion is minimized at maximum throughput. But the claim

> “a provider scheduling its own first-party workloads pays both capex and energy and should target the Joule point directly”

is not generally correct. The Joule point minimizes **energy**, while holding throughput there costs roughly 1.22–1.24× as many cards/card-hours. A provider that owns the hardware also values capacity and capex, so its total-cost optimum is generally above the Joule point unless energy/power has a sufficiently high shadow price.

Write the objective explicitly. If \(c_g\) is amortized GPU cost per second and \(c_e\) is energy cost per joule,

\[
C(p)=\frac{c_g}{R(p)}+c_e\frac{P(p)}{R(p)}.
\]

A power-capacity or carbon shadow price can be folded into \(c_e\). The Joule point is recovered when the energy term dominates; the throughput-maximizing point is recovered when the GPU-time term dominates. This is more rigorous and actually strengthens the paper’s “energy as a priced resource” thesis.

### 12. Cut or radically shorten Section 11 (“energy-to-entropy”)
**Priority: major for focus/tone.** The entropy/Landauer discussion is speculative and disconnected from the evidence. Early exit, conditional computation, and speculative decoding are not all naturally parameterized by entropy reduction, and Landauer’s limit is far removed from the measured system regime. The section makes the paper sound grander but less rigorous.

Replace it with one short future-work paragraph on **joint algorithmic and hardware operating points**: batch size, precision, early exit/speculative decoding, and power/clock control. That is a credible extension of ELF without claiming an information-thermodynamic theory the paper does not develop.

---

## 3. Section-by-section concrete edits

### Abstract

#### A. Opening overgeneralizes both utilization and “waste”
**Original**

> “AI data centers run their GPUs flat out, and that default quietly wastes roughly a quarter to a third of the energy behind every inference.”

**Suggested change**

> “GPU power limits are commonly left at their maximum even when service objectives do not require the maximum per-GPU rate. In our loaded A100 and A10G sweeps, the energy-minimizing operating point reduces GPU-board energy per measured work unit by a median 29% and 31%, respectively, relative to the uncapped point, at roughly a 20% reduction in per-GPU throughput.”

Why: “flat out” conflates configured limit with actual draw; “waste” hides the throughput/capacity trade; “every inference” exceeds the dataset scope.

#### B. Do not call the fitted model a governing law
**Original**

> “from it establish the governing law: board power P grows superlinearly with the achieved inference rate R…”

**Suggested change**

> “Across fixed workload/card cap sweeps, a compact empirical model, \(P(R)=P_0+aR^\beta\), fits the measured power–rate relationship well (median \(R^2=0.986\)).”

Then add one clause explaining that the model is per workload state, not universal across batches/models.

#### C. Replace “modest throughput cost” with the actual number
**Original**

> “capping to it cuts energy per inference by roughly a quarter to a third at a modest throughput cost…”

**Suggested change**

> “On A100/A10G, the measured Joule cap reduces board energy by 29%/31% while lowering per-card throughput by about 20%; maintaining aggregate throughput therefore requires about 1.24×/1.22× the cards under the paper’s fixed-batch scaling model.”

This makes the trade transparent rather than editorializing it as modest.

#### D. Correct the criterion for when a cap helps
**Original**

> “The cap pays wherever a workload draws a GPU’s full rated power; a workload that peaks below it needs no cap.”

**Issue**: false as written. A workload can naturally draw below TDP and still sit above its energy-optimal power. Figure 5 shows exactly such cases.

**Suggested change**

> “A cap changes behavior only when it is set below the workload’s natural draw; it is unnecessary when the uncapped workload already operates at or below its energy-optimal board power.”

#### E. Scope the per-card constant claim
**Original**

> “one fixed cap per card serves every job”

**Suggested change**

> “Across the 20 measured loaded workloads, one per-card cap on A100 and A10G incurs only 0.9% and 0.4% mean energy penalty relative to model-specific cap optima.”

Avoid “every job.”

#### F. Soften the economic conclusion
**Original**

> “the one thing the physics cannot fix is the incentive: paying for GPUs by the hour rewards running them at the energy-worst point.”

**Suggested change**

> “Per-hour GPU pricing can favor the throughput-maximizing point over the energy-minimizing point, creating a split incentive when the tenant does not pay the corresponding power cost.”

This is precise and does not claim economics is the only deployment barrier.

---

### 1. Introduction

#### A. Distinguish a maximum power *limit* from actual maximum draw
**Original**

> “The usual practice is to run each GPU at its full rated power (its thermal design power, TDP)…”

**Suggested change**

> “A common default is to leave the GPU power limit at its maximum. Whether the workload actually reaches that limit depends on the model, batch, and serving state.”

This distinction is foundational to the rest of the paper.

#### B. Fix awkward ordering and scope
**Original**

> “cuts the energy per inference by 31 to 29 per cent…”

**Suggested change**

> “reduces GPU-board energy per work unit by a median 29% on A100 and 31% on A10G…”

#### C. Remove value-laden “good decisions / bad mistakes” language
**Original**

> “In these terms the good decisions become simple calculations and the bad ones become visible mistakes.”

**Suggested change**

> “In these terms, cap selection, marginal power value, and SLO-constrained energy become explicit quantities on the measured response surface.”

#### D. Recast contribution 2
**Original**

> “This is the empirical license for scheduling in energy space rather than hardware space.”

**Suggested change**

> “This shows that hardware identity and achieved rate alone do not determine energy; the operating state must also be represented explicitly.”

That claim is easier to defend and still carries the thesis.

#### E. Recast contribution 5
**Original**

> “identifying the central obstacle to adoption as economic rather than technical.”

**Suggested change**

> “showing that per-hour tenant pricing can favor a higher-throughput, higher-energy operating point, even when a lower-energy point exists.”

Do not claim this is the central obstacle before production-stack validation.

---

### 2. Related work

#### A. P0 is analogous to fixed power, not necessarily “exactly” the classical fixed share
**Original**

> “The fixed floor \(P_0\) in our law is exactly that share measured for modern GPUs…”

**Suggested change**

> “The fitted intercept \(P_0\) plays an analogous role to a rate-independent power component, although in ELF it is inferred from active-run curves rather than independently measured as idle/static board power.”

Unless you separately measure idle/leakage, do not equate a fit intercept with a physical component exactly.

#### B. Do not say the online controller problem is “solved in closed form”
**Original**

> “What these controllers search for online, this paper measures and solves in closed form…”

**Suggested change**

> “ELF complements these online controllers with dense offline characterization and a compact analytic summary of the single-workload power–rate curve, including its energy-minimizing point.”

The fleet scheduling problem, batch selection, queueing, and SLO control are not solved by the scalar closed form.

#### C. “Orthogonal and compose” is too strong
**Original**

> “The two axes are orthogonal and compose…”

**Suggested change**

> “The mechanisms are complementary in principle, but their interaction is not measured here: changing the amount or pattern of computation can also change the power–rate response surface.”

#### D. Remove Landauer from related work unless Section 11 is retained and developed
It currently signals a theoretical scope the paper does not substantiate. The practical comparison should end with adaptive inference as another control dimension.

#### E. Consider replacing prose comparisons among energy datasets with a small table
Columns: dataset, number of models, GPUs, production/standalone workload, power-cap sweep, clock sweep, batch sweep, raw power traces, response-curve density, open data/code. This would make ELF’s novelty easier to assess and reduce claims such as “report energy at whatever operating points a deployment happens to use,” which may be too broad about prior work.

---

### 3. Energy space and its calculus

This section should become the conceptual center of the paper, but it currently contains the most important definitional problem.

#### A. Replace the opening paragraph with a mathematically exact definition
**Original**

> “An execution is a point in energy space: an operating point (GPU g, power cap p) that realizes an achieved inference rate R … and draws board power P; where the cap binds, P=p.”

**Suggested replacement**

> “For a fixed workload state \(w\) (including model, batch, precision, and serving configuration) on GPU \(g\), an actuator setting \(u\) induces measured board power \(P(u)\), rate \(R(u)\), and latency \(L(u)\). We call \((g,u)\), conditional on \(w\), an operating point. In the power-cap experiments \(u=p\); when the cap binds, \(P=p\), while above the workload’s natural draw the cap is inert.”

This accommodates both batch and clock control.

#### B. Use actual-power elasticity
**Original**

> “The compute-power elasticity \(\mathrm{CPE}=d\ln R/d\ln p\)… The minimum sits exactly where CPE falls through one…”

**Suggested replacement**

> “Define the actual-power elasticity \(\epsilon_P=d\ln R/d\ln P\). Since \(E=P/R\), \(d\ln E/d\ln P=1-\epsilon_P\); therefore any interior energy minimum occurs at \(\epsilon_P=1\). When a configured power limit binds so that \(P=p\), the same identity can be written with respect to \(p\).”

This is both shorter and more rigorous.

#### C. “Energy has an interior minimum” needs a scope condition
**Original**

> “Energy per inference E=P/R has an interior minimum…”

**Suggested change**

> “Under a sufficiently convex power–rate response, energy per work unit can have an interior minimum; actuator bounds can instead place the measured optimum at a boundary.”

This is necessary because the small-card cap curves are boundary-limited.

#### D. Define SLO-constrained *energy*, not only least power
Add both \(p_{\mathrm{SLO}}\) and \(p_{E\mid\mathrm{SLO}}\). This one addition will make Sections 7–8 much cleaner.

#### E. Replace the final sentence
**Original**

> “Hardware-space scheduling discards all of them.”

**Suggested change**

> “A hardware-only placement model does not expose these operating-point derivatives; power-aware schedulers can use them directly.”

This acknowledges the prior work cited in Section 2.

---

### 4. The ELF dataset

#### A. Clarify the work unit underlying throughput and energy
**Original**

> “throughput is iterations completed over the same window.”

This is not enough if the paper reports “inferences/s” and compares batches. State the conversion explicitly. If a batch-32 forward pass is one iteration, then report either `iterations/s` and `J/iteration` everywhere, or convert to samples/s and J/sample. For LLM decode, define whether the work unit is a token step per sequence, a batched decode iteration, or tokens/s.

This is especially important to the pooled-batch argument in Section 5.

#### B. Introduce the clock sweeps here or do not rely on them later
Section 10 contains new T4/L4/A10G clock-sweep results, but Section 4 says ELF consists of “two sweep collections and one trace” and describes only power-cap sweeps. Add a fourth collection or revise the schema description.

At minimum report:

- GPUs covered by clock sweep;
- min/max graphics clock and number of levels;
- batch/load state;
- repetitions;
- warm-up and settling period;
- whether voltage remained automatic;
- whether clock and cap were ever active simultaneously;
- row count and release location.

#### C. Explain why there are 232 fitted configurations
With 20 workloads × 4 GPUs × three named sweep states/collections, a reader can naturally expect 240 curves. If eight are missing because of OOM, unsupported caps, failed runs, or absent workload/GPU pairs, state this explicitly.

#### D. Fix the timing protocol
**Original**

> “board power was sampled … over a two-second window per operating point”

combined with 191–242 ms actuation means a large fraction of each window is transient. Add a stabilization delay, use a longer steady-state interval, or recompute from samples after settlement. Report variation over repetitions.

#### E. Clarify the A100 p4d methodology
**Original**

> “a p4d whose eight A100s shared the sweep”

State whether the eight GPUs were swept simultaneously or serially, whether the other GPUs were idle, and whether any node-level power/thermal coupling could affect board behavior.

#### F. Add a compact “measurement scope” table
Per GPU: TDP, minimum power limit, cap levels, clock range (if used), instance type, repetitions, measurement duration, number of valid workload curves. This would eliminate several later ambiguities.

---

### 5. The response surface and its law

#### A. Replace the “derivation” with physical motivation + empirical test
**Original**

> “The functional form is not assumed; it follows from how a GPU turns power into rate.”

**Suggested change**

> “CMOS switching motivates a convex relationship between effective compute rate and dynamic power, but the end-to-end GPU response also depends on utilization, memory behavior, and the device’s internal control policy. We therefore test the empirical family \(P(R)=P_0+aR^\beta\) on each fixed workload/card sweep.”

#### B. Remove or qualify the cubic statement
**Original**

> “where voltage tracks the clock it reaches and can exceed the cubic \(f^3\)”

If \(V\propto f\), the switching term is cubic; exceeding cubic requires a superlinear \(V(f)\) relation and does not follow merely because voltage “tracks” clock. This is unnecessary to the paper. Replace with:

> “As voltage rises with frequency, dynamic power becomes increasingly convex in frequency.”

#### C. Do not identify beta with a card-level physical exponent
**Original**

> “the exponent \(\beta\) is set by the card’s V-f characteristic, so it travels with the card rather than with the model.”

**Suggested change**

> “In ELF, fitted curvature varies more across cards than across models, but it also has substantial within-card spread; we therefore treat \(\beta\) as an empirical curve parameter rather than a device constant.”

This matches your later variance statement and avoids contradiction.

#### D. Correct the boundary-fit interpretation
**Original**

> “58 of 232 fits reach the \(\beta=8\) ceiling … so … the true superlinearity there is stronger still.”

**Suggested change**

> “58 of 232 fits hit the imposed \(\beta=8\) ceiling, so \(\beta\) is not numerically identifiable for those curves under this parameterization. We report them as ceiling-censored and avoid interpreting their fitted exponent literally.”

Also test whether a saturating response model fits these near-plateau curves better.

#### E. Saturation exposes a limitation of the single-valued P(R) notation
**Original**

> “throughput plateaus … so the top of the power range … buys no rate at all.”

If the measured rate is genuinely flat while power changes, then multiple power values correspond to the same \(R\), so \(P(R)\) is not strictly single-valued. This is exactly where a parametric description \((P(p),R(p))\) is safer than treating \(P\) as a global function of \(R\).

#### F. Figure 2 should show the nontrivial case
The current figure proves that *different models* can consume different power at the same numerical throughput. That is unsurprising because the work per “inference” is different.

The stronger and relevant result is the one mentioned only in prose: **same workload + same GPU + same useful-work rate, different batch/cap operating points, different power**. Make that the main panel. Ideally:

- panel A: one model/GPU with two `(batch, cap)` points at nearly equal samples/s but materially different board power;
- panel B: pooled-batch rate-only fit and its poor residual/R².

Then the “rate alone is insufficient” claim becomes genuinely eye-opening rather than tautological.

---

### 6. The Joule point, and the price of reaching it

#### A. The closed-form rate is correct
The derivation

\[
R^*=\left(\frac{P_0}{a(\beta-1)}\right)^{1/\beta}
\]

from \(E(R)=P_0/R+aR^{\beta-1}\) is correct for \(\beta>1\).

Add the equally useful implied board power:

\[
P_J=P(R^*)=\frac{\beta}{\beta-1}P_0.
\]

This is potentially one of the paper’s best analytic observations because `a` cancels. It also provides an immediate internal sanity check for the claim that the Joule point is mostly card-specific.

#### B. Perform that sanity check against the reported summary values
Later the paper says A100 \(P_0/TDP\approx0.19\) and Section 5 gives median \(\beta\approx5.2\). If those numbers refer to the **same loaded curves** as the reported 46%-TDP Joule point, the formula above would imply

\[
P_J/TDP\approx0.19\times\frac{5.2}{4.2}\approx0.235,
\]

not 0.46. This is too large a gap to ignore.

This may simply be a collection mismatch (for example, \(P_0\) summarized on one batch collection and the Joule point on the saturated collection). If so, say so explicitly and report matched \((P_0,\beta,P_J)\) statistics from the same curves. If not, there is a deeper inconsistency between the fitted parameters and the measured optimum.

#### C. Fix the small-card direction error
**Original**

> “on the small cards it lies below the reachable power-cap floor.”

As discussed above, this does not match the direction of the cap-sweep curves. Replace with a statement limited to what is observed:

> “Within the supported power-limit range, T4 and L4 do not show the same useful sub-TDP cap optimum as A100/A10G. Separate clock sweeps probe lower-power operating states and are reported in Section 10.”

Do not infer whether the missing continuation lies below or above the cap range from the current cap plot.

#### D. “The well deepens with beta” needs qualification
Well depth depends on \(P_0\), \(a\), \(\beta\), and the accessible maximum rate/power. Say “larger fitted curvature co-occurs with deeper wells in ELF” rather than making beta the sole cause.

#### E. Make the capacity trade explicit and use consistent notation
**Original**

> “The A100 median is 29 per cent energy for 1.24× cards and latency…”

**Suggested change**

> “On A100, the median energy reduction is 29%, while maintaining the same aggregate rate under the fixed-batch model requires about 1.24× as many cards and increases per-request service time by the same factor; the corresponding A10G values are 31% and 1.22×.”

Also use \(R_{\max}/R_J\), not `Tmax/Teff`, for a throughput ratio; `T` is easily read as time.

The card-multiplier = latency-ratio identity is exact only under the paper’s fixed-work, independent-card model where service time is inversely proportional to measured rate. State that scope.

#### F. Remove “for free”
**Original**

> “One fixed cap per card places it for free.”

**Suggested change**

> “One per-card cap captures nearly all of the model-specific energy optimum on the measured loaded A100/A10G workloads.”

The actual measured penalty (0.9% / 0.4%) is stronger than the rhetorical phrase.

#### G. Table 2 currently overclaims prediction
The `n=4` card-level correlations are exploratory, not evidence that \(P_0\) or \(R_{\max}\) is “predictable without a sweep.” “Read off the hardware” is especially problematic if \(P_0\) is a fitted intercept rather than a vendor field.

Either move Table 2 to an appendix labeled **exploratory correlates**, or provide a clearly defined predictive protocol with held-out cards/workloads. Explain what “63% skill over the per-card mean” means, including metric and cross-validation split.

#### H. Figures 3 and 4 are partly redundant
Both establish the U-shape / card-size contrast. Merge them into a two-panel figure:

- panel A: one representative workload showing raw/normalized U-shapes;
- panel B: median loaded behavior across models.

Use the freed figure slot for the within-workload batch/cap non-uniqueness or clock-sweep validation, both of which are more important to the argument.

---

### 7. Decision I: the policy space

This section has the right role but needs formal definitions.

#### A. Define the three utilities mathematically
**Original**

> “a throughput utility that values every millisecond equally; a soft deadline …; and a hard SLO…”

“Throughput utility” and “values every millisecond equally” are not the same mathematical statement. Give the exact functions \(U(L)\) or \(U(R)\), including normalization.

#### B. State the optimization problem
Replace the loose wording with:

\[
\min_p E(p) \quad \text{s.t.}\quad U(p)\ge u_0.
\]

Then the 95%-utility numbers are directly reproducible.

#### C. Do not call the constrained point “the energy-optimal cap”
**Original**

> “The energy-optimal cap therefore moves with the objective…”

The unconstrained energy optimum is the Joule point and does not move with utility. What moves is the **utility-constrained operating point**.

**Suggested change**

> “At a 95%-utility constraint, the selected operating point moves from about 78% of TDP under proportional-throughput utility to about 57% under the deadline utilities.”

#### D. Give this section a figure or a compact table
The “policy frontier” is conceptually important but currently has no visual support. A single energy-vs-latency curve with the Joule point and three utility-constrained points would make the transition from Section 6 to Section 8 much clearer.

---

### 8. Decision II: allocation, routing, and control

#### A. Fix Table 3 before using “dispersion predicts gain” as a claim
**Original**

> “Driving every job to saturation collapses that spread and the gain with it (Table 3)”

but A10G is:

- draw CV: `0.15 -> 0.08` (less dispersion)
- dispersion gain: `4 -> 11%` (more gain)

That is the opposite of the claim. The caption also says saturation collapses both dispersion and gain “on the large cards,” which is false for A10G as displayed.

Do not explain this away in prose. Recompute and diagnose it. Possible causes include a different objective, nonlinearity in the cap grid, natural-draw truncation, or a metric bug. Until resolved, remove “roughly in proportion to spread.”

#### B. Keep Figure 6, but make its aggregate claim visible
Figure 6 itself shows one ResNet-50 trace, while the text reports median 191 ms and maximum 242 ms over 12 steps on two workloads. Add a small inset/distribution of settling times or state the aggregate numbers in the caption. The figure then supports the claim it is cited for.

#### C. Rename the fleet trace if it is synthetic
The manuscript calls it “trace-driven,” but the described 200-tick workload mix appears to be a seeded constructed trace over measured response curves, not a trace collected from a production service. If so, call it a **seeded controlled simulation** or **synthetic workload trace replay**. “Trace-driven” conventionally suggests an externally observed trace.

#### D. Fix the controller objective
**Original**

> “energy-space controller (cap each admitted job to the least power meeting its SLO)”

Use the SLO-constrained energy optimum defined in Section 3, unless the actual objective is to minimize instantaneous fleet power. If the latter, rename the policy and do not present it as the energy-optimal rule.

#### E. Make “same work” a paired comparison
The text says the capped controller “matches or exceeds” SLO-met throughput and reports energy “per served job.” If the job sets differ, the abstract’s “serves the same work for 18 to 36 per cent less energy” is not established.

Add a paired experiment on the common completed job set and separately report admission gain. Ideally show:

| Metric | Uncapped placement | Operating-point controller |
|---|---:|---:|
| matched jobs completed | same | same |
| energy for matched jobs | baseline | -X% |
| total SLO goodput under same budget | N | N+Y |
| total board energy | ... | ... |

#### F. State which measurement collection drives the simulator
Does the simulator use batch-32 latency/power curves, saturated curves, or a combination? Section 4 says only one collection contains the latency split. State the exact input table and interpolation/nearest-cap rule.

#### G. Figure 7 uses a different SLO from the preceding simulation
The simulation paragraph defines the default SLO as 1.25× best achievable latency. Figure 7 instead uses latency within 10% of full-power latency. That is acceptable, but the prose preceding the figure must explicitly say:

> “For this static sensitivity experiment only, we tighten the SLO to 1.10× the full-power latency.”

Otherwise readers will assume the 8-vs-18 result uses the earlier 25% slack.

#### H. Fix the impossible routing example
**Original**

> “For LLM decode across a wide latency budget (roughly 45 to 120 ms), the greenest option is … the A100 capped near 46 per cent of TDP, which is both fast enough (about 158 ms)…”

A 158 ms point is not feasible for a 120 ms upper bound. Do not rewrite around it; identify which measured number is wrong and regenerate the sentence/figure from the source data.

#### I. Support or remove the 17% warming-drift claim
**Original**

> “uncapped A10G draw drifts by about 17 per cent as the card warms…”

Figure 6 is an actuation trace and does not clearly establish a 17% warm-up drift. If this is a separate measurement, show it or cite a dataset statistic. Otherwise remove it from the main argument.

---

### 9. The obstacle is economic, not technical

The section contains a valuable split-incentive result, but the title and conclusion are too categorical.

#### A. Rename the section
Suggested title:

> **9. Per-hour pricing can favor the energy-inefficient operating point**

or

> **9. Energy and card-hour objectives are misaligned**

#### B. Qualify the “uncapped = cost-optimal” statement
**Original**

> “cost per inference … is minimized at maximum throughput, the uncapped point…”

Maximum-throughput is the right statement. “Uncapped” is not always, because the paper itself reports full-power regressions.

**Suggested change**

> “With a fixed per-hour rental price and no energy charge, cost per work unit is minimized at the throughput-maximizing cap, which is at TDP for the median A100 and A10G workload but can be interior when full power reduces throughput.”

#### C. Do not say the cost-optimal point is always the “least energy-efficient” point
That is true only where the measured energy curve rises toward the throughput-maximizing end. It is not true for every card/workload, especially under the small-card power-cap sweeps.

#### D. Add the total-cost equation
Use

\[
C(p)=\frac{c_g}{R(p)}+c_eE(p)
=\frac{c_g}{R(p)}+c_e\frac{P(p)}{R(p)}.
\]

This single equation makes the section substantially stronger. It shows cleanly why:

- per-hour tenants push toward maximum throughput;
- pure energy minimization gives the Joule point;
- a provider with nonzero hardware opportunity cost generally chooses an intermediate point;
- a power/carbon shadow price shifts the optimum toward the Joule point.

#### E. Correct the first-party provider claim
**Original**

> “a provider scheduling its own first-party workloads pays both capex and energy and should target the Joule point directly”

**Suggested change**

> “A first-party provider internalizes both energy and capacity cost. Its economic optimum therefore depends on their relative shadow prices; the Joule point is optimal only for the energy component, while scarce accelerator capacity pushes the optimum toward higher throughput.”

#### F. Quantify the “cents per hour” argument
State the electricity price assumption, date/source for cloud rental price, and the implied break-even energy or power shadow price. Without this, “too weak to close the gap” is qualitative.

#### G. Figure 8 remains useful, but its caption should not imply AWS price magnitude determines the red points
With pure per-hour billing, the throughput-maximizing cap is independent of whether the hourly price is $2 or $8. AWS price magnitude matters for the *strength* of the incentive and the break-even analysis, not for which cap maximizes throughput.

---

### 10. Limitations

This section currently reads defensively and contains claims that should be corrected in the analysis, not framed as conservative caveats.

#### A. Remove the opening claim that limitations are conservative
**Original**

> “Four choices scope these claims, and each either biases the headline numbers in the conservative direction or carries its own finding.”

**Suggested change**

> “The results are scoped by the following measurement and deployment limitations.”

Then use short paragraphs or bullets. Production-stack absence, privilege constraints, synthetic fleet traces, and board-only power do not all bias estimates conservatively.

#### B. Do not call the settling artifact a lower bound without reanalysis
As noted above, remove the first 300–500 ms and recompute. This belongs in Methods/Results, not as a post hoc direction-of-bias assertion.

#### C. Scope privilege claims
**Original**

> “bare-metal and VM tenants hold [nvidia-smi privileges] … containerized serverless platforms do not”

**Suggested change**

> “The required controls were available in the AWS VM environments used for ELF, but access to power/clock controls varies by provider and tenancy model and is often unavailable in managed or serverless environments.”

#### D. Expand the production-serving limitation
A live production stack adds more than “arrival-process realism.” It can add continuous batching, prefill/decode phase changes, KV-cache pressure, queueing, multi-tenancy, scheduler overhead, CPU/network energy, model parallelism, thermal coupling, and changing natural draw. Remove:

> “what a live deployment adds is arrival-process realism rather than new physics.”

Suggested change:

> “The response-curve mechanism is hardware-grounded, but the stability of the measured Joule point under production serving dynamics remains an empirical question.”

#### E. Add board-energy versus system-energy scope
ELF uses NVML GPU board power. The paper should state explicitly that 29–31% refers to **GPU-board energy per measured work unit**, not whole-server or facility energy. CPU, memory, networking, storage, and cooling/PUE add fixed and variable components that will attenuate or alter facility-level savings.

#### F. Add hardware-generation scope
Four NVIDIA GPUs are useful, but the per-card constancy claim should not be generalized to H100/H200/B100/B200 or AMD accelerators without validation.

#### G. Integrate the clock sweep into the main methods
Do not introduce a major new dataset/result only in Limitations. If it is strong enough to support “the energy sweet spot is thus a property of every card,” it belongs in Methods + Results with a figure/table.

#### H. Soften the last sentence
**Original**

> “The energy sweet spot is thus a property of every card; which actuator reaches it is an implementation detail.”

**Suggested change**

> “Clock sweeps on T4/L4 reveal additional lower-power operating points not reachable through their power-limit interface. Whether cap and clock control trace an equivalent energy surface is actuator- and device-dependent and should be treated as an empirical property rather than assumed.”

---

### 11. The next frontier: a design space of energy-to-entropy operating points

Recommendation: **remove this section from the main paper** or reduce it to 4–5 sentences.

#### A. Current language overreaches
**Original**

> “Each is a knob on energy per unit of entropy reduction, the information-theoretic analogue of energy per inference, with a floor set by the thermodynamics of prediction and ultimately Landauer’s limit.”

Problems:

- many early-exit methods use confidence/margin, not entropy;
- speculative decoding changes accepted-token computation and synchronization, not a monotone “entropy reduction” quantity;
- entropy reduction is not a common comparable utility measure across classification, diffusion, and autoregressive generation;
- Landauer is not operationally relevant at the measured GPU scale.

#### B. Suggested replacement

> “A natural extension is to enlarge the operating point beyond hardware power to algorithmic controls that change the work performed per request, including batch size, precision, early exit, and speculative decoding. These controls can alter the power–rate surface itself, so jointly characterizing algorithmic and hardware operating points is a useful next step. ELF isolates the hardware-power axis; future work should test whether its local regularities persist when the computation is also adaptive.”

That preserves the forward-looking idea without speculative thermodynamics.

---

### 12. Conclusion

The current conclusion is punchy but too absolute.

#### A. Remove “the cost that matters is joules”
Hardware capacity, latency, and card-hours also matter; the paper itself quantifies them.

#### B. Remove “a pile of scheduling heuristics into a single calculation”
Fleet scheduling still includes admission, placement, queueing, SLOs, and workload dynamics.

#### C. Suggested replacement conclusion

> “Across 20 standalone inference workloads and four NVIDIA GPUs, changing the GPU operating power materially changes board energy even when hardware placement is fixed. On the loaded A100 and A10G sweeps, the measured energy minimum clusters near 46% and 43% of TDP and reduces GPU-board energy per work unit by 29% and 31% relative to the uncapped point, at a measurable capacity cost. A compact empirical power–rate model explains the existence of this interior minimum and yields a useful local elasticity for deciding when additional power buys proportional throughput. These results support treating operating power as a first-class scheduling variable alongside GPU placement. Validating the stability of these operating points under production serving stacks and total-cost objectives is the next step.”

This is still strong and memorable, but every clause is traceable to evidence.

---

### Data and code availability

The availability statement is good in structure. Two edits:

1. If clock sweeps support Section 10, explicitly say they are included in the archived ELF release and schema.
2. “Every figure and headline number … is computed … by the accompanying build script” is excellent reproducibility language; retain it, but ensure the script also regenerates the corrected Table 3 and all simulation seeds/configuration.

---

### Figures and tables: keep / change / merge

### Figure 1 — **Keep, with wording changes**
It earns its place as the empirical-fit figure. Change “law” to “empirical response model.” Add a visual indication for ceiling-censored \(\beta=8\) fits. If space permits, add residual or leave-one-cap-out error rather than only \(R^2\).

### Figure 2 — **Redesign; current version is not the strongest evidence**
Cross-model same-throughput/different-power is expected because an “inference” has different work across models. Replace or add a panel showing the same workload/GPU at different batch/cap combinations with nearly equal useful-work rate but different board power. This directly supports the coordinate argument.

### Figures 3 and 4 — **Merge**
They largely make the same point. Correct the small-card interpretation: the current cap sweep shows a boundary-limited/high-end optimum, not evidence that the hidden optimum lies below the cap floor. Use the freed space for clock-sweep validation or the Section 7 utility frontier.

### Figure 5 — **Keep, but scope to what it proves**
This is one of the paper’s strongest figures for A100/A10G. If the small-card true optimum is clock-based, either remove T4/L4 from this plot or label their y-values as “best power-limit setting” rather than actuator-independent Joule points.

### Figure 6 — **Keep if dynamic control is a contribution**
Add the 12-step aggregate settling statistic to the visual/caption. Otherwise it can move to an appendix.

### Figure 7 — **Keep**
It is a useful demonstration that heterogeneity can be induced by per-job operating choices even on identical hardware. Explicitly state the 10%-latency SLO difference from the preceding 25%-slack simulation. Consider plotting the power budget in the conventional increasing left-to-right direction or making the reversed axis visually unmistakable. Explain any shaded region in the caption.

### Figure 8 — **Keep after economics rewrite**
The median gap is visually effective. Separate “which cap maximizes throughput under hourly billing” from the later AWS-price magnitude / break-even argument.

### Table 1 — **Keep**
The displayed A100 values are internally consistent with a median energy reduction of about 29% and a median throughput reduction around 20%. Add a column or footnote with `R_uncapped/R_J` if the 1.24× capacity claim is central. If the full-power regressions matter, give their actual best-throughput cap and gain rather than mentioning them only in the caption/limitations.

### Table 2 — **Demote, redesign, or remove**
The card-level correlations are based on four GPUs and do not justify categorical “predictable without a sweep” conclusions. Label as exploratory or provide held-out prediction.

### Table 3 — **Must be corrected**
The A10G row contradicts the stated dispersion mechanism and the caption. This is a visible reviewer red flag because it undercuts the result the table is meant to support.

---

## 4. Correctness and internal-consistency checklist

| Item | Status | Finding | Required action |
|---|---|---|---|
| Closed-form \(R^*=\left(P_0/[a(\beta-1)]\right)^{1/\beta}\) | **PASS** | Algebra is correct for \(E=P/R\), \(P=P_0+aR^\beta\), \(\beta>1\). | Keep; add conditions and perhaps derive \(P_J=\beta P_0/(\beta-1)\). |
| Claim that an interior minimum exists for \(\beta>1\) | **CONDITIONAL** | True for the unconstrained fitted curve, but the *reachable actuator range* may put the measured optimum at a boundary. | Say “unconstrained/model-implied minimum”; distinguish reachable cap/clock range. |
| CPE definition \(d\ln R/d\ln p\) and minimum at CPE=1 | **CONDITIONAL / CURRENTLY OVERSTATED** | Exact only when configured cap binds and \(P=p\). In inert regions, cap and actual power differ. | Define elasticity using actual \(P\), or explicitly restrict cap-based identity to binding region. |
| Marginal compute \(dR/dp\) as “rate bought by the next watt” | **CONDITIONAL** | A watt of *cap* is not necessarily a watt of actual board power. | Use \(dR/dP\) for physical marginal value. |
| Physical derivation of \(P=P_0+aR^\beta\) | **OVERCLAIM** | CMOS motivates convexity but does not derive a single pure power law in achieved inference rate across mixed workloads. | Recast as empirical model motivated by CMOS. |
| Interpretation of \(P_0\) as static/fixed physical floor | **NOT ESTABLISHED** | \(P_0\) appears to be a fitted intercept, not an independently measured leakage/idle component. | Call it a fitted rate-independent intercept unless independently validated. |
| Interpretation of \(\beta\) as card V–f exponent | **OVERCLAIM** | Within-card spread is wide and 58/232 fits hit the imposed bound. | Treat beta as empirical curvature; report uncertainty/censoring. |
| “58 fits hit beta=8, so true superlinearity is stronger” | **FAIL** | Boundary hits do not prove the true exponent exceeds the bound. | Remove; diagnose identifiability or compare alternative saturating models. |
| Saturation described by single-valued \(P(R)\) | **TENSION** | If R plateaus while P changes, the same R maps to multiple P values. | Use parametric \((P(p),R(p))\) response or qualify the fit region. |
| A100 Joule cap ≈46% TDP | **INTERNALLY CONSISTENT AS REPORTED** | Repeated in Sections 6, Figure 4/5, and economics; Figure 8 gap ≈54 points is consistent with 100−46. | Keep after matched-curve fit sanity check. |
| A10G Joule cap ≈43% TDP | **INTERNALLY CONSISTENT AS REPORTED** | Repeated consistently; Figure 8 gap ≈57 points is consistent with 100−43. | Keep after matched-curve fit sanity check. |
| A100 energy saving ≈29% | **CONSISTENT** | Table 1 median displayed \(\Delta E\) is −29%; rounded energy ratios also give ≈29%. | Keep, scope to GPU-board energy and measured workload state. |
| A10G energy saving ≈31% | **NO INTERNAL CONTRADICTION FOUND** | Repeated consistently, though no per-model A10G table is shown. | Consider supplementary per-model table/CI. |
| A100 1.24× capacity/card-hour penalty | **CONSISTENT WITH DISPLAYED THROUGHPUT LOSS** | Median displayed throughput reduction is about 19.5%; reciprocal ≈1.24. | State fixed-batch/linear-scaling assumptions. |
| “card multiplier equals latency ratio” | **CONDITIONAL** | Holds when latency/service time is inverse of the measured per-card rate for the same work unit and cards scale independently. | State assumptions; avoid presenting as universal end-to-end latency identity. |
| One fixed cap penalty 0.9% A100 / 0.4% A10G | **INTERNALLY CONSISTENT** | Repeated consistently in contributions, Section 6, Table 2. | Scope to the 20 measured loaded workloads. |
| “one fixed cap serves every job” | **OVERGENERALIZED** | Data are 20 standalone workloads, mostly CNNs, with one LLM decode micro-workload. | Replace with measured-set language. |
| “cap only helps workloads that draw full TDP” | **FAIL** | A cap can help whenever natural draw is above the energy-optimal operating power, even if natural draw is below TDP. | Correct criterion in abstract/results. |
| Small-card optimum “below power-cap floor” | **FAIL / CONTRADICTS FIGURES** | Figures 3–4 show energy improving toward higher cap on L4 and roughly flat/high-end on T4. That derivative does not support a hidden optimum below the minimum cap on the same curve. | Rewrite cap result; treat clock sweep as separate actuator path and show it. |
| Figure 3 “energy-optimal cap pinned by driver power floor” | **FAIL** | Plot visually puts small-card best measured cap near the high-power end, not at the lower cap boundary. | Correct caption. |
| Clock-sweep result in Section 10 | **METHODS MISSING** | Section 4 describes two cap-sweep collections + trace, not the clock-sweep protocol used to claim 24%/8% savings on small cards. | Add dataset/methods/figure or remove claim. |
| A100 \(P_0/TDP\approx0.19\), median \(\beta\approx5.2\), Joule cap 46% | **VERIFY COLLECTION MATCH** | If these summarize the same curves, \(P_J=\beta P_0/(\beta-1)\) implies ≈23.5% TDP, not 46%. They may come from different collections. | Report matched-curve statistics and explain collection dependence. |
| 232 fitted configurations | **UNEXPLAINED COUNT** | Natural full cross-product of 20×4×3 states would be 240 if all exist. | State eight missing/invalid configurations and why. |
| Measurement window 2 s vs 191–242 ms settling | **MAJOR VALIDITY ISSUE** | ~10%+ of window can be transient. “Lower bound” direction is not proven for E=P/R. | Remove transient from analysis or rerun steady-state measurement; report uncertainty. |
| Table 3 dispersion mechanism | **FAIL** | A10G CV falls 0.15→0.08 while gain rises 4→11, contradicting text/caption. | Recompute or revise claim/metric. |
| Figure 7 10% SLO vs simulation 25% SLO | **CONSISTENT ONLY IF EXPLICITLY SEPARATED** | Caption states a different SLO, but prose does not flag the change clearly. | State this is a separate tighter-SLO experiment. |
| Routing example 45–120 ms vs 158 ms | **FAIL** | 158 ms cannot satisfy the stated upper latency budget. | Correct source number(s) and regenerate sentence. |
| A10G 17% warm-up drift | **UNSUPPORTED IN MAIN FIGURE/TEXT** | Figure 6 primarily shows cap actuation, not a clear 17% warm-up experiment. | Add evidence or remove. |
| Fleet 18–36% energy saving | **PLAUSIBLE BUT CLAIM FORM NEEDS FIX** | No arithmetic contradiction found; homogeneous A100 result of 26% lies inside range. But “same work” is not proven when admission differs. | Add matched-work comparison and total-goodput metric. |
| “least power meeting SLO” is energy-optimal | **FAIL IN GENERAL** | Below the Joule point, less power can increase energy per completion. | Use SLO-constrained energy argmin or explicitly call it a power/headroom policy. |
| “saving is independent of hardware mix” | **TOO BROAD** | Homogeneous A100 shows the capping mechanism does not require heterogeneity; this does not prove identical magnitude across arbitrary hardware mixes. | Say “the mechanism persists on a homogeneous A100 fleet.” |
| Per-hour billing -> throughput-maximizing cap | **PASS WITH QUALIFICATION** | Correct for fixed hourly price and no energy line item. Throughput max need not be uncapped if full-power regression exists. | Say “throughput-maximizing cap,” then report median at TDP. |
| Provider should target Joule point | **FAIL AS ECONOMIC CLAIM** | Provider also bears capacity/capex opportunity cost; 1.22–1.24× cards matter. | Optimize total cost with GPU-time + energy/shadow-power terms. |
| Figure 8 median gaps 54/57 points | **CONSISTENT** | Matches 100−46 and 100−43 medians. | Keep. |
| AWS price needed to choose red cost-optimal cap | **NO** | Under pure per-hour billing, price magnitude cancels; only throughput matters. | Use AWS price magnitude for incentive strength/break-even analysis, not cap location. |
| Board energy -> data-center energy | **OVERGENERALIZED** | Measurements are NVML board power, excluding host/network/cooling. | Scope all headline savings to GPU-board energy; discuss translation to facility energy. |
| LLM “inference” scope | **OVERGENERALIZED** | ELF includes an autoregressive decode step with prefetched KV, not full request serving. | Say “LLM decode micro-workload” and validate under vLLM/TensorRT-LLM before broad LLM-serving claims. |
| Throughput unit “iterations/s” vs “inferences/s” | **VERIFY / POTENTIALLY BLOCKING** | If batch iterations are not converted to samples/tokens, energy units and pooled-batch comparisons are mislabeled. | Audit build script; define useful-work unit and use it consistently. |
| Section 11 entropy/Landauer argument | **UNSUPPORTED BY PRESENT EVIDENCE** | No entropy metric or experiment connects these controls to the measured surface. | Cut or reduce to conventional future work. |
| Overall claim “economic, not technical obstacle” | **OVERCLAIM** | Production-stack stability, privileges, total-system energy, multi-tenancy, and newer hardware remain technical/open questions. | Reframe as a demonstrated split incentive, not the sole/central obstacle. |