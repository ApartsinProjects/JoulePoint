The overall story is understandable, but I would not submit the figure in its present form. The sequence itself is strong:

**grid stress → imposed facility envelope → facility response → mechanism of reduction → service consequences → recovery/rebound**

The main problem is not aesthetics. There is one potentially serious systems issue in panels 3–4: **the controller appears to curtail much more power than the envelope requires, and the resulting queues/rebound may therefore be partly self-inflicted.**

### (a) Clarity and correctness

| Panel / issue | Assessment | Severity |
|---|---|---:|
| Overall causal sequence | Very good. The shared stress shading makes the 20–40 min intervention visually obvious. | Low |
| Scenario vs measured/simulated provenance | Conceptually correct, but `[S on M]` is cryptic and the scenario curves visually resemble data. | Medium |
| Panel 2 allowance > installed capacity | The normal-state allowance appears ≈102 MW while installed capacity is 100 MW. Technically it can mean “non-binding,” but visually it looks inconsistent. | Medium |
| Panel 3 controlled power ≈22–25 MW under a 45 MW allowance | This is the biggest issue. An envelope is an upper bound, so it is feasible, but it looks highly suboptimal if QoS is being preserved. | **High** |
| Panels 3–4 consistency | They appear internally consistent: baseline ≈70–80 MW minus ≈45–55 MW reduction gives ≈20–30 MW controlled power. Unfortunately, that reinforces the over-curtailment concern. | **High** |
| Post-event jump to ≈100 MW | This is plausible and actually tells the rebound story well: baseline ≈75 MW + ≈25 MW recovery = physical 100 MW limit. | Low |
| Panel 5 protected classes | Critical/interactive queues appear to stay at zero, but the lines effectively disappear on the x-axis. The protection result is therefore visually weak. | Medium |
| Queue length as evidence of protection | Zero queue does **not necessarily mean unchanged latency/SLO**, particularly if GPU power capping slows running requests. | **High scientifically** |
| Panel 6 aggregate backlog | Appears consistent with panel 5: at t≈40, elastic + offline queues sum to roughly the ≈35k aggregate backlog. | Low |
| Panel 6 “recovery power” | The green series has nonzero noisy values before the stress event and after backlog recovery. If this means power specifically used to clear deferred work, that is confusing/inconsistent. | **High/needs definition** |
| Dual y-axis | Understandable, but not ideal; it makes the causal interpretation less immediate. | Medium |
| Electrical boundary | Must be explicit. “Facility MW,” 100-MW allowance, and reductions inferred from measured GPU power have to refer to the same electrical boundary or be converted consistently. | **High scientifically** |

## The biggest issue: why are you running at 22 MW when allowed 45 MW?

During the event:

- uncontrolled load is roughly **70–80 MW**;
- allowance is roughly **45 MW**;
- therefore required reduction is only about **25–35 MW**;
- panel 4 shows GPU power capping alone providing roughly **30–35 MW** of reduction;
- then another roughly **10–20 MW** is obtained by deferring work.

That gives the ≈22–25 MW facility load seen in panel 3.

A systems reviewer is very likely to ask:

> Why defer thousands of requests if GPU power capping already gets you to approximately the requested envelope?

Unless there is a reason not evident from the figure, this makes the controller look inefficient. It also means the large backlog and subsequent rebound are not simply consequences of meeting the grid request; they are consequences of **overshooting the grid request**.

That matters because the scientific story should ideally be:

\[
\text{meet }P(t)\leq A(t)\quad\text{while minimizing service degradation}
\]

rather than merely:

\[
\text{make }P(t)\text{ much smaller than }A(t).
\]

For an e-Energy paper, I would make this a central correctness criterion. A better controller should use approximately:

\[
R_{\mathrm{required}}(t)
=
\max(0,P_\mathrm{uncontrolled}(t)-A(t))
\]

and choose power caps + deferral so that actual reduction stays close to that requirement, perhaps with a small safety margin.

Then panel 4 should contain a dashed line for **“required reduction to meet envelope”**. The stacked achieved reduction should nearly touch that line. This immediately demonstrates controller efficiency.

If the current underfill is unavoidable because GPU caps are discrete, power models are uncertain, or you deliberately reserve a safety margin, say that explicitly and quantify it. A ≈20 MW safety margin on a 45 MW envelope is too large to leave unexplained.

## The rebound itself is not a problem

The post-event rise toward 100 MW is actually useful. It demonstrates the central temporal tradeoff:

**curtailment does not destroy deferred work; it moves energy/work in time.**

What needs to be distinguished is:

- **baseline facility demand** ≈75 MW;
- **additional recovery/rebound power** ≈25 MW;
- **total facility power** ≈100 MW.

Panel 6 should therefore say something like:

**additional recovery power above contemporaneous baseline (MW)**

rather than simply `recovery MW`.

There is also an unexplained feature: the recovery-power trace is noisy and nonzero before minute 20 and again after roughly minute 54, when panel 6 shows no deferred backlog. If that green quantity is actually *available recovery headroom*, *low-priority power*, or something else, rename it. If it is truly backlog-recovery power, I would expect it to be zero when backlog is zero.

---

# Panel-by-panel changes I would make

### Panel 1 — Israeli grid context

The basic idea works, but the smooth demand and PV curves look like measurements even though they are scenario trajectories.

I would label the panel directly:

**(a) Israeli grid context — illustrative scenario, Noga-anchored**

Then distinguish the actual anchor from the constructed profile. For example, place a marker at the real peak:

**Noga anchor: 15.0 GW, 18:49**

and possibly:

**PV at peak: 0.66 GW**

The rest can be dashed or otherwise visually coded as scenario.

A stronger option, if feasible, is to use the actual Noga 60-minute trace. A reviewer will trust that considerably more than a smooth synthetic bell-shaped curve.

Also consider whether **net load** is actually the quantity that motivates the constraint. If PV is only contextual, the green area adds relatively little. If low PV is essential to why the system is stressed, make that relationship explicit.

### Panel 2 — Power envelope

Change the outside-event value from ≈102 MW to exactly **100 MW**.

At present:

- dashed line = installed 100 MW;
- red envelope = ≈102 MW.

Although mathematically an allowance can exceed physical capacity, it creates needless cognitive friction.

I'd use:

**Site capacity: 100 MW**

and:

**Grid-requested maximum: 45 MW, t=20–40 min**

Outside the event, either draw the envelope at 100 MW or omit it entirely.

`IL-2 research envelope` is not very informative to a reader seeing the figure independently. Put `IL-2` in the caption if needed.

### Panel 3 — Facility power

This is probably the most important panel.

Rename the y-axis to something electrically precise:

**site power (MW)**

or

**IT power (MW)**

rather than `facility MW`, depending on what the simulator actually represents.

Make the three important quantities unmistakable:

- uncontrolled baseline: dashed gray/black;
- controlled: solid blue;
- allowance: strong dashed red.

Currently the allowance line is too subtle, and the inline `allowance` text overlaps the controlled trace.

I would also annotate the two major transitions:

**constraint begins**

at minute 20,

and

**constraint released / recovery begins**

at minute 40.

If you fix envelope tracking, this panel should become visually compelling: blue tracks just below the red line during the stress window and reaches 100 MW temporarily during recovery.

### Panel 4 — Reduction mechanism

Change the axis from:

`reduction MW`

to:

**power reduction vs. uncontrolled baseline (MW)**.

Most importantly, add:

**required reduction = uncontrolled − allowance**

as a dashed black line.

Then the reader sees three things immediately:

1. how much reduction was required;
2. how much came from GPU power caps;
3. how much came from workload deferral.

This panel becomes much more scientifically informative than the current stack alone.

It would also make a good quantitative callout:

**GPU capping supplies X% of required reduction; deferral supplies Y%.**

That is a potentially important systems result.

### Panel 5 — Service impact

This is currently the weakest panel for your intended statement **“critical protected, low-priority deferred.”**

Critical and interactive appear to be essentially zero, which means the most important result is invisible.

I would explicitly annotate directly in the panel:

**critical: 0 deferred**  
**interactive: 0 deferred**

Then label the two growing curves directly:

**elastic**  
**offline**

rather than forcing the reader to repeatedly consult the legend.

More importantly, if you want to claim that critical/interactive service is protected, queue depth alone is insufficient. Power capping can increase service time even while queue length remains zero.

For a systems paper, I would ideally show at least one QoS metric such as:

- p95 latency inflation;
- p99 latency;
- deadline/SLO violation rate;
- fraction deferred;
- admission delay.

A very strong panel would say something like:

**Critical / interactive:** p95 latency +1.8%, 0% deferred  
**Elastic / offline:** deferred as necessary

if that is what the simulation produces.

That makes the scheduling policy's intended asymmetry much more convincing.

### Panel 6 — rebound

The scientific idea is good, but you currently repeat information from panel 5: aggregate backlog is essentially the sum of the previous queues.

The useful result here is not merely that backlog exists. It is:

**how large it gets, how long it takes to clear, and what rebound power is required.**

Annotate those quantities directly, for example:

**peak backlog ≈35k requests**

**cleared ≈14 min after release**

**recovery increment ≈25 MW**

Those three numbers make the panel much more memorable.

I would strongly consider eliminating the dual axis. Two alternatives are cleaner:

1. plot backlog only here and show recovery/rebound power as a band in panel 3; or
2. make panel 6 two very short aligned strips: backlog and extra recovery power.

The first is probably best because rebound power naturally belongs with **facility power**.

---

# Visual structure I would use

I would keep the current order. It reads well:

**(a) grid context**  
↓  
**(b) requested power envelope**  
↓  
**(c) achieved facility power**  
↓  
**(d) control mechanism**  
↓  
**(e) class-level service impact**  
↓  
**(f) deferred-work recovery**

But add two cross-panel event markers:

**20 min — grid constraint starts**

**40 min — grid constraint released**

and optionally a third:

**≈54 min — backlog cleared**

A very light second shading from 40 to ≈54 labelled **recovery** would make the complete three-phase story visually immediate:

**normal → constrained → recovery**.

That is more informative than shading only the constrained phase.

---

# Provenance labels

I would remove the repeated tiny `[S on M]` labels. They are not self-explanatory and look like internal notation.

Instead use panel subtitles or one figure-level statement:

**(a–b) scenarios; (c–f) discrete-event simulation driven by production Azure arrivals and measured GPU power-cap elasticity.**

Then, in panel 1:

**Noga-anchored scenario**

and panel 2:

**research envelope scenario**

are enough.

This is both more transparent and visually cleaner.

---

# Colour and accessibility

There are currently several unrelated meanings assigned to the same colours: green represents PV, critical traffic, and recovery power; blue represents controlled facility power, GPU-cap reduction, and interactive requests.

Within individual panels it is understandable, but across six tightly aligned panels it increases cognitive load.

I would use colour semantically:

- black/gray: uncontrolled/reference quantities;
- red: constraint/envelope;
- blue: controlled power / physical control;
- orange: deferred work;
- separate colorblind-safe class colours in the queue panel.

Make the stress window light gray or very pale neutral rather than pink. Then red can consistently mean the actual grid constraint.

Also use line style as well as colour so the figure survives grayscale printing.

---

# Two other details I would fix

There are suspicious vertical strokes at the beginning/end of several traces, especially the allowance at minutes 0 and 60 and controlled facility power at the boundaries. They look like plotting artefacts rather than system behavior. Remove them; otherwise readers may think the system transitions at t=0 and t=60.

The large figure title is also visibly clipped. For the paper version I would remove the in-figure title completely and let the ACM figure caption carry the narrative. Use only `(a)`–`(f)` panel headings inside the figure.

## Bottom line

The **conceptual figure is strong**, and the top-to-bottom story is much better than a collection of unrelated performance plots. But I would resolve three things before using it as a flagship figure:

1. **Fix or explain the ~20 MW under-utilization during the 45 MW constraint.** As drawn, the scheduler seems to create unnecessary service degradation and rebound.
2. **Demonstrate “critical protected” with an actual QoS/SLO measure**, not only zero queue length.
3. **Define recovery power precisely and eliminate the inexplicable pre/post-event recovery signal.**

If those are fixed, the figure can make a quite clean systems argument: **an externally imposed Israeli grid envelope is translated into measurable compute controls, priorities determine who absorbs the service cost, and the figure exposes both the immediate curtailment and the delayed rebound rather than pretending that curtailed work disappears.** memcite