## 1. Novelty

The **ANOVA/additive result is not a contribution by itself**. A reviewer will correctly say that
\[
E_{ij}=\mu+r_i+c_j
\]
implies \(\arg\min_j E_{ij}=\arg\min_j c_j\), independent of \(i\). Calling this a theorem will not help; present it as an elementary observation that exposes a flaw in variance-based evaluation.

Likewise, matrix completion, inductive MF, CF, etc. provide no methodological novelty.

What **is potentially publishable** is the empirical/conceptual result:

> **Prediction difficulty and decision value are different quantities. Across seven heterogeneous AI-energy corpora, nearly all energy variation is decision-invariant; the residual interaction is real enough to reverse hardware rankings, yet the oracle value of exploiting it is usually only ≈2%.**

That is stronger than simply saying “our recommender did not help.” It says **even a perfect recommender could hardly help**, and demonstrates it repeatedly across independent datasets. This is directly aligned with the decision-focused-learning argument that predictive accuracy can be badly misaligned with downstream decision quality. 

I would therefore characterize it as a **negative measurement result with a useful general methodological lesson**, not a new ML method. It becomes a null result dressed up if the paper mainly says “interaction variance is small.” It becomes publishable if the paper establishes:

1. high \(R^2\) is almost meaningless for this placement decision;
2. interaction can be statistically/structurally real while its **economic decision value is tiny**;
3. the oracle ceiling is consistently small over seven substantially different corpora;
4. the apparent interaction is traced experimentally to configuration rather than merely observed;
5. this changes system-design priorities: composition/configuration before sophisticated placement.

The reviewer attack I would worry about most is **scope**: your claim is not that scheduling generally has ≤2% value. Queueing, contention, batching, SLOs, power states, concurrency, locality, network cost, carbon intensity, etc. can create much larger scheduling opportunities. Your ceiling applies to **per-workload hardware selection under your measured action set and energy objective**. Overstate “ANY placement method” and a systems reviewer will reject it.

**Venue tier:** I would target a strong specialist measurement/energy venue. ACM e-Energy is a natural fit; SIGMETRICS/POMACS is plausible if the measurement methodology and decision decomposition are made sufficiently general. Both explicitly cover energy/performance measurement and computing systems.  MLSys is possible but harder: its scope fits AI-system efficiency and benchmarking, but without a system or methodological advance I would expect substantial resistance. 

**NeurIPS/ICML/ICLR:** I would expect rejection as a main-track ML contribution.  
**OSDI/SOSP/NSDI:** I would expect rejection without an actual system contribution.

So: **good specialist-paper material; not currently a flagship-ML methodological contribution.**

---

## 2. Prior art: the ceiling already has a close standard concept

Your quantity has a very clean decision-analysis interpretation.

Let
\[
C_{\rm fixed}=\min_j E_i[C_{ij}]
\]
and
\[
C_{\rm oracle}=E_i[\min_j C_{ij}].
\]

Then
\[
C_{\rm fixed}-C_{\rm oracle}
\]

is essentially the **Expected Value of Perfect Information (EVPI)** in a cost-minimization formulation: how much perfect knowledge of the state before acting could possibly improve the decision. EVPI is explicitly treated as an **upper bound on the value of imperfect information** in classical decision analysis. See Merkhofer, *The Value of Information Given Decision Flexibility*, Management Science, 1977, and later decision-analysis literature. 

So I would **not introduce HEADROOM as though the underlying concept were new**.

Your particular normalization,
\[
H={C_{\rm fixed}\over C_{\rm oracle}}-1
={\mathrm{EVPI}\over C_{\rm oracle}},
\]
does not, as far as I can establish, have a universally standard name. I would call it something transparent such as **relative oracle gap** or **normalized EVPI**, while explicitly defining the denominator.

There is an especially useful existing concept for evaluating a learned placement method. Bickel (2008), *The Relationship Between Perfect and Imperfect Information in a Two-Action Risk-Sensitive Problem*, Decision Analysis 5(3):116–128, explicitly studies the **relative value of imperfect information (RVOI)**—the value obtained from imperfect information relative to the value of perfect information. 

For your policy \(f\), define
\[
C_f=E_i[C_{i,f(i)}].
\]

Then there is a trivial but very informative decomposition:
\[
\underbrace{C_{\rm fixed}-C_{\rm oracle}}_{\text{available decision value}}
=
\underbrace{C_{\rm fixed}-C_f}_{\text{value captured}}
+
\underbrace{C_f-C_{\rm oracle}}_{\text{oracle regret}}.
\]

Therefore
\[
\boxed{
\text{fraction of achievable value captured}
=
{C_{\rm fixed}-C_f\over C_{\rm fixed}-C_{\rm oracle}}
}
\]

is probably the metric I would emphasize. A model saving 1% sounds poor or good depending on context; if EVPI is 1.1%, it has captured ~91% of everything available.

This sits squarely inside established **predict-then-optimize / decision-focused learning**. Donti, Amos & Kolter (NeurIPS 2017) optimize the downstream task rather than predictive loss; Wilder, Dilkina & Tambe (AAAI 2019) formalize decision-focused learning; Elmachtoub & Grigas, *Smart “Predict, then Optimize”*, Management Science 68(1):9–26 (2022), define SPO loss in terms of induced decision error rather than ordinary prediction error. 

Contextual-bandit work likewise evaluates **regret against an optimal policy** rather than prediction RMSE—for example Agarwal et al., *Taming the Monster*, ICML 2014.  Off-policy evaluation/CRM is related but less directly relevant to your contribution: Li et al. evaluate contextual-bandit policies from logged data, and Swaminathan & Joachims optimize counterfactual policy risk. Your full energy matrix means you do not have their fundamental missing-counterfactual problem. 

So my terminology recommendation is:

**absolute ceiling:** EVPI / oracle advantage  
**your percentage:** relative oracle gap or normalized EVPI  
**model quality:** fraction of EVPI captured / normalized oracle regret

**Uncertain:** I do not find evidence that `EVPI/C_oracle` itself has one accepted standard name. I would not claim that “normalized EVPI” is standardized.

One numerical issue: your definition is not percentage energy saving. If \(H=2\%\), the oracle saves
\[
1-{1\over1.02}=1.96\%
\]
relative to the fixed policy. Minor, but reviewers may notice.

---

## 3. What to measure next

The highest-value experiment is **not another placement algorithm**. It is a **controlled full-factorial configuration experiment**:

\[
\text{workload}\times\text{hardware}\times
\{\text{precision, quantization, batch, sequence/input size}\}.
\]

Measure the **same workloads on the same hardware under matched configurations**, with repeated energy measurements.

This gives you three increasingly broad oracle ceilings:

\[
H_{\rm hardware\mid matched\ config}
\]

for pure hardware placement;

\[
H_{\rm hardware+configuration}
\]

when the decision can jointly choose device and execution configuration; and, separately, your already-large fleet-composition opportunity.

This experiment matters because your statement that **configuration causes the interaction** is currently much more consequential than the CF result. Unless configuration was experimentally manipulated rather than merely observed, a reviewer can say configuration and hardware are confounded. A factorial intervention turns that from an interpretation into evidence.

If you deliberately want to **maximize headroom**, sample the regimes where crossovers should occur: FP32/TF32/BF16/FP16/INT8/INT4, small versus saturation batch sizes, short versus long sequences, memory-bound versus compute-bound models, and hardware generations with genuinely different low-precision support. Enforce equal accuracy/quality/SLO constraints; otherwise the “energy optimum” is not a legitimate decision comparison.

If even this deliberately interaction-rich experiment produces, say, a <5% upper confidence bound for pure hardware-placement EVPI, that is substantially stronger evidence than seven more naturally occurring corpora.

But you cannot prove that headroom **cannot ever** be raised. Someone can always change the workload distribution, hardware set or action space. The defensible claim is narrower:

> Across representative corpora, and even under a prospective stress test deliberately designed to maximize workload×hardware interaction, the value of perfect workload-specific hardware selection remains small.

That would make the negative result much harder to dismiss.

The most important next figure, in my view, would therefore be **headroom as the action space expands**:

\[
\text{fixed hardware}
\rightarrow
\text{workload-aware hardware}
\rightarrow
\text{hardware+configuration}
\rightarrow
\text{fleet composition}.
\]

Given your existing ~2% versus ~34% numbers, that hierarchy is potentially the paper's strongest result—not the prediction model. memcite