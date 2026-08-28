# Pending register

Living record of every outstanding task, open question, known defect and unexplored direction.
Rewritten 19 August 2026 after a full session audit.

**Paper state:** 13,209 body words · 84 references, all resolving under bibtest · 5 figures ·
7 tables · 6 corpora in Table 2 · Sections 1–10 with 2.1–2.9, 8.1–8.2.

---

## 1. Running now

| ID | Item | Status |
|---|---|---|
| R1 | G2 refresh timing, G4 multi-tenant cost sharing, G5 minimum sufficient descriptor, G6 utilisation as a carbon lever | agent running |
| R2 | **DONE 19 Aug, folded as Sections 7.3 and 7.4 with Tables 8 and 9.** CF model fitted on all six corpora. The predictive model has only ever been fitted on two (our grid, MLPerf); the other four were used only for variance decomposition. Grid'5000 at 5x18 is the right corpus for a serious cold-start-on-machines test | agent running |

## 2. Ready to fold into the paper, results in hand

| ID | Item |
|---|---|
| P1 | **DONE 19 Aug.** B3/B4 folded into Section 8.1. Recipes 2 and 3 are complete (27/28 and 55/60 checks) but the paper still says "only the first is implemented here". Headlines: a few hundred log lines suffice, and below ~250 the dominant failure is an under-provisioned fleet missing the SLA rather than wasted energy; an uninformed uniform prior costs 11.1% (vision) to 33.4% (fp16); probing beats extra passive logging only once logging noise approaches the fp16/fp32 runtime separation, which is exactly Section 7's precision claim |
| P2 | **DONE 19 Aug.** CAPLAI cited in Section 2.8 as [85]. Reviewed in full, `references/CAPLAI_review.md`. Nie et al., e-Energy '25: stochastic optimisation of GPU purchase, retirement and cooling retrofit under demand uncertainty, 27–40% lower lifecycle cost. Closest AI-systems work to our formulation; must be cited. Its demand constraint gives each GPU type one scalar performance figure and prices energy as rated power x hours, so by Proposition 1 it is exactly a fixed ranking — it illustrates the gap rather than closing it |

## 3. Open, free

| ID | Item | Note |
|---|---|---|
| F1 | DeepEn2023 kernel tier | **Now accessible.** The Drive folder opens via `embeddedfolderview`; file IDs captured for Kernel_energy, Kernel_latency, Kernel_power, Predictor, benchmark_model. Still: no licence (repo `license: null`), and kernel granularity is operator-level, so it cannot be a Table 2 row. Could support a narrower mechanism claim — does the interaction structure hold at operator level across 8 SoCs |
| F2 | Second reviewer pass on the whole manuscript now that it has roughly doubled | `paper-reviewer` skill |
| F3 | Regenerate DOCX/PDF deliverables from the current HTML | `paper-build` skill |

## 4. Open, paid (~$25 of $30 remaining on the `apartsin` Modal workspace; C1+C4+T4 cost about $5)

| ID | Item | Cost | Verdict |
|---|---|---|---|
| C2 | Second provider (RunPod / Vast) | ~$10 | **Worth it, for a reason that changed.** Not mainly domain shift: RunPod gives root on a dedicated pod, so `nvidia-smi -pl` may work where Modal denies it. That would unblock E1, the allocation-amount axis (power caps, clocks), which the paper currently cites from others rather than measuring. Secondary benefit: the T4 card-variance result predicts a different card population on a different provider, which is directly testable |
| C3 | Co-location pairs | $10–20 | Still likely redundant against Han et al.; lowest priority |

## 5. Blocked, not actionable

| ID | Item | Why |
|---|---|---|
| E1 | Power caps, clocks, MIG on Modal | **Definitively blocked.** Denied on all seven machine types including H100 and H200, on a second workspace. A Modal serverless platform restriction, not an account or GPU-generation effect. Only route is C2 |
| E2 | Watt Counts (50 LLMs x 10 GPUs) | Paper public and cited as [68]; data releases "upon acceptance". Poll the GitHub |
| E3 | Exact-part MLPerf transfer test | **Permanently partial from our side.** MLPerf has no power-measured resnet or bert submission for the H200 at all, and no resnet for the 40 GB A100. No spending on our side closes it |

## 6. User decisions outstanding

| ID | Item |
|---|---|
| U1 | **Venue choice — DEFERRED at the user's request; they will say when.** Gates the 250-word abstract (currently 317) and how much MRTA and queueing vocabulary survives Sections 2.6 and 3. Do not raise again until asked |
| U2 | Whether to rotate the Modal token that was pasted into the session transcript |

## 7. Unexplored directions still open

G2, G4, G5, G6 are with an agent (R1). Remaining: cross-provider card-population comparison
(follows from the T4 result); performance-degradation-over-time as an uncertainty axis, which
CAPLAI models and we do not.

---

## Completed this session

**Measurement (Modal, `apartsin` workspace, about $5 total).**
C1 cross-corpus bridge: 210/210 cells, 7 accelerators, added H100, H200 and BERT-Large.
C4 training grid: 136/140 cells, 5 workloads in training mode on 7 accelerators over a 10x
power-limit range. T4 replicates: 8 independent containers, 8 distinct physical cards.

**Findings that changed the paper.**
- F3 broken: the training grid has two winners (L4 13/18, H200 5/18), not one; fastest is
  cheapest in only 5 of 18.
- Cross-run reproducibility: median 2.4%, ordering preserved on 96.7% of within-row pairs.
- **T4 card-to-card variance 10.2% median CV, 27% between the thirstiest and leanest of 8
  cards**, against 1.6% within-run. Within-run replicate noise substantially understates
  reproducibility on rented cloud hardware — a general methodological point now in Section 9.
- Figure 2's reversal independently replicated on different physical cards, all six configurations.
- E1 closed as definitively blocked.

**Corpora: 3 → 6.** Added ejhusom (2.49%), Wilkins (2.66%), Grid'5000 (2.16%, 5x18 dense,
joules direct, CC-BY-4.0, and CPU/HPC so the structure is not specific to AI accelerators).
Proposition 1 holds exactly on all six. DeepEn2023 assessed and rejected.

**Analysis.** B5 Shapley under an SLA sweep (24/24): marginal value depends on the contract, four
distinct type orderings across six SLA levels. B8 two-stage stochastic program (5/5): LP prunes
and the simulator picks, recovering the enumeration optimum from 25 simulations not 1001 and
scaling to 100 slots; VSS 21.3%, EVPI 16.8%. B3/B4 (see P1). G1: the carbon-optimal fleet is the
same mixed fleet from 20 to 700 gCO2e/kWh, saving 34.6–39.9%. G3, new Section 8.2: composition
37.3%, spatial placement 0.70%, temporal shifting 0.04% — the ordering sharpens under a carbon
objective, and an energy-chosen fleet captures 37.29 of the 37.32 points a carbon-chosen fleet
gets, so energy is a near-perfect proxy.

**Corrections made rather than defended.**
- Novelty repositioned against del Rey (interaction <3% already reported), EcoServe (already
  right-sizes heterogeneous GPUs under SLOs), Wilkins (already measured ~7.5% from heterogeneous
  routing), Zine (already studies configuration interactions). Section 2.5's "no work makes
  purchase the decision variable" claim **withdrawn**.
- B8's formulation grounded in stochastic programming and **not claimed as novel**: Swaminathan
  (EJOR 2000) is structurally identical in semiconductor tool procurement.
- Section 6's "maximum over configurations is precisely the operation that removes the
  interaction" **corrected**: measured on our own grid it removes about a third (1.20% → 0.80%),
  not all.
- A100 memory bandwidth corrected from 1555 to 2039 GB/s (the 80 GB part), verified across all
  401 corpus rows.
- Section 8's single-seed point estimates now carry seed spread (34.5%, sd 0.4, fleet identical
  in 12/12 seeds); the 1.9% simulator gap traced to one line of arrival sampling.
- B8 docstring claimed EVPI must bound VSS; there is no such ordering.

**My own checks that were wrong and were fixed, not weakened:** the permutation null for an
interaction (invalid whenever the column main effect is large, not just at two columns); OOM
monotonicity scored in the wrong direction; an additive-policy check comparing a value with
itself; interaction-residual correlation on a 2x2 where it is forced to ±1; a
configuration-collapse test run at a size where both quantities are near zero.

**Infrastructure.** All 43 filesystem skills converted from bash symlinks to NTFS junctions and
now discovered. Modal `apartsin` profile configured alongside `llmcourse`. Secrets ignore rules
added to both repos, verified no secret tracked. 51 result files and raw run logs preserved and
pushed.
