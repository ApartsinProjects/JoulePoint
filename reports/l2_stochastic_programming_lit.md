# L2. Citation scaffolding for the B8 two-stage stochastic program

Scope: references to ground `experiments/b8_stochastic_program.py` (fleet counts chosen first stage,
LP routing recourse in the fluid limit, outsourcing penalty, VSS = 21.3%, EVPI = 16.8%) in the
paper's Section 2. Every DOI below was resolved through Crossref via `bibtest check-doi`; titles and
first authors match what is claimed. Anything not verified is isolated in the last section.

## 1. Novelty verdict (read first)

**The framing is safe as a domain instantiation. It is NOT safe as a new formulation.**

- No verified work poses *accelerator/GPU procurement* as a two-stage stochastic program with the
  **workload mix** as the scenario and **routing across the purchased fleet** as recourse. The
  cell is open in the AI-systems literature.
- But the identical OR structure is 25 years old in **semiconductor tool procurement**. Swaminathan
  (EJOR 2000) [Swam00] places tool orders before demand is realised and, once demand is known,
  assigns wafer types to tools, "formulated as a stochastic integer program with recourse". Swap
  "tool type" for "accelerator type" and "wafer type" for "workload cell" and it is our model.
  Barahona, Bermon, Günlük and Hood [BBGH05] are the robust-optimisation counterpart in the same
  domain.
- The generic combinatorial precedent is the stochastic server location problem [NS05]: binary
  first stage (which servers to install), assignment recourse after demand realises.

**Recommended positioning.** Do not write "we formulate fleet design as a two-stage stochastic
program" as though the formulation were the contribution. Write that fleet design *is* a two-stage
stochastic program in the classical sense, cite [D55, B55, BL11] for the shape and [Swam00] for the
isomorphic procurement instance, and claim only (a) the instantiation on an accelerator-by-workload
**energy** matrix rather than a cost or shortage objective, and (b) the quantification: 21.3% VSS
and 16.8% EVPI say how much of the energy saving is attributable to planning on the mix
distribution versus to a perfect mix forecast. That claim survives every item found.

Nearest AI-systems neighbours, all short of a direct hit:

| Work | Why it is not a hit |
|---|---|
| Mélange, arXiv:2404.14527 (Griggs et al., 2024) | Same structure (how many of each GPU type, which slice runs where) but a single-shot deterministic cost ILP on a known workload distribution. No scenario set, no expectation, no recourse. |
| Chaisiri, Lee, Niyato 2012 [CLN12]; Bülbül, Noyan, Erol 2021 [BNE21] | Genuine two-stage/multi-stage SPs, but the first stage is a *rental commitment* (reserved instances), the uncertainty is aggregate demand and price, and the recourse is buying on-demand top-up rather than routing across a heterogeneous fleet. Cost objective. |
| CAPLAI / "AI-assisted Stochastic Optimization for GPU Data Centers Lifecycle Planning", ACM e-Energy 2025, 10.1145/3679240.3735099 (Nie, Xing, Latif, Liu) | Stochastic optimisation over GPU *upgrade and retirement timing* under demand-growth, decay, price and resale scenarios. Four-page short paper (pp. 870-873). Closest AI-systems item; **model details not verified** (see section 4). |
| Rearchitecting Datacenter Lifecycle for AI, arXiv:2509.26534 (Stojkovic et al., 2025) | Fleet composition under uncertainty, but by Monte Carlo simulation of policies, not an optimisation model with recourse. |
| arXiv:2604.07472 (Cheng and Nguyen, 2026) | Joint GPU provisioning, model selection and routing for SLO-constrained LLM inference, but one deterministic stage with provisioned headroom. |

## 2. Recommended prose for Section 2

> Recipe 3 is a two-stage stochastic program in the classical sense of Dantzig and Beale [D55, B55]:
> the fleet counts are a here-and-now first-stage decision and routing work across the purchased
> accelerators, once a mix scenario is revealed, is the recourse [BL11]. Pricing off-fleet work at a
> penalty makes the recourse relatively complete, so every fleet has a finite second-stage value and
> the objective is defined across the whole first-stage lattice [Wets74, BL11]. We report the two
> standard diagnostics for such a model, the value of the stochastic solution [Birge82] and the
> expected value of perfect information [Mad60]. Posing capacity as a first-stage investment with
> per-scenario allocation recourse is the canonical shape in capacity planning under demand
> uncertainty [EMS89, VM03]; it is the same structure Swaminathan uses for semiconductor tool
> procurement, where tool orders precede demand and wafer types are assigned to tools afterwards
> [Swam00], and it has been carried into cloud resource provisioning as a reservation-versus-on-demand
> decision [CLN12, BNE21]. What differs here is the content rather than the form: the uncertainty is
> the workload *mix* across an accelerator-by-workload energy matrix, and the objective is energy, so
> the VSS and EVPI we report measure how much of the attainable energy saving is bought by planning
> on the mix distribution and how much would additionally be bought by a perfect mix forecast. The
> first stage is integer over a composition lattice and each recourse evaluation is a small LP, so we
> use steepest-descent local search over one-slot swaps rather than L-shaped or Benders decomposition
> [VSW69, Ben62], which target continuous first-stage variables.

Optional sentence answering the robust-optimisation reviewer:

> We adopt a stochastic rather than a robust formulation because the scenario set is explicit, small
> and carries meaningful probabilities, and the operator's objective is expected energy over the
> fleet's service life; a robust or distributionally robust formulation [BBC11, MEK18], including the
> two-stage adjustable form that most closely mirrors our recourse [BTGGN04], would instead optimise a
> worst case over an ambiguity set, which answers a different question.

## 3. Proposed references

| Key | Authors | Title | Venue / Year | DOI / ID | Verified | Supports |
|---|---|---|---|---|---|---|
| D55 | George B. Dantzig | Linear Programming under Uncertainty | Management Science 1(3-4):197-206, 1955 | 10.1287/mnsc.1.3-4.197 | YES | Originating paper for the two-stage recourse formulation |
| B55 | E. M. L. Beale | On Minimizing a Convex Function Subject to Linear Inequalities | J. Royal Statistical Society B 17(2):173-184, 1955 | 10.1111/j.2517-6161.1955.tb00191.x | YES | Independent co-origin; cite alongside Dantzig |
| BL11 | John R. Birge, François Louveaux | Introduction to Stochastic Programming, 2nd edition | Springer Series in Operations Research and Financial Engineering, Springer New York, 2011 | 10.1007/978-1-4614-0237-4 | YES | Canonical textbook: two-stage recourse, relatively complete recourse, VSS and EVPI definitions and non-negativity. **2nd edition is the one to cite** |
| SDR21 | Alexander Shapiro, Darinka Dentcheva, Andrzej Ruszczyński | Lectures on Stochastic Programming: Modeling and Theory, 3rd edition | SIAM, 2021 | 10.1137/1.9781611976595 | YES | Second textbook anchor. (1st ed. 2009 is 10.1137/1.9780898718751, also verified) |
| Wets74 | Roger J.-B. Wets | Stochastic Programs with Fixed Recourse: The Equivalent Deterministic Program | SIAM Review 16(3):309-339, 1974 | 10.1137/1016053 | YES | Primary source for fixed / relatively complete recourse and the deterministic equivalent |
| Birge82 | John R. Birge | The value of the stochastic solution in stochastic linear programs with fixed recourse | Mathematical Programming 24:314-325, 1982 | 10.1007/BF01585113 | YES | Originating paper for VSS; the reference for VSS = 21.3% |
| Mad60 | Albert Madansky | Inequalities for Stochastic Linear Programming Problems | Management Science 6(2):197-204, 1960 | 10.1287/mnsc.6.2.197 | YES | Wait-and-see versus here-and-now bound underlying EVPI; the reference for EVPI = 16.8% |
| VSW69 | R. M. Van Slyke, Roger Wets | L-Shaped Linear Programs with Applications to Optimal Control and Stochastic Programming | SIAM J. Applied Mathematics 17(4):638-663, 1969 | 10.1137/0117061 | YES | The L-shaped method; the "why not decomposition" answer |
| Ben62 | J. F. Benders | Partitioning procedures for solving mixed-variables programming problems | Numerische Mathematik 4:238-252, 1962 | 10.1007/BF01386316 | YES | Benders decomposition, of which L-shaped is the stochastic specialisation |
| EMS89 | Gary D. Eppen, R. Kipp Martin, Linus Schrage | OR Practice — A Scenario Approach to Capacity Planning | Operations Research 37(4):517-527, 1989 | 10.1287/opre.37.4.517 | YES | Canonical scenario-based capacity planning: first-stage capacity, second-stage production allocation. Keep the "OR Practice —" prefix |
| VM03 | Jan A. Van Mieghem | Commissioned Paper: Capacity Management, Investment, and Hedging: Review and Recent Developments | Manufacturing & Service Operations Management 5(4):269-302, 2003 | 10.1287/msom.5.4.269.24882 | YES | Survey establishing first-stage capacity / second-stage allocation as the standard shape. Keep the "Commissioned Paper:" prefix |
| Swam00 | Jayashankar M. Swaminathan | Tool capacity planning for semiconductor fabrication facilities under demand uncertainty | European Journal of Operational Research 120(3):545-558, 2000 | 10.1016/S0377-2217(98)00389-0 | YES | **The defensive citation.** Isomorphic prior formulation: buy tool counts before demand, assign wafer types to tools as recourse, stochastic integer program with recourse |
| BBGH05 | Francisco Barahona, Stuart Bermon, Oktay Günlük, Sarah Hood | Robust capacity planning in semiconductor manufacturing | Naval Research Logistics 52(5):459-468, 2005 | 10.1002/nav.20086 | YES | Robust counterpart of the same tool-procurement problem; useful in the robust-versus-stochastic sentence |
| NS05 | Lewis Ntaimo, Suvrajeet Sen | The Million-Variable "March" for Stochastic Combinatorial Optimization | Journal of Global Optimization 32(3):385-400, 2005 | 10.1007/s10898-004-5910-6 | YES | Stochastic server location problem: binary first stage, assignment recourse. Generic structural precedent. Optional |
| CLN12 | Sivadon Chaisiri, Bu-Sung Lee, Dusit Niyato | Optimization of Resource Provisioning Cost in Cloud Computing | IEEE Trans. Services Computing 5(2):164-177, 2012 | 10.1109/TSC.2011.7 | YES | The standard cloud two-stage SP: reserve instances first stage, on-demand recourse. Cost, not energy; homogeneous VMs |
| BNE21 | Kerem Bülbül, Nilay Noyan, Hazal Erol | Multi-stage stochastic programming models for provisioning cloud computing resources | European Journal of Operational Research 288(3):886-901, 2021 | 10.1016/j.ejor.2020.06.027 | YES | Recent multi-stage version of the same cloud provisioning problem |
| FF90 | Charles H. Fine, Robert M. Freund | Optimal Investment in Product-Flexible Manufacturing Capacity | Management Science 36(4):449-466, 1990 | 10.1287/mnsc.36.4.449 | YES | Classical analogue: buy a *mix* of dedicated versus flexible capacity before the demand scenario is known. Optional |
| Luss82 | Hanan Luss | Operations Research and Capacity Expansion Problems: A Survey | Operations Research 30(5):907-947, 1982 | 10.1287/opre.30.5.907 | YES | Older capacity-expansion survey. Optional |
| BTEN09 | Aharon Ben-Tal, Laurent El Ghaoui, Arkadi Nemirovski | Robust Optimization | Princeton University Press, 2009 | 10.1515/9781400831050 | YES | Robust optimisation textbook anchor |
| BBC11 | Dimitris Bertsimas, David B. Brown, Constantine Caramanis | Theory and Applications of Robust Optimization | SIAM Review 53(3):464-501, 2011 | 10.1137/080734510 | YES | Survey; the one-reference answer to "why not robust optimisation" |
| BTGGN04 | A. Ben-Tal, A. Goryashko, E. Guslitzer, A. Nemirovski | Adjustable robust solutions of uncertain linear programs | Mathematical Programming 99:351-376, 2004 | 10.1007/s10107-003-0454-y | YES | The robust counterpart of two-stage recourse specifically; sharpest comparison point |
| MEK18 | Peyman Mohajerin Esfahani, Daniel Kuhn | Data-driven distributionally robust optimization using the Wasserstein metric: performance guarantees and tractable reformulations | Mathematical Programming 171:115-166, 2018 | 10.1007/s10107-017-1172-1 | YES | DRO built on a finite empirical scenario set; the "we only have five scenarios" version of the reviewer's question |
| DY10 | Erick Delage, Yinyu Ye | Distributionally Robust Optimization Under Moment Uncertainty with Application to Data-Driven Problems | Operations Research 58(3):595-612, 2010 | 10.1287/opre.1090.0741 | YES | Alternative DRO anchor (moment ambiguity). Pick one of MEK18 / DY10 |
| BS04 | Dimitris Bertsimas, Melvyn Sim | The Price of Robustness | Operations Research 52(1):35-53, 2004 | 10.1287/opre.1030.0065 | YES | Budget-of-uncertainty robustness; only if a reviewer pushes on conservatism |
| Mel24 | Tyler Griggs, Xiaoxuan Liu, Jiaxiang Yu, Doyoung Kim, Wei-Lin Chiang, Alvin Cheung, Ion Stoica | Mélange: Cost Efficient Large Language Model Serving by Exploiting GPU Heterogeneity | arXiv preprint, 2024 | arXiv:2404.14527 | YES | Already cited as [34]; note explicitly that it is deterministic, which is what makes B8 a step rather than a restatement |

Minimal set if space is tight (nine): **D55, BL11, Wets74, Birge82, Mad60, VSW69, EMS89, Swam00, BBC11**.
Swam00 is not optional; it is the reference that keeps the novelty claim honest.

## 4. Could not verify

- **CAPLAI (10.1145/3679240.3735099).** Crossref metadata verified (Nie, Xing, Latif, Liu; ACM e-Energy 2025; pp. 870-873, so a four-page short paper). The **model itself is not verified**: ACM DL returned 403, the NSF PAR mirror refused connection, IEEE Xplore returned an empty body. The description of its scenarios and decision variables above comes from search snippets, not a fetched full text. Read the PDF before finalising anything that turns on this paper. It is the only AI-systems item that could plausibly move the verdict.
- **Swaminathan 2000 formulation detail.** Bibliographic metadata verified through Crossref. The
  characterisation of its two stages comes from a search-surfaced abstract, not from the article
  itself (ScienceDirect returned 403 and both Crossref and Semantic Scholar have the abstract
  elided). The wording is consistent and specific enough to rely on, but the full text has not been
  read.
- **Barahona et al. 2005 formulation detail.** Metadata verified; abstract elided by the publisher in
  every API checked. Cited above only for what its title supports (robust capacity planning in
  semiconductor manufacturing).
- **Dong, Wang, Zhang, Zeng 2024** (10.1016/j.jclepro.2024.141482, "A two-stage stochastic
  collaborative planning approach for data centers and distribution network..."), metadata verified,
  full text 403. It appears to be a data-centre siting-and-dispatch problem on the power-network
  side rather than accelerator procurement, so it is **not** included above. Worth a glance if you
  want a second data-centre-flavoured two-stage SP citation.
- **Coverage gap.** Pre-2015 systems venues (ASPLOS/ISCA-era heterogeneous-datacentre provisioning,
  e.g. the Guevara line of work) were not searched in full text, because ACM DL, IEEE Xplore and
  Google Scholar all block automated fetch. If a hidden direct hit exists, that is where it sits.

## 5. Side note on the B8 code, not a literature item

The docstring of `experiments/b8_stochastic_program.py` states as sanity check S5 that EVPI "must
bound VSS". The recorded result has VSS = 1420.15 and EVPI = 880.04 J/job, so VSS > EVPI. The
check as actually recorded in `b8_stochastic_program.json` only asserts EVPI >= 0, which is correct
and passes; there is no general ordering between VSS and EVPI in the standard theory [BL11,
Birge82]. The docstring wording should be corrected before a reviewer reads it as a violated
invariant.
