# CAPLAI review — the one item the literature pass could not verify

**Nie, C., Xing, A., Latif, I., Liu, Z. "AI-assisted Stochastic Optimization for GPU Data
Centers Lifecycle Planning." E-ENERGY '25, Rotterdam, pp. 870–873.
doi:10.1145/3679240.3735099.** Open access; PDF saved alongside this file.

Stony Brook University and Brookhaven National Laboratory. Four pages, short-paper track.

## What it actually does

Formulates GPU **lifecycle planning** as a two-stage stochastic program over `T` quarters.

- **Decision variables** (§3.1): `x_t^i` GPUs of type `i` purchased at `t`, `r_t^i` retired,
  `g_t^i` operational at end of `t`, plus a binary `y_t` for a one-time liquid-cooling retrofit.
- **Objective** (eq. 1): minimise expected **total cost in USD** = purchase + cooling upgrade
  + operational electricity − salvage.
- **Uncertainty** (§3.5): workload demand `d_t`, GPU performance degradation `p_t^i`, and
  residual resale value `v_res(t)`. Scenarios are drafted by an LLM and then given explicit
  noise distributions by the authors (§4) — that scenario-generation pipeline is the paper's
  claimed novelty.
- **Evaluation** (§5): simulation only. GPU types {V100, A100, H100}, 12 quarters, A100 $10k,
  H100 $20k, $0.10/kWh, 2 MW total. Reports 27–40 per cent lower total lifecycle cost than a
  threshold heuristic ("buy when utilisation exceeds 90 per cent").

## Why it is a genuine adjacency

It is the closest work in AI systems to our Section 8.1 formulation: a stochastic program in
which GPU purchase quantities are first-stage decisions under demand uncertainty, with power
and cooling as constraints. Any claim of ours that framed *stochastic optimisation of GPU
procurement* as new would be unsafe, and this paper is the reason. It should be cited in
Section 2.8 next to Swaminathan.

It is also the direct predecessor for direction G2 (refresh timing), which it addresses head-on
and we have not.

## Why it does not threaten the contribution — and in fact illustrates it

The decisive detail is the demand-fulfilment constraint, eq. (2):

> Σ_i g_t^i · p_t^i ≥ d_t

`p_t^i` is "the effective performance of one GPU of type `i` in scenario `s` at time `t`": a
**single scalar per GPU type**. Likewise the operational cost, eq. (11), is
`c_o(t) · h_quarter · Σ_i g_t^i · w^i`, where `w^i` is the **rated power** of type `i`.

So in this model:

1. Energy is `rated power × hours`. It is never measured, and it does not depend on the
   workload.
2. Performance is one number per GPU type. Demand is one number per quarter.
3. Consequently the model **cannot express a workload-dependent hardware preference at all**.
   In our terms it is exactly the additive form of Proposition 1: a fixed accelerator ranking
   by `p^i / w^i`, identical for every workload. Whichever GPU type has the best scalar ratio
   under the power constraint is preferred for *all* demand, always.

This is not a criticism of the paper, which is about *when* to buy and retire, and for which a
scalar capacity model is a reasonable abstraction. It is that the paper's formulation and ours
are complementary along the axis each treats as scalar:

| | CAPLAI | This work |
|---|---|---|
| Objective | total cost (USD) | measured facility energy |
| Energy model | rated power × hours | measured, per workload × hardware cell |
| Performance | one scalar per GPU type | measured, workload-dependent |
| Workload | scalar aggregate demand `d_t` | a mix over 24+ (model, configuration) cells |
| Time | 12 quarters, buy/retire/retrofit | single-shot composition |
| Validation | simulation with example parameters | measurement, 6 corpora |

## Recommended positioning

Cite in Section 2.8 as the closest AI-systems instance of stochastic GPU procurement, and use
it to sharpen rather than defend the contribution:

> Closest in framing, Nie et al. [CAPLAI] pose GPU lifecycle planning as a stochastic program
> over purchase, retirement and cooling retrofit under demand uncertainty, with LLM-generated
> scenarios, and report 27 to 40 per cent lower lifecycle cost than a threshold heuristic.
> Their demand constraint gives each GPU type a single scalar performance figure and prices
> energy as rated power times hours, so by Proposition 1 the model is exactly a fixed
> accelerator ranking and cannot prefer different hardware for different workloads. That is
> the quantity this paper measures, and supplying it is complementary to their treatment of
> *when* to buy rather than competitive with it.

## Consequences for our open work

- **G2 (refresh timing)** now has a direct predecessor. Any G2 result must be positioned
  against CAPLAI's 27–40 per cent, and must be explicit that ours is an energy objective on
  measured data where theirs is a cost objective on assumed parameters.
- **B8**: our VSS (21.3 per cent) and EVPI (16.8 per cent) remain unclaimed by them; they
  report neither, so the value-of-information framing stays ours.
- Their `p_t^i` performance-degradation uncertainty is a dimension we do not model at all and
  should acknowledge as out of scope.

## Verification status

Metadata confirmed via OpenAlex (DOI resolves, four authors as listed, E-ENERGY '25,
conference-paper, open access). Full text read from the ACM PDF, not from an abstract.
This closes the "could not verify" item left open by the stochastic-programming literature pass.
