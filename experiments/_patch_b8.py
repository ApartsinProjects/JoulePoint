# -*- coding: utf-8 -*-
"""
DISARMED 19 August 2026. DO NOT RUN.

This was a one-shot patch script that rewrote the analysis tail of b8_stochastic_program.py.
It still carries the PRE-FIX `local_search`, which applies swap moves mid-scan and therefore
produces negative fleet counts and a search that "beats" exhaustive enumeration. Running it
would silently revert the script behind the VSS 21.3 per cent and EVPI 16.8 per cent figures
in Section 2.8 and Section 8.1.

The audit of 19 August classified it as a regression landmine. It is kept for provenance only.
"""
import sys
print(__doc__)
sys.exit("refusing to run: this script would revert a fixed bug (see docstring)")

# ---- original content preserved below, unreachable ----

# # -*- coding: utf-8 -*-
# """Replace the analysis tail of b8_stochastic_program.py."""
# import io
# 
# p = "experiments/b8_stochastic_program.py"
# s = io.open(p, encoding="utf-8").read()
# head = s[:s.index("# ------------------------------------------------------------------ S2 fidelity")]
# 
# TAIL = r'''# ------------------------------------------------------------------ S2 fidelity
# lpv, simv = [], []
# for c, e, d in sim:
#     if d > SLA_S:
#         continue
#     v, _ = recourse(np.array(c, float), wcal, RHO)
#     if np.isfinite(v):
#         lpv.append(v); simv.append(e)
# lpv, simv = np.array(lpv), np.array(simv)
# rank = float(np.corrcoef(np.argsort(np.argsort(lpv)), np.argsort(np.argsort(simv)))[0, 1])
# 
# def pair_acc(margin):
#     n = ok = 0
#     for i in range(len(simv)):
#         for j in range(i + 1, len(simv)):
#             if abs(simv[i] - simv[j]) / simv[j] > margin:
#                 n += 1
#                 ok += (lpv[i] < lpv[j]) == (simv[i] < simv[j])
#     return ok / n, n
# 
# acc5, n5 = pair_acc(0.05)
# acc1, n1p = pair_acc(0.01)
# print("LP-vs-simulator fidelity: Spearman {:.3f}, median |rel err| {:.1%}".format(
#     rank, float(np.median(np.abs(lpv - simv) / simv))))
# print("  pairs differing by >1%: {:.1%} ordered correctly ({:,} pairs)".format(acc1, n1p))
# print("  pairs differing by >5%: {:.1%} ordered correctly ({:,} pairs)".format(acc5, n5))
# sane("S2 LP orders clearly separated fleets correctly", acc5 > 0.90,
#      "{:.1%} of pairs differing by more than 5% ({:,} pairs); Spearman over all feasible "
#      "fleets is {:.3f}, depressed by near-ties".format(acc5, n5, rank))
# 
# # ------------------------------------------------------------------ first stage
# def local_search(scen, probs, slots, ntypes, seed=0, starts=3):
#     rng = np.random.default_rng(seed)
#     bestn, bestv, evals = None, float("inf"), 0
#     for st in range(starts):
#         n = np.zeros(ntypes)
#         if st == 0:
#             n[rng.integers(ntypes)] = slots
#         else:
#             for _ in range(slots):
#                 n[rng.integers(ntypes)] += 1
#         v = expected_recourse(n, scen, probs, RHO); evals += 1
#         improved = True
#         while improved:
#             improved = False
#             for a in range(ntypes):
#                 if n[a] <= 0:
#                     continue
#                 for b in range(ntypes):
#                     if a == b:
#                         continue
#                     cand = n.copy(); cand[a] -= 1; cand[b] += 1
#                     cv = expected_recourse(cand, scen, probs, RHO); evals += 1
#                     if cv < v - 1e-9:
#                         v, n, improved = cv, cand, True
#         if v < bestv:
#             bestv, bestn = v, n
#     return bestn, bestv, evals
# 
# def shortlist(scen, probs, slots, ntypes, k=25):
#     """The LP prunes; the simulator picks among the survivors. This is the division of
#     labour the fidelity result supports: the LP is reliable on clear separations and
#     unreliable on near-ties, so it should never make the final call itself."""
#     n, v, ev = local_search(scen, probs, slots, ntypes, seed=0, starts=3)
#     cands = {tuple(int(x) for x in n)}
#     for a in range(ntypes):
#         for b in range(ntypes):
#             if a == b or n[a] <= 0:
#                 continue
#             c2 = n.copy(); c2[a] -= 1; c2[b] += 1
#             cands.add(tuple(int(x) for x in c2))
#             for a2 in range(ntypes):
#                 for b2 in range(ntypes):
#                     if a2 == b2 or c2[a2] <= 0:
#                         continue
#                     c3 = c2.copy(); c3[a2] -= 1; c3[b2] += 1
#                     cands.add(tuple(int(x) for x in c3))
#     scored = sorted((expected_recourse(np.array(c, float), scen, probs, RHO), c) for c in cands)
#     return [c for _, c in scored[:k]], ev + len(cands)
# 
# print("\nS1: LP search plus simulator refinement versus exhaustive enumeration")
# enum_best, enum_c = float("inf"), None
# for c, e, d in sim:
#     if d <= SLA_S and e < enum_best:
#         enum_best, enum_c = e, c
# cands, ev = shortlist([wcal], np.array([1.0]), 10, 5, k=25)
# best_sim, best_c = float("inf"), None
# for c in cands:
#     e, d = facility({MACH[i]: c[i] for i in range(5) if c[i] > 0}, wcal, seed=0)
#     if d <= SLA_S and e < best_sim:
#         best_sim, best_c = e, c
# sane("S1 LP shortlist plus simulation recovers the enumeration optimum",
#      best_c == enum_c and abs(best_sim - enum_best) < 1e-6,
#      "enumeration {} at {:.0f} J/job needing 1001 simulations; LP+refine {} at {:.0f} "
#      "needing {} LP solves and {} simulations".format(enum_c, enum_best, best_c, best_sim,
#                                                       ev, len(cands)))
# 
# # ------------------------------------------------------------------ S3
# n1, v1, _ = local_search([wcal], np.array([1.0]), 10, 5, seed=1)
# lp_enum = min(recourse(np.array(c, float), wcal, RHO)[0] for c in COMPS10)
# sane("S3 single scenario reduces to the deterministic problem", abs(v1 - lp_enum) < 1e-6,
#      "local search {:.1f} vs LP enumeration {:.1f} J/job".format(v1, lp_enum))
# 
# # ------------------------------------------------------------------ VSS and EVPI
# scen = [SCENARIOS[k] for k in SCENARIOS]
# names = list(SCENARIOS)
# wbar = sum(p * w for p, w in zip(PROBS, scen))
# n_sp, v_sp, _ = local_search(scen, PROBS, 10, 5, seed=0)
# n_ev, _, _ = local_search([wbar], np.array([1.0]), 10, 5, seed=0)
# v_ev = expected_recourse(n_ev, scen, PROBS, RHO)
# wait_and_see = sum(p * local_search([w], np.array([1.0]), 10, 5, seed=0)[1]
#                    for p, w in zip(PROBS, scen))
# VSS, EVPI = v_ev - v_sp, v_sp - wait_and_see
# 
# def fl(n):
#     return "+".join("{}x{}".format(int(v), MACH[i]) for i, v in enumerate(n) if v > 0)
# 
# print("\ntwo-stage stochastic program, 10 slots, {} mix scenarios".format(len(scen)))
# print("  here-and-now fleet (stochastic)   {:<30} {:.0f} J/job".format(fl(n_sp), v_sp))
# print("  mean-mix fleet (deterministic)    {:<30} {:.0f} J/job on the distribution".format(
#     fl(n_ev), v_ev))
# print("  wait-and-see (perfect foresight)  {:<30} {:.0f} J/job".format("-", wait_and_see))
# print("  value of the stochastic solution   {:.0f} J/job ({:.2f}% of the mean-mix plan)".format(
#     VSS, 100 * VSS / v_ev))
# print("  expected value of perfect information {:.0f} J/job ({:.2f}%)".format(EVPI, 100 * EVPI / v_sp))
# for nm, w, p in zip(names, scen, PROBS):
#     v, info = recourse(n_sp, w, RHO)
#     print("    scenario {:<26} p={:.2f}  {:.0f} J/job, off-fleet share {:.1%}".format(
#         nm, p, v, info["outsourced"]))
# sane("S4 value of the stochastic solution is non-negative", VSS >= -1e-9, "{:.3f} J/job".format(VSS))
# sane("S5 EVPI is non-negative", EVPI >= -1e-9, "{:.3f} J/job".format(EVPI))
# 
# # ------------------------------------------------------------------ scaling
# print("\nscaling beyond what enumeration can reach")
# import math
# scal = []
# for slots, ntypes in [(10, 5), (20, 5), (50, 5), (100, 5)]:
#     n, v, ev2 = local_search(scen, PROBS, slots, ntypes, seed=0)
#     ncomp = math.comb(slots + ntypes - 1, ntypes - 1)
#     scal.append(dict(slots=slots, types=ntypes, lp_evals=ev2, enumeration_size=ncomp,
#                      fleet=fl(n), energy=v))
#     print("  {:>4} slots: {:>11,} compositions to enumerate vs {:>5} LP solves "
#           "({:>9,.0f}x fewer)   {} at {:.0f} J/job".format(
#               slots, ncomp, ev2, ncomp / ev2, fl(n), v))
# 
# OUT = dict(rho=RHO, rho_agreement=best_agree / len(COMPS10), lp_vs_sim_spearman=rank,
#            pair_accuracy_1pct=acc1, pair_accuracy_5pct=acc5, penalty=PENALTY,
#            scenarios=names, probs=PROBS.tolist(),
#            enumeration_optimum=list(enum_c), enumeration_energy=enum_best,
#            lp_refined_optimum=list(best_c), lp_refined_energy=best_sim,
#            stochastic_fleet=fl(n_sp), stochastic_value=v_sp,
#            mean_mix_fleet=fl(n_ev), mean_mix_value_on_distribution=v_ev,
#            wait_and_see=wait_and_see, VSS=VSS, EVPI=EVPI,
#            VSS_pct=100 * VSS / v_ev, EVPI_pct=100 * EVPI / v_sp,
#            scaling=scal, sanity=SANITY)
# json.dump(OUT, io.open("experiments/results/b8_stochastic_program.json", "w", encoding="utf-8"), indent=1)
# print("\nsaved -> experiments/results/b8_stochastic_program.json")
# print("sanity: {} passed, {} failed".format(sum(x["passed"] for x in SANITY),
#                                             sum(not x["passed"] for x in SANITY)))
# '''
# 
# io.open(p, "w", encoding="utf-8", newline="\n").write(head + TAIL)
# print("rewritten, {} bytes".format(len(head + TAIL)))