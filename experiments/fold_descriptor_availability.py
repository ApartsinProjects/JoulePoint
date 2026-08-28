# -*- coding: utf-8 -*-
"""
Add the motivation the CF machinery has been missing: descriptor availability depends on how
the workload reaches the machine, and it is falling.

Section 7 recommends adding numerical precision to the scheduling interface, and calls it the
cheapest intervention in the paper. That is true where a workload is SUBMITTED with a
declaration. It is unavailable where a model call is a library invocation inside an ordinary
application, which is the direction the industry is moving. The paper needs to say which regime
each of its methods serves, because they are not interchangeable:

  descriptors declared    -> the feature-based model of Section 7, which also does cold start
  descriptors absent,
     workload recurs      -> free-embedding collaborative filtering, learned from behaviour
  descriptors absent,
     workload novel       -> nothing works; fall back to the fixed ranking

The measured case for free embeddings is modest, at best 0.0075 in regret on our extended grid,
and it must be reported that way. The case is not magnitude, it is availability: it is the only
estimator that runs at all when nothing is declared.
"""
import io

P = "paper/greenmatch-paper.html"
s = io.open(P, encoding="utf-8").read()
applied, failed = [], []

def sub(tag, old, new):
    global s
    if old in s:
        s = s.replace(old, new, 1); applied.append(tag)
    else:
        failed.append(tag)

NEW = """<h3>7.5 What the operator is actually told, and what follows</h3>
<p>
The recommendation of Section 7 is to put numerical precision into the scheduling interface, and
where a workload arrives as a declared job that is nearly free. But how much is declared depends on
how the workload reaches the machine, and the trend is downward. A model call is increasingly a
library invocation inside an ordinary service that also authenticates users and queries databases,
rather than a job submitted to a machine-learning platform. There is no model identity to declare
because, from the operator's side, there is no machine-learning job: there is a container that
happens to touch an accelerator.
</p>
<div class="tablewrap">
<table>
<caption><b>Table 10.</b> What an operator knows about a workload, by how it arrives. The right-hand
column lists what can be recovered by observation instead, with the section that measures it.</caption>
<thead><tr><th>How the workload arrives</th><th>Declared to the operator</th>
<th>Recoverable by observation</th></tr></thead>
<tbody>
<tr><td>Managed inference service</td><td>model, precision, batch: the provider owns the serving
stack</td><td>not needed</td></tr>
<tr><td>Managed training service</td><td>framework, model configuration, hyperparameters</td>
<td>not needed</td></tr>
<tr><td>Orchestrated container with resource requests</td><td>image, memory request, device count,
time limit; <i>not</i> precision</td><td>precision, from runtime on a known machine, which separates
the two by a median factor of 2.2 (Section 8.1)</td></tr>
<tr><td>Serverless accelerator function</td><td>container image and an invocation; user code is
opaque</td><td>peak memory, essentially a workload property with a cross-machine coefficient of
variation of 0.000 and predictable to 9.3 per cent (Section 7.2)</td></tr>
<tr><td>Rented instance or bare metal</td><td>nothing; the tenant owns the kernel</td>
<td>external power and utilisation traces only</td></tr>
<tr><td>Model embedded in an application</td><td>nothing; there is no machine-learning job to
declare</td><td>the workload's own history, if it recurs, through a free per-workload latent
factor</td></tr>
</tbody>
</table>
</div>
<p>
Reading the table downward gives the estimator that applies. Where descriptors are declared, the
feature-based model of Section 7.3 is preferred, and it alone can place a workload seen for the
first time. Where they are not declared but the workload recurs, which is the common case for an
embedded model called continuously by one application, a free per-workload factor learned from
observed placements is the only remaining option: it needs no descriptors, and on our extended grid
it improves regret by up to 0.0075 once three or more of a workload's cells have been seen. Where
descriptors are absent <i>and</i> the workload is novel, neither method has anything to work with,
the latent factor is unidentified, and the honest fallback is the fixed ranking of Proposition 1.
</p>
<p>
The size of that free-factor gain is small and we do not claim otherwise; on a three-machine corpus
it is negative, because with one cell hidden per workload a fixed ranking is already near-optimal and
the variance of a fitted factor costs more than it recovers. The argument for it is not magnitude but
availability. It is the estimator that still runs when the declarative channel is gone, and the
declarative channel is precisely what disappears as models move from platforms into applications.
</p>

"""
sub("sec7.5", "<h3>7.5 Placement is algorithmically saturated</h3>",
    NEW + "<h3>7.6 Placement is algorithmically saturated</h3>")
s = s.replace("Section 7.5's", "Section 7.6's").replace("in Section 7.5", "in Section 7.6")

io.open(P, "w", encoding="utf-8", newline="\n").write(s)
print("applied: {}".format(", ".join(applied)))
print("FAILED : {}".format(", ".join(failed) if failed else "none"))
