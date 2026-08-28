# -*- coding: utf-8 -*-
"""
Rewrite the Introduction with a fuller logical arc, and finish the term-introduction audit.

The arc the previous version skipped: WHY the performance premise is reasonable (so the
reader understands why it is universally held rather than merely wrong), and an intuitive
account of the mechanism before any number is quoted.
"""
import io, re

p = "paper/greenmatch-paper.html"
s = io.open(p, encoding="utf-8").read()
n = 0

start = s.index("<h2>1. Introduction</h2>")
end = s.index("<h3>1.1 Contributions</h3>")
NEW = """<h2>1. Introduction</h2>

<p>
Electricity is now a binding constraint on artificial-intelligence capacity rather than a line in its
operating budget. Global data-center demand is projected to approach 945 TWh by 2030, with accelerated
servers driving most of the growth [3]. The constraint binds at the interconnection rather than at the
meter: capacity studies of two grid operators, Dominion Energy, the utility serving Northern Virginia
and with it the largest data-center concentration in the world, and EirGrid, the Irish transmission
system operator, find that neither can host projected load under firm-reliability assumptions, and that
relaxing those guarantees for new facilities would raise hostable capacity by factors of 1.5 to 4.6
[2]. Operators now plan power six to twelve months ahead of accelerator availability on fleets reaching
150 MW [1]. Where interconnection capacity is rationed rather than merely expensive, the operative
quantity becomes useful computation per connected megawatt.
</p>

<p>
Two decisions determine that quantity, and they are made by different people at different times. A
facility decides which accelerators to buy, typically by comparing advertised throughput per watt
across vendor datasheets. Its scheduler then decides, thousands of times an hour, which of the machines
it owns should run each arriving job, typically by taking whichever is free or whichever finishes
soonest. Both decisions rest on the same premise: that a device advertising more computation per watt,
or completing work sooner, will consume less energy on the work actually submitted.
</p>

<p>
The premise is not naive. It follows from a plausible physical argument. If two accelerators draw
comparable power, the faster one occupies for less time and therefore integrates less energy, so
throughput is a sufficient statistic and energy needs no separate treatment. That argument holds when
power is roughly constant across devices, which was broadly true when server fleets were built from
processors of similar power envelope. It is the reason performance per watt became the purchasing
metric and availability became the scheduling rule, and it is why neither practice is usually
questioned.
</p>

<p>
Modern accelerator fleets violate the argument's premise rather than its logic. The devices in a single
facility now span an order of magnitude in power draw, from inference parts rated in tens of watts to
training parts rated in hundreds. Once power varies as much as speed does, the two effects can cancel,
and the faster device is no longer reliably the cheaper one. Worse, which device is cheapest stops
being a fixed property at all: it comes to depend on the workload and on the software settings the
workload runs under, because those settings determine which hardware resource is the limiting one.
A device that is efficient when its memory bandwidth is the bottleneck may be wasteful when its
arithmetic units are, and a change of numerical precision can move a workload from one regime to the
other without touching the hardware.
</p>

<p>
<b>This paper measures whether the premise survives, and finds that it does not at either decision
point.</b> Vendor specifications carry essentially no information about the energy a device actually
uses. Measured throughput carries little more. The reason is that energy is a property not of a device
but of the match between a workload, its configuration and a device, and that match is expressed in a
component of the data so small that ordinary model fitting discards it, yet large enough to reverse
which device is cheapest for nearly half the device pairs we tested.
</p>

<p>
The practical consequence is this paper's headline, and it reorders the levers available for saving
energy. If efficiency is a property of the match, it can only be exploited where a facility holds
devices that differ, and the size of the opportunity is therefore fixed by the fleet before any
scheduler runs. Measured against replayed arrivals and directly measured idle power, choosing the fleet
is worth roughly a third of the energy bill, letting idle machines sleep is worth about an eighth, and
choosing where each job runs is worth less than a tenth, falling to exactly zero on a fleet of
identical machines. We further show that placement is not merely limited by opportunity but
algorithmically saturated: replacing the simple placement rule with an exact optimal assignment
recovers nothing at all. Most published work on data-center artificial-intelligence energy optimises
the last and smallest of the three levers. <b>A facility cannot schedule its way out of a homogeneous
fleet.</b>
</p>

"""
s = s[:start] + NEW + s[end:]
n += 1

# name the workloads on first use
old_m = """Four inference workloads spanning distinct computational profiles were executed on five accelerator
families across three NVIDIA generations with a 5.7x range of enforced power limits, at two precisions
and three batch sizes, giving 120 cells."""
new_m = """Four inference workloads spanning distinct computational profiles were executed: ResNet-50 and
ConvNeXt-Tiny, both convolutional networks, ViT-B/16, a vision transformer whose cost is dominated by
attention, and a six-layer Transformer encoder. Each ran on five accelerator families across three
NVIDIA generations with a 5.7x range of enforced power limits, at two numerical precisions and three
batch sizes, giving 120 cells."""
if old_m in s:
    s = s.replace(old_m, new_m); n += 1

# introduce SLA and Shapley at their first occurrence in the contributions list
s = s.replace("subject to a service constraint", "subject to a service-level constraint on queueing delay", 1)
s2 = s.replace("the marginal value of each accelerator type by Shapley decomposition",
               "the marginal value of each accelerator type, obtained by treating types as players in a cooperative game and computing the Shapley value", 1)
if s2 != s:
    s = s2; n += 1

# descriptor ablation table
if "Descriptor ablation" not in s:
    m = re.search(r"<b>The descriptor that matters most", s)
    if m:
        tbl = """<div class="tablewrap">
<table>
<caption><b>Table 8.</b> Descriptor ablation. Removing numerical precision collapses ranking accuracy to
the additive ceiling of Proposition 1; removing workload identity costs little.</caption>
<thead><tr><th>Descriptor set available to the scheduler</th><th>Pairwise accuracy</th></tr></thead>
<tbody>
<tr><td>Full: workload family, precision, batch size</td><td>87.9</td></tr>
<tr><td>Precision and batch size, no workload identity</td><td>85.4</td></tr>
<tr><td>Precision only</td><td>83.3</td></tr>
<tr><td>Workload family and batch size, no precision</td><td>81.7</td></tr>
<tr><td>Memory request and batch size, as a scheduler observes</td><td>81.7</td></tr>
<tr><td>Nothing (additive ceiling)</td><td>81.7</td></tr>
</tbody>
</table>
</div>
<p>
"""
        s = s[:m.start()] + tbl + s[m.start():]
        n += 1

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
import html
plain = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s)))
intro = plain[plain.index("1. Introduction"):plain.index("1.1 Contributions")]
print("applied {} edits".format(n))
print("introduction: {} words, {} paragraphs".format(len(intro.split()), s[s.index("<h2>1."):s.index("<h3>1.1")].count("<p>")))
print("tables: {}".format(s.count("<caption>")))
print("Israel mentions: {}".format(s.lower().count("israel")))
