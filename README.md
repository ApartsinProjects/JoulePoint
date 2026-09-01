# The Joule Point: an Energy-Optimal Operating Point for AI Inference

Alexander Apartsin (HIT) and Yehudit Aperstein (Afeka). Paper source, builds, dataset, measurement
harnesses, figures, fleet simulation, and review artifacts.

**Paper (HTML):** [docs/GPTEnergy.html](docs/GPTEnergy.html) (GitHub Pages serves it as the site
index, with a PDF link at the top right).
**Paper (PDF/Word):** [docs/JoulePoint_1col.pdf](docs/JoulePoint_1col.pdf) (primary, single-column),
[docs/JoulePoint_2col.pdf](docs/JoulePoint_2col.pdf), plus the matching `.docx` files.
**Dataset:** ELF (Energy-Latency Frontier), archived at Zenodo,
[doi:10.5281/zenodo.22058568](https://doi.org/10.5281/zenodo.22058568), CC-BY-4.0 (data) and MIT (code).

## Headline

GPU board power follows P(&theta;) = P&#8320; + a&theta;^&beta; over the operating point (GPU, power
cap), so energy per inference is U-shaped with a minimum, the **Joule Point**, at 43 to 46 per cent
of TDP on large GPUs. Capping to it cuts energy per inference by 29 to 31 per cent at an exactly
priced cost (about 1.2x latency and the same factor in cards). Under load the Joule Point is nearly
a per-card constant, so one static cap per card replaces the online per-job search prior systems
run; a trace-driven fleet simulation over the measured curves serves equal work for 18 to 45 per
cent less energy under a power budget.

## Layout

```
docs/            GPTEnergy.html            the paper (built HTML, single source of truth)
                 GPTEnergy_{1col,2col}.{pdf,docx}   built deliverables
                 build_docx.py             HTML -> Word/PDF build
power_shaping/   build_gptenergy.py        writes docs/GPTEnergy.html
                 aws_*.py                  AWS measurement harnesses (power-cap and clock sweeps)
                 make_*.py                 figure builders
                 figures/                  built figures used by the paper
                 data/raw, data/processed  measured sweeps; elf_master.csv is the merged dataset
                 elf_release/              the Zenodo deposit package (README, dictionary, LICENSE)
                 energy_scheduler_sim.py   fleet simulation of Section 7
                 results/                  per-experiment JSON results
                 reviews/                  external review passes on the draft
reports/         dataset verification, novelty and related-work scouting, audit notes
experiments/     earlier pilot line (configuration-dependent energy rankings), with BUGFIXES.md
paper/           earlier draft lineage (greenmatch)
references/      cited external material
```

## Reproducing

Every empirical figure and measurement-derived number in the paper is computed from the ELF data by
the build scripts:

```
python power_shaping/build_gptenergy.py   # rebuilds docs/GPTEnergy.html
python docs/build_docx.py                 # rebuilds the Word/PDF deliverables
```

The `power_shaping/aws_*.py` harnesses reproduce the measurements themselves on rented AWS GPU
instances (g6, g5, p4d, g4dn); see `power_shaping/DATA_PROVENANCE.md`.
