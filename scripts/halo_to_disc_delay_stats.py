import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from hestia.settings import Settings


# ============================================================
# Load all datasets
# ============================================================

all_data = {}

simulations = Settings.HIGH_RES_SIMULATIONS
galaxies = ["MW", "M31"]

for simulation in simulations:
    for galaxy in galaxies:

        filename = f"results/{simulation}_{galaxy}/halo_to_disk_tracers.json"

        print(f"Loading {filename}")

        with open(filename) as f:
            data = json.load(f)

        total_per_snap = defaultdict(int)
        reached_per_snap = defaultdict(int)
        delays_per_snap = defaultdict(list)
        halo_time_per_snap = {}

        t_final = 0

        for tid, tracer in data.items():

            h_snap = tracer["halo_snap"]
            h_time = tracer["halo_time_Gyr"]
            delay = tracer["delay_Gyr"]

            total_per_snap[h_snap] += 1
            halo_time_per_snap[h_snap] = h_time

            if h_time is not None:
                t_final = max(t_final, h_time)

            if delay is not None:
                reached_per_snap[h_snap] += 1
                delays_per_snap[h_snap].append(delay)

        all_data[(simulation, galaxy)] = {
            "total_per_snap": total_per_snap,
            "reached_per_snap": reached_per_snap,
            "delays_per_snap": delays_per_snap,
            "halo_time_per_snap": halo_time_per_snap,
            "t_final": t_final,
        }

# ============================================================
# Create output directories
# ============================================================

os.makedirs("images/halo_to_disk", exist_ok=True)
os.makedirs("images/halo_to_disk/delay_distributions", exist_ok=True)

# ============================================================
# Fraction reaching disk figure
# ============================================================

fig, axes = plt.subplots(
    nrows=len(simulations),
    ncols=2,
    figsize=(10, 4 * len(simulations)),
    sharex=True,
    sharey=True
)

if len(simulations) == 1:
    axes = np.array([axes])

for i, simulation in enumerate(simulations):
    for j, galaxy in enumerate(galaxies):

        ax = axes[i, j]

        d = all_data[(simulation, galaxy)]

        total_per_snap = d["total_per_snap"]
        reached_per_snap = d["reached_per_snap"]
        halo_time_per_snap = d["halo_time_per_snap"]

        snaps = sorted(total_per_snap.keys())

        times = []
        fractions = []

        for snap in snaps:

            ntot = total_per_snap[snap]

            if ntot == 0:
                continue

            frac = reached_per_snap[snap] / ntot

            times.append(halo_time_per_snap[snap])
            fractions.append(frac)

        ax.plot(times, fractions)

        ax.set_ylim(0, 1)
        ax.set_title(f"{simulation} {galaxy}")

        if j == 0:
            ax.set_ylabel("Fraction reaching disk")

        if i == len(simulations) - 1:
            ax.set_xlabel("Cosmic time [Gyr]")

fig.tight_layout()

fig.savefig(
    "images/halo_to_disk/fraction_reaching_disk.pdf"
)

plt.close(fig)

# ============================================================
# Determine common snapshots
# ============================================================

reference = all_data[(simulations[0], galaxies[0])]
snaps_to_plot = sorted(reference["total_per_snap"].keys())[::10]

# ============================================================
# Delay distributions
# ============================================================

for snap in snaps_to_plot:

    fig, axes = plt.subplots(
        nrows=len(simulations),
        ncols=2,
        figsize=(10, 4 * len(simulations)),
        sharex=True,
        sharey=True
    )

    if len(simulations) == 1:
        axes = np.array([axes])

    for i, simulation in enumerate(simulations):
        for j, galaxy in enumerate(galaxies):

            ax = axes[i, j]

            d = all_data[(simulation, galaxy)]

            total_per_snap = d["total_per_snap"]
            reached_per_snap = d["reached_per_snap"]
            delays_per_snap = d["delays_per_snap"]
            halo_time_per_snap = d["halo_time_per_snap"]
            t_final = d["t_final"]

            if snap not in total_per_snap:
                ax.set_axis_off()
                continue

            ntot = total_per_snap[snap]
            nreach = reached_per_snap[snap]

            frac = nreach / ntot

            delays = delays_per_snap[snap]

            if len(delays) < 10:
                ax.set_axis_off()
                continue

            h_time = halo_time_per_snap[snap]

            max_delay = t_final - h_time

            ax.hist(
                delays,
                bins=30,
                density=True,
                histtype="step",
                linewidth=1.5
            )

            ax.axvline(
                max_delay,
                color="red",
                linestyle="--",
                linewidth=2
            )

            ax.set_title(
                f"{simulation} {galaxy}\n"
                f"N={ntot}, reached={nreach}\n"
                f"f={frac:.3f}"
            )

            if j == 0:
                ax.set_ylabel("PDF")

            if i == len(simulations) - 1:
                ax.set_xlabel("Delay [Gyr]")

    t_snap = reference["halo_time_per_snap"][snap]

    fig.suptitle(
        f"t = {t_snap:.2f} Gyr (snap {snap})",
        fontsize=16
    )

    fig.tight_layout()

    fig.savefig(
        f"images/halo_to_disk/delay_distributions/snap_{snap:03d}.pdf"
    )

    plt.close(fig)

print("Done.")