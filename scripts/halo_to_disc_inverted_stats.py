import json
import os

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

        filename = f"results/{simulation}_{galaxy}/disk_arrival_delay_stats.json"

        print(f"Loading {filename}")

        with open(filename) as f:
            data = json.load(f)

        disk_arrivals_per_snap = {}
        with_halo_per_snap = {}
        delays_per_snap = {}
        disk_time_per_snap = {}

        for snap_str, d in data.items():

            snap = int(snap_str)

            disk_arrivals_per_snap[snap] = d["N_disk"]
            with_halo_per_snap[snap] = d["N_with_halo"]
            delays_per_snap[snap] = d["delays_Gyr"]
            disk_time_per_snap[snap] = d["disk_time_Gyr"]

        all_data[(simulation, galaxy)] = {
            "disk_arrivals_per_snap": disk_arrivals_per_snap,
            "with_halo_per_snap": with_halo_per_snap,
            "delays_per_snap": delays_per_snap,
            "disk_time_per_snap": disk_time_per_snap,
        }

# ============================================================
# Create output directories
# ============================================================

os.makedirs("images/disk_arrival", exist_ok=True)
os.makedirs(
    "images/disk_arrival/time_since_first_halo_entry_distributions",
    exist_ok=True
)

# ============================================================
# Fraction with previous halo
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

        times = []
        fractions = []

        for snap in sorted(d["disk_arrivals_per_snap"]):

            ntot = d["disk_arrivals_per_snap"][snap]

            if ntot == 0:
                continue

            t = d["disk_time_per_snap"][snap]

            if t is None:
                continue

            frac = d["with_halo_per_snap"][snap] / ntot

            times.append(t)
            fractions.append(frac)

        print(simulation, galaxy, len(times))

        ax.plot(times, fractions)

        ax.set_ylim(0, 1)
        ax.set_title(f"{simulation} {galaxy}")

        if j == 0:
            ax.set_ylabel("Fraction with previous halo")

        if i == len(simulations) - 1:
            ax.set_xlabel("Disk arrival time [Gyr]")

fig.tight_layout()

fig.savefig(
    "images/disk_arrival/fraction_with_previous_halo.pdf"
)

plt.close(fig)

# ============================================================
# Mean and median time since first halo entry
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

        times = []
        means = []
        medians = []
        p16, p84 = [], []

        for snap in sorted(d["disk_arrivals_per_snap"]):

            delays = d["delays_per_snap"][snap]

            if len(delays) == 0:
                continue

            t = d["disk_time_per_snap"][snap]

            if t is None:
                continue

            times.append(t)
            means.append(np.mean(delays))
            medians.append(np.median(delays))
            p16.append(np.percentile(delays, 16))
            p84.append(np.percentile(delays, 84))

        ax.plot(times, means, label="mean")
        ax.plot(times, medians, label="median")
        ax.fill_between(times, p16, p84, alpha=0.2, color='gray', label="16-84th percentile")

        ax.set_title(f"{simulation} {galaxy}")

        if j == 0:
            ax.set_ylabel("Time since first halo entry [Gyr]")

        if i == len(simulations) - 1:
            ax.set_xlabel("Disk arrival time [Gyr]")

        ax.legend()

fig.tight_layout()

fig.savefig(
    "images/disk_arrival/mean_time_since_first_halo_entry.pdf"
)

plt.close(fig)

# ============================================================
# Distribution plots
# ============================================================

reference = all_data[(simulations[0], galaxies[0])]
snaps_to_plot = sorted(reference["disk_arrivals_per_snap"])[::10]

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

            if snap not in d["disk_arrivals_per_snap"]:
                ax.set_axis_off()
                continue

            ntot = d["disk_arrivals_per_snap"][snap]

            if ntot == 0:
                ax.set_axis_off()
                continue

            nwith = d["with_halo_per_snap"][snap]

            frac = nwith / ntot

            delays = d["delays_per_snap"][snap]

            if len(delays) == 0:

                ax.text(
                    0.5,
                    0.5,
                    "No previous halo entries",
                    transform=ax.transAxes,
                    ha="center",
                    va="center"
                )

            else:

                ax.hist(
                    delays,
                    bins=30,
                    density=True,
                    histtype="step",
                    linewidth=1.5
                )

            ax.set_title(
                f"{simulation} {galaxy}\n"
                f"N={ntot}, with halo={nwith}\n"
                f"f={frac:.3f}"
            )

            if j == 0:
                ax.set_ylabel("PDF")

            if i == len(simulations) - 1:
                ax.set_xlabel(
                    "Time since first halo entry [Gyr]"
                )

    t_snap = reference["disk_time_per_snap"][snap]

    fig.suptitle(
        f"Disk arrival time = {t_snap:.2f} Gyr (snap {snap})",
        fontsize=16
    )

    fig.tight_layout()

    fig.savefig(
        f"images/disk_arrival/time_since_first_halo_entry_distributions/snap_{snap:03d}.pdf"
    )

    plt.close(fig)

print("Done.")