import json
import numpy as np
import matplotlib.pyplot as plt
import csv

from hestia.settings import Settings


def load_snapshot_times(filepath):
    snap_time = {}

    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["Time_Gyr"] or not row["SnapshotNumber"]:
                continue

            try:
                snap = int(row["SnapshotNumber"])
                time = float(row["Time_Gyr"])
            except ValueError:
                continue

            snap_time[snap] = time

    return snap_time


# ---------- FIGURE SETUP ----------
fig, axes = plt.subplots(
    1,
    len(Settings.HIGH_RES_SIMULATIONS),
    figsize=(12, 4),
    sharex=True,
    sharey=True
)

if len(Settings.HIGH_RES_SIMULATIONS) == 1:
    axes = [axes]  # asegurar iterable


# ---------- LOOP OVER SIMULATIONS ----------
for i, simulation in enumerate(Settings.HIGH_RES_SIMULATIONS):

    ax = axes[i]

    for galaxy in Settings.GALAXIES:

        # ---------- LOAD DATA ----------
        filepath = f"results/{simulation}_{galaxy}/halo_to_disk_tracers.json"

        with open(filepath) as f:
            data = json.load(f)

        time_file = f"data/hestia/r200_t/r200_t_{galaxy}_{simulation}.csv"
        snap_time = load_snapshot_times(time_file)

        # ---------- GROUP BY HALO SNAP ----------
        delays_by_snap = {}

        for tid, v in data.items():

            h_snap = v["halo_snap"]
            delay = v["delay_Gyr"]

            if delay is None:
                continue

            if h_snap not in delays_by_snap:
                delays_by_snap[h_snap] = []

            delays_by_snap[h_snap].append(delay)

        # ---------- COMPUTE MEAN ----------
        snaps = sorted(delays_by_snap.keys())

        times = []
        mean_delays = []

        for s in snaps:
            delays = delays_by_snap[s]

            if len(delays) < 5:  # evitar ruido fuerte
                continue

            if s not in snap_time:
                continue

            times.append(snap_time[s])
            mean_delays.append(np.mean(delays))

        # ---------- PLOT ----------
        ax.plot(times, mean_delays, label=galaxy)
        ax.set_xlim(0, 14)

    # label de simulación dentro del plot
    ax.text(
        0.05, 0.95,
        simulation,
        transform=ax.transAxes,
        va='top',
        ha='left'
    )


# ---------- LABELS ----------
axes[0].set_ylabel("Mean delay halo → disk [Gyr]")

for ax in axes:
    ax.set_xlabel("Cosmic time [Gyr]")

# ---------- LEGEND (solo en el primero) ----------
axes[0].legend()

# ---------- TIGHT LAYOUT ----------
plt.subplots_adjust(wspace=0.05)

# ---------- SAVE ----------
plt.savefig("images/mean_delay_vs_time_all_sims.pdf", dpi=200)
plt.close()

print("Saved combined plot")