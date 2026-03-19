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


for simulation in Settings.HIGH_RES_SIMULATIONS:

    plt.figure()

    for galaxy in Settings.GALAXIES:

        # ---------- LOAD DATA ----------
        filepath = f"results/{simulation}_{galaxy}/halo_to_disk_tracers.json"

        with open(filepath) as f:
            data = json.load(f)

        time_file = f"r200_t_{galaxy}_{simulation}.csv"
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

            if len(delays) == 0:
                continue

            times.append(snap_time[s])
            mean_delays.append(np.mean(delays))

        # ---------- PLOT ----------
        plt.plot(times, mean_delays, label=galaxy)

    plt.xlabel("Cosmic time [Gyr]")
    plt.ylabel("Mean delay halo → disk [Gyr]")
    plt.title(f"{simulation}")
    plt.legend()

    plt.savefig(f"results/{simulation}_mean_delay_vs_time.png")
    plt.close()

    print(f"Saved plot for {simulation}")