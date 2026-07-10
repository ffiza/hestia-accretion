import json
import os
import yaml
import csv
import numpy as np

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

                print("Skipping bad row:", row)
                continue

            snap_time[snap] = time

    return snap_time


GLOBAL_CONFIG = yaml.safe_load(open("configs/global.yml"))


for simulation in Settings.HIGH_RES_SIMULATIONS:

    for galaxy in Settings.GALAXIES:

        print(f"{simulation} {galaxy}")

        time_file = (
            f"data/hestia/r200_t/r200_t_{galaxy}_{simulation}.csv"
        )

        snap_time = load_snapshot_times(time_file)

        # -----------------------------------
        # first halo entry
        # -----------------------------------

        halo_entry = {}

        # -----------------------------------
        # first disk entry after halo
        # -----------------------------------

        disk_entry = {}

        for snap in range(
            GLOBAL_CONFIG["FIRST_SNAPSHOT"],
            GLOBAL_CONFIG["N_SNAPSHOTS"]
        ):

            # ---------- HALO ----------

            halo_file = (
                f"results/{simulation}_{galaxy}"
                f"/accreted_ids_halo_snap{snap}.json"
            )

            if os.path.exists(halo_file):

                with open(halo_file) as f:

                    data = json.load(f)

                for tid in data["InflowingTracerIDs"]:

                    if tid not in halo_entry:

                        halo_entry[tid] = snap

            # ---------- DISK ----------

            disk_file = (
                f"results/{simulation}_{galaxy}"
                f"/accreted_ids_stellar_disc_snap{snap}.json"
            )

            if os.path.exists(disk_file):

                with open(disk_file) as f:

                    data = json.load(f)

                for tid in data["InflowingTracerIDs"]:

                    if tid in halo_entry:

                        if snap > halo_entry[tid]:

                            if tid not in disk_entry:

                                disk_entry[tid] = snap

        # ==================================================
        # Group by disk entry snapshot
        # ==================================================

        results = {}

        for snap in range(
            GLOBAL_CONFIG["FIRST_SNAPSHOT"],
            GLOBAL_CONFIG["N_SNAPSHOTS"]
        ):

            results[snap] = {

                "disk_time_Gyr": snap_time.get(snap),

                "N_disk": 0,

                "N_with_halo": 0,

                "fraction_with_halo": None,

                "mean_delay_Gyr": None,

                "median_delay_Gyr": None,

                "p16_delay_Gyr": None,

                "p84_delay_Gyr": None,

                "delays_Gyr": []

            }

        # ----------------------------------------
        # loop over tracers that reach the disk
        # ----------------------------------------

        for tid, d_snap in disk_entry.items():

            results[d_snap]["N_disk"] += 1

            if tid in halo_entry:

                h_snap = halo_entry[tid]

                if h_snap < d_snap:

                    delay = (
                        snap_time[d_snap]
                        - snap_time[h_snap]
                    )

                    results[d_snap]["N_with_halo"] += 1

                    results[d_snap]["delays_Gyr"].append(delay)

        # ----------------------------------------
        # statistics
        # ----------------------------------------

        for snap in results:

            n_disk = results[snap]["N_disk"]

            n_halo = results[snap]["N_with_halo"]

            if n_disk > 0:

                results[snap]["fraction_with_halo"] = (
                    n_halo / n_disk
                )

            delays = results[snap]["delays_Gyr"]

            if len(delays) > 0:

                results[snap]["mean_delay_Gyr"] = (
                    float(np.mean(delays))
                )

                results[snap]["median_delay_Gyr"] = (
                    float(np.median(delays))
                )

                results[snap]["p16_delay_Gyr"] = (
                    float(np.percentile(delays, 16))
                )

                results[snap]["p84_delay_Gyr"] = (
                    float(np.percentile(delays, 84))
                )

        # ==================================================
        # save
        # ==================================================

        outfile = (
            f"results/{simulation}_{galaxy}"
            "/disk_arrival_delay_stats.json"
        )

        with open(outfile, "w") as f:

            json.dump(results, f, indent=4)

        print("Saved", outfile)