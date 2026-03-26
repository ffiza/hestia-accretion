import json
import os
import yaml

from hestia.settings import Settings
import csv

def load_snapshot_times(filepath):
    snap_time = {}

    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:

            if not row["Time_Gyr"] or not row["SnapshotNumber"]:
                continue  # skip filas rotas

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

        time_file = f"data/hestia/r200_t/r200_t_{galaxy}_{simulation}.csv"
        snap_time = load_snapshot_times(time_file)


        # store first halo entry
        halo_entry = {}

        # store first disk entry after halo entry
        disk_entry = {}

        for snap in range(GLOBAL_CONFIG["FIRST_SNAPSHOT"], GLOBAL_CONFIG["N_SNAPSHOTS"]):

            # ---------- HALO ----------
            # halo_file = halo_template.format(snap)
            halo_file = f"results/{simulation}_{galaxy}/accreted_ids_halo_snap{snap}.json"
            if os.path.exists(halo_file):
                with open(halo_file) as f:
                    data = json.load(f)

                for tid in data["InflowingTracerIDs"]:
                    if tid not in halo_entry:
                        halo_entry[tid] = snap

            # ---------- DISK ----------
            # disk_file = disk_template.format(snap)
            disk_file = f"results/{simulation}_{galaxy}/accreted_ids_stellar_disc_snap{snap}.json"
            if os.path.exists(disk_file):
                with open(disk_file) as f:
                    data = json.load(f)

                for tid in data["InflowingTracerIDs"]:
                    if tid in halo_entry:
                        if snap > halo_entry[tid]:
                            if tid not in disk_entry:
                                disk_entry[tid] = snap


        # ---------- STATISTICS ----------

        total_halo = len(halo_entry)
        reached_disk = len(disk_entry)

        print("Total tracers accreted to halo:", total_halo)
        print("Reached disk later:", reached_disk)
        print("Fraction reaching disk:", reached_disk / total_halo)


        # ---------- DELAY TIMES ----------

        delays = []

        for tid in disk_entry:
            delays.append(disk_entry[tid] - halo_entry[tid])

        if len(delays) != 0:
            print("Mean delay (snapshots):", sum(delays) / len(delays))


        # ---------- SAVE RESULTS ----------


        results = {}

        for tid in halo_entry:

            h_snap = halo_entry[tid]
            d_snap = disk_entry.get(tid, None)

            if d_snap is not None:
                delay = snap_time[d_snap] - snap_time[h_snap]
            else:
                delay = None

            results[tid] = {
                "halo_snap": h_snap,
                "disk_snap": d_snap,
                "halo_time_Gyr": snap_time.get(h_snap, None),
                "disk_time_Gyr": snap_time.get(d_snap, None) if d_snap else None,
                "delay_Gyr": delay
            }

        with open(f"results/{simulation}_{galaxy}/halo_to_disk_tracers.json", "w") as f:
            json.dump(results, f)

        print("Saved tracer catalog: halo_to_disk_tracers.json")