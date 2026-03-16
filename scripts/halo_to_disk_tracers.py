import json
import os
import yaml

from hestia.settings import Settings


GLOBAL_CONFIG = yaml.safe_load(open("configs/global.yml"))

for simulation in Settings.HIGH_RES_SIMULATIONS:
    for galaxy in Settings.GALAXIES:



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
            results[tid] = {
                "halo_snap": halo_entry[tid],
                "disk_snap": disk_entry.get(tid, None)
            }

        with open(f"results/{simulation}_{galaxy}/halo_to_disk_tracers.json", "w") as f:
            json.dump(results, f)

        print("Saved tracer catalog: halo_to_disk_tracers.json")