"""Environment

This script computes the overdensity of the environment using the delta_1200
observable, defined in Creasey et al. (2015).
"""
import yaml
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool

from hestia.settings import Settings
from hestia.dataframe import make_dataframe


def calculate_overdensity(df: pd.DataFrame, distance: float) -> np.ndarray:
    GLOBAL_CONFIG = yaml.safe_load(open("configs/global.yml"))
    df["SphericalRadius_ckpc"] = np.linalg.norm(
        df[["xPosition_ckpc", "yPosition_ckpc", "zPosition_ckpc"]].values,
        axis=1
    )

    omega0 = GLOBAL_CONFIG["OMEGA_0"]
    hubbleparam = GLOBAL_CONFIG["SMALL_HUBBLE_CONST"]
    omegalambda = GLOBAL_CONFIG["OMEGA_LAMBDA"]

    omega0 = omega0 * (1 + df.redshift)**3
    H = 100 * hubbleparam * np.sqrt(omega0 + omegalambda)  # km s^-1 Mpc^-1
    critical_density = 3 * H**2 / (1E3 * 8 * np.pi * 4.3E-3)  # Msun kpc^-3

    mass = df["Mass_Msun"][df["SphericalRadius_ckpc"] < distance].sum()
    vol = (4/3 * np.pi * df.expansion_factor**3 * distance**3)
    mean_density = mass / vol

    overdensity = mean_density / critical_density / omega0

    return np.asarray(
        [df.snapshot_number, df.time, overdensity])


def calculate_overdensity_in_simulation(simulation: str,
                                        galaxy: str,
                                        snapshot_number: int,
                                        distance: float):
    GLOBAL_CONFIG = yaml.safe_load(open("configs/global.yml"))
    if snapshot_number < GLOBAL_CONFIG["FIRST_SNAPSHOT"]:
        return np.array([np.nan] * 3)
    df = make_dataframe(
        SimName=simulation, SnapNo=snapshot_number,
        MW_or_M31=galaxy, max_radius=distance)
    return calculate_overdensity(df=df, distance=distance)


def calculate_overdensity_evolution(simulation: str,
                                    galaxy: str,
                                    config: dict,
                                    distance: float) -> None:
    GLOBAL_CONFIG = yaml.safe_load(open("configs/global.yml"))
    n_processes = GLOBAL_CONFIG["N_PROCESSES"]
    n_snapshots = GLOBAL_CONFIG["N_SNAPSHOTS"]

    arguments = [(simulation, galaxy, i, distance) for i in range(n_snapshots)]
    data = np.array(Pool(n_processes).starmap(
        calculate_overdensity_in_simulation, arguments))

    df = pd.DataFrame(
        data, columns=["SnapshotNumbers", "Times_Gyr", "Delta"]
    )
    # df["SnapshotNumbers"] = df["SnapshotNumbers"].astype(int)

    df.to_csv(f"results/{simulation}_{galaxy}/delta_{int(distance)}.csv",
              index=False)


if __name__ == "__main__":
    DISTANCE: float = 1200.0  # ckpc

    # Get arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = yaml.safe_load(open(f"configs/{args.config}.yml"))

    for simulation in Settings.SIMULATIONS:
        for galaxy in Settings.GALAXIES:
            calculate_overdensity_evolution(
                simulation=simulation, galaxy=galaxy,
                config=config, distance=DISTANCE)
