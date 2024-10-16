import numpy as np
import yaml
import argparse
import pandas as pd
import json

from hestia.dataframe import make_dataframe

GLOBAL_CONFIG = yaml.safe_load(open("configs/global.yml"))


def calculate_disc_size_by_percentiles(
        positions: np.ndarray,
        masses: np.ndarray,
        config: dict) -> float:
    """
    This method calculates the radius and height of the disc given the position
    of the particles.

    Parameters
    ----------
    positions : np.ndarray
        The positions of the particles, of size (N, 3).
    masses : np.ndarray
        The masses of the particles, of size (N,).
    config : dict
        A dictionary with the configuration parameters.

    Returns
    -------
    rd : float
        The radius of the disc.
    hd : float
        The height of the disc.
    """

    rxy = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
    z = positions[:, 2]
    r = np.linalg.norm(positions, axis=1)

    mask = (r <= config["DISC_SIZE_SPHERICAL_CUT_CKPC"]) \
        & (np.abs(z) <= config["DISC_SIZE_ZCOORD_CUT_CKPC"])

    radius = np.percentile(
        a=rxy[mask],
        q=config["DISC_ENCLOSED_MASS_PERCENTILE"],
        weights=masses[mask],
    )
    heights = np.percentile(
        a=z[mask],
        q=[int((100 - config["DISC_ENCLOSED_MASS_PERCENTILE"]) // 2),
           int(100 - (100 - config["DISC_ENCLOSED_MASS_PERCENTILE"]) // 2)],
        weights=masses[mask],
    )

    return radius, heights[0], heights[1]


def calculate_disc_size(simulation: str, galaxy: str, config: dict):
    data = {"Configuration": config["RUN_CODE"],
            "Simulation": simulation,
            "Galaxy": galaxy,
            "Times_Gyr": [np.nan] * config["FIRST_SNAPSHOT"],
            "Redshift": [np.nan] * config["FIRST_SNAPSHOT"],
            "ExpansionFactor": [np.nan] * config["FIRST_SNAPSHOT"],
            "SnapshotNumbers": [np.nan] * config["FIRST_SNAPSHOT"],
            "VirialRadius_ckpc": [np.nan] * config["FIRST_SNAPSHOT"],
            "PercentileRadius_ckpc": [np.nan] * config["FIRST_SNAPSHOT"],
            "PercentileUpperHeight_ckpc": [np.nan] * config["FIRST_SNAPSHOT"],
            "PercentileLowerHeight_ckpc": [np.nan] * config["FIRST_SNAPSHOT"],
            "DiscRadius_ckpc": [np.nan] * config["FIRST_SNAPSHOT"],
            "DiscHeight_ckpc": [np.nan] * config["FIRST_SNAPSHOT"]}

    # Read virial radius
    virial_radius_data = pd.read_csv(
        f"data/{simulation}_{galaxy}/virial_radius.csv")

    for i in range(GLOBAL_CONFIG["FIRST_SNAPSHOT"],
                   GLOBAL_CONFIG["N_SNAPSHOTS"]):
        virial_radius = virial_radius_data["VirialRadius_ckpc"].loc[i]
        df = make_dataframe(simulation, i, galaxy)

        pos = df[["xPosition_ckpc", "yPosition_ckpc", "zPosition_ckpc"]]
        pos = pos.to_numpy()

        is_star = df["ParticleType"] == 4

        radius, lower_height, upper_height = \
            calculate_disc_size_by_percentiles(
                positions=pos[is_star],
                masses=df["Mass_Msun"][is_star],
                config=config)

        # Decide on disc radius and height based on time and virial radius
        rd = radius
        if df.time < config["VIRIAL_RADIUS_TIME_THRESHOLD_GYR"]:
            rd = np.min(
                [rd, config["VIRIAL_RADIUS_FRACTION"] * virial_radius])

        hd = np.mean([lower_height, upper_height])
        if df.time < config["VIRIAL_RADIUS_TIME_THRESHOLD_GYR"]:
            hd = np.min(
                [hd, config["VIRIAL_RADIUS_FRACTION"] * virial_radius])

        # Add data to dictionary
        data["Times_Gyr"][i] = df.time
        data["Redshift"][i] = df.redshift
        data["ExpansionFactor"][i] = df.expansion_factor
        data["SnapshotNumbers"][i] = i
        data["PercentileRadius_ckpc"][i] = radius
        data["PercentileLowerHeight_ckpc"][i] = lower_height
        data["PercentileUpperHeight_ckpc"][i] = upper_height
        data["DiscRadius_ckpc"] = rd
        data["DiscHeight_ckpc"] = hd

    # Save dictionary
    path = f"data/{simulation}_{galaxy}/" \
        + f"disc_size_config{config['RUN_CODE']}.json"
    with open(path, "w") as f:
        json.dump(data, f)


def main():
    # Get arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = yaml.safe_load(open(f"configs/{args.config}.yml"))

    calculate_disc_size(simulation="17_11", galaxy="MW", config=config)
    calculate_disc_size(simulation="17_11", galaxy="M31", config=config)
