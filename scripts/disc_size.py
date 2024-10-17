import numpy as np
import yaml
import argparse
import pandas as pd
import json
from scipy.signal import savgol_filter

from hestia.dataframe import make_dataframe
from hestia.tools import weighted_percentile

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

    is_radius = (r <= config["DISC_SIZE_SPHERICAL_CUT_CKPC"])
    is_height = (np.abs(z) <= config["DISC_SIZE_ZCOORD_CUT_CKPC"])

    radius = weighted_percentile(
        x=rxy[is_radius], w=masses[is_radius],
        q=config["DISC_ENCLOSED_MASS_PERCENTILE"],
    )
    lower_height = weighted_percentile(
        x=z[is_radius & is_height], w=masses[is_radius & is_height],
        q=int((100 - config["DISC_ENCLOSED_MASS_PERCENTILE"]) // 2),
    )
    upper_height = weighted_percentile(
        x=z[is_radius & is_height], w=masses[is_radius & is_height],
        q=int(100 - (100 - config["DISC_ENCLOSED_MASS_PERCENTILE"]) // 2),
    )

    return radius, lower_height, upper_height


def calculate_disc_size(simulation: str, galaxy: str, config: dict):
    n_snapshots = GLOBAL_CONFIG["N_SNAPSHOTS"]

    data = {"Configuration": config["RUN_CODE"],
            "Simulation": simulation,
            "Galaxy": galaxy,
            "Times_Gyr": [np.nan] * n_snapshots,
            "Redshift": [np.nan] * n_snapshots,
            "ExpansionFactor": [np.nan] * n_snapshots,
            "SnapshotNumbers": [np.nan] * n_snapshots,
            "VirialRadius_ckpc": [np.nan] * n_snapshots,
            "PercentileRadius_ckpc": [np.nan] * n_snapshots,
            "PercentileUpperHeight_ckpc": [np.nan] * n_snapshots,
            "PercentileLowerHeight_ckpc": [np.nan] * n_snapshots,
            "DiscRadius_ckpc": [np.nan] * n_snapshots,
            "DiscHeight_ckpc": [np.nan] * n_snapshots}

    # Read virial radius
    virial_radius_data = pd.read_csv(
        f"data/{simulation}_{galaxy}/virial_radius.csv")

    for i in range(GLOBAL_CONFIG["FIRST_SNAPSHOT"], n_snapshots):
        virial_radius = virial_radius_data["VirialRadius_ckpc"].loc[i]
        df = make_dataframe(simulation, i, galaxy)

        pos = df[["xPosition_ckpc", "yPosition_ckpc", "zPosition_ckpc"]]

        is_star = df["ParticleType"] == 4

        radius, lower_height, upper_height = \
            calculate_disc_size_by_percentiles(
                positions=pos.to_numpy()[is_star],
                masses=df["Mass_Msun"].to_numpy()[is_star],
                config=config)

        # Decide on disc radius and height based on time and virial radius
        rd = radius
        if df.time < config["VIRIAL_RADIUS_TIME_THRESHOLD_GYR"]:
            rd = np.min(
                [rd, config["VIRIAL_RADIUS_FRACTION"] * virial_radius])

        hd = np.mean([np.abs(lower_height), np.abs(upper_height)])
        if df.time < config["VIRIAL_RADIUS_TIME_THRESHOLD_GYR"]:
            hd = np.min(
                [hd, config["VIRIAL_RADIUS_FRACTION"] * virial_radius])

        # Add data to dictionary
        data["Times_Gyr"][i] = df.time
        data["Redshift"][i] = df.redshift
        data["ExpansionFactor"][i] = df.expansion_factor
        data["SnapshotNumbers"][i] = i
        data["VirialRadius_ckpc"][i] = virial_radius
        data["PercentileRadius_ckpc"][i] = radius
        data["PercentileLowerHeight_ckpc"][i] = lower_height
        data["PercentileUpperHeight_ckpc"][i] = upper_height
        data["DiscRadius_ckpc"][i] = rd
        data["DiscHeight_ckpc"][i] = hd

    # Apply smoothing to the disc sizes
    data["DiscRadius_ckpc"] = _apply_filter(
        data["DiscRadius_ckpc"],
        config["DISC_SIZE_SMOOTHING_WINDOW_LENGTH"],
        config["DISC_SIZE_SMOOTHING_POLYORDER"])
    data["DiscHeight_ckpc"] = _apply_filter(
        data["DiscHeight_ckpc"],
        config["DISC_SIZE_SMOOTHING_WINDOW_LENGTH"],
        config["DISC_SIZE_SMOOTHING_POLYORDER"])

    # Save dictionary
    path = f"data/{simulation}_{galaxy}/" \
        + f"disc_size_config{config['RUN_CODE']}.json"
    with open(path, "w") as f:
        json.dump(data, f)


def _apply_filter(x: list, window_length: int, polyorder: int) -> list:
    y = np.array(x.copy())
    y_filtered = savgol_filter(y[~np.isnan(y)], window_length, polyorder)
    y_new = np.array([np.nan] * len(x))
    y_new[~np.isnan(y)] = y_filtered
    return list(y_new)


def main():
    # Get arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = yaml.safe_load(open(f"configs/{args.config}.yml"))

    calculate_disc_size(simulation="17_11", galaxy="MW", config=config)
    calculate_disc_size(simulation="17_11", galaxy="M31", config=config)


if __name__ == "__main__":
    main()
