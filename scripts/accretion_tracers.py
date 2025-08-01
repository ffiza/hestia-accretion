import yaml
import json
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool

from hestia.df_type import DFType
from hestia.settings import Settings
from hestia.dataframe import make_dataframe
from hestia.tools import timer

GLOBAL_CONFIG = yaml.safe_load(open("configs/global.yml"))
DF_COLUMNS = ["TracerID", "xPosition_ckpc", "yPosition_ckpc", "zPosition_ckpc",
              "ParentCellType"]

@timer
def calculate_accretion(df1: pd.DataFrame, df2: pd.DataFrame,
                        t1_gyr: float, t2_gyr: float,
                        geometry_ckpc: tuple) -> tuple:
    """
    This method calculates the accretion rates between two snapshots using
    tarcer particles. It takes two data frames, `df1` and `df2`, with the
    information of the particles in each snapshot. Each data frame must
    contain the following columns: `TracerID`, `xPosition_ckpc`,
    `yPosition_ckpc`, `zPosition_ckpc`, `ParentCellType`; also, as metadata,
    the dataframes must contain the value of the target gas mass of the
    corresponding simulation in units of Msun.

    The method also takes the time of each snapshot (`t1_gyr` and `t2_gyr`) in
    Gyr and a tuple that indicates the geometry of the region in which to
    calculate the accretion.

    The method returns the inflow and outflow accretion rate in Msun/yr.

    Parameters
    ----------
    df1 : pd.DataFrame
        Data frame with the tracer particles in the first snapshot.
    df2 : pd.DataFrame
        Data frame with the tracer particles in the second snapshot.
    t1_gyr : float
        Time of the first snapshot in Gyr.
    t2_gyr : float
        Time of the second snapshot in Gyr.
    geometry_ckpc : tuple
        Tuple that indicates the geometry of the region in which to calculate
        the accretion. If the tuple has only one element, the method asumes an
        spheroid of that radius. If the tuple cotains two elements, the method
        assumes a disc with the first element as the radius and the second as
        the height. All the dimensions should be expressed in ckpc.
    """

    for df in [df1, df2]:  # Check all columns are in the data frames
        for column in DF_COLUMNS:
            if column not in list(df.columns):
                raise ValueError(f"`{column}` is not in the dataframe.")

    if t2_gyr <= t1_gyr:  # Check if the times are correct
        raise ValueError("`t2_gyr` must be greater than `t1_gyr`.")

    # Define the geometry
    if len(geometry_ckpc) == 1:
        geometry_type = "spheroid"
    elif len(geometry_ckpc) == 2:
        geometry_type = "disc"
    else:
        raise ValueError(
            "`geometry` should be a tuple of length 1 "
            "for spheroids or length 2 for discs.")

    # Calculate new coordinates
    for df in [df1, df2]:
        pos = df[["xPosition_ckpc",
                  "yPosition_ckpc",
                  "zPosition_ckpc"]].to_numpy()
        if geometry_type == "spheroid":
            df["SphericalRadius_ckpc"] = np.linalg.norm(pos, axis=1)
        if geometry_type == "disc":
            df["CylindricalRadius_ckpc"] = np.linalg.norm(pos[:, :2], axis=1)

    # Calculate the time between snapshots in yr
    delta_time = (t2_gyr - t1_gyr) * 1E9

    # Tag particles inside geometry
    for df in [df1, df2]:
        if geometry_type == "spheroid":
            df["IsGeometry"] = df["SphericalRadius_ckpc"] <= geometry_ckpc[0]
        if geometry_type == "disc":
            df["IsGeometry"] = (
                df["CylindricalRadius_ckpc"] <= geometry_ckpc[0]) \
                & (np.abs(df["zPosition_ckpc"]) <= geometry_ckpc[1])

    # Compute the amount of inflowing/outflowing particles
    inflowing_number = (
        (df1["IsGeometry"] == 0) &
        (df1["ParentCellType"] == GLOBAL_CONFIG["GAS_PARTICLE_TYPE"]) &
        (df2["IsGeometry"] == 1) &
        ((df2["ParentCellType"] == GLOBAL_CONFIG["GAS_PARTICLE_TYPE"]) |
         (df2["ParentCellType"] == GLOBAL_CONFIG["STAR_PARTICLE_TYPE"]))).sum()
    outflowing_number = (
        (df1["IsGeometry"] == 1) &
        (df1["ParentCellType"] == GLOBAL_CONFIG["GAS_PARTICLE_TYPE"]) &
        (df2["IsGeometry"] == 0) &
        ((df2["ParentCellType"] == GLOBAL_CONFIG["GAS_PARTICLE_TYPE"]) |
         (df2["ParentCellType"] == GLOBAL_CONFIG["STAR_PARTICLE_TYPE"]))).sum()

    # Compute rates in Msun/yr
    in_rate = inflowing_number * df1.target_gas_mass / delta_time
    out_rate = outflowing_number * df1.target_gas_mass / delta_time

    return (in_rate, out_rate)

@timer
def calculate_accretion_evolution(simulation: str,
                                  galaxy: str,
                                  config: dict) -> None:
    """
    This method calculates the evolution of the accretion rates for tracers
    for a given simulation.

    Parameters
    ----------
    simulation : str
        The name of the simulation: `17_11`, `37_11` or `9_18`.
    galaxy : str
        The name of the galaxy: `MW` or `M31`.
    geometry_ckpc : tuple
        Tuple that indicates the geometry of the region in which to calculate
        the accretion. If the tuple has only one element, the method asumes an
        spheroid of that radius. If the tuple cotains two elements, the method
        assumes a disc with the first element as the radius and the second as
        the height. All the dimensions should be expressed in ckpc.
    config : dict
        A dictionary with the configuration parameters.
    """
    n_snapshots = GLOBAL_CONFIG["N_SNAPSHOTS"]

    data = {"Configuration": config["RUN_CODE"],
            "Simulation": simulation,
            "Galaxy": galaxy,
            "Times_Gyr": [np.nan] * n_snapshots,
            "Redshift": [np.nan] * n_snapshots,
            "ExpansionFactor": [np.nan] * n_snapshots,
            "SnapshotNumbers": [np.nan] * n_snapshots,
            "InflowRate_Msun/yr": [np.nan] * n_snapshots,
            "OutflowRate_Msun/yr": [np.nan] * n_snapshots}

    path = f"results/{simulation}_{galaxy}/" \
        + f"disc_size_{config['RUN_CODE']}.json"
    with open(path) as f:
        disc_size = json.load(f)

    df1 = make_dataframe(
        simulation, GLOBAL_CONFIG["FIRST_SNAPSHOT"], galaxy,
        DFType.TRACERS, np.inf)
    for i in range(GLOBAL_CONFIG["FIRST_SNAPSHOT"] + 1, n_snapshots):

        # Define geometry
        rd = disc_size["DiscRadius_ckpc"][i]
        hd = disc_size["DiscHeight_ckpc"][i]

        df2 = make_dataframe(simulation, i, galaxy, DFType.TRACERS, np.inf)
        inflow_rate, outflow_rate = calculate_accretion(
            df1=df1, df2=df2, t1_gyr=df1.time, t2_gyr=df2.time,
            geometry_ckpc=(rd, hd))

        data["Times_Gyr"][i] = df2.time
        data["Redshift"][i] = df2.redshift
        data["ExpansionFactor"][i] = df2.expansion_factor
        data["SnapshotNumbers"][i] = i
        data["InflowRate_Msun/yr"][i] = inflow_rate
        data["OutflowRate_Msun/yr"][i] = outflow_rate

        # Save df2 as df1 for the next step
        df1 = df2.copy()
        df1.__dict__.update(df2.__dict__)

    # Save data
    path = f"results/{simulation}_{galaxy}/" \
        + f"accretion_tracers_{config['RUN_CODE']}.json"
    with open(path, "w") as f:
        json.dump(data, f)


def main():
    # Get arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = yaml.safe_load(open(f"configs/{args.config}.yml"))

    for simulation in Settings.SIMULATIONS:
        for galaxy in Settings.GALAXIES:
            calculate_accretion_evolution(
                simulation=simulation, galaxy=galaxy, config=config)

    arguments = [(simulation, galaxy, config)
                 for simulation in Settings.SIMULATIONS
                 for galaxy in Settings.GALAXIES]
    Pool(GLOBAL_CONFIG["N_PROCESSES"]).starmap(
        calculate_accretion_evolution, arguments)


if __name__ == "__main__":
    main()
