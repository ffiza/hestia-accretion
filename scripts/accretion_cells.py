import numpy as np
import pandas as pd
import yaml

GLOBAL_CONFIG = yaml.safe_load(open("configs/global.yml"))
DF_COLUMNS = ["xPosition_ckpc", "yPosition_ckpc", "zPosition_ckpc",
              "Mass_Msun", "ParticleType", "StellarBirthTime_Gyr"]


def calculate_net_accretion(df1: pd.DataFrame, df2: pd.DataFrame,
                            t1_gyr: float, t2_gyr: float,
                            geometry_ckpc: tuple) -> float:
    """
    This method calculates the net accretion rate between two snapshots. It
    takes two data frames, `df1` and `df2`, with the information of the
    gas and stars in each snapshot. Each data frame must contain the following
    columns: `xPosition_ckpc`, `yPosition_ckpc`, `zPosition_ckpc`,
    `Mass_Msun`, `ParticleType`, and `StellarBirthTime_Gyr`.

    The columns are:
    - `xPosition_ckpc`: x-coordinate of the particles in ckpc.
    - `yPosition_ckpc`: y-coordinate of the particles in ckpc.
    - `zPosition_ckpc`: z-coordinate of the particles in ckpc
    - `Mass_Msun`: mass of the particles in Msun.
    - `ParticleType`: type of the particle (0 for gas and 4 for stars).
    - `StellarBirthTime_Gyr`: time of birth of the star in Gyr. If the particle
      is not a star, this element should be set to `np.nan`.

    The method also takes the time of each snapshot (`t1_gyr` and `t2_gyr`) in
    Gyr and a tuple that indicates the geometry of the region in which to
    calculate the accretion.

    The method returns the net accretion rate in Msun/yr.

    Parameters
    ----------
    df1 : pd.DataFrame
        Data frame with the gas and stars in the first snapshot.
    df2 : pd.DataFrame
        Data frame with the gas and stars in the second snapshot.
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

    for df in [df1, df2]:  # Check data type of field `ParticleType`
        df["ParticleType"] = df["ParticleType"].astype(np.int8)

    if t2_gyr - t1_gyr <= 0:  # Check if the times are correct
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
                & (np.abs(df["zPosition_ckpc"]) <= geometry_ckpc[1] / 2)

    # Calculate mass of new stars
    is_new_star = (
        df2["ParticleType"] == GLOBAL_CONFIG["STAR_PARTICLE_TYPE"]) \
        & (df2["StellarBirthTime_Gyr"] >= t1_gyr)
    new_star_mass = (df2["Mass_Msun"][
        is_new_star & (df2["IsGeometry"] == 1)]).sum()

    # Calculate mass of gas inside geometry
    gas_mass_1 = (df1["Mass_Msun"][
        (df1["ParticleType"] == GLOBAL_CONFIG["GAS_PARTICLE_TYPE"])
        & (df1["IsGeometry"] == 1)]).sum()
    gas_mass_2 = (df2["Mass_Msun"][(
        df2["ParticleType"] == GLOBAL_CONFIG["GAS_PARTICLE_TYPE"])
        & (df2["IsGeometry"] == 1)]).sum()

    # Calculate the net accretion rate in Msun/yr
    net_accretion_rate = (gas_mass_2 - gas_mass_1 + new_star_mass) \
        / delta_time

    return net_accretion_rate


def calculate_net_accretion_evolution(simulation: str,
                                      geometry_ckpc: tuple) -> None:
    """
    This method calculates the evolution of the net accretion rate for a
    given simulation.

    Parameters
    ----------
    simulation : str
        The name of the simulation: `17_11`, `37_11` or `9_18`.
    geometry_ckpc : tuple
        Tuple that indicates the geometry of the region in which to calculate
        the accretion. If the tuple has only one element, the method asumes an
        spheroid of that radius. If the tuple cotains two elements, the method
        assumes a disc with the first element as the radius and the second as
        the height. All the dimensions should be expressed in ckpc.

    """
    data = np.zeros(
        (GLOBAL_CONFIG["N_SNAPSHOTS"] - GLOBAL_CONFIG["FIRST_SNAPSHOT"],
         4))
    for i in range(GLOBAL_CONFIG["FIRST_SNAPSHOT"],
                   GLOBAL_CONFIG["N_SNAPSHOTS"]):

        t1_gyr = ...
        t2_gyr = ...
        redshift2 = ...
        exp_fact2 = ...

        # Milky-Way
        df1 = ...
        df2 = ...
        net_accretion_mw = calculate_net_accretion(
            df1=df1, df2=df2, t1_gyr=t1_gyr, t2_gyr=t2_gyr,
            geometry_ckpc=geometry_ckpc)

        # M31
        df1 = ...
        df2 = ...
        net_accretion_m31 = calculate_net_accretion(
            df1=df1, df2=df2, t1_gyr=t1_gyr, t2_gyr=t2_gyr,
            geometry_ckpc=geometry_ckpc)

        data[i] = np.array([t2_gyr, redshift2, exp_fact2,
                            net_accretion_mw, net_accretion_m31,])

    data = pd.DataFrame(data=data,
                        columns=["Time_Gyr", "Redshift", "ExpansionFactor",
                                 "NetAccretion_Msun/yr_MW",
                                 "NetAccretion_Msun/yr_M31",])

    data.to_csv(f"results/{simulation}/net_accretion.csv")


if __name__ == "__main__":
    # TODO: Run the calculation for all galaxies.
    pass
