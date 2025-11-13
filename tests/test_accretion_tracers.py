import yaml
import unittest
import numpy as np
import pandas as pd

from hestia.accretion_region import AccretionRegion, HaloRegion, FilterType


GLOBAL_CONFIG = yaml.safe_load(open("configs/global.yml"))
DF_COLUMNS = ["TracerID", "xPosition_ckpc", "yPosition_ckpc",
              "zPosition_ckpc", "ParentCellType"]


def calculate_accretion(df1: pd.DataFrame, df2: pd.DataFrame,
                        t1_gyr: float, t2_gyr: float,
                        accretion_region: AccretionRegion) -> tuple:
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
    accretion_region : AccretionRegion
        The region onto which to calculate the accretion rate.
    """

    for df in [df1, df2]:  # Check all columns are in the data frames
        for column in DF_COLUMNS:
            if column not in list(df.columns):
                raise ValueError(f"`{column}` is not in the dataframe.")

    if t2_gyr <= t1_gyr:  # Check if the times are correct
        raise ValueError("`t2_gyr` must be greater than `t1_gyr`.")

    # Calculate new coordinates
    for df in [df1, df2]:
        pos = df[["xPosition_ckpc",
                  "yPosition_ckpc",
                  "zPosition_ckpc"]].to_numpy()
        df["SphericalRadius_ckpc"] = np.linalg.norm(pos, axis=1)
        df["CylindricalRadius_ckpc"] = np.linalg.norm(pos[:, :2], axis=1)

    # Calculate the time between snapshots in yr
    delta_time = (t2_gyr - t1_gyr) * 1E9

    # Tag particles inside geometry
    for df in [df1, df2]:
        df["IsGeometry"] = accretion_region.select(df, FilterType.IN)

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


class AccretionTracersTests(unittest.TestCase):
    def test_01(self):
        df1 = pd.DataFrame({
            "TracerID": [1, 2],
            "xPosition_ckpc": [500, 500],
            "yPosition_ckpc": [500, 500],
            "zPosition_ckpc": [500, 500],
            "ParentCellType": [0, 0],
        })
        df1.target_gas_mass = 1.0
        t1 = 1.0 / 1E9

        df2 = pd.DataFrame({
            "TracerID": [1, 2],
            "xPosition_ckpc": [5, 5],
            "yPosition_ckpc": [5, 5],
            "zPosition_ckpc": [5, 5],
            "ParentCellType": [0, 0],
        })
        df1.target_gas_mass = 1.0
        t2 = 2.0 / 1E9

        accretion_region = HaloRegion(100.0)

        in_rate, _ = calculate_accretion(df1, df2, t1, t2, accretion_region)
        self.assertAlmostEqual(in_rate, 2.0)

    def test_02(self):
        df1 = pd.DataFrame({
            "TracerID": [1, 2],
            "xPosition_ckpc": [500, 500],
            "yPosition_ckpc": [500, 500],
            "zPosition_ckpc": [500, 500],
            "ParentCellType": [0, 0],
        })
        df1.target_gas_mass = 1.0
        t1 = 1.0 / 1E9

        df2 = pd.DataFrame({
            "TracerID": [1, 2],
            "xPosition_ckpc": [5, 5],
            "yPosition_ckpc": [5, 5],
            "zPosition_ckpc": [5, 5],
            "ParentCellType": [0, 0],
        })
        df1.target_gas_mass = 1.0
        t2 = 2.0 / 1E9

        accretion_region = HaloRegion(100.0)

        _, out_rate = calculate_accretion(df1, df2, t1, t2, accretion_region)
        self.assertAlmostEqual(out_rate, 0.0)

    def test_03(self):
        df1 = pd.DataFrame({
            "TracerID": [1, 2],
            "xPosition_ckpc": [500, 500],
            "yPosition_ckpc": [500, 500],
            "zPosition_ckpc": [500, 500],
            "ParentCellType": [0, 0],
        })
        df1.target_gas_mass = 1.0
        t1 = 1.0 / 1E9

        df2 = pd.DataFrame({
            "TracerID": [1, 2],
            "xPosition_ckpc": [5, 5],
            "yPosition_ckpc": [5, 5],
            "zPosition_ckpc": [5, 5],
            "ParentCellType": [0, 4],
        })
        df1.target_gas_mass = 1.0
        t2 = 2.0 / 1E9

        accretion_region = HaloRegion(100.0)

        _, out_rate = calculate_accretion(df1, df2, t1, t2, accretion_region)
        self.assertAlmostEqual(out_rate, 0.0)

    def test_04(self):
        df1 = pd.DataFrame({
            "TracerID": [1, 2],
            "xPosition_ckpc": [500, 500],
            "yPosition_ckpc": [500, 500],
            "zPosition_ckpc": [500, 500],
            "ParentCellType": [4, 4],
        })
        df1.target_gas_mass = 1.0
        t1 = 1.0 / 1E9

        df2 = pd.DataFrame({
            "TracerID": [1, 2],
            "xPosition_ckpc": [5, 5],
            "yPosition_ckpc": [5, 5],
            "zPosition_ckpc": [5, 5],
            "ParentCellType": [0, 4],
        })
        df1.target_gas_mass = 1.0
        t2 = 2.0 / 1E9

        accretion_region = HaloRegion(100.0)

        in_rate, _ = calculate_accretion(df1, df2, t1, t2, accretion_region)
        self.assertAlmostEqual(in_rate, 0.0)


if __name__ == '__main__':
    unittest.main()
