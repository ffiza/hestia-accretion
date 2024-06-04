import numpy as np
import pandas as pd
import yaml

GLOBAL_CONFIG = yaml.safe_load(open("configs/global.yml"))


def generate_dummy_df(size: int) -> pd.DataFrame:
    """
    This method creates an artificial data frame that emulates the
    contents of the data obtained from the simulation. The only
    purpouse of this function is to test the accretion calculation.

    Parameters
    ----------
    size : int
        The number of rows in the data frame.

    Returns
    -------
    pd.DataFrame
        The data frame with the particle information.
    """

    data = {
        "xPosition_ckpc": np.random.uniform(0, 500, size),
        "yPosition_ckpc": np.random.uniform(0, 500, size),
        "zPosition_ckpc": np.random.uniform(0, 500, size),
        "Mass_Msun": np.ones(size),
        "ParticleType": GLOBAL_CONFIG["GAS_PARTICLE_TYPE"]
            * np.ones(size, dtype=np.int8),
        "StellarBirthTime_Gyr": np.random.uniform(0, 14, size),
    }

    data["ParticleType"][
        np.random.randint(
            low=0, high=size, size=int(0.5 * size))] \
                = GLOBAL_CONFIG["STAR_PARTICLE_TYPE"]
    data["StellarBirthTime_Gyr"][
        data["ParticleType"] == GLOBAL_CONFIG["GAS_PARTICLE_TYPE"]] = np.nan

    return pd.DataFrame(data)
