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
        np.random.randint(low=0, high=size, size=int(0.5 * size))] \
        = GLOBAL_CONFIG["STAR_PARTICLE_TYPE"]
    data["StellarBirthTime_Gyr"][
        data["ParticleType"] == GLOBAL_CONFIG["GAS_PARTICLE_TYPE"]] = np.nan

    return pd.DataFrame(data)


def weighted_percentile(x: np.ndarray, w: np.ndarray, q: int) -> float:
    """
    Calculates the weighted percentile of array `x` using
    weights `w`.

    Parameters
    ---------
    x : np.ndarray
        The array of values.
    w : np.ndarray
        The weight of each value of `x`.
    q : int
        The percentile.
    """
    if (q < 0) or (q > 100):
        raise ValueError("Percentile must be between 0 and 100, inclusive.")
    if (q == 0):
        return x.min()
    if (q == 100):
        return x.max()
    idxs = np.argsort(x)
    x_sorted = x[idxs]
    w_sorted = w[idxs]
    cumsum = np.cumsum(w_sorted)
    target_idx = np.where(cumsum >= q * np.sum(w) / 100)[0][0] + 1
    return x_sorted[target_idx]
