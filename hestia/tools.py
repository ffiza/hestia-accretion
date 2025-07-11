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
    if (x.shape[0] != w.shape[0]):
        raise ValueError("`x` and `w` must have the same dimensions.")
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


def windowed_average(x: np.ndarray, y: np.ndarray,
                     window_length: float) -> np.ndarray:
    """
    Compute a moving average of `y` over a sliding window centered at each
    point in `x`.

    For each value in `x`, this function computes the average of `y` values
    whose corresponding `x` values fall within a window of width
    `window_length`, centered at the current `x[i]`. If no points fall in the
    window, `NaN` is returned at that index.

    Parameters
    ----------
    x : np.ndarray
        A 1D array of x-values (must be sorted if spatial locality is assumed).
    y : np.ndarray
        A 1D array of y-values, same length as `x`.
    window_length : float
        Width of the window over which to compute the local average.

    Returns
    -------
    y_avg : np.ndarray
        A 1D array of the same shape as `x`, containing the windowed average
        of `y`. If no points fall within the window at a given `x[i]`, the
        result is `np.nan`.
    """
    y_avg = np.nan * np.ones(x.shape[0])
    for i in range(len(x)):
        mask = (x >= x[i] - window_length / 2) \
            & (x <= x[i] + window_length / 2)
        if np.sum(mask) > 0:
            y_avg[i] = np.nanmean(y[mask])
    return y_avg
