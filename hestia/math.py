import numpy as np


def schechter(x: np.ndarray, amplitude: float,
              scale: float, exponent: float) -> np.ndarray:
    """
    Calculate the y values of x for a Schechter function with given
    parameters.

    Parameters
    ----------
    x : np.ndarray
        An array with the x coordinates.
    amplitude : float
        The amplitude of the Schechter function.
    scale : float
        The scale of the Schechter function.
    exponent : float
        The exponent of the Schechter function.

    Returns
    -------
    y : np.ndarray
        The result of applying the function to x.
    """
    y = amplitude * (x / scale)**exponent * np.exp(-x / scale)
    return y
