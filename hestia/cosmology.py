import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM


class Cosmology:
    """
    A class used to manage the cosmological configuration.

    Attributes
    ----------
    hubble_factor : float
        The small Huble factor.
    hubble_constant : float
        The Hubble constant in km/s/Mpc.
    omega0 : float
        The cosmological matter density.
    omega_baryons : float
        The cosmological baryon density.
    omega_lambda : float
        The cosmological dark matter density.
    cosmology : FlatLambdaCDM
        The cosmology from AstroPy.
    present_time : float
        The present-day time in Gyr.

    Methods
    -------
    redshift_to_time(redshift)
        Returns the corresponding age of the universe for this redshift in Gyr.
    redshift_to_expansion_factor(redshift)
        Returns the corresponding expansion factor for this redshift.
    """

    SMALL_HUBBLE_CONST: float = 0.6777
    HUBBLE_CONST: float = 67.77 * u.km / u.s / u.Mpc
    OMEGA_LAMBDA: float = 0.682
    OMEGA_MATTER: float = 0.27
    OMEGA_BARYONS: float = 0.048
    OMEGA_0: float = 0.318
    GRAV_CONST = 4.3E-3 * u.pc * u.km**2 / u.s**2 / u.solMass

    def __init__(self) -> None:
        self.cosmology = FlatLambdaCDM(H0=Cosmology.HUBBLE_CONST,
                                       Om0=Cosmology.OMEGA_0)
        self.present_time: float = self.cosmology.age(0)

    def redshift_to_time(self, redshift: float) -> float:
        """
        This method calculates the corresponding age of the universe in Gyr
        for a given redshift value.

        Parameters
        ----------
        redshift : float
            The redshift to transform.

        Returns
        -------
        float
            The corresponding age of the universe in Gyr.
        """

        return self.cosmology.age(redshift)

    def redshift_to_lookback_time(self, redshift: float) -> float:
        """
        This method calculates the lookback time for a given redshift.

        Parameters
        ----------
        redshift : float
            The redshift to transform.

        Returns
        -------
        float
            The corresponding lookback time.
        """

        return self.cosmology.lookback_time(redshift)  # Gyr

    def expansion_factor_to_redshift(self, a: float) -> float:
        """
        This method calculatest the redshift asociated with a given value of
        the expansion factor.

        Parameters
        ----------
        a : float
            The expansion factor.

        Returns
        -------
        float
            The corresponding redshift.
        """

        return 1 / a - 1

    def expansion_factor_to_time(self, a: float) -> float:
        """
        This method calculates the cosmic time in Gyr asociated with a
        given value of the expansion factor.

        Parameters
        ----------
        a : float
            The expansion factor.

        Returns
        -------
        float
            The corresponding cosmic time.
        """

        redshift = self.expansion_factor_to_redshift(a)
        return self.redshift_to_time(redshift)

    def redshift_to_expansion_factor(self, redshift: float) -> float:
        """
        This method calculates the corresponding expansion factor
        for a given redshift value.

        Parameters
        ----------
        redshift : float
            The refshift to transform.

        Returns
        -------
        float
            The corresponding expansion factor.
        """

        return 1 / (1 + redshift)

    def omega_matter(self, redshift: float) -> float:
        """
        This method calculates the matter density parameter at a given
        redshift.

        Parameters
        ----------
        redshift : float
            The redshift to transform.

        Returns
        -------
        float
            The matter density parameter at the given redshift.
        """

        return Cosmology.OMEGA_0 * (1 + redshift)**3

    def hubble_constant(self, redshift: float) -> float:
        """
        This method calculates the Hubble constant at a given redshift.

        Parameters
        ----------
        redshift : float
            The redshift to transform.

        Returns
        -------
        float
            The Hubble constant at the given redshift in km/s/Mpc.
        """

        omega_matter = self.omega_matter(redshift)
        return Cosmology.HUBBLE_CONST \
            * np.sqrt(omega_matter + Cosmology.OMEGA_LAMBDA)

    def critical_density(self, redshift: float) -> float:
        """
        This method calculates the critical density at a given redshift.

        Parameters
        ----------
        redshift : float
            The redshift to transform.

        Returns
        -------
        float
            The critical density at the given redshift in Msun/kpc^3.
        """

        H = self.hubble_constant(redshift)
        rho_c = 3 * H**2 / (8 * np.pi * Cosmology.GRAV_CONST)
        return rho_c.to(u.solMass / u.kpc**3)


if __name__ == "__main__":
    print(Cosmology.HUBBLE_CONST)
    c = Cosmology()
    print(c.present_time)
    print(c.redshift_to_time(1.0))
    print(c.redshift_to_lookback_time(1.0))
    print(c.expansion_factor_to_time(1.0))
    print(c.hubble_constant(0.0))
    print(c.critical_density(0))
