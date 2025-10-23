import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM


class Cosmology:
    """
    A class used to manage the cosmological configuration.

    Methods
    -------
    redshift_to_time(redshift)
        Returns the corresponding age of the universe for this redshift in Gyr.
    redshift_to_lookback_time(redshift)
        Returns the corresponding lookback time for this redshift in Gyr.
    expansion_factor_to_redshift(a)
        Returns the corresponding redshift for this expansion factor.
    expansion_factor_to_time(a)
        Returns the corresponding age of the universe for this expansion
        factor in Gyr.
    redshift_to_expansion_factor(redshift)
        Returns the corresponding expansion factor for this redshift.
    rho_matter(redshift)
        Returns the matter density at this redshift in Msun/kpc^3.
    omega_matter(redshift)
        Returns the matter density parameter at this redshift.
    hubble(redshift)
        Returns the Hubble constant at this redshift in km/s/Mpc.
    critical_density(redshift)
        Returns the critical density at this redshift in Msun/kpc^3.
    """

    SMALL_HUBBLE_CONST: float = 0.6777
    HUBBLE_CONST: float = 67.77 * u.km / u.s / u.Mpc
    OMEGA_LAMBDA: float = 0.682
    OMEGA_BARYONS: float = 0.048
    OMEGA_MATTER: float = 0.318
    STATE_PARAM_DARK_ENERGY: float = -1.0
    STATE_PARAM_MATTER: float = 0.0
    GRAV_CONST = 4.3E-3 * u.pc * u.km**2 / u.s**2 / u.solMass
    CRITICAL_DENSITY = (3 * HUBBLE_CONST**2 / (8 * np.pi * GRAV_CONST)).to(
        u.solMass / u.kpc**3)

    def __init__(self) -> None:
        self.cosmology = FlatLambdaCDM(H0=Cosmology.HUBBLE_CONST,
                                       Om0=Cosmology.OMEGA_MATTER)
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

        return self.cosmology.lookback_time(redshift)

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

    def _aux_evol_calc(self, redshift: float) -> float:
        """
        Auxiliary method to calculate the evolution of the Hubble parameter.
        """
        return Cosmology.OMEGA_LAMBDA * (1 + redshift)**(3 * (
            1 + Cosmology.STATE_PARAM_DARK_ENERGY)) \
            + Cosmology.OMEGA_MATTER * (1 + redshift)**(3 * (
                1 + Cosmology.STATE_PARAM_MATTER))

    def rho_matter(self, redshift: float) -> float:
        """
        This method calculates the matter density at a given redshift.

        Parameters
        ----------
        redshift : float
            The redshift.

        Returns
        -------
        rho_matter : float
            The matter density at the given redshift.
        """

        return Cosmology.OMEGA_MATTER * Cosmology.CRITICAL_DENSITY \
            * (1 + redshift)**(3 * (1 + Cosmology.STATE_PARAM_MATTER))

    def rho_dark_energy(self, redshift: float) -> float:
        """
        This method calculates the dark energy density at a given redshift.

        Parameters
        ----------
        redshift : float
            The redshift.

        Returns
        -------
        rho_dark_energy : float
            The matter density at the given redshift.
        """

        return Cosmology.OMEGA_LAMBDA * Cosmology.CRITICAL_DENSITY \
            * (1 + redshift)**(3 * (1 + Cosmology.STATE_PARAM_DARK_ENERGY))

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

        return self.rho_matter(redshift) / self.critical_density(redshift)

    def omega_dark_energy(self, redshift: float) -> float:
        """
        This method calculates the dark energy density parameter at a given
        redshift.

        Parameters
        ----------
        redshift : float
            The redshift to transform.

        Returns
        -------
        float
            The dark energy density parameter at the given redshift.
        """

        return self.rho_dark_energy(redshift) / self.critical_density(redshift)

    def hubble(self, redshift: float) -> float:
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

        return Cosmology.HUBBLE_CONST * np.sqrt(self._aux_evol_calc(redshift))

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

        H = self.hubble(redshift)
        rho_c = 3 * H**2 / (8 * np.pi * Cosmology.GRAV_CONST)
        return rho_c.to(u.solMass / u.kpc**3)


if __name__ == "__main__":
    print(f"HUBBLE_CONSTANT: {Cosmology.HUBBLE_CONST}")
    print(f"CRITICAL_DENSITY: {Cosmology.CRITICAL_DENSITY}")
    c = Cosmology()
    print(f"AGE_OF_UNIVERSE: {c.present_time}")
    print()
    print(f"PRESENT_DAY_HUBBLE_PARAM: {c.hubble(0.0)}")
    print(f"PRESENT_DAY_CRIT_DENS: {c.critical_density(0)}")
    print(f"PRESENT_DAY_MATTER_DENS: {c.rho_matter(0)}")
    print(f"PRESENT_DAY_OMEGA_MATTER: {c.omega_matter(0)}")
    print(f"PRESENT_DAY_DARK_ENERGY_DENS: {c.rho_dark_energy(0)}")
    print(f"PRESENT_DAY_OMEGA_LAMBDA: {c.omega_dark_energy(0)}")
    print()
    print(f"Z=1 HUBBLE_PARAM: {c.hubble(1.0)}")
    print(f"Z=1 CRIT_DENS: {c.critical_density(1.0)}")
