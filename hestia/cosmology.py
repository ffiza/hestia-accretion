from astropy.cosmology import FlatLambdaCDM
import yaml

GLOBAL_CONFIG = yaml.safe_load(open("configs/global.yml"))


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

    def __init__(self) -> None:
        self.hubble_factor = GLOBAL_CONFIG["SMALL_HUBBLE_CONST"]
        self.hubble_constant = GLOBAL_CONFIG["HUBBLE_CONST"]
        self.omega0 = GLOBAL_CONFIG["OMEGA_0"]
        self.omega_baryons = GLOBAL_CONFIG["OMEGA_BARYONS"]
        self.omega_lambda = GLOBAL_CONFIG["OMEGA_LAMBDA"]
        self.cosmology = FlatLambdaCDM(H0=self.hubble_constant,
                                       Om0=self.omega0)
        self.present_time: float = self.cosmology.age(0).value  # Gyr

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

        return self.cosmology.age(redshift).value  # Gyr

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

        return self.cosmology.lookback_time(redshift).value  # Gyr

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
        return self.redshift_to_time(redshift)  # Gyr

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
