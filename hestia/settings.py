class Settings:
    """
    A class to manage generic configuration variables.
    """

    def __init__(self) -> None:
        self.galaxies: list = ["17_11_MW", "17_11_M31"]
        self.galaxy_colors: list = ["tab:blue", "tab:red"]
        self.galaxy_lss: list = ["-", "--"]
