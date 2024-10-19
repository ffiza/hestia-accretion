class Settings:
    """
    A class to manage generic configuration variables.
    """

    def __init__(self) -> None:
        self.galaxies: list = [
            "17_11_MW", "17_11_M31",
            "09_18_MW", "09_18_M31",
            "37_11_MW", "37_11_M31"]
        self.galaxy_colors: dict = {
            "17_11_MW": "tab:red", "17_11_M31": "tab:red",
            "09_18_MW": "tab:green", "09_18_M31": "tab:green",
            "37_11_MW": "tab:blue", "37_11_M31": "tab:blue"}
        self.galaxy_lss: dict = {
            "17_11_MW": "-", "17_11_M31": "--",
            "09_18_MW": "-", "09_18_M31": "--",
            "37_11_MW": "-", "37_11_M31": "--"}
