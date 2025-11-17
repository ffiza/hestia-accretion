class Settings:
    # SIMULATIONS: list = ["17_11", "09_18", "37_11"]
    SIMULATIONS: list = ["17_11"]
    GALAXIES: list = ["MW", "M31"]
    SIMULATION_COLORS: dict = {
        "17_11": "tab:red",
        "09_18": "tab:green",
        "37_11": "tab:blue"
    }
    GALAXY_LINESTYLES: dict = {
        "MW": "-",
        "M31": "--"
    }
    GALAXY_SYMBOLS: dict = {
        "MW": "^",
        "M31": "s"
    }
