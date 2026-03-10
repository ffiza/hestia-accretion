class Settings:
    HIGH_RES_SIMULATIONS: list = [
        "17_11",
        "09_18",
        "37_11",
    ]
    SIMULATIONS: list = ["17_11", "09_18", "37_11", "i_09_10", "i_09_16",
                         "i_09_17", "i_09_18", "i_09_19", "i_17_10",
                         "i_17_11", "i_17_13", "i_17_14", "i_37_11",
                         "i_37_12", "i_37_16", "i_37_17"]
    GALAXIES: list = ["MW", "M31"]
    SIMULATION_COLORS: dict = {

        "17_11": "tab:red",
        "09_18": "tab:green",
        "37_11": "tab:blue",

        # --- i_09_* ---
        "i_09_10": "tab:green",
        "i_09_16": "tab:green",
        "i_09_17": "tab:green",
        "i_09_18": "tab:green",
        "i_09_19": "tab:green",

        # --- i_17_* ---
        "i_17_10": "tab:red",
        "i_17_11": "tab:red",
        "i_17_13": "tab:red",
        "i_17_14": "tab:red",

        # --- i_37_* ---
        "i_37_11": "tab:blue",
        "i_37_12": "tab:blue",
        "i_37_16": "tab:blue",
        "i_37_17": "tab:blue",
    }
    GALAXY_LINESTYLES: dict = {
        "MW": "-",
        "M31": "--"
    }
    GALAXY_SYMBOLS: dict = {
        "MW": "^",
        "M31": "s"
    }
