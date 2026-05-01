import json
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr

from hestia.images import figure_setup
from hestia.settings import Settings


class DataLoader():

    def __init__(self, config: dict) -> None:
        self.config = config

    def get_data(self) -> pd.DataFrame:

        df = pd.read_csv(
            "data/auriga/cold_gas_mass_evolution.csv",
            usecols=["Snapshot", "Time_Gyr", "Redshift", "ExpansionFactor",
                     "SubhaloGasMassUnder20000K_Msun", "Suite", "Simulation",
                     "Galaxy"]
        )

        subhalo_gas_mass_msun = []
        subhalo_stellar_mass_msun = []
        simulation = []
        snapshot = []
        delta_1200 = []
        for i in range(1, 31):
            data = np.loadtxt(
                f"data/auriga/au{i}/baryon_mass.csv",
                delimiter=" ",)
            subhalo_gas_mass_msun += list(data[:, 0] * 1E10)
            subhalo_stellar_mass_msun += list(data[:, 1] * 1E10)
            simulation += [i] * len(data[:, 0])
            snapshot += list(range(len(data[:, 0])))

            delta_1200 += pd.read_csv(
                f"data/auriga/au{i}/environment_evolution.csv",
                usecols=["Delta1200"])["Delta1200"].to_list()
        data = pd.DataFrame(
            {"Snapshot": snapshot,
             "SubhaloGasMass_Msun": subhalo_gas_mass_msun,
             "SubhaloStellarMass_Msun": subhalo_stellar_mass_msun,
             "Suite": ["Au"] * len(subhalo_gas_mass_msun),
             "Simulation": simulation,
             "Galaxy": np.nan * np.ones(len(subhalo_gas_mass_msun)),
             "Delta1200": delta_1200})

        df = df.merge(
            data[["Snapshot", "Suite", "Simulation", "SubhaloGasMass_Msun",
                  "SubhaloStellarMass_Msun", "Delta1200"]],
            left_on=["Snapshot", "Suite", "Simulation"],
            right_on=["Snapshot", "Suite", "Simulation"],
            how="left"
        )

        for simulation in Settings.HIGH_RES_SIMULATIONS:
            for galaxy in Settings.GALAXIES:
                data = pd.read_csv(
                    f"data/hestia/{galaxy}_M_SFR_t_Hestia{simulation}.csv",
                    usecols=["SnapNo", "Mgas", "Mstar", "Mcold"])
                delta_1200 = pd.read_csv(
                    f"results/{simulation}_{galaxy}/delta_1200_01.csv")[
                        "Delta"].iloc[-1]
                new_row = pd.DataFrame([{
                    "Snapshot": 127,
                    "Time_Gyr": np.nan,
                    "Redshift": np.nan,
                    "ExpansionFactor": np.nan,
                    "SubhaloGasMassUnder20000K_Msun": data["Mcold"].iloc[-1],
                    "Suite": "He",
                    "Simulation": simulation,
                    "Galaxy": galaxy,
                    "SubhaloGasMass_Msun": data["Mgas"].iloc[-1],
                    "SubhaloStellarMass_Msun": data["Mstar"].iloc[-1],
                    "Delta1200": delta_1200,
                }])

                
                df = pd.concat([df, new_row], ignore_index=True)
                

        df["Snapshot"] = df["Snapshot"].astype(np.uint8)
        df["Suite"] = df["Suite"].astype("category")
        df["Simulation"] = df["Simulation"].astype("category")
        df["Galaxy"] = df["Galaxy"].astype("category")

        return df


def plot_gas_mass_fraction_vs_environment(
        df: pd.DataFrame,
        config: dict,
        snapnum: int) -> None:

    df["GasMassFraction"] = df["SubhaloGasMass_Msun"] \
        / (df["SubhaloGasMass_Msun"] + df["SubhaloStellarMass_Msun"])
    df["ColdGasMassFraction"] = df["SubhaloGasMassUnder20000K_Msun"] \
        / (df["SubhaloGasMassUnder20000K_Msun"] \
            + df["SubhaloStellarMass_Msun"])

    df = df.drop(["SubhaloGasMassUnder20000K_Msun", "SubhaloGasMass_Msun",
                  "SubhaloStellarMass_Msun"],
                 axis=1)

    fig = plt.figure(
        figsize=(4, 2))
    gs = fig.add_gridspec(
        nrows=1,
        ncols=2,
        hspace=0,
        wspace=0.4)
    axs = np.array(gs.subplots(
        sharex=True,
        sharey=False))

    axs[0].set_xlim(0.6, 1.5)
    axs[0].set_ylim(0, 1)
    axs[0].set_xticks(
        ticks=(0.8, 1.0, 1.2, 1.4),
        labels=("0.8", "1.0", "1.2", "1.4"),
        fontsize=5)
    axs[0].set_yticks(
        ticks=(0, 0.25, 0.5, 0.75, 1.0),
        labels=("0.0", "0.25", "0.5", "0.75", "1.0"),
        fontsize=5)
    axs[0].set_xlabel(
        r"$\log_{10} \delta_{1200}$",
        fontsize=8)
    axs[0].set_ylabel(
        r"$f_\mathrm{gas}$",
        fontsize=8)

    axs[1].set_xlim(0.6, 1.5)
    axs[1].set_ylim(0, 1)
    axs[1].set_xticks(
        ticks=(0.8, 1.0, 1.2, 1.4),
        labels=("0.8", "1.0", "1.2", "1.4"),
        fontsize=5)
    axs[1].set_yticks(
        ticks=(0, 0.25, 0.5, 0.75, 1.0),
        labels=("0.0", "0.25", "0.5", "0.75", "1.0"),
        fontsize=5)
    axs[1].set_xlabel(
        r"$\log_{10} \delta_{1200}$",
        fontsize=8)
    axs[1].set_ylabel(
        r"$f_\mathrm{cold \; gas}$",
        fontsize=8)


    for i, row in df.iterrows():
        facecolor = '#4d4d4d' if row["Suite"] == "Au" \
            else Settings.SIMULATION_COLORS[row["Simulation"]]
        marker = "o" if row["Suite"] == "Au" \
            else Settings.GALAXY_SYMBOLS[row["Galaxy"]]
        label = row["Simulation"] + "_" + row["Galaxy"] \
            if row["Suite"] == "He" else "_" 
        axs[0].scatter(
            np.log10(row["Delta1200"]),
            row["GasMassFraction"],
            s=8,
            edgecolor="none",
            facecolor=facecolor,
            marker=marker,
            label=label,
            zorder=10,
        )
        axs[1].scatter(
            np.log10(row["Delta1200"]),
            row["ColdGasMassFraction"],
            s=8,
            edgecolor="none",
            facecolor=facecolor,
            marker=marker,
            zorder=10,
        )

    axs[0].legend(
        frameon=False,
        fontsize=4,
        loc='lower center',
        ncols=2)

    plt.savefig(
        f"images/gas_mass_vs_environment_s{snapnum}_{config['RUN_CODE']}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    figure_setup()

    # Get arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = yaml.safe_load(open(f"configs/{args.config}.yml"))

    df = DataLoader(config).get_data()
    df = df.drop(["Time_Gyr", "Redshift", "ExpansionFactor"],
                 axis=1)

    plot_gas_mass_fraction_vs_environment(
        df[df["Snapshot"] == 127].copy(),
        config,
        127)
    # plot_cold_gas_mass_fraction_vs_environment(config, 127)
