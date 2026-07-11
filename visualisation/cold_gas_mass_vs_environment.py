import yaml
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from hestia.images import figure_setup
from hestia.settings import Settings
from hestia.get_data import get_data


def plot_cold_gas_mass_vs_environment(
        df: pd.DataFrame,
        config: dict) -> None:
    fig = plt.figure(
        figsize=(5.0, 3.5))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        hspace=0,
        wspace=0)
    axs = gs.subplots(
        sharex=True,
        sharey=False)

    for ax in axs.flatten():
        ax.set_axisbelow(True)
        ax.set_xlim(0, 25)
        ax.set_ylim(0, 1)
        ax.set_xticks(
            ticks=[5, 10, 15, 20],
            labels=["5", "10", "15", "20"],
            fontsize=6)
        ax.set_yticks(
            ticks=[0.2, 0.4, 0.6, 0.8],
            labels=["0.2", "0.4", "0.6", "0.8"],
            fontsize=6)
        ax.set_ylabel(
            r'$M_\mathrm{cold} / \left(M_\star + M_\mathrm{cold} \right)$',
            fontsize=8)
        ax.set_xlabel(
            r'$\delta_{1200}$',
            fontsize=8)
        ax.label_outer()

    for j, simulation in enumerate(Settings.HIGH_RES_SIMULATIONS):
        for i, galaxy in enumerate(Settings.GALAXIES):
            mask = (df["Simulation"] == simulation) & (df["Galaxy"] == galaxy)
            axs[i, j].plot(
                df[mask]["Delta1200"],
                df[mask]["ColdGasMassFraction"],
                color=Settings.SIMULATION_COLORS[simulation],
                lw=0.75,
                zorder=10,
                label=r"$\texttt{" + f"{simulation}" + f"_{galaxy}" + "}$")
            axs[i, j].scatter(
                df[mask]["Delta1200"].to_numpy()[-1],
                df[mask]["ColdGasMassFraction"].to_numpy()[-1],
                color=Settings.SIMULATION_COLORS[simulation],
                s=3,
                zorder=10)
            axs[i, j].scatter(
                df[(df["Suite"] == "Au")]["Delta1200"],
                df[(df["Suite"] == "Au")]["ColdGasMassFraction"],
                color="#8a8a8a",
                s=0.5,
                zorder=9,
                alpha=0.25,
                label="Auriga")

    for ax in axs.flatten():
        ax.legend(
            loc="lower right",
            framealpha=0,
            fontsize=5,
            ncol=1)

    plt.savefig(
        f"images/cold_gas_mass_vs_environment_{config['RUN_CODE']}.pdf")
    plt.close(fig)


def plot_gas_mass_vs_environment(
        df: pd.DataFrame,
        config: dict) -> None:
    fig = plt.figure(
        figsize=(5.0, 3.5))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        hspace=0,
        wspace=0)
    axs = gs.subplots(
        sharex=True,
        sharey=False)

    for ax in axs.flatten():
        ax.set_axisbelow(True)
        ax.set_xlim(0, 25)
        ax.set_ylim(0, 0.3)
        ax.set_xticks(
            ticks=[5, 10, 15, 20],
            labels=["5", "10", "15", "20"],
            fontsize=6)
        ax.set_yticks(
            ticks=[0.05, 0.1, 0.15, 0.2, 0.25],
            labels=["0.05", "0.1", "0.15", "0.2", "0.25"],
            fontsize=6)
        ax.set_ylabel(
            r'$M_\mathrm{gas} / M_{200}$',
            fontsize=8)
        ax.set_xlabel(
            r'$\delta_{1200}$',
            fontsize=8)
        ax.label_outer()

    for j, simulation in enumerate(Settings.HIGH_RES_SIMULATIONS):
        for i, galaxy in enumerate(Settings.GALAXIES):
            mask = (df["Simulation"] == simulation) & (df["Galaxy"] == galaxy)
            axs[i, j].plot(
                df[mask]["Delta1200"],
                df[mask]["GasMassFraction"],
                color=Settings.SIMULATION_COLORS[simulation],
                lw=0.75,
                zorder=10,
                label=r"$\texttt{" + f"{simulation}" + f"_{galaxy}" + "}$")
            axs[i, j].scatter(
                df[mask]["Delta1200"].to_numpy()[-1],
                df[mask]["GasMassFraction"].to_numpy()[-1],
                color=Settings.SIMULATION_COLORS[simulation],
                s=3,
                zorder=10)
            axs[i, j].scatter(
                df[(df["Suite"] == "Au")]["Delta1200"],
                df[(df["Suite"] == "Au")]["GasMassFraction"],
                color="#8a8a8a",
                s=0.5,
                zorder=9,
                alpha=0.25,
                label="Auriga")

    for ax in axs.flatten():
        ax.legend(
            loc="upper right",
            framealpha=0,
            fontsize=5,
            ncol=1)

    plt.savefig(
        f"images/gas_mass_vs_environment_{config['RUN_CODE']}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    figure_setup()

    # Get arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = yaml.safe_load(open(f"configs/{args.config}.yml"))

    df = get_data(config)
    df["ColdGasMassFraction"] = df[
        "SubhaloGasMassUnder20000K_1E10Msun"] \
        / (df["StellarMass_1E10Msun"] \
            + df["SubhaloGasMassUnder20000K_1E10Msun"])
    df["GasMassFraction"] = df["GasMass_1E10Msun"] / df["M200_1E10Msun"]
    print(df.sample(n=20))
    plot_cold_gas_mass_vs_environment(df, config)
    plot_gas_mass_vs_environment(df, config)
