import json
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hestia.settings import Settings
from hestia.images import figure_setup


def _get_data(galaxy: str, config: dict) -> pd.DataFrame:
    path = f"results/{galaxy}/disc_size_{config['RUN_CODE']}.json"
    with open(path) as f:
        data = json.load(f)
        time = np.array(data["Times_Gyr"])
        rd = np.array(data["DiscRadius_ckpc"])
        hd = np.array(data["DiscHeight_ckpc"])
        a = np.array(data["ExpansionFactor"])
    df = pd.DataFrame({
        "Time_Gyr": time,
        "DiscRadius_ckpc": rd,
        "DiscHeight_ckpc": hd,
        "ExpansionFactor": a
    })
    return df


def plot_disc_radius(config: dict) -> None:
    fig = plt.figure(figsize=(5.0, 2.0))
    gs = fig.add_gridspec(nrows=1, ncols=3, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=False)

    for ax in axs.flatten():
        ax.set_axisbelow(True)
        ax.grid(True)
        ax.set_xlim(0, 14)
        ax.set_xticks([2, 4, 6, 8, 10, 12])
        ax.set_xlabel(r'Time [Gyr]')
        ax.set_ylim(0, 40)
        ax.set_yticks([0, 10, 20, 30, 40])
        ax.set_ylabel(r'$R_\mathrm{d}$ [kpc]')
        ax.label_outer()

    for i, simulation in enumerate(Settings.SIMULATIONS):
        ax = axs[i]
        for galaxy in Settings.GALAXIES:
            df = _get_data(
                galaxy=f"{simulation}_{galaxy}", config=config)
            ax.plot(
                df["Time_Gyr"], df["ExpansionFactor"] * df["DiscRadius_ckpc"],
                ls=Settings.GALAXY_LINESTYLES[galaxy],
                color=Settings.SIMULATION_COLORS[simulation],
                lw=1, label=galaxy)
        ax.text(x=0.95, y=0.95, s=r"$\texttt{" + f"{simulation}" + "}$",
                transform=ax.transAxes, fontsize=7.0,
                verticalalignment='top', horizontalalignment='right',
                color=Settings.SIMULATION_COLORS[simulation])
        ax.legend(loc="upper left", framealpha=0, fontsize=5)

    plt.savefig(f"images/disc_radius_{config['RUN_CODE']}.pdf")
    plt.close(fig)


def plot_disc_height(config: dict) -> None:
    fig = plt.figure(figsize=(5.0, 2.0))
    gs = fig.add_gridspec(nrows=1, ncols=3, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=False)

    for ax in axs.flatten():
        ax.set_axisbelow(True)
        ax.grid(True)
        ax.set_xlim(0, 14)
        ax.set_xticks([2, 4, 6, 8, 10, 12])
        ax.set_xlabel(r'Time [Gyr]')
        ax.set_ylim(0, 4)
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_ylabel(r'$h_\mathrm{d}$ [kpc]')
        ax.label_outer()

    for i, simulation in enumerate(Settings.SIMULATIONS):
        ax = axs[i]
        for galaxy in Settings.GALAXIES:
            df = _get_data(
                galaxy=f"{simulation}_{galaxy}", config=config)
            ax.plot(
                df["Time_Gyr"], df["ExpansionFactor"] * df["DiscHeight_ckpc"],
                ls=Settings.GALAXY_LINESTYLES[galaxy],
                color=Settings.SIMULATION_COLORS[simulation],
                lw=1, label=galaxy)
        ax.text(x=0.95, y=0.95, s=r"$\texttt{" + f"{simulation}" + "}$",
                transform=ax.transAxes, fontsize=7.0,
                verticalalignment='top', horizontalalignment='right',
                color=Settings.SIMULATION_COLORS[simulation])
        ax.legend(loc="upper left", framealpha=0, fontsize=5)

    plt.savefig(f"images/disc_height_{config['RUN_CODE']}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    figure_setup()

    # Get arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = yaml.safe_load(open(f"configs/{args.config}.yml"))

    plot_disc_radius(config=config)
    plot_disc_height(config=config)
