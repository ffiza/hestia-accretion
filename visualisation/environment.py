import yaml
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hestia.settings import Settings
from hestia.images import figure_setup


def _get_data(galaxy: str, config: dict) -> pd.DataFrame:
    path = f"results/{galaxy}/delta_1200_{config['RUN_CODE']}.csv"
    return pd.read_csv(path)


def _get_auriga_data() -> pd.DataFrame:
    df = dict()
    for i in range(1, 31):
        data = pd.read_csv(f"data/auriga/au{i}/environment_evolution.csv")
        if "Time_Gyr" not in df:
            df["Time_Gyr"] = data["Time_Gyr"]
            df["Redshift"] = data["Redshift"]
            df["ExpansionFactor"] = data["ExpansionFactor"]
        df[f"Delta1200_Au{i}"] = data["Delta1200"]
    df = pd.DataFrame(df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df["Delta1200Mean"] = np.nanmean(
            df[[f"Delta1200_Au{i}" for i in range(1, 31)]].to_numpy(),
            axis=1)
        df["Delta1200Std"] = np.nanstd(
            df[[f"Delta1200_Au{i}" for i in range(1, 31)]].to_numpy(),
            axis=1)
    return df


def make_plot(config: dict) -> None:
    auriga = _get_auriga_data()

    fig = plt.figure(figsize=(5.0, 2.0))
    gs = fig.add_gridspec(nrows=1, ncols=3, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=False)

    for ax in axs.flatten():
        ax.set_axisbelow(True)
        ax.set_xlim(0, 14)
        ax.set_ylim(1, 30)
        ax.set_yscale("log")
        ax.set_xticks(ticks=[2, 4, 6, 8, 10, 12],
                      labels=["2", "4", "6", "8", "10", "12"],
                      fontsize=6)
        ax.set_yticks(ticks=[1, 3, 5, 10, 20],
                      labels=["1", "3", "5", "10", "20"],
                      fontsize=6)
        ax.set_ylabel(r'$\delta_{1200}$', fontsize=8)
        ax.set_xlabel(r'Time [Gyr]', fontsize=8)
        ax.label_outer()

    for i, simulation in enumerate(Settings.SIMULATIONS):
        ax = axs[i]
        for galaxy in Settings.GALAXIES:
            try:
                df = _get_data(f"{simulation}_{galaxy}", config)
            except FileNotFoundError:
                warnings.warn(
                    f"Data for {simulation}_{galaxy} not found. "
                    "Ignoring in figure.")
                continue
            ax.plot(df["Times_Gyr"],
                    df["Delta"],
                    ls=Settings.GALAXY_LINESTYLES[galaxy],
                    color=Settings.SIMULATION_COLORS[simulation],
                    lw=0.75, label=galaxy, zorder=11)
        ax.text(
            x=0.05, y=0.95, s=r"$\texttt{" + f"{simulation}" + "}$",
            transform=ax.transAxes, fontsize=6,
            verticalalignment='top', horizontalalignment='left',
            color=Settings.SIMULATION_COLORS[simulation])

        ax.fill_between(
            auriga["Time_Gyr"],
            auriga["Delta1200Mean"] - auriga["Delta1200Std"],
            auriga["Delta1200Mean"] + auriga["Delta1200Std"],
            color="k", alpha=0.1, label="Auriga", lw=0, zorder=10)
        ax.plot(auriga["Time_Gyr"],
                auriga["Delta1200Mean"],
                ls="-", color="darkgray", lw=0.75, zorder=10)

        ax.legend(loc="lower right", framealpha=0, fontsize=5)

    plt.savefig(f"images/delta1200_{config['RUN_CODE']}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    figure_setup()

    # Get arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = yaml.safe_load(open(f"configs/{args.config}.yml"))

    make_plot(config)
