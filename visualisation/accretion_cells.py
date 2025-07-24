import json
import yaml
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hestia.settings import Settings
from hestia.images import figure_setup
from hestia.tools import windowed_average


def _get_data(galaxy: str, config: dict) -> pd.DataFrame:
    path = f"results/{galaxy}/net_accretion_cells" \
        + f"_{config['RUN_CODE']}.json"
    with open(path) as f:
        data = json.load(f)
        time = np.array(data["Times_Gyr"])
        net_accretion = np.array(data["NetAccretionCells_Msun/yr"])
    df = pd.DataFrame({
        "Time_Gyr": time,
        "NetAccretionCells_Msun/yr": net_accretion
    })
    return df


def _get_auriga_data(config: dict) -> pd.DataFrame:
    window_length = config["TEMPORAL_AVERAGE_WINDOW_LENGTH"]
    df = pd.read_csv("data/iza_et_al_2022/net_accretion_rate_cells.csv")
    for i in range(1, 31):
        time = df["Time_Gyr"].to_numpy()
        rate = df[f"AccretionRate_Au{i}_Msun/yr"].to_numpy()
        rate[rate < 0.1] = np.nan
        df[f"AccretionRateSmoothed_Au{i}_Msun/yr"] = windowed_average(
            time, rate, window_length)
    df["AccretionRateSmoothedMin_Msun/yr"] = df[[
        f"AccretionRateSmoothed_Au{i}_Msun/yr" for i in range(1, 31)]].min(
            axis=1)
    df["AccretionRateSmoothedMax_Msun/yr"] = df[[
        f"AccretionRateSmoothed_Au{i}_Msun/yr" for i in range(1, 31)]].max(
            axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df["AccretionRateSmoothedMean_Msun/yr"] = np.nanmean(
            df[[f"AccretionRateSmoothed_Au{i}_Msun/yr" for i in range(
                1, 31)]].to_numpy(),
            axis=1)
        df["AccretionRateSmoothedStd_Msun/yr"] = np.nanstd(
            df[[f"AccretionRateSmoothed_Au{i}_Msun/yr" for i in range(
                1, 31)]].to_numpy(),
            axis=1)
    return df


def make_plot(config: dict) -> None:
    df_auriga = _get_auriga_data(config)
    window_length = config["TEMPORAL_AVERAGE_WINDOW_LENGTH"]

    fig = plt.figure(figsize=(5.0, 2.0))
    gs = fig.add_gridspec(nrows=1, ncols=3, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=False)

    for ax in axs:
        ax.set_axisbelow(True)
        ax.grid(True)
        ax.set_xlim(0, 14)
        ax.set_ylim(0.1, 100)
        ax.set_yscale("log")
        ax.set_xticks([2, 4, 6, 8, 10, 12])
        ax.set_yticks([0.1, 1, 10, 100])
        ax.set_yticklabels(["0.1", "1", "10", "100"])
        ax.set_ylabel(
            r'$\dot{M}_\mathrm{net}$ [$\mathrm{M}_\odot \, \mathrm{yr}^{-1}$]')
        ax.set_xlabel(r'Time [Gyr]')
        ax.label_outer()

    for i, simulation in enumerate(Settings.SIMULATIONS):
        ax = axs[i]
        for galaxy in Settings.GALAXIES:
            df = _get_data(
                galaxy=f"{simulation}_{galaxy}", config=config)
            is_positive = df["NetAccretionCells_Msun/yr"] >= 0.1
            ax.plot(df["Time_Gyr"][is_positive],
                    windowed_average(
                        df["Time_Gyr"][is_positive].to_numpy(),
                        df["NetAccretionCells_Msun/yr"][
                            is_positive].to_numpy(),
                        window_length
                    ),
                    ls=Settings.GALAXY_LINESTYLES[galaxy],
                    color=Settings.SIMULATION_COLORS[simulation],
                    lw=1, label=galaxy, zorder=12)
        ax.text(
            x=0.05, y=0.95, s=r"$\texttt{" + f"{simulation}" + "}$",
            transform=ax.transAxes, fontsize=7.0,
            verticalalignment='top', horizontalalignment='left',
            color=Settings.SIMULATION_COLORS[simulation])

        #region TestAurigaData
        ax.fill_between(
            df_auriga["Time_Gyr"],
            df_auriga["AccretionRateSmoothedMean_Msun/yr"]
            - df_auriga["AccretionRateSmoothedStd_Msun/yr"],
            df_auriga["AccretionRateSmoothedMean_Msun/yr"]
            + df_auriga["AccretionRateSmoothedStd_Msun/yr"],
            color="k", alpha=0.1, label="Auriga", lw=0)
        ax.plot(df_auriga["Time_Gyr"],
                df_auriga["AccretionRateSmoothedMean_Msun/yr"],
                ls="-", color="darkgray", lw=1, zorder=10)
        #endregion

        ax.legend(loc="lower right", framealpha=0, fontsize=5)

    plt.savefig(f"images/net_accretion_cells_{config['RUN_CODE']}.png")
    plt.close(fig)


if __name__ == "__main__":
    figure_setup()

    # Get arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = yaml.safe_load(open(f"configs/{args.config}.yml"))

    make_plot(config=config)
