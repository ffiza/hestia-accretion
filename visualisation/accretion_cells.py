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
from hestia.accretion_region import (AccretionRegionType,
                                     get_accretion_region_suffix)


def _get_data(galaxy: str, config: dict,
              accretion_region_type: AccretionRegionType) -> pd.DataFrame:

    suffix = get_accretion_region_suffix(accretion_region_type)

    path = f"results/{galaxy}/net_accretion_cells" \
        + f"{suffix}_{config['RUN_CODE']}.json"
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
    df["AccretionRateMin_Msun/yr"] = df[[
        f"AccretionRate_Au{i}_Msun/yr" for i in range(1, 31)]].min(
            axis=1)
    df["AccretionRateMax_Msun/yr"] = df[[
        f"AccretionRate_Au{i}_Msun/yr" for i in range(1, 31)]].max(
            axis=1)
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
        df["AccretionRateMean_Msun/yr"] = np.nanmean(
            df[[f"AccretionRate_Au{i}_Msun/yr" for i in range(
                1, 31)]].to_numpy(),
            axis=1)
        df["AccretionRateStd_Msun/yr"] = np.nanstd(
            df[[f"AccretionRate_Au{i}_Msun/yr" for i in range(
                1, 31)]].to_numpy(),
            axis=1)
    return df


def _add_auriga_data_to_ax(ax, config: dict) -> None:
    df_auriga = _get_auriga_data(config)
    ax.fill_between(
        df_auriga["Time_Gyr"],
        df_auriga["AccretionRateSmoothedMean_Msun/yr"]
        - df_auriga["AccretionRateSmoothedStd_Msun/yr"],
        df_auriga["AccretionRateSmoothedMean_Msun/yr"]
        + df_auriga["AccretionRateSmoothedStd_Msun/yr"],
        color="k", alpha=0.1, label="Auriga", lw=0)
    ax.plot(df_auriga["Time_Gyr"],
            df_auriga["AccretionRateSmoothedMean_Msun/yr"],
            ls="-", color="darkgray", lw=0.75, zorder=10)


def make_plot(config: dict,
              accretion_region_type: AccretionRegionType) -> None:
    window_length = config["TEMPORAL_AVERAGE_WINDOW_LENGTH"]

    fig = plt.figure(figsize=(5.0, 2.0))
    gs = fig.add_gridspec(nrows=1, ncols=3, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=False)

    match accretion_region_type:
        case AccretionRegionType.STELLAR_DISC:
            ylabel = r'$\dot{M}_\mathrm{net}$ '
            r'[$\mathrm{M}_\odot \, \mathrm{yr}^{-1}$]'
        case AccretionRegionType.HALO:
            ylabel = r'$\dot{M}_\mathrm{net}^\mathrm{halo}$ '
            r'[$\mathrm{M}_\odot \, \mathrm{yr}^{-1}$]'
        case _:
            raise ValueError("Invalid accretion region type.")

    for ax in axs:
        ax.set_axisbelow(True)
        ax.set_xlim(0, 14)
        ax.set_ylim(0.1, 400)
        ax.set_yscale("log")
        ax.set_xticks(ticks=[2, 4, 6, 8, 10, 12],
                      labels=["2", "4", "6", "8", "10", "12"],
                      fontsize=6)
        ax.set_yticks(ticks=[0.1, 1, 10, 100],
                      labels=["0.1", "1", "10", "100"],
                      fontsize=6)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlabel(r'Time [Gyr]',
                      fontsize=8)
        ax.label_outer()

    for i, simulation in enumerate(Settings.SIMULATIONS):
        ax = axs[i]
        for galaxy in Settings.GALAXIES:
            df = _get_data(
                f"{simulation}_{galaxy}", config, accretion_region_type)
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
                    lw=0.75, label=galaxy, zorder=12)
        ax.text(
            x=0.05, y=0.95, s=r"$\texttt{" + f"{simulation}" + "}$",
            transform=ax.transAxes, fontsize=6,
            verticalalignment='top', horizontalalignment='left',
            color=Settings.SIMULATION_COLORS[simulation])

        if accretion_region_type == AccretionRegionType.STELLAR_DISC:
            _add_auriga_data_to_ax(ax, config)

        ax.legend(loc="lower right", framealpha=0, fontsize=5)

    suffix = get_accretion_region_suffix(accretion_region_type)
    plt.savefig(f"images/net_accretion_cells{suffix}_{config['RUN_CODE']}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    figure_setup()

    # Get arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = yaml.safe_load(open(f"configs/{args.config}.yml"))

    make_plot(config, AccretionRegionType.STELLAR_DISC)
    make_plot(config, AccretionRegionType.HALO)
