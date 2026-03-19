import json
import yaml
import argparse
import numpy as np
import pandas as pd
from enum import Enum
import matplotlib.pyplot as plt

from hestia.settings import Settings
from hestia.images import figure_setup
from hestia.tools import windowed_average
from hestia.accretion_region import (AccretionRegionType,
                                     get_accretion_region_suffix)
from hestia.auriga import AurigaData
from hestia.data import HestiaData


class RateType(Enum):
    INFLOW = 1
    OUTFLOW = 2


class Helpers:
    @staticmethod
    def get_rate_type_string(rate_type: RateType) -> str:
        if rate_type == RateType.INFLOW:
            return 'Inflow'
        elif rate_type == RateType.OUTFLOW:
            return 'Outflow'
        else:
            raise ValueError("Invalid RateType.")

    @staticmethod
    def get_file_prefix(rate_type: RateType) -> str:
        if rate_type == RateType.INFLOW:
            return 'inflow_accretion_tracers'
        elif rate_type == RateType.OUTFLOW:
            return 'outflow_accretion_tracers'
        else:
            raise ValueError("Invalid RateType.")

    @staticmethod
    def get_feat_name(rate_type: RateType) -> str:
        if rate_type == RateType.INFLOW:
            return 'InflowRate_Msun/yr'
        elif rate_type == RateType.OUTFLOW:
            return 'OutflowRate_Msun/yr'
        else:
            raise ValueError("Invalid RateType.")

    @staticmethod
    def get_ylabel(rate_type: RateType,
                   accretion_region_type: AccretionRegionType) -> str:
        if rate_type == RateType.INFLOW \
                and accretion_region_type == AccretionRegionType.STELLAR_DISC:
            label = r'$\dot{M}_\mathrm{in}$ ' \
                r'[$\mathrm{M}_\odot \, \mathrm{yr}^{-1}$]'
        elif rate_type == RateType.INFLOW \
                and accretion_region_type == AccretionRegionType.HALO:
            label = r'$\dot{M}_\mathrm{in}^\mathrm{halo}$ ' \
                r'[$\mathrm{M}_\odot \, \mathrm{yr}^{-1}$]'
        elif rate_type == RateType.OUTFLOW \
                and accretion_region_type == AccretionRegionType.STELLAR_DISC:
            label = r'$\dot{M}_\mathrm{out}$ ' \
                r'[$\mathrm{M}_\odot \, \mathrm{yr}^{-1}$]'
        elif rate_type == RateType.OUTFLOW \
                and accretion_region_type == AccretionRegionType.HALO:
            label = r'$\dot{M}_\mathrm{out}^\mathrm{halo}$ ' \
                r'[$\mathrm{M}_\odot \, \mathrm{yr}^{-1}$]'
        else:
            raise ValueError("Invalid combination of inputs.")
        return label


def _get_data(galaxy: str, config: dict,
              accretion_region_type: AccretionRegionType) -> pd.DataFrame:

    suffix = get_accretion_region_suffix(accretion_region_type)

    path = f"results/{galaxy}/accretion_tracers" \
        + f"{suffix}_{config['RUN_CODE']}.json"
    with open(path) as f:
        data = json.load(f)
        time = np.array(data["Times_Gyr"])
        inflow_rate = np.array(data["InflowRate_Msun/yr"])
        outflow_rate = np.array(data["OutflowRate_Msun/yr"])
    df = pd.DataFrame({
        "Time_Gyr": time,
        "InflowRate_Msun/yr": inflow_rate,
        "OutflowRate_Msun/yr": outflow_rate,
    })
    return df


def _add_auriga_data_to_ax(
        ax,
        rate_type: RateType,
        config: dict) -> None:
    df_disc = AurigaData.get_accretion(
        config, AccretionRegionType.STELLAR_DISC)
    df_halo = AurigaData.get_accretion(
        config, AccretionRegionType.HALO)

    data = pd.DataFrame({})
    data["Time_Gyr"] = df_disc["Time_Gyr"]
    for i in Settings.AURIGA_RERUNS:
        data[f"{Helpers.get_rate_type_string(rate_type)}"
             f"AccretionHaloDiscFraction_Au{i}"] = \
            df_halo[f"{Helpers.get_rate_type_string(rate_type)}"
                    f"RateSmoothed_Au{i}_Msun/yr"] / \
            df_disc[f"{Helpers.get_rate_type_string(rate_type)}"
                    f"RateSmoothed_Au{i}_Msun/yr"]
    data[f"{Helpers.get_rate_type_string(rate_type)}"
         "AccretionHaloDiscFraction_Median"] = np.nanmedian(
        data[[f"{Helpers.get_rate_type_string(rate_type)}"
              f"AccretionHaloDiscFraction_Au{i}" for i in AurigaData.RERUNS]],
        axis=1)
    data[f"{Helpers.get_rate_type_string(rate_type)}"
         "AccretionHaloDiscFraction_Perc16"] = np.nanpercentile(
        data[[f"{Helpers.get_rate_type_string(rate_type)}"
              f"AccretionHaloDiscFraction_Au{i}" for i in AurigaData.RERUNS]],
        16,
        axis=1)
    data[f"{Helpers.get_rate_type_string(rate_type)}"
         "AccretionHaloDiscFraction_Perc84"] = np.nanpercentile(
        data[[f"{Helpers.get_rate_type_string(rate_type)}"
              f"AccretionHaloDiscFraction_Au{i}" for i in AurigaData.RERUNS]],
        84,
        axis=1)

    ax.fill_between(
        data["Time_Gyr"],
        data[f"{Helpers.get_rate_type_string(rate_type)}"
             "AccretionHaloDiscFraction_Perc16"],
        data[f"{Helpers.get_rate_type_string(rate_type)}"
             "AccretionHaloDiscFraction_Perc84"],
        color="#e6e6e6", label="Auriga", lw=0)
    ax.plot(data["Time_Gyr"],
            data[
                f"{Helpers.get_rate_type_string(rate_type)}"
                "AccretionHaloDiscFraction_Median"],
            ls="-", color="#4d4d4d", lw=0.75, zorder=10)


def plot_halo_and_disc_time_series(
        config: dict,
        rate_type: RateType) -> None:
    window_length = config["TEMPORAL_AVERAGE_WINDOW_LENGTH"]

    fig = plt.figure(figsize=(5.0, 2.0))
    gs = fig.add_gridspec(nrows=1, ncols=3, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=False)

    xlabel = "Time [Gyr]"
    if rate_type == RateType.INFLOW:
        ylabel = r'$\dot{M}_\mathrm{in}^\mathrm{halo} / \dot{M}_\mathrm{in}$'
    elif rate_type == RateType.OUTFLOW:
        ylabel = r'$\dot{M}_\mathrm{out}^\mathrm{halo} / \dot{M}_\mathrm{out}$'

    for ax in axs.flatten():
        ax.set_axisbelow(True)
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 6)
        ax.set_xticks(ticks=[2, 4, 6, 8, 10, 12],
                      labels=[2, 4, 6, 8, 10, 12],
                      fontsize=6)
        ax.set_yticks(ticks=[1, 2, 3, 4, 5],
                      labels=[1, 2, 3, 4, 5],
                      fontsize=6)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.label_outer()

    for i, simulation in enumerate(Settings.HIGH_RES_SIMULATIONS):
        ax = axs[i]
        for _, galaxy in enumerate(Settings.GALAXIES):
            df_disc = _get_data(f"{simulation}_{galaxy}", config,
                                AccretionRegionType.STELLAR_DISC)
            df_halo = _get_data(f"{simulation}_{galaxy}", config,
                                AccretionRegionType.HALO)
            time = df_disc["Time_Gyr"].to_numpy()
            f_acc = windowed_average(
                df_halo["Time_Gyr"].to_numpy(),
                df_halo[Helpers.get_feat_name(rate_type)].to_numpy(),
                window_length) / windowed_average(
                    df_disc["Time_Gyr"].to_numpy(),
                    df_disc[Helpers.get_feat_name(rate_type)].to_numpy(),
                    window_length)

            ax.plot(
                time, f_acc,
                ls=Settings.GALAXY_LINESTYLES[galaxy],
                color=Settings.SIMULATION_COLORS[simulation],
                lw=0.75, label=galaxy, zorder=12)
        ax.text(
            x=0.05, y=0.95,
            s=r"$\texttt{" + f"{simulation}" + "}$",
            transform=ax.transAxes, fontsize=6,
            verticalalignment='top', horizontalalignment='left',
            color=Settings.SIMULATION_COLORS[simulation])

        _add_auriga_data_to_ax(ax, rate_type, config)

    plt.savefig(
        f"images/{Helpers.get_file_prefix(rate_type)}_time_series"
        f"_{config['RUN_CODE']}.pdf")
    plt.close(fig)


def plot_halo_vs_disc_scatter(
        config: dict, rate_type: RateType) -> None:
    fig = plt.figure(figsize=(5.0, 3.0))
    gs = fig.add_gridspec(nrows=2, ncols=3, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=False)

    if rate_type == RateType.INFLOW:
        xlabel = r'$\dot{M}_\mathrm{in}$ [$\mathrm{M}_\odot' + \
            r'\, \mathrm{yr}^{-1}$]'
        ylabel = r'$\dot{M}_\mathrm{in}^\mathrm{halo}$ [$\mathrm{M}_\odot' + \
            r'\, \mathrm{yr}^{-1}$]'
    elif rate_type == RateType.OUTFLOW:
        xlabel = r'$\dot{M}_\mathrm{out}$ [$\mathrm{M}_\odot' + \
            r'\, \mathrm{yr}^{-1}$]'
        ylabel = r'$\dot{M}_\mathrm{out}^\mathrm{halo}$ [$\mathrm{M}_\odot' + \
            r'\, \mathrm{yr}^{-1}$]'

    for ax in axs.flatten():
        ax.set_axisbelow(True)
        ax.set_xlim(0.1, 400)
        ax.set_ylim(0.1, 400)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(ticks=[0.1, 1, 10, 100],
                      labels=["0.1", "1", "10", "100"],
                      fontsize=6)
        ax.set_yticks(ticks=[0.1, 1, 10, 100],
                      labels=["0.1", "1", "10", "100"],
                      fontsize=6)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.label_outer()

    for i, simulation in enumerate(Settings.HIGH_RES_SIMULATIONS):
        for j, galaxy in enumerate(Settings.GALAXIES):
            ax = axs[j, i]
            df_disc = _get_data(f"{simulation}_{galaxy}", config,
                                AccretionRegionType.STELLAR_DISC)
            df_halo = _get_data(f"{simulation}_{galaxy}", config,
                                AccretionRegionType.HALO)
            ax.scatter(
                df_disc[Helpers.get_feat_name(rate_type)].to_numpy(),
                df_halo[Helpers.get_feat_name(rate_type)].to_numpy(),
                c=df_disc["Time_Gyr"].to_numpy(),
                s=1, cmap="viridis", zorder=11)
            ax.text(
                x=0.05, y=0.95,
                s=r"$\texttt{" + f"{simulation}_{galaxy}" + "}$",
                transform=ax.transAxes, fontsize=6,
                verticalalignment='top', horizontalalignment='left',
                color=Settings.SIMULATION_COLORS[simulation])

    # Add Auriga data as background scatter
    au_d = AurigaData.get_accretion(config, AccretionRegionType.STELLAR_DISC)
    au_h = AurigaData.get_accretion(config, AccretionRegionType.HALO)
    for g in AurigaData.RERUNS:
        for ax in axs.flatten():
            ax.scatter(
                au_d[f"{Helpers.get_rate_type_string(rate_type)}"
                     f"Rate_Au{g}_Msun/yr"],
                au_h[f"{Helpers.get_rate_type_string(rate_type)}"
                     f"Rate_Au{g}_Msun/yr"],
                s=5, marker="X", alpha=0.25, edgecolor="none",
                c="tab:gray", label="Auriga", zorder=10,
            )

    for ax in axs.flatten():
        ax.plot(ax.get_xlim(), ax.get_ylim(), c='k', ls='--',
                lw=0.5, zorder=12)

    plt.savefig(
        f"images/{Helpers.get_file_prefix(rate_type)}_relation"
        f"_{config['RUN_CODE']}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    figure_setup()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, default="01")
    args = parser.parse_args()

    config = yaml.safe_load(open(f"configs/{args.config}.yml"))
    plot_halo_vs_disc_scatter(config, RateType.INFLOW)
    plot_halo_vs_disc_scatter(config, RateType.OUTFLOW)
    plot_halo_and_disc_time_series(config, RateType.INFLOW)
    plot_halo_and_disc_time_series(config, RateType.OUTFLOW)
