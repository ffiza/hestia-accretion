import json
import yaml
import argparse
import numpy as np
import pandas as pd
from enum import Enum
from scipy.stats import binned_statistic, ks_2samp
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
        accretion_region_type: AccretionRegionType,
        config: dict) -> None:
    df_auriga = AurigaData.get_accretion(config, accretion_region_type)
    ax.fill_between(
        df_auriga["Time_Gyr"],
        df_auriga[f"{Helpers.get_rate_type_string(rate_type)}"
                  "RateSmoothedMean_Msun/yr"]
        - df_auriga[f"{Helpers.get_rate_type_string(rate_type)}"
                    "RateSmoothedStd_Msun/yr"],
        df_auriga[f"{Helpers.get_rate_type_string(rate_type)}"
                  "RateSmoothedMean_Msun/yr"]
        + df_auriga[
            f"{Helpers.get_rate_type_string(rate_type)}"
            "RateSmoothedStd_Msun/yr"],
        color="k", alpha=0.1, label="Auriga", lw=0)
    ax.plot(df_auriga["Time_Gyr"],
            df_auriga[
                f"{Helpers.get_rate_type_string(rate_type)}"
                "RateSmoothedMean_Msun/yr"],
            ls="-", color="darkgray", lw=0.75, zorder=10)


def plot_accretion_evolution(
        config: dict,
        rate_type: RateType,
        accretion_region_type: AccretionRegionType) -> None:
    window_length = config["TEMPORAL_AVERAGE_WINDOW_LENGTH"]

    fig = plt.figure(figsize=(5.0, 2.0))
    gs = fig.add_gridspec(nrows=1, ncols=3, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=False)

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
        ax.set_ylabel(Helpers.get_ylabel(rate_type, accretion_region_type),
                      fontsize=8)
        ax.set_xlabel(r'Time [Gyr]', fontsize=8)
        ax.label_outer()

    for i, simulation in enumerate(Settings.SIMULATIONS):
        ax = axs[i]
        for galaxy in Settings.GALAXIES:
            df = _get_data(f"{simulation}_{galaxy}", config,
                           accretion_region_type)
            ax.plot(df["Time_Gyr"].to_numpy(),
                    windowed_average(
                        df["Time_Gyr"].to_numpy(),
                        df[Helpers.get_feat_name(rate_type)].to_numpy(),
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

        _add_auriga_data_to_ax(ax, rate_type, accretion_region_type, config)

        ax.legend(loc="lower right", framealpha=0, fontsize=5)

    suffix = get_accretion_region_suffix(accretion_region_type)
    plt.savefig(
        f"images/{Helpers.get_file_prefix(rate_type)}{suffix}"
        f"_{config['RUN_CODE']}.pdf")
    plt.close(fig)


def plot_halo_disc_relation(
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

    for i, simulation in enumerate(Settings.SIMULATIONS):
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


def plot_simulation_comparison(config: dict) -> None:
    fig = plt.figure(figsize=(5, 2))
    gs = fig.add_gridspec(nrows=1, ncols=2, hspace=0.2, wspace=0.3)
    axs = gs.subplots(sharex=True, sharey=False)

    au = AurigaData.get_accretion(config, AccretionRegionType.STELLAR_DISC)
    he = HestiaData.get_accretion(config, AccretionRegionType.STELLAR_DISC)

    axs[0].set_xlabel(
        r'$\dot{M}_\mathrm{in}^\mathrm{Au}$ [$\mathrm{M}_\odot'
        r'\, \mathrm{yr}^{-1}$]', fontsize=7)
    axs[0].set_ylabel(
        r'$\dot{M}_\mathrm{in}^\mathrm{He}$ [$\mathrm{M}_\odot'
        r'\, \mathrm{yr}^{-1}$]', fontsize=7)
    axs[1].set_xlabel(
        r'$\dot{M}_\mathrm{out}^\mathrm{Au}$ [$\mathrm{M}_\odot'
        r'\, \mathrm{yr}^{-1}$]', fontsize=7)
    axs[1].set_ylabel(
        r'$\dot{M}_\mathrm{out}^\mathrm{He}$ [$\mathrm{M}_\odot'
        r'\, \mathrm{yr}^{-1}$]', fontsize=7)
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

    au_binned_inflow = binned_statistic(
            au["Time_Gyr"].to_numpy(),
            au["InflowRateMean_Msun/yr"].to_numpy(),
            statistic="mean", bins=100, range=(0, 14),
            )
    bin_centers = au_binned_inflow[1][1:] - np.diff(au_binned_inflow[1]) / 2

    axs[0].scatter(
        au_binned_inflow[0],
        binned_statistic(
            he["Time_Gyr"].to_numpy(),
            he["InflowRateMean_Msun/yr"].to_numpy(),
            statistic="mean", bins=100, range=(0, 14),
        )[0],
        c=bin_centers, s=1.5, zorder=11, cmap="gnuplot", vmin=0, vmax=14)
    s = axs[1].scatter(
        binned_statistic(
            au["Time_Gyr"].to_numpy(),
            au["OutflowRateMean_Msun/yr"].to_numpy(),
            statistic="mean", bins=100, range=(0, 14),
        )[0],
        binned_statistic(
            he["Time_Gyr"].to_numpy(),
            he["OutflowRateMean_Msun/yr"].to_numpy(),
            statistic="mean", bins=100, range=(0, 14),
        )[0],
        c=bin_centers, s=1.5, zorder=11, cmap="gnuplot", vmin=0, vmax=14)

    cbax = axs[0].inset_axes(
        [0.1, 0.1, 0.8, 0.025],
        transform=axs[0].transAxes)
    cb = plt.colorbar(s, cax=cbax, orientation="horizontal")
    cbax.set_xlim(0, 14)
    cb.set_ticks(ticks=[0, 2, 4, 6, 8, 10, 12, 14],
                 labels=['0', '2', '4', '6', '8', '10', '12', '14'],
                 fontsize=5)
    cbax.set_xlabel("Time [Gyr]", fontsize=5)
    cbax.xaxis.set_label_position('top')

    for ax in axs.flatten():
        ax.plot(ax.get_xlim(), ax.get_ylim(), c='k', ls='--',
                lw=0.5, zorder=12)

    plt.savefig(
        f"images/accretion_tracers_simulation_comparison"
        f"_{config['RUN_CODE']}.pdf")
    plt.close(fig)


def perform_ks_test(config: dict) -> None:
    # Perform Kolmogorov-Smirnov test
    au = AurigaData.get_accretion(config, AccretionRegionType.STELLAR_DISC)
    he = HestiaData.get_accretion(config, AccretionRegionType.STELLAR_DISC)

    au_accumulated = pd.DataFrame({
        "Time_Gyr": au[[
            "Time_Gyr" for _ in AurigaData.RERUNS]].values.ravel(),
        "InflowRate_Msun/yr": au[[
            f"InflowRate_Au{i}_Msun/yr"
            for i in AurigaData.RERUNS]].values.ravel(),
    })
    he_accumulated = pd.DataFrame({
        "Time_Gyr": he[[
            "Time_Gyr" for _ in Settings.GALAXIES
            for _ in Settings.SIMULATIONS]].values.ravel(),
        "InflowRate_Msun/yr": he[[
            f"InflowRate_{s}_{g}_Msun/yr"
            for s in Settings.SIMULATIONS
            for g in Settings.GALAXIES]].values.ravel(),
    })

    bin_edges = np.linspace(0, 13.9, 57)
    bin_centers = bin_edges[1:] - np.diff(bin_edges) / 2
    pvalues = []
    for bin_edge_start, bin_edge_end in zip(bin_edges[:-1], bin_edges[1:]):
        au_values = au_accumulated[
            (au_accumulated["Time_Gyr"] >= bin_edge_start)
            & (au_accumulated["Time_Gyr"] < bin_edge_end)][
                "InflowRate_Msun/yr"].to_numpy()
        he_values = he_accumulated[
            (he_accumulated["Time_Gyr"] >= bin_edge_start)
            & (he_accumulated["Time_Gyr"] < bin_edge_end)][
                "InflowRate_Msun/yr"].to_numpy()
        pvalues.append(ks_2samp(au_values, he_values).pvalue)
    print(pvalues)

    # Plot results
    fig = plt.figure(figsize=(3, 3))
    gs = fig.add_gridspec(nrows=2, ncols=1, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=False)


    axs[0].set_ylabel(
        r'$\dot{M}_\mathrm{in}$ [$\mathrm{M}_\odot'
        r'\, \mathrm{yr}^{-1}$]', fontsize=7)
    axs[0].set_yscale("log")
    axs[0].set_yticks(
        ticks=[0.1, 1, 10, 100],
        labels=["0.1", "1", "10", "100"],
        fontsize=6)

    axs[1].set_xlabel("Time [Gyr]", fontsize=7)
    axs[1].set_ylabel(r'$p$-value', fontsize=7)
    axs[1].set_ylim(0, 1)
    axs[1].set_yticks(
        ticks=[0, 0.2, 0.4, 0.6, 0.8],
        labels=["0", "0.2", "0.4", "0.6", "0.8"],
        fontsize=6)

    for ax in axs.flatten():
        ax.set_axisbelow(True)
        ax.set_xlim(0, 14)
        ax.set_xticks(ticks=[2, 4, 6, 8, 10, 12],
                      labels=["2", "4", "6", "8", "10", "12"],
                      fontsize=6)

    axs[0].scatter(
        au_accumulated["Time_Gyr"], au_accumulated["InflowRate_Msun/yr"],
        s=5, zorder=11, edgecolor="none", facecolor="tab:blue", alpha=0.25)
    au_binned_inflow = binned_statistic(
            au_accumulated["Time_Gyr"].to_numpy(),
            au_accumulated["InflowRate_Msun/yr"].to_numpy(),
            statistic="mean", bins=bin_edges,
            )
    axs[0].plot(
        bin_centers, au_binned_inflow[0],
        lw=0.75, zorder=12, color="tab:blue")

    axs[0].scatter(
        he_accumulated["Time_Gyr"], he_accumulated["InflowRate_Msun/yr"],
        s=5, zorder=11, edgecolor="none", facecolor="tab:red", alpha=0.25)
    he_binned_inflow = binned_statistic(
            he_accumulated["Time_Gyr"].to_numpy(),
            he_accumulated["InflowRate_Msun/yr"].to_numpy(),
            statistic="mean", bins=bin_edges,
            )
    axs[0].plot(
        bin_centers, he_binned_inflow[0],
        lw=0.75, zorder=12, color="tab:red")

    axs[1].scatter(
        bin_centers, pvalues,
        s=5, zorder=11, edgecolor="none", facecolor="k")

    plt.savefig(
        f"images/accretion_tracers_ks_test_{config['RUN_CODE']}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    figure_setup()

    # Get arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = yaml.safe_load(open(f"configs/{args.config}.yml"))

    # plot_accretion_evolution(
    #     config, RateType.INFLOW, AccretionRegionType.STELLAR_DISC)
    # plot_accretion_evolution(
    #     config, RateType.OUTFLOW, AccretionRegionType.STELLAR_DISC)
    # plot_accretion_evolution(
    #     config, RateType.INFLOW, AccretionRegionType.HALO)
    # plot_accretion_evolution(
    #     config, RateType.OUTFLOW, AccretionRegionType.HALO)
    # plot_halo_disc_relation(config, RateType.INFLOW)
    # plot_halo_disc_relation(config, RateType.OUTFLOW)
    # plot_simulation_comparison(config)
    perform_ks_test(config)
