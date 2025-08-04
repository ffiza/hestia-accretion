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


class RateType(Enum):
    INFLOW = 1
    OUTFLOW = 2


RATE_TYPE_STRING: dict = {
    RateType.INFLOW: "Inflow",
    RateType.OUTFLOW: "Outflow"
}


RATE_TYPE_AX_LABEL: dict = {
    RateType.INFLOW:
        r'$\dot{M}_\mathrm{in}$ [$\mathrm{M}_\odot \, \mathrm{yr}^{-1}$]',
    RateType.OUTFLOW:
        r'$\dot{M}_\mathrm{out}$ [$\mathrm{M}_\odot \, \mathrm{yr}^{-1}$]',
}


RATE_TYPE_FEAT_NAME: dict = {
    RateType.INFLOW: 'InflowRate_Msun/yr',
    RateType.OUTFLOW: 'OutflowRate_Msun/yr',
}

RATE_TYPE_FILE_PREFIX: dict = {
    RateType.INFLOW: 'inflow_accretion_tracers',
    RateType.OUTFLOW: 'outflow_accretion_tracers',
}


def _get_data(galaxy: str, config: dict) -> pd.DataFrame:
    path = f"results/{galaxy}/accretion_tracers" \
        + f"_{config['RUN_CODE']}.json"
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


def _get_auriga_data(config: dict) -> pd.DataFrame:
    AURIGA_RERUNS: list = [5, 6, 9, 13, 17, 23, 24, 26, 28]
    window_length = config["TEMPORAL_AVERAGE_WINDOW_LENGTH"]
    df = pd.read_csv("data/iza_et_al_2022/accretion_rate_tracers.csv")
    for i in AURIGA_RERUNS:
        inflow_rate = df[f"InflowRate_Au{i}_Msun/yr"].to_numpy()
        outflow_rate = df[f"OutflowRate_Au{i}_Msun/yr"].to_numpy()
        time = df["Time_Gyr"].to_numpy()
        df[f"InflowRateSmoothed_Au{i}_Msun/yr"] = windowed_average(
            time, inflow_rate, window_length)
        df[f"OutflowRateSmoothed_Au{i}_Msun/yr"] = windowed_average(
            time, outflow_rate, window_length)
    df["InflowRateSmoothedMin_Msun/yr"] = df[[
        f"InflowRateSmoothed_Au{i}_Msun/yr" for i in AURIGA_RERUNS]].min(
            axis=1)
    df["InflowRateSmoothedMax_Msun/yr"] = df[[
        f"InflowRateSmoothed_Au{i}_Msun/yr" for i in AURIGA_RERUNS]].max(
            axis=1)
    df["InflowRateSmoothedMean_Msun/yr"] = np.nanmean(
        df[[f"InflowRateSmoothed_Au{i}_Msun/yr" for i in
            AURIGA_RERUNS]].to_numpy(),
        axis=1)
    df["InflowRateSmoothedStd_Msun/yr"] = np.nanstd(
        df[[f"InflowRateSmoothed_Au{i}_Msun/yr" for i in
            AURIGA_RERUNS]].to_numpy(),
        axis=1)
    df["OutflowRateSmoothedMin_Msun/yr"] = df[[
        f"OutflowRateSmoothed_Au{i}_Msun/yr" for i in AURIGA_RERUNS]].min(
            axis=1)
    df["OutflowRateSmoothedMax_Msun/yr"] = df[[
        f"OutflowRateSmoothed_Au{i}_Msun/yr" for i in AURIGA_RERUNS]].max(
            axis=1)
    df["OutflowRateSmoothedMean_Msun/yr"] = np.nanmean(
        df[[f"OutflowRateSmoothed_Au{i}_Msun/yr" for i in
            AURIGA_RERUNS]].to_numpy(),
        axis=1)
    df["OutflowRateSmoothedStd_Msun/yr"] = np.nanstd(
        df[[f"OutflowRateSmoothed_Au{i}_Msun/yr" for i in
            AURIGA_RERUNS]].to_numpy(),
        axis=1)
    df = df.dropna()
    return df


def make_plot(config: dict, rate_type: RateType) -> None:
    df_auriga = _get_auriga_data(config)
    window_length = config["TEMPORAL_AVERAGE_WINDOW_LENGTH"]

    fig = plt.figure(figsize=(5.0, 2.0))
    gs = fig.add_gridspec(nrows=1, ncols=3, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=False)

    for ax in axs:
        ax.set_axisbelow(True)
        ax.grid(True)
        ax.set_xlim(0, 14)
        ax.set_ylim(0.1, 200)
        ax.set_yscale("log")
        ax.set_xticks([2, 4, 6, 8, 10, 12])
        ax.set_yticks([0.1, 1, 10, 100])
        ax.set_yticklabels(["0.1", "1", "10", "100"])
        ax.set_ylabel(RATE_TYPE_AX_LABEL[rate_type])
        ax.set_xlabel(r'Time [Gyr]')
        ax.label_outer()

    for i, simulation in enumerate(Settings.SIMULATIONS):
        ax = axs[i]
        for galaxy in Settings.GALAXIES:
            df = _get_data(galaxy=f"{simulation}_{galaxy}", config=config)
            ax.plot(df["Time_Gyr"].to_numpy(),
                    windowed_average(
                        df["Time_Gyr"].to_numpy(),
                        df[RATE_TYPE_FEAT_NAME[rate_type]].to_numpy(),
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
            df_auriga[f"{RATE_TYPE_STRING[rate_type]}RateSmoothedMean_Msun/yr"]
            - df_auriga[
                f"{RATE_TYPE_STRING[rate_type]}RateSmoothedStd_Msun/yr"],
            df_auriga[f"{RATE_TYPE_STRING[rate_type]}RateSmoothedMean_Msun/yr"]
            + df_auriga[
                f"{RATE_TYPE_STRING[rate_type]}RateSmoothedStd_Msun/yr"],
            color="k", alpha=0.1, label="Auriga", lw=0)
        ax.plot(df_auriga["Time_Gyr"],
                df_auriga[
                    f"{RATE_TYPE_STRING[rate_type]}RateSmoothedMean_Msun/yr"],
                ls="-", color="darkgray", lw=1, zorder=10)
        #endregion

        ax.legend(loc="lower right", framealpha=0, fontsize=5)

    plt.savefig(
        f"images/{RATE_TYPE_FILE_PREFIX[rate_type]}_{config['RUN_CODE']}.png")
    plt.close(fig)


if __name__ == "__main__":
    figure_setup()

    # Get arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = yaml.safe_load(open(f"configs/{args.config}.yml"))

    make_plot(config=config, rate_type=RateType.INFLOW)
    make_plot(config=config, rate_type=RateType.OUTFLOW)
