import json
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from hestia.math import schechter
from hestia.settings import Settings
from hestia.images import figure_setup


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


def make_plot(config: dict) -> None:
    fig = plt.figure(figsize=(5.0, 2.0))
    gs = fig.add_gridspec(nrows=1, ncols=3, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=False)

    for ax in axs.flatten():
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
            window_length = config["NET_ACCRETION_SMOOTHING_WINDOW_LENGTH"]
            polyorder = config["NET_ACCRETION_SMOOTHING_POLYORDER"]
            ax.plot(df["Time_Gyr"][is_positive],
                    savgol_filter(
                        df["NetAccretionCells_Msun/yr"].to_numpy()[
                            is_positive],
                        window_length, polyorder),
                    ls=Settings.GALAXY_LINESTYLES[galaxy],
                    color=Settings.SIMULATION_COLORS[simulation],
                    lw=1, label=galaxy)
        ax.text(
            x=0.05, y=0.95, s=r"$\texttt{" + f"{simulation}" + "}$",
            transform=ax.transAxes, fontsize=7.0,
            verticalalignment='top', horizontalalignment='left',
            color=Settings.SIMULATION_COLORS[simulation])

        #region Load External Data
        with open("data/iza_et_al_2022/accretion_fits.json") as f:
            ref = json.load(f)
        time = np.linspace(0, 14, 100)
        amplitude = ref["NetAccretionSchechterFits"]["G1"]["Amplitude"]
        alpha = ref["NetAccretionSchechterFits"]["G1"]["Alpha"]
        timescale = ref["NetAccretionSchechterFits"]["G1"]["TimeScale"]
        net_accretion = schechter(time, amplitude, timescale, alpha)
        ax.plot(time, net_accretion, ls="-.", lw=1, label=ref["Label"], c='k')
        #endregion
        
        ax.legend(loc="lower right", framealpha=0, fontsize=5)

    plt.savefig(f"images/net_accretion_cells_{config['RUN_CODE']}.pdf")
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
