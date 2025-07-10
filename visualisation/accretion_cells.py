import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import yaml
from scipy.signal import savgol_filter

from hestia.settings import Settings
from hestia.images import figure_setup
from hestia.math import schechter


def make_plot(config: dict) -> None:
    settings = Settings()

    fig = plt.figure(figsize=(2.5, 2.5))
    gs = fig.add_gridspec(nrows=1, ncols=1, hspace=0, wspace=0)
    ax = gs.subplots(sharex=True, sharey=True)

    ax.label_outer()
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.set_xlim(0, 14)
    ax.set_ylim(0.1, 100)
    ax.set_yscale("log")
    ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])
    ax.set_yticks([0.1, 1, 10, 100])
    ax.set_yticklabels(["0.1", "1", "10", "100"])
    ax.set_ylabel(
        r'$\dot{M}_\mathrm{net}$ [$\mathrm{M}_\odot \, \mathrm{yr}^{-1}$]')
    ax.set_xlabel(r'Time [Gyr]')

    for galaxy in ["17_11_MW", "17_11_M31", "09_18_MW", "09_18_M31", "37_11_MW", "37_11_M31"]:
        path = f"results/{galaxy}/" \
            + f"net_accretion_cells_{config['RUN_CODE']}.json"
        with open(path) as f:
            data = json.load(f)
            time = np.array(data["Times_Gyr"])
            net_accretion = np.array(data["NetAccretionCells_Msun/yr"])

        is_positive = net_accretion >= 0.1
        window_length = config["NET_ACCRETION_SMOOTHING_WINDOW_LENGTH"]
        polyorder = config["NET_ACCRETION_SMOOTHING_POLYORDER"]
        ax.plot(time[is_positive],
                savgol_filter(net_accretion[is_positive],
                              window_length, polyorder),
                ls=settings.galaxy_lss[galaxy],
                color=settings.galaxy_colors[galaxy],
                lw=1, label=r"$\texttt{" + f"{galaxy}" + "}$")

    with open("data/iza_et_al_2022/accretion_fits.json") as f:
        ref = json.load(f)
    time = np.linspace(0, 14, 100)
    amplitude = ref["NetAccretionSchechterFits"]["G1"]["Amplitude"]
    alpha = ref["NetAccretionSchechterFits"]["G1"]["Alpha"]
    timescale = ref["NetAccretionSchechterFits"]["G1"]["TimeScale"]
    net_accretion = schechter(time, amplitude, timescale, alpha)
    ax.plot(time, net_accretion, ls="-.", lw=1, label=ref["Label"], c='k')

    ax.legend(loc="lower right", framealpha=0, fontsize=6.0)

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
