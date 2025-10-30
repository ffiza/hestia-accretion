import json
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hestia.images import figure_setup
from hestia.settings import Settings


def _get_data(config: dict) -> pd.DataFrame:
    galaxies = []
    delta = []
    time = []
    in_rate = []
    out_rate = []

    for simulation in Settings.SIMULATIONS:
        for galaxy in Settings.GALAXIES:
            environment = pd.read_csv(
                f"results/{simulation}_{galaxy}/delta_1200.csv")
            delta += environment["Delta"].to_list()
            time += environment["Times_Gyr"].to_list()
            galaxies += [f"{simulation}_{galaxy}"] * len(environment)

            with open(f'results/{simulation}_{galaxy}/accretion_tracers_{config["RUN_CODE"]}.json', 'r') as file:
                data = json.load(file)
                in_rate += data["InflowRate_Msun/yr"]
                out_rate += data["OutflowRate_Msun/yr"]

    # for i in range(1, 31):
    #     data = pd.read_csv("data/iza_et_al_2022/accretion_rate_tracers.csv")
    #     if f"InflowRate_Au{i}_Msun/yr" in data.columns:
    #         print(len(data))
    #         in_rate += data[f"InflowRate_Au{i}_Msun/yr"].to_list()
    #         out_rate += data[f"OutflowRate_Au{i}_Msun/yr"].to_list()
    #         galaxies += [f"Au{i}"] * len(data)
    #         time += data["Time_Gyr"].to_list()
    #         env = pd.read_csv(f"data/auriga/au{i}/environment_evolution.csv")
    #         print(len(env))
    #         delta += env["Delta1200"].to_list()

    df = pd.DataFrame({
        "Galaxy": galaxies,
        "Delta1200": delta,
        "Time_Gyr": time,
        "InflowRate_Msun/yr": in_rate,
        "OutflowRate_Msun/yr": out_rate,
    })

    return df


def plot_prop_comparison(config: dict) -> None:
    df = _get_data(config)

    fig = plt.figure(figsize=(3.0, 4.0))
    gs = fig.add_gridspec(nrows=2, ncols=1, hspace=0, wspace=0.4)
    axs = np.array(gs.subplots(sharex=True, sharey=False))

    axs[0].set_xlim(0, 1.5)
    axs[0].set_ylim(1E-1, 600)
    axs[0].set_yscale("log")
    axs[0].set_yticks([1, 10, 100])
    axs[0].set_yticklabels(["1", "10", "100"])
    axs[0].set_ylabel(
        r"$\dot{M}_\mathrm{in} \, [\mathrm{M}_\odot"
        r" \, \mathrm{yr}^{-1}]$")
    axs[0].scatter(
        np.log10(df["Delta1200"].to_numpy()),
        df["InflowRate_Msun/yr"].to_numpy(),
        s=5, marker="o", alpha=0.75, edgecolor="none", c=df["Time_Gyr"],
        vmin=0, vmax=14, cmap="gnuplot2",
    )

    axs[1].set_ylim(1E-1, 600)
    axs[1].set_yscale("log")
    axs[1].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
    axs[1].set_yticks([1, 10, 100])
    axs[1].set_yticklabels(["1", "10", "100"])
    axs[1].set_ylabel(
        r"$\dot{M}_\mathrm{out} \, [\mathrm{M}_\odot"
        r" \, \mathrm{yr}^{-1}]$")
    axs[1].set_xlabel(r"$\log_{10} \delta_{1200}$")
    s = axs[1].scatter(
        np.log10(df["Delta1200"].to_numpy()),
        df["OutflowRate_Msun/yr"].to_numpy(),
        s=5, marker="o", alpha=0.75, edgecolor="none", c=df["Time_Gyr"],
        vmin=0, vmax=14, cmap="gnuplot2",
    )

    cbax = axs[0].inset_axes([0.45, 0.1, 0.5, 0.025],
                             transform=axs[0].transAxes)
    cb = plt.colorbar(s, cax=cbax, orientation="horizontal")
    cbax.set_xlim(0, 14)
    cb.set_ticks([0, 2, 4, 6, 8, 10, 12, 14])
    cb.set_ticklabels(['0', '2', '4', '6', '8', '10', '12', '14'],
                      fontsize=5.0)
    cbax.set_xlabel("Time [Gyr]", fontsize=6)
    cbax.xaxis.set_label_position('top')

    for ax in axs.flatten():
        ax.set_axisbelow(True)

    plt.savefig("images/accretion_vs_environment.pdf")
    plt.close(fig)


if __name__ == "__main__":
    figure_setup()

    # Get arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = yaml.safe_load(open(f"configs/{args.config}.yml"))

    plot_prop_comparison(config)
