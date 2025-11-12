import json
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hestia.settings import Settings
from hestia.images import figure_setup


def _get_data(galaxy: str, config: dict) -> pd.DataFrame:
    with open(f"results/{galaxy}/disc_size_{config['RUN_CODE']}.json") as f:
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

    virial_radius = pd.read_csv(
        f"results/{galaxy}/virial_radius.csv")
    df["VirialRadius_ckpc"] = virial_radius["VirialRadius_ckpc"].values

    environment = pd.read_csv(
        f"results/{galaxy}/delta_1200_{config['RUN_CODE']}.csv")
    df["Delta1200"] = environment["Delta"].values

    path = f"results/{galaxy}/accretion_tracers_{config['RUN_CODE']}.json"
    with open(path) as f:
        data = json.load(f)
        time = np.array(data["Times_Gyr"])
        inflow_rate = np.array(data["InflowRate_Msun/yr"])
        outflow_rate = np.array(data["OutflowRate_Msun/yr"])
    df["InflowRate_Disc_Msun/yr"] = inflow_rate
    df["OutflowRate_Disc_Msun/yr"] = outflow_rate

    path = f"results/{galaxy}/accretion_tracers_halo_{config['RUN_CODE']}.json"
    with open(path) as f:
        data = json.load(f)
        time = np.array(data["Times_Gyr"])
        inflow_rate = np.array(data["InflowRate_Msun/yr"])
        outflow_rate = np.array(data["OutflowRate_Msun/yr"])
    df["InflowRate_Halo_Msun/yr"] = inflow_rate
    df["OutflowRate_Halo_Msun/yr"] = outflow_rate

    return df


def make_plot(config: dict) -> None:
    TICK_LABEL_FONTSIZE: float = 6
    LABEL_FONTSIZE: float = 6

    fig = plt.figure(figsize=(5.0, 7.0))
    gs = fig.add_gridspec(nrows=8, ncols=3, hspace=0.1, wspace=0)
    axs = gs.subplots(sharex=True, sharey=False)

    for i in range(3):
        axs[0, i].set_ylim(0, 300)
        axs[0, i].set_yticks(ticks=[0, 100, 200, 300],
                             labels=["0", "100", "200", "300"],
                             fontsize=TICK_LABEL_FONTSIZE)
        axs[1, i].set_ylim(0, 40)
        axs[1, i].set_yticks(ticks=[0, 10, 20, 30],
                             labels=["0", "10", "20", "30"],
                             fontsize=TICK_LABEL_FONTSIZE)
        axs[2, i].set_ylim(0, 4)
        axs[2, i].set_yticks(ticks=[0, 1, 2, 3],
                             labels=["0", "1", "2", "3"],
                             fontsize=TICK_LABEL_FONTSIZE)
        axs[3, i].set_yscale('log')
        axs[3, i].set_ylim(1, 40)
        axs[3, i].set_yticks(ticks=[1, 10],
                             labels=["1", "10"],
                             fontsize=TICK_LABEL_FONTSIZE)
        axs[4, i].set_yscale('log')
        axs[4, i].set_ylim(1, 400)
        axs[4, i].set_yticks(ticks=[1, 10, 100],
                             labels=["1", "10", "100"],
                             fontsize=TICK_LABEL_FONTSIZE)
        axs[5, i].set_yscale('log')
        axs[5, i].set_ylim(1, 400)
        axs[5, i].set_yticks(ticks=[1, 10, 100],
                             labels=["1", "10", "100"],
                             fontsize=TICK_LABEL_FONTSIZE)
        axs[6, i].set_yscale('log')
        axs[6, i].set_ylim(1, 400)
        axs[6, i].set_yticks(ticks=[1, 10, 100],
                             labels=["1", "10", "100"],
                             fontsize=TICK_LABEL_FONTSIZE)
        axs[7, i].set_yscale('log')
        axs[7, i].set_ylim(1, 400)
        axs[7, i].set_yticks(ticks=[1, 10, 100],
                             labels=["1", "10", "100"],
                             fontsize=TICK_LABEL_FONTSIZE)
    axs[0, 0].set_ylabel(r'$R_{200}$' + '\n[kpc]', fontsize=LABEL_FONTSIZE)
    axs[1, 0].set_ylabel(
        r'$R_\mathrm{d}$' + '\n[kpc]', fontsize=LABEL_FONTSIZE)
    axs[2, 0].set_ylabel(
        r'$h_\mathrm{d}$' + '\n[kpc]', fontsize=LABEL_FONTSIZE)
    axs[3, 0].set_ylabel(r'$\delta_{1200}$', fontsize=LABEL_FONTSIZE)
    axs[4, 0].set_ylabel(
        r'$\dot{M}_\mathrm{in}^\mathrm{disc}$' + '\n'
        r'[$\mathrm{M}_\odot \, \mathrm{yr}^{-1}$]', fontsize=LABEL_FONTSIZE)
    axs[5, 0].set_ylabel(
        r'$\dot{M}_\mathrm{out}^\mathrm{disc}$' + '\n'
        r'[$\mathrm{M}_\odot \, \mathrm{yr}^{-1}$]', fontsize=LABEL_FONTSIZE)
    axs[6, 0].set_ylabel(
        r'$\dot{M}_\mathrm{in}^\mathrm{halo}$' + '\n'
        r'[$\mathrm{M}_\odot \, \mathrm{yr}^{-1}$]', fontsize=LABEL_FONTSIZE)
    axs[7, 0].set_ylabel(
        r'$\dot{M}_\mathrm{out}^\mathrm{halo}$' + '\n'
        r'[$\mathrm{M}_\odot \, \mathrm{yr}^{-1}$]', fontsize=LABEL_FONTSIZE)

    for ax in axs.flatten():
        ax.set_axisbelow(True)
        ax.set_xlim(0, 14)
        ax.set_xticks(ticks=[2, 4, 6, 8, 10, 12],
                      labels=["2", "4", "6", "8", "10", "12"],
                      fontsize=TICK_LABEL_FONTSIZE)
        ax.set_xlabel('Time [Gyr]', fontsize=LABEL_FONTSIZE)
        ax.label_outer()
        ax.yaxis.set_label_coords(-0.2, 0.5)

    for i, simulation in enumerate(Settings.SIMULATIONS):
        for galaxy in Settings.GALAXIES:
            df = _get_data(galaxy=f"{simulation}_{galaxy}", config=config)
            axs[0, i].plot(
                df["Time_Gyr"],
                df["ExpansionFactor"] * df["VirialRadius_ckpc"],
                ls=Settings.GALAXY_LINESTYLES[galaxy],
                color=Settings.SIMULATION_COLORS[simulation], lw=0.75,
                label=r"$\texttt{" + f"{simulation}_{galaxy}" + "}$",
                zorder=11)
            axs[1, i].plot(
                df["Time_Gyr"], df["ExpansionFactor"] * df["DiscRadius_ckpc"],
                ls=Settings.GALAXY_LINESTYLES[galaxy],
                color=Settings.SIMULATION_COLORS[simulation], lw=0.75,
                label=r"$\texttt{" + f"{simulation}_{galaxy}" + "}$",
                zorder=11)
            axs[2, i].plot(
                df["Time_Gyr"], df["ExpansionFactor"] * df["DiscHeight_ckpc"],
                ls=Settings.GALAXY_LINESTYLES[galaxy],
                color=Settings.SIMULATION_COLORS[simulation], lw=0.75,
                label=r"$\texttt{" + f"{simulation}_{galaxy}" + "}$",
                zorder=11)
            axs[3, i].plot(
                df["Time_Gyr"], df["Delta1200"],
                ls=Settings.GALAXY_LINESTYLES[galaxy],
                color=Settings.SIMULATION_COLORS[simulation], lw=0.75,
                label=r"$\texttt{" + f"{simulation}_{galaxy}" + "}$",
                zorder=11)
            axs[4, i].plot(
                df["Time_Gyr"], df["InflowRate_Disc_Msun/yr"],
                ls=Settings.GALAXY_LINESTYLES[galaxy],
                color=Settings.SIMULATION_COLORS[simulation], lw=0.75,
                label=r"$\texttt{" + f"{simulation}_{galaxy}" + "}$",
                zorder=11)
            axs[5, i].plot(
                df["Time_Gyr"], df["OutflowRate_Disc_Msun/yr"],
                ls=Settings.GALAXY_LINESTYLES[galaxy],
                color=Settings.SIMULATION_COLORS[simulation], lw=0.75,
                label=r"$\texttt{" + f"{simulation}_{galaxy}" + "}$",
                zorder=11)
            axs[6, i].plot(
                df["Time_Gyr"], df["OutflowRate_Halo_Msun/yr"],
                ls=Settings.GALAXY_LINESTYLES[galaxy],
                color=Settings.SIMULATION_COLORS[simulation], lw=0.75,
                label=r"$\texttt{" + f"{simulation}_{galaxy}" + "}$",
                zorder=11)
            axs[7, i].plot(
                df["Time_Gyr"], df["OutflowRate_Halo_Msun/yr"],
                ls=Settings.GALAXY_LINESTYLES[galaxy],
                color=Settings.SIMULATION_COLORS[simulation], lw=0.75,
                label=r"$\texttt{" + f"{simulation}_{galaxy}" + "}$",
                zorder=11)

    for i in range(3):
        axs[0, i].legend(loc="lower right", framealpha=0, fontsize=6)

    plt.savefig(f"images/time_series_{config['RUN_CODE']}.pdf")
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
