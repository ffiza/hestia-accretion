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
                f"results/{simulation}_{galaxy}/"
                f"delta_1200_{config['RUN_CODE']}.csv")
            delta += environment["Delta"].to_list()
            time += environment["Times_Gyr"].to_list()
            galaxies += [f"{simulation}_{galaxy}"] * len(environment)

            with open(f'results/{simulation}_{galaxy}/accretion_tracers_'
                      f'{config["RUN_CODE"]}.json', 'r') as file:
                data = json.load(file)
                in_rate += data["InflowRate_Msun/yr"]
                out_rate += data["OutflowRate_Msun/yr"]

    for i in range(1, 31):
        data = pd.read_csv("data/iza_et_al_2022/accretion_rate_tracers.csv")
        if f"InflowRate_Au{i}_Msun/yr" in data.columns:
            in_rate += data[f"InflowRate_Au{i}_Msun/yr"].to_list()
            out_rate += data[f"OutflowRate_Au{i}_Msun/yr"].to_list()
            galaxies += [f"Au{i}"] * len(data)
            time += data["Time_Gyr"].to_list()
            env = pd.read_csv(
                f"data/auriga/au{i}_rerun/environment_evolution.csv")
            env = env.iloc[1:]
            delta += env["Delta1200"].to_list()

    df = pd.DataFrame({
        "Galaxy": galaxies,
        "Delta1200": delta,
        "Time_Gyr": time,
        "InflowRate_Msun/yr": in_rate,
        "OutflowRate_Msun/yr": out_rate,
    })

    return df


def plot_accretion_vs_environment(config: dict) -> None:
    df = _get_data(config)
    is_auriga = df["Galaxy"].str.startswith("Au")

    fig = plt.figure(figsize=(3.0, 4.0))
    gs = fig.add_gridspec(nrows=2, ncols=1, hspace=0, wspace=0.4)
    axs = np.array(gs.subplots(sharex=True, sharey=False))

    axs[0].set_xlim(0, 1.4)
    axs[0].set_ylim(1E-1, 600)
    axs[0].set_yscale("log")
    axs[0].set_yticks(ticks=[1, 10, 100],
                      labels=["1", "10", "100"],
                      fontsize=7)
    axs[0].set_ylabel(
        r"$\dot{M}_\mathrm{in}^\mathrm{disc} \, [\mathrm{M}_\odot"
        r" \, \mathrm{yr}^{-1}]$")
    axs[0].scatter(
        np.log10(df[is_auriga]["Delta1200"].to_numpy()),
        df[is_auriga]["OutflowRate_Msun/yr"].to_numpy(),
        s=5, marker="X", alpha=0.25, edgecolor="none",
        c="tab:gray", label="Auriga",
    )
    axs[0].scatter(
        np.log10(df[~is_auriga]["Delta1200"].to_numpy()),
        df[~is_auriga]["InflowRate_Msun/yr"].to_numpy(),
        s=1, cmap="viridis", c=df[~is_auriga]["Time_Gyr"], label="_Hestia",
    )
    axs[0].legend(fontsize=5, loc="upper left", frameon=False)

    axs[1].set_ylim(1E-1, 600)
    axs[1].set_yscale("log")
    axs[1].set_xticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4],
                      labels=["0", "0.2", "0.4", "0.6", "0.8",
                              "1.0", "1.2", "1.4"],
                      fontsize=7)
    axs[1].set_yticks(ticks=[1, 10, 100],
                      labels=["1", "10", "100"],
                      fontsize=7)
    axs[1].set_ylabel(
        r"$\dot{M}_\mathrm{out}^\mathrm{disc} \, [\mathrm{M}_\odot"
        r" \, \mathrm{yr}^{-1}]$")
    axs[1].set_xlabel(r"$\log_{10} \delta_{1200}$")

    axs[1].scatter(
        np.log10(df[is_auriga]["Delta1200"].to_numpy()),
        df[is_auriga]["OutflowRate_Msun/yr"].to_numpy(),
        s=5, marker="X", alpha=0.2, edgecolor="none",
        c="tab:gray", label="Auriga",
    )
    axs[1].scatter(
        np.log10(df[~is_auriga]["Delta1200"].to_numpy()),
        df[~is_auriga]["OutflowRate_Msun/yr"].to_numpy(),
        s=1, cmap="viridis", c=df[~is_auriga]["Time_Gyr"], label="_Hestia",
    )

    plt.savefig(
        f"images/accretion_vs_environment_{config['RUN_CODE']}.pdf")
    plt.close(fig)


def plot_inflows_vs_environment_by_galaxy(config: dict) -> None:
    df = _get_data(config)
    is_auriga = df["Galaxy"].str.startswith("Au")

    fig, axs = plt.subplots(figsize=(5.0, 3.0), nrows=2, ncols=3,
                            sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0, wspace=0)

    for ax in axs.flatten():
        ax.set_xlim(0, 1.4)
        ax.set_ylim(1E-1, 600)
        ax.set_yscale("log")
        ax.set_xticks(ticks=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2],
                      labels=["0.2", "0.4", "0.6", "0.8", "1.0", "1.2"],
                      fontsize=7)
        ax.set_yticks(ticks=[1, 10, 100],
                      labels=["1", "10", "100"],
                      fontsize=7)
        ax.set_xlabel(r"$\log_{10} \delta_{1200}$", fontsize=8)
        ax.set_ylabel(
            r"$\dot{M}_\mathrm{in}^\mathrm{disc} \, [\mathrm{M}_\odot"
            r" \, \mathrm{yr}^{-1}]$", fontsize=8)
        ax.set_axisbelow(True)
        ax.label_outer()
        ax.scatter(
            np.log10(df[is_auriga]["Delta1200"].to_numpy()),
            df[is_auriga]["InflowRate_Msun/yr"].to_numpy(),
            s=5, marker="X", alpha=0.2, edgecolor="none",
            color="tab:gray", zorder=10, label="Auriga",
        )

    for simulation in Settings.SIMULATIONS:
        for galaxy in Settings.GALAXIES:
            subset = df[(df["Galaxy"] == f"{simulation}_{galaxy}")]
            ax = axs[Settings.GALAXIES.index(galaxy),
                     Settings.SIMULATIONS.index(simulation)]
            ax.scatter(
                np.log10(subset["Delta1200"].to_numpy()),
                subset["InflowRate_Msun/yr"].to_numpy(),
                s=1, zorder=11, label="_Hestia",
                c=subset["Time_Gyr"], cmap="viridis",
            )
            ax.text(
                x=0.05, y=0.95,
                s=r"$\texttt{" + f"{simulation}_{galaxy}" + "}$",
                transform=ax.transAxes, fontsize=7.0,
                verticalalignment='top', horizontalalignment='left',
                color=Settings.SIMULATION_COLORS[simulation])

    plt.savefig(
        f"images/accretion_vs_environment_by_galaxy_{config['RUN_CODE']}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    figure_setup()

    # Get arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = yaml.safe_load(open(f"configs/{args.config}.yml"))

    plot_accretion_vs_environment(config)
    plot_inflows_vs_environment_by_galaxy(config)
