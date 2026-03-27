import json
import yaml
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hestia.images import figure_setup
from hestia.settings import Settings


class Helpers:
    @staticmethod
    def read_auriga_data() -> pd.DataFrame:
        df = pd.read_csv(
            "data/auriga/directional_density.csv",
            index_col=False, comment="#")
        df["Avg_Rho50_Msun/ckpc3"] = df[
            [f"Au{i}_Rho50_Msun/ckpc3" for i in range(1, 31)]].mean(axis=1)
        return df

    @staticmethod
    def read_hestia_data(simulation: str,
                         galaxy: str,
                         config: dict) -> pd.DataFrame:
        if galaxy == "M31":
            print("Here")
            raise ValueError(
                "These files are only available for MW."
            )

        with open(f"results/{simulation}_{galaxy}/"
                  f"density_profile_{config['RUN_CODE']}.json",
                  "r",
                  encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)

        # Add the distance between M31 and MW as metadata
        path = f"data/hestia/r200_t/r200_t_MW_{simulation}.csv"
        data = pd.read_csv(
            path,
            index_col=False,
            comment="#")
        df.distance_kpc = data["d_MW_M31_kpc"].to_list()[-1]
        return df


def make_plot(config: dict) -> None:

    au = Helpers.read_auriga_data()

    fig = plt.figure(figsize=(5.0, 2.0))
    gs = fig.add_gridspec(nrows=1, ncols=3, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=False)

    for idx, simulation in enumerate(Settings.HIGH_RES_SIMULATIONS):
        ax = axs.flat[idx]

        ax.set_xlabel(
            "Distance [ckpc]",
            fontsize=8)
        ax.set_xlim(0, 1000)
        ax.set_xticks(
            ticks=[200, 400, 600, 800],
            labels=["200", "400", "600", "800"],
            fontsize=6)

        ax.set_ylabel(
            r"$\rho$ [$\mathrm{M}_\odot ~ \mathrm{ckpc}^{-3}$]",
            fontsize=8)
        ax.set_yscale("log")
        ax.set_ylim(5E-1, 5E6)
        ax.set_yticks(
            ticks=[1E2, 1E4, 1E6],
            labels=[r"$10^2$", r"$10^4$", r"$10^6$"],
            fontsize=6)

        ax.label_outer()
        ax.text(
            0.05,
            0.95,
            r"$\texttt{" + simulation + r"}$",
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="left",
            color=Settings.SIMULATION_COLORS[simulation])

        # Auriga
        ax.plot(
            au["Radius_ckpc"],
            au["Avg_Rho50_Msun/ckpc3"],
            ls='-',
            color='#4d4d4d',
            lw=0.75,
            zorder=12,
            label="Auriga")

        # Hestia
        he = Helpers.read_hestia_data(
            simulation,
            "MW",
            config)
        ax.plot(
            he["radii"],
            he["rho_med"],
            ls='-',
            color=Settings.SIMULATION_COLORS[simulation],
            lw=0.75,
            zorder=12,
            label="Hestia")
        ax.fill_between(
            he["radii"],
            he["rho_p16"],
            he["rho_p84"],
            zorder=10,
            color=Settings.SIMULATION_COLORS[simulation],
            edgecolor=None,
            alpha=0.2)
        ax.plot(
            he["radii"],
            he["rho_med_cone"],
            ls='--',
            color=Settings.SIMULATION_COLORS[simulation],
            lw=0.75,
            zorder=12,
            label="Hestia (MW to M31)")
        ax.plot(
            he["radii"],
            he["rho_med_anticone"],
            ls=':',
            color=Settings.SIMULATION_COLORS[simulation],
            lw=0.75,
            zorder=12,
            label="Hestia (MW to AB)")

        m31_pos_idx = np.argmin(np.abs(he["radii"] - he.distance_kpc))
        y_max = he["rho_med_cone"].loc[m31_pos_idx]
        factor = 4
        ax.annotate(
            "",
            xy=(he.distance_kpc, y_max),
            xytext=(he.distance_kpc, y_max * factor),
            arrowprops=dict(
                arrowstyle="->, head_width=0.15, head_length=0.3",
                color="black",
                lw=0.75)
        )

    axs[0].legend(
        loc="lower left",
        framealpha=0,
        fontsize=5)

    plt.savefig(f"images/directional_density_profile_{config['RUN_CODE']}.pdf")
    plt.close(fig)


def main() -> None:
    figure_setup()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(open(f"configs/{args.config}.yml"))

    make_plot(config)


if __name__ == "__main__":
    main()
