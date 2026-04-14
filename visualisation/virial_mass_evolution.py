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
            "data/auriga/virial_properties.csv",
            index_col=False, comment="#")
        features = [f"M200_Au{i}_1E10Msun" for i in range(1, 31)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            df["M200_Median_1E10Msun"] = np.nanmedian(
                df[features],
                axis=1)
            df["M200_16thPerc_1E10Msun"] = np.nanpercentile(
                df[features],
                16, axis=1)
            df["M200_84thPerc_1E10Msun"] = np.nanpercentile(
                df[features],
                84, axis=1)
        return df

    @staticmethod
    def read_hestia_data(simulation: str,
                         galaxy: str,
                         config: dict) -> pd.DataFrame:
        df = pd.read_csv(
            f"data/hestia/r200_t/r200_t_{galaxy}_{simulation}.csv",
            index_col=False,
            comment="#")
        df["M200_1E10Msun"] = df["Mvir"] / 1e10
        return df


def make_plot(config: dict) -> None:

    au = Helpers.read_auriga_data()

    fig = plt.figure(figsize=(5.0, 3.0))
    gs = fig.add_gridspec(nrows=2, ncols=3, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=False)

    for idx, simulation in enumerate(Settings.HIGH_RES_SIMULATIONS):
        ax = axs[0, idx]

        ax.set_ylabel(
            r"$M_{200}$ [$10^{10} ~ \mathrm{M}_\odot$]",
            fontsize=8)
        ax.set_ylim(0, 300)
        ax.set_yticks(
            ticks=[0, 50, 100, 150, 200, 250],
            labels=[r"$0$", r"$50$", r"$100$", r"$150$", r"$200$", r"$250$"],
            fontsize=6)

        axs[1, idx].set_xlabel(
            "Time [Gyr]",
            fontsize=8)
        axs[1, idx].set_xlim(0, 14)
        axs[1, idx].set_xticks(
            ticks=[2, 4, 6, 8, 10, 12],
            labels=["2", "4", "6", "8", "10", "12"],
            fontsize=6)

        axs[1, idx].set_ylabel(
            r"$M_{200} / M_{200}(z=0)$",
            fontsize=8)
        axs[1, idx].set_ylim(0, 1.2)
        axs[1, idx].set_yticks(
            ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=[r"$0$", r"$0.2$", r"$0.4$", r"$0.6$", r"$0.8$", r"$1.0$"],
            fontsize=6)

        ax.label_outer()
        axs[1, idx].label_outer()
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
            au["Time_Gyr"],
            au["M200_Median_1E10Msun"],
            ls='-',
            color='#4d4d4d',
            lw=0.75,
            zorder=11,
            label="Auriga")
        axs[1, idx].plot(
            au["Time_Gyr"],
            au["M200_Median_1E10Msun"] / au["M200_Median_1E10Msun"].iloc[-1],
            ls='-',
            color='#4d4d4d',
            lw=0.75,
            zorder=11,
            label="Auriga")
        ax.fill_between(
            au["Time_Gyr"],
            au["M200_16thPerc_1E10Msun"],
            au["M200_84thPerc_1E10Msun"],
            zorder=10,
            color="#e6e6e6",
            edgecolor=None)

        # Hestia
        for galaxy in Settings.GALAXIES:
            he = Helpers.read_hestia_data(
                simulation,
                galaxy,
                config)
            ax.plot(
                he["Time_Gyr"],
                he["M200_1E10Msun"],
                ls=Settings.GALAXY_LINESTYLES[galaxy],
                color=Settings.SIMULATION_COLORS[simulation],
                lw=0.75,
                zorder=12,
                label=galaxy)
            axs[1, idx].plot(
                he["Time_Gyr"],
                he["M200_1E10Msun"] / he["M200_1E10Msun"].iloc[-1],
                ls=Settings.GALAXY_LINESTYLES[galaxy],
                color=Settings.SIMULATION_COLORS[simulation],
                lw=0.75,
                zorder=12,
                label=galaxy)

        ax.plot(
            ax.get_xlim(),
            [1] * 2,
            ls=':',
            color='k',
            lw=0.5,
            zorder=9)
        axs[1, idx].plot(
            axs[1, idx].get_xlim(),
            [1] * 2,
            ls=':',
            color='k',
            lw=0.5,
            zorder=9)

    axs[0, 0].legend(
        loc="lower right",
        framealpha=0,
        fontsize=5)

    plt.savefig(f"images/virial_mass_evolution_{config['RUN_CODE']}.pdf")
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
