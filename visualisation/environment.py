import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hestia.settings import Settings
from hestia.images import figure_setup


def _get_data(galaxy: str) -> pd.DataFrame:
    path = f"results/{galaxy}/delta_1200.csv"
    return pd.read_csv(path)


def _get_auriga_data() -> pd.DataFrame:
    df = pd.read_csv("data/iza_et_al_2022/environment_delta_1200.csv")
    df["Delta1200Mean"] = np.nanmean(
        df[[f"Delta1200_Au{i}" for i in range(1, 31)]].to_numpy(),
        axis=1)
    df["Delta1200Std"] = np.nanstd(
        df[[f"Delta1200_Au{i}" for i in range(1, 31)]].to_numpy(),
        axis=1)
    return df


def make_plot() -> None:
    auriga = _get_auriga_data()

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
            r'$\delta_{1200}$')
        ax.set_xlabel(r'Time [Gyr]')
        ax.label_outer()

    for i, simulation in enumerate(Settings.SIMULATIONS):
        ax = axs[i]
        for galaxy in Settings.GALAXIES:
            try:
                df = _get_data(galaxy=f"{simulation}_{galaxy}")
            except FileNotFoundError:
                warnings.warn(
                    f"Data for {simulation}_{galaxy} not found. "
                    "Ignoring in figure.")
                continue
            ax.plot(df["Times_Gyr"],
                    df["Delta"],
                    ls=Settings.GALAXY_LINESTYLES[galaxy],
                    color=Settings.SIMULATION_COLORS[simulation],
                    lw=1, label=galaxy, zorder=11)
        ax.text(
            x=0.05, y=0.95, s=r"$\texttt{" + f"{simulation}" + "}$",
            transform=ax.transAxes, fontsize=7.0,
            verticalalignment='top', horizontalalignment='left',
            color=Settings.SIMULATION_COLORS[simulation])

        #region TestAurigaData
        ax.fill_between(
            auriga["Time_Gyr"],
            auriga["Delta1200Mean"] - auriga["Delta1200Std"],
            auriga["Delta1200Mean"] + auriga["Delta1200Std"],
            color="k", alpha=0.1, label="Auriga", lw=0, zorder=10)
        ax.plot(auriga["Time_Gyr"],
                auriga["Delta1200Mean"],
                ls="-", color="darkgray", lw=1, zorder=10)
        #endregion

        ax.legend(loc="lower right", framealpha=0, fontsize=5)

    plt.savefig("images/delta1200.pdf")
    plt.close(fig)


if __name__ == "__main__":
    figure_setup()
    make_plot()
