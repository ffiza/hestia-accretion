import json
import yaml
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from hestia.settings import Settings
from hestia.images import figure_setup


def load_data(simulation: str):
    # Can read any galaxy, both have the same distance to the other
    path = f"data/hestia/r200_t/r200_t_MW_{simulation}.csv"
    df = pd.read_csv(
        path,
        index_col=False,
        comment="#")
    df = df.drop(
        columns=["Mvir"])
    df = df.rename(
        columns={"d_MW_M31_kpc": "ObjectDistance_kpc"}
    )

    # Read the expansion factor from somewhere else
    with open("results/09_18_M31/disc_size_01.json", "r") as f:
        data = json.load(f)
    df["ExpansionFactor"] = data["ExpansionFactor"]

    return df


def make_plot(config: dict) -> None:
    fig = plt.figure(figsize=(5.0, 2.0))
    gs = fig.add_gridspec(nrows=1, ncols=3, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=False)

    for idx, s in enumerate(Settings.HIGH_RES_SIMULATIONS):
        ax = axs.flat[idx]

        ax.set_xlabel(
            "Time [Gyr]",
            fontsize=8)
        ax.set_xlim(0, 14)
        ax.set_xticks(
            ticks=[2, 4, 6, 8, 10, 12],
            labels=[2, 4, 6, 8, 10, 12],
            fontsize=6)

        ax.set_ylabel(
            r"MW-M31 Distance [kpc]",
            fontsize=8)
        ax.set_ylim(0, 1_200)
        ax.set_yticks(
            ticks=[200, 400, 600, 800, 1_000],
            labels=[200, 400, 600, 800, 1_000],
            fontsize=6)
        ax.label_outer()
        ax.text(
            0.95,
            0.95,
            r"$\texttt{" + s + r"}$",
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="right",
            color=Settings.SIMULATION_COLORS[s])

        df = load_data(s)
        ax.plot(
            df["Time_Gyr"],
            df["ObjectDistance_kpc"],
            ls="-",
            color=Settings.SIMULATION_COLORS[s],
            lw=0.75,
            zorder=12,
            )

    plt.savefig(f"images/object_distance_{config['RUN_CODE']}.pdf")
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
