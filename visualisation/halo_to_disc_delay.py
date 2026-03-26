import yaml
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from hestia.settings import Settings
from hestia.images import figure_setup


def load_data(simulation: str, galaxy: str, config: dict):
    run_code = config["RUN_CODE"]
    path = f"results/{simulation}_{galaxy}/halo_to_disc_delays_{run_code}.csv"
    df = pd.read_csv(path, index_col=False, comment="#")
    df = df.dropna()
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
            r"$\left< \tau_{\mathrm{halo} \, \to \, \mathrm{disc}} \right>$ [Gyr]",
            fontsize=8)
        ax.set_ylim(1, 5)
        ax.set_yticks(
            ticks=[1, 2, 3, 4, 5],
            labels=[1, 2, 3, 4, 5],
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

        for g in Settings.GALAXIES:
            df = load_data(s, g, config)
            ax.plot(
                df["Time_Gyr"],
                df["MeanDelay_Gyr"],
                ls=Settings.GALAXY_LINESTYLES[g],
                color=Settings.SIMULATION_COLORS[s],
                lw=0.75,
                zorder=12,
                label=g)

    axs[0].legend(
        loc="upper left",
        framealpha=0,
        fontsize=5)

    plt.savefig(f"images/halo_to_disc_delay_{config['RUN_CODE']}.pdf")
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
