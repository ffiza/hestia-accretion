import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import yaml

from hestia.settings import Settings
from hestia.images import figure_setup


def make_plot(config: dict) -> None:
    settings = Settings()

    fig = plt.figure(figsize=(4.0, 2.0))
    gs = fig.add_gridspec(nrows=1, ncols=2, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=True)

    for ax in axs.flatten():
        ax.label_outer()
        ax.set_axisbelow(True)
        ax.grid(True)
        ax.set_xlim(0, 14)
        ax.set_ylim(0, 40)
        ax.set_xticks([2, 4, 6, 8, 10, 12])
        ax.set_yticks([10, 20, 30])

    for i, galaxy in enumerate(settings.galaxies):
        ax = axs[i]
        ax2 = ax.twinx()
        ax2.set_ylim(0, 4)
        ax2.set_yticks([1, 2, 3])
        ax2.tick_params(axis="y", direction="in")

        # Load galaxy data
        path = f"data/{galaxy}/disc_size_config{config['RUN_CODE']}.json"
        with open(path) as f:
            data = json.load(f)
            time = np.array(data["Times_Gyr"])
            rd = np.array(data["DiscRadius_ckpc"])
            hd = np.array(data["DiscHeight_ckpc"])
            a = np.array(data["ExpansionFactor"])

        ax.plot(time, a * rd, '-', color='tab:blue', lw=1.5)
        ax2.plot(time, a * hd, '--', color='tab:red', lw=1.5)

        ax.text(0.05, 0.95, r"$\texttt{" + f"{galaxy}" + "}$",
                ha='left', va='top', transform=ax.transAxes)

        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel(r'$R_\mathrm{d}$ [kpc]', color='tab:blue')
        if ax.get_subplotspec().is_last_row():
            ax.set_xlabel(r'Time [Gyr]')
        if ax.get_subplotspec().is_last_col():
            ax2.set_ylabel(r'$h_\mathrm{d}$ [kpc]', color='tab:red')
            ax2.set_yticklabels([1, 2, 3])
        else:
            ax2.set_yticklabels([])

    plt.savefig(f"images/disc_size_config{config['RUN_CODE']}.pdf")
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
