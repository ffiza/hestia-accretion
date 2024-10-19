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
    gs = fig.add_gridspec(nrows=1, ncols=2, hspace=0, wspace=0.3)
    axs = gs.subplots(sharex=True, sharey=False)

    for ax in axs.flatten():
        ax.set_axisbelow(True)
        ax.grid(True)
        ax.set_xlim(0, 14)
        ax.set_xticks([2, 4, 6, 8, 10, 12])
        ax.set_xlabel(r'Time [Gyr]')

    axs[0].set_ylim(0, 40)
    axs[0].set_yticks([0, 10, 20, 30, 40])
    axs[0].set_ylabel(r'$R_\mathrm{d}$ [kpc]')

    axs[1].set_ylim(0, 4)
    axs[1].set_yticks([0, 1, 2, 3, 4])
    axs[1].set_ylabel(r'$h_\mathrm{d}$ [kpc]')

    for galaxy in ["17_11_MW", "17_11_M31"]:
        # Load galaxy data
        path = f"results/{galaxy}/disc_size_config{config['RUN_CODE']}.json"
        with open(path) as f:
            data = json.load(f)
            time = np.array(data["Times_Gyr"])
            rd = np.array(data["DiscRadius_ckpc"])
            hd = np.array(data["DiscHeight_ckpc"])
            a = np.array(data["ExpansionFactor"])

        axs[0].plot(time, a * rd, ls=settings.galaxy_lss[galaxy],
                    color=settings.galaxy_colors[galaxy], lw=1,
                    label=r"$\texttt{" + f"{galaxy}" + "}$")
        axs[1].plot(time, a * hd, ls=settings.galaxy_lss[galaxy],
                    color=settings.galaxy_colors[galaxy], lw=1)

    axs[0].legend(loc="upper left", framealpha=0, fontsize=6.0)

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
