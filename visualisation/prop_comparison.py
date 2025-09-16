import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hestia.images import figure_setup
from hestia.settings import Settings


def _get_hestia_data(config: dict) -> pd.DataFrame:
    simulations = []
    galaxies = []
    sfr = []
    m_gas = []
    m_star = []
    delta = []
    for simulation in Settings.SIMULATIONS:
        for galaxy in Settings.GALAXIES:
            data = pd.read_csv(
                f"data/hestia/{galaxy}_M_SFR_t_Hestia{simulation}.csv")
            simulations.append(simulation)
            galaxies.append(galaxy)
            sfr.append(data["SFR"].to_numpy()[-1])
            m_gas.append(data["Mgas"].to_numpy()[-1] / 1e10)
            m_star.append(data["Mstar"].to_numpy()[-1] / 1e10)
            environment = pd.read_csv(
                f"results/{simulation}_{galaxy}/delta_1200.csv")
            delta.append(environment["Delta"].to_numpy()[-1])
    df = pd.DataFrame({
        "Simulation": simulations,
        "Galaxy": galaxies,
        "SFR_Msun/yr": sfr,
        "Mgas_10^10Msun": m_gas,
        "Mstar_10^10Msun": m_star,
        "Delta1200": delta,
    })
    return df


def _get_auriga_data(config: dict) -> pd.DataFrame:
    df = pd.read_csv("data/iza_et_al_2022/table_1.csv")

    sfr = pd.read_csv("data/iza_et_al_2022/sfr.csv")
    present_day_sfr = []
    for galaxy in df["Galaxy"]:
        present_day_sfr.append(
            sfr[f"SFR_Au{galaxy}_Msun/yr"].to_numpy()[-1]
        )
    df["SFR_Msun/yr"] = present_day_sfr

    delta = pd.read_csv("data/iza_et_al_2022/environment_delta_1200.csv")
    present_day_delta = []
    for galaxy in df["Galaxy"]:
        present_day_delta.append(
            delta[f"Delta1200_Au{galaxy}"].to_numpy()[-1]
        )
    df["Delta1200"] = present_day_delta

    return df


def plot_prop_comparison(config: dict) -> None:
    auriga = _get_auriga_data(config)
    hestia = _get_hestia_data(config)

    fig = plt.figure(figsize=(5.0, 4.0))
    gs = fig.add_gridspec(nrows=2, ncols=2, hspace=0, wspace=0.4)
    axs = np.array(gs.subplots(sharex=False, sharey=False))

    axs[0, 0].set_xlim(10.4, 11.2)
    axs[0, 0].set_ylim(9.6, 11.4)
    axs[1, 0].set_xticks([10.6, 10.8, 11])
    axs[0, 0].set_xticklabels([])
    axs[0, 0].set_yticks([9.8, 10, 10.2, 10.4, 10.6, 10.8, 11, 11.2])
    axs[0, 0].set_ylabel(r"$\log_{10} M_\mathrm{gas} \, [\mathrm{M}_\odot]$")
    axs[0, 0].scatter(
        np.log10(auriga["SubhaloStellarMass_10^10Msun"].to_numpy() * 1e10),
        np.log10(auriga["SubhaloGasMass_10^10Msun"].to_numpy() * 1e10),
        s=12, c="darkgray", edgecolor="none", label="Auriga",
    )
    for _, row in hestia.iterrows():
        axs[0, 0].scatter(
            np.log10(row["Mstar_10^10Msun"] * 1e10),
            np.log10(row["Mgas_10^10Msun"] * 1e10),
            s=14, facecolors="none",
            marker=Settings.GALAXY_SYMBOLS[row["Galaxy"]],
            edgecolor=Settings.SIMULATION_COLORS[row["Simulation"]],
            label=r"$\texttt{" + f"{row['Simulation']}_{row['Galaxy']}" + "}$",
        )
    axs[0, 0].legend(frameon=False, fontsize=6, loc="lower center", ncol=2)

    axs[1, 0].set_xlim(10.4, 11.2)
    axs[1, 0].set_ylim(-0.3, 1.3)
    axs[1, 0].set_xticks([10.6, 10.8, 11])
    axs[1, 0].set_yticks([-0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1])
    axs[1, 0].set_ylabel(
        r"$\log_{10} \mathrm{SFR} \, [\mathrm{M}_\odot \, \mathrm{yr}^{-1}]$")
    axs[1, 0].set_xlabel(r"$\log_{10} M_\mathrm{star} \, [\mathrm{M}_\odot]$")
    axs[1, 0].scatter(
        np.log10(auriga["SubhaloStellarMass_10^10Msun"].to_numpy() * 1e10),
        np.log10(auriga["SFR_Msun/yr"].to_numpy()),
        s=12, c="darkgray", edgecolor="none",
    )
    for _, row in hestia.iterrows():
        axs[1, 0].scatter(
            np.log10(row["Mstar_10^10Msun"] * 1e10),
            np.log10(row["SFR_Msun/yr"]),
            s=14, facecolors="none",
            marker=Settings.GALAXY_SYMBOLS[row["Galaxy"]],
            edgecolor=Settings.SIMULATION_COLORS[row["Simulation"]],
            label=r"$\texttt{" + f"{row['Simulation']}_{row['Galaxy']}" + "}$",
        )

    axs[0, 1].set_xlim(0.7, 1.4)
    axs[0, 1].set_ylim(-0.3, 1.3)
    axs[0, 1].set_xticklabels([])
    axs[0, 1].set_yticks([-0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1])
    axs[0, 1].set_ylabel(
        r"$\log_{10} \mathrm{SFR} \, [\mathrm{M}_\odot \, \mathrm{yr}^{-1}]$")
    axs[0, 1].scatter(
        np.log10(auriga["Delta1200"].to_numpy()),
        np.log10(auriga["SFR_Msun/yr"].to_numpy()),
        s=12, c="darkgray", edgecolor="none",
    )
    for _, row in hestia.iterrows():
        axs[0, 1].scatter(
            np.log10(row["Delta1200"]),
            np.log10(row["SFR_Msun/yr"]),
            s=14, facecolors="none",
            marker=Settings.GALAXY_SYMBOLS[row["Galaxy"]],
            edgecolor=Settings.SIMULATION_COLORS[row["Simulation"]],
            label=r"$\texttt{" + f"{row['Simulation']}_{row['Galaxy']}" + "}$",
        )

    axs[1, 1].set_xlim(0.7, 1.4)
    axs[1, 1].set_ylim(-1.9, -0.7)
    axs[1, 1].set_xticks([0.8, 1.0, 1.2, 1.4])
    axs[1, 1].set_yticks([-1.8, -1.6, -1.4, -1.2, -1, -0.8])
    axs[1, 1].set_ylabel(r"$\log_{10} \mathrm{sSFR} \, [\mathrm{Gyr}^{-1}]$")
    axs[1, 1].set_xlabel(r"$\log_{10} \delta_{1200}$")
    ssfr = auriga["SFR_Msun/yr"].to_numpy() \
        / auriga["SubhaloStellarMass_10^10Msun"].to_numpy() / 10
    axs[1, 1].scatter(
        np.log10(auriga["Delta1200"].to_numpy()), np.log10(ssfr),
        s=12, c="darkgray", edgecolor="none",
    )
    for _, row in hestia.iterrows():
        axs[1, 1].scatter(
            np.log10(row["Delta1200"]),
            np.log10(row["SFR_Msun/yr"] / row["Mstar_10^10Msun"] / 10),
            s=14, facecolors="none",
            marker=Settings.GALAXY_SYMBOLS[row["Galaxy"]],
            edgecolor=Settings.SIMULATION_COLORS[row["Simulation"]],
            label=r"$\texttt{" + f"{row['Simulation']}_{row['Galaxy']}" + "}$",
        )

    for ax in axs.flatten():
        ax.set_axisbelow(True)

    plt.savefig("images/prop_comparison.png")
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
