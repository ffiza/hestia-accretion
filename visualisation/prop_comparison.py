import json
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

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


def _get_data_for_corr_analysis(snapnum: int, config: dict) -> pd.DataFrame:
    galaxies = []

    sfr = []
    stellar_mass = []
    delta = []

    for galaxy in range(1, 31):
        data = pd.read_csv("data/iza_et_al_2022/sfr.csv")
        sfr.append(data[f"SFR_Au{galaxy}_Msun/yr"].to_numpy()[snapnum - 1])
        data = pd.read_csv(f"data/auriga/au{galaxy}/baryon_mass.csv", sep=" ")
        stellar_mass.append(data["TotalStar"].to_numpy()[snapnum])
        data = pd.read_csv("data/iza_et_al_2022/environment_delta_1200.csv")
        delta.append(data[f"Delta1200_Au{galaxy}"].to_numpy()[snapnum])
        galaxies.append(f"Au{galaxy}")
    for simulation in Settings.SIMULATIONS:
        for galaxy in Settings.GALAXIES:
            data = pd.read_csv(
                f"data/hestia/{galaxy}_M_SFR_t_Hestia{simulation}.csv")
            sfr.append(data["SFR"][data["SnapNo"] == snapnum].values[0])
            stellar_mass.append(
                data["Mstar"][data["SnapNo"] == snapnum].values[0] / 1e10)
            environment = pd.read_csv(
                f"results/{simulation}_{galaxy}/delta_1200.csv")
            delta.append(
                environment["Delta"].to_numpy()[snapnum])
            galaxies.append(f"{simulation}_{galaxy}")

    colors = ["darkgray"] * 30
    for s in Settings.SIMULATIONS:
        for _ in Settings.GALAXIES:
            colors.append(Settings.SIMULATION_COLORS[s])

    symbols = ["o"] * 30
    for _ in Settings.SIMULATIONS:
        for g in Settings.GALAXIES:
            symbols.append(Settings.GALAXY_SYMBOLS[g])

    df = pd.DataFrame({
        "Galaxy": galaxies,
        "SFR_Msun/yr": sfr,
        "Mstar_10^10Msun": stellar_mass,
        "Delta1200": delta,
        "Colors": colors,
        "Symbols": symbols,
    })

    with open('data/auriga/simulation_data.json', 'r') as file:
        data = json.load(file)
    df.time = data["Original"]["Time_Gyr"][snapnum]
    df.redshift = data["Original"]["Redshift"][snapnum]
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


def plot_time_correlation(config: dict) -> None:
    fig, ax = plt.subplots(figsize=(2.5, 2.5))

    ax.set_xlabel("Time [Gyr]")
    ax.set_ylabel("Regression Slope")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.set_xticks([2, 4, 6, 8, 10, 12])
    ax.set_yticks([0, 1, 2, 3, 4])

    ax1 = ax.inset_axes([1, 0, 1, 1/3])
    ax1.tick_params(axis="y", labelleft=False, labelright=True)
    ax1.set_xlabel(r"$\log_{10} \delta_{1200}$")
    ax1.set_xlim(0, 1.5)
    ax1.set_ylim(-0.4, 1.6)
    ax1.set_xticks([0.25, 0.5, 0.75, 1, 1.25])
    ax1.set_yticks([0, 0.5, 1])
    ax2 = ax.inset_axes([1, 1/3, 1, 1/3], sharex=ax1, sharey=ax1)
    ax2.tick_params(axis="x", labelbottom=False)
    ax2.tick_params(axis="y", labelleft=False, labelright=True)
    ax2.set_ylabel(
        r"$\log_{10} \mathrm{SFR} \, [\mathrm{M}_\odot \, \mathrm{yr}^{-1}]$")
    ax2.yaxis.set_label_position("right")
    ax3 = ax.inset_axes([1, 2/3, 1, 1/3], sharex=ax1, sharey=ax1)
    ax3.tick_params(axis="x", labelbottom=False)
    ax3.tick_params(axis="y", labelleft=False, labelright=True)

    axs = [ax1, ax2, ax3]
    snapnums = [77, 95, 127]

    for i, snapnum in enumerate(snapnums):
        df = _get_data_for_corr_analysis(snapnum, config)
        for _, row in df.iterrows():
            axs[i].scatter(
                np.log10(row["Delta1200"]), np.log10(row["SFR_Msun/yr"]), s=14,
                color=row["Colors"], facecolors="none", marker=row["Symbols"])
        r = linregress(np.log10(df["Delta1200"]), np.log10(df["SFR_Msun/yr"]))
        axs[i].text(0.025, 0.95, f"$z =$ {round(df.redshift, 1)}",
                    transform=axs[i].transAxes,
                    ha="left", va='top', fontsize=6)
        axs[i].plot(axs[i].get_xlim(),
                    [r.slope * axs[i].get_xlim()[0] + r.intercept,
                     r.slope * axs[i].get_xlim()[1] + r.intercept],
                    c="black", lw=0.75)

    with open('data/auriga/simulation_data.json', 'r') as file:
        data = json.load(file)
    time = data["Original"]["Time_Gyr"]

    slopes = [np.nan] * 128
    pvalues = [np.nan] * 128
    for snapnum in range(50, 128):
        df = _get_data_for_corr_analysis(snapnum, config)
        r = linregress(np.log10(df["Delta1200"]), np.log10(df["SFR_Msun/yr"]))
        slopes[snapnum] = r.slope
        pvalues[snapnum] = r.pvalue
        if snapnum in snapnums:
            ax.scatter(time[snapnum], slopes[snapnum], color="k",
                       s=10, facecolor="none", zorder=11, lw=0.75)
            ax.annotate(
                f"$z =$ {round(df.redshift, 1)}",
                xy=(time[snapnum], slopes[snapnum]), xycoords='data',
                xytext=(-40, -20), textcoords='offset points',
                arrowprops=dict(arrowstyle="->", linewidth=0.75),
                fontsize=6, zorder=11)
    s = ax.scatter(time, slopes, c=pvalues, s=10, zorder=10, vmin=0, vmax=0.1,
                   cmap="RdYlGn_r")

    cbax = ax.inset_axes([0.35, 0.89, 0.6, 0.025],
                         transform=ax.transAxes)
    cb = plt.colorbar(s, cax=cbax, orientation="horizontal")
    cbax.set_xlim(0, 0.1)
    cb.set_ticks([0, 0.02, 0.04, 0.06, 0.08, 0.1])
    cb.set_ticklabels(['0', '0.02', '0.04', '0.06', '0.08', '0.1'],
                      fontsize=5.0)
    cbax.set_xlabel(r"$p$-value", fontsize=6)
    cbax.xaxis.set_label_position('top')

    plt.savefig("images/prop_correlation_sfr_vs_delta.pdf")
    plt.close(fig)


if __name__ == "__main__":
    figure_setup()

    # Get arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = yaml.safe_load(open(f"configs/{args.config}.yml"))

    # plot_prop_comparison(config)
    plot_time_correlation(config)
