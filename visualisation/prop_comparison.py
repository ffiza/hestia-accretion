import json
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr

from hestia.images import figure_setup
from hestia.settings import Settings


def _get_data(snapnum: int, config: dict) -> pd.DataFrame:
    if snapnum != 127:
        raise NotImplementedError("Only snapnum 127 is implemented.")

    galaxies = []

    sfr = []
    m_star = []
    m_gas = []
    delta = []
    r200 = []

    for galaxy in range(1, 31):
        data = pd.read_csv("data/iza_et_al_2022/sfr.csv")
        sfr.append(data[f"SFR_Au{galaxy}_Msun/yr"].to_numpy()[snapnum - 1])
        data = np.loadtxt(f"data/auriga/au{galaxy}/baryon_mass.csv")
        m_star.append(data[snapnum, 1])
        m_gas.append(data[snapnum, 0])
        data = pd.read_csv(f"data/auriga/au{galaxy}/environment_evolution.csv")
        delta.append(data["Delta1200"].to_numpy()[snapnum])
        galaxies.append(f"Au{galaxy}")
        data = pd.read_csv("data/auriga/virial_radius.csv")
        r200.append(data[f"VirialRadius_Au{galaxy}_ckpc"].to_numpy()[snapnum])

    for simulation in Settings.SIMULATIONS:
        for galaxy in Settings.GALAXIES:
            data = pd.read_csv(
                f"data/hestia/{galaxy}_M_SFR_t_Hestia{simulation}.csv")
            sfr.append(data["SFR"][data["SnapNo"] == snapnum].values[0])
            m_star.append(
                data["Mstar"][data["SnapNo"] == snapnum].values[0] / 1e10)
            m_gas.append(
                data["Mgas"][data["SnapNo"] == snapnum].values[0] / 1e10)
            environment = pd.read_csv(
                f"results/{simulation}_{galaxy}/"
                f"delta_1200_{config['RUN_CODE']}.csv")
            delta.append(environment["Delta"].to_numpy()[snapnum])
            galaxies.append(f"{simulation}_{galaxy}")
            data = pd.read_csv(
                f"results/{simulation}_{galaxy}/virial_radius.csv")
            r200.append(data["VirialRadius_ckpc"].to_numpy()[snapnum])

    colors = ["tab:gray"] * 30
    for s in Settings.SIMULATIONS:
        for _ in Settings.GALAXIES:
            colors.append(Settings.SIMULATION_COLORS[s])

    symbols = ["X"] * 30
    for _ in Settings.SIMULATIONS:
        for g in Settings.GALAXIES:
            symbols.append(Settings.GALAXY_SYMBOLS[g])

    df = pd.DataFrame({
        "Galaxy": galaxies,
        "SFR_Msun/yr": sfr,
        "Mstar_10^10Msun": np.array(m_star, np.float64),
        "Mgas_10^10Msun": np.array(m_gas, np.float64),
        "Delta1200": delta,
        "VirialRadius_ckpc": r200,
        "Colors": colors,
        "Symbols": symbols,
        "sSFR_Gyr^-1": sfr / np.array(m_star, np.float64) / 10,
    })

    df["logSFR_Msun/yr"] = np.log10(df["SFR_Msun/yr"])
    df["logMstar_Msun"] = np.log10(df["Mstar_10^10Msun"] * 1e10)
    df["logMgas_Msun"] = np.log10(df["Mgas_10^10Msun"] * 1e10)
    df["logDelta1200"] = np.log10(df["Delta1200"])
    df["logsSFR_Gyr^-1"] = np.log10(df["sSFR_Gyr^-1"])

    with open('data/auriga/simulation_data.json', 'r') as file:
        data = json.load(file)
    df.time = data["Original"]["Time_Gyr"][snapnum]
    df.redshift = data["Original"]["Redshift"][snapnum]
    return df


def plot_prop_comparison(config: dict) -> None:
    df = _get_data(127, config)
    df_au = df[df["Galaxy"].str.contains("Au")]
    df_he = df[~df["Galaxy"].str.contains("Au")]

    FEATS = [
        "logSFR_Msun/yr",
        "logMstar_Msun",
        "logMgas_Msun",
        "logDelta1200",
        "VirialRadius_ckpc",
        "logsSFR_Gyr^-1"]
    AX_LIMIT = [
        (-0.3, 1.6),
        (10.4, 11.4),
        (10.4, 11.5),
        (0.7, 1.6),
        (200, 320),
        (-1.9, -0.4)]
    AX_LABEL = [
        r"$\log_{10} \mathrm{SFR}$" + "\n" + r"$[\mathrm{M}_\odot \, \mathrm{yr}^{-1}]$",
        r"$\log_{10} M_\mathrm{star}$" + "\n" + r"$[\mathrm{M}_\odot]$",
        r"$\log_{10} M_\mathrm{gas}$" + "\n" + r"$[\mathrm{M}_\odot]$",
        r"$\log_{10} \delta_{1200}$",
        r"$R_{200}$" + "\n" + r"$[\mathrm{ckpc}]$",
        r"$\log_{10} \mathrm{sSFR}$" + "\n" + r"$[\mathrm{Gyr}^{-1}]$"]
    AX_TICKS = [
        [-0.1, 0.3, 0.7, 1.1],
        [10.6, 10.8, 11.0],
        [10.6, 10.8, 11.0, 11.2],
        [0.8, 1.0, 1.2, 1.4],
        [220, 240, 260, 280],
        [-1.8, -1.6, -1.4, -1.2, -1.0, -0.8],
    ]
    AX_TICK_LABELS = [
        ["$-0.1$", "0.3", "0.7", "1.1"],
        ["10.6", "10.8", "11.0"],
        ["10.6", "10.8", "11.0", "11.2"],
        ["0.8", "1.0", "1.2", "2.4"],
        ["220", "240", "260", "280"],
        ["$-1.8$", "$-1.6$", "$-1.4$", "$-1.2$", "$-1.0$", "$-0.8$"],
    ]

    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(nrows=6, ncols=6, hspace=0, wspace=0)
    axs = np.array(gs.subplots(sharex=False, sharey=False))

    for i, f1 in enumerate(FEATS):
        for j, f2 in enumerate(FEATS):
            ax = np.array(axs)[i, j]
            ax.set_xlim(AX_LIMIT[j])
            ax.set_ylim(AX_LIMIT[i])
            ax.set_xticks(
                ticks=AX_TICKS[j], labels=AX_TICK_LABELS[j], fontsize=5,
                rotation=45)
            ax.set_yticks(
                ticks=AX_TICKS[i], labels=AX_TICK_LABELS[i], fontsize=5)
            ax.set_xlabel(AX_LABEL[j], fontsize=8)
            ax.set_ylabel(AX_LABEL[i], fontsize=8)
            ax.yaxis.set_label_coords(-0.4, 0.5)
            ax.xaxis.set_label_coords(0.5, -0.4)
            if i != j:
                ax.scatter(
                    df_au[f2].to_numpy(), df_au[f1].to_numpy(),
                    s=12, edgecolor="none",
                    facecolor=df_au["Colors"].values[0],
                    marker="X", label="Auriga", zorder=10,
                )
                for _, row in df_he.iterrows():
                    ax.scatter(
                        row[f2], row[f1],
                        s=12, facecolors="none", marker=row["Symbols"],
                        edgecolor=row["Colors"], zorder=11,
                        label=r"$\texttt{" + f"{row['Galaxy']}" + "}$",
                    )
                correlation = pearsonr(df[f2], df[f1])
                rho = correlation.__getattribute__("statistic")
                pvalue = correlation.__getattribute__("pvalue")
                color = "tab:green" if pvalue < 0.05 else "tab:red"
                stat_text = r"$\rho = $ " + f"{np.round(rho, 2)}" \
                    if rho > 0 else r"$\rho = -$" + f"{np.abs(rho):.2f}"
                ax.text(0.03, 0.97,
                        stat_text,
                        transform=ax.transAxes, color=color,
                        ha="left", va='top', fontsize=4, zorder=12)
                pvalue_text = r"$p$-value $ =$" + f" {np.round(pvalue, 2)}" \
                    if pvalue > 0.01 else r"$p$-value $ <0.01$"
                ax.text(0.03, 0.90,
                        pvalue_text,
                        transform=ax.transAxes, color=color,
                        ha="left", va='top', fontsize=4, zorder=12)
            ax.label_outer()

    plt.savefig("images/prop_comparison.pdf")
    plt.close(fig)


def plot_time_correlation_sfr_vs_delta(config: dict) -> None:
    fig, ax = plt.subplots(figsize=(2.5, 2.5))

    ax.set_xlabel("Time [Gyr]", fontsize=8)
    ax.set_ylabel("Regression Slope", fontsize=8)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.set_xticks(ticks=[2, 4, 6, 8, 10, 12],
                  labels=["2", "4", "6", "8", "10", "12"],
                  fontsize=6)
    ax.set_yticks(ticks=[0, 1, 2, 3, 4, 5],
                  labels=["0", "1", "2", "3", "4", "5"],
                  fontsize=6)

    ax1 = ax.inset_axes([1, 0, 1, 1/3])
    ax1.tick_params(axis="y", labelleft=False, labelright=True)
    ax1.set_xlabel(r"$\log_{10} \delta_{1200}$", fontsize=8)
    ax1.set_xlim(0.4, 1.2)
    ax1.set_ylim(-0.4, 1.6)
    ax1.set_xticks(ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1],
                   labels=["0.5", "0.6", "0.7", "0.8", "0.9", "1", "1.1"],
                   fontsize=6)
    ax1.set_yticks(ticks=[0, 0.5, 1],
                   labels=["0.0", "0.5", "1.0"],
                   fontsize=6)
    ax2 = ax.inset_axes([1, 1/3, 1, 1/3], sharex=ax1, sharey=ax1)
    ax2.tick_params(axis="x", labelbottom=False)
    ax2.tick_params(axis="y", labelleft=False, labelright=True, labelsize=6)
    ax2.set_ylabel(
        r"$\log_{10} \mathrm{SFR} \, [\mathrm{M}_\odot \, \mathrm{yr}^{-1}]$",
        fontsize=8)
    ax2.yaxis.set_label_position("right")
    ax3 = ax.inset_axes([1, 2/3, 1, 1/3], sharex=ax1, sharey=ax1)
    ax3.tick_params(axis="x", labelbottom=False)
    ax3.tick_params(axis="y", labelleft=False, labelright=True, labelsize=6)

    axs = [ax1, ax2, ax3]
    snapnums = [61, 77, 95]

    for j, snapnum in enumerate(snapnums):
        i = len(snapnums) - j - 1
        df = _get_data(snapnum, config)
        for _, row in df.iterrows():
            if row["Galaxy"].startswith("Au"):
                axs[i].scatter(
                    np.log10(row["Delta1200"]), np.log10(row["SFR_Msun/yr"]),
                    s=12, facecolor=row["Colors"], edgecolors="none",
                    marker=row["Symbols"])
            else:
                axs[i].scatter(
                    np.log10(row["Delta1200"]), np.log10(row["SFR_Msun/yr"]),
                    s=12, color=row["Colors"], facecolors="none",
                    marker=row["Symbols"])
        r = linregress(np.log10(df["Delta1200"]), np.log10(df["SFR_Msun/yr"]))
        axs[i].text(0.025, 0.95, f"$z =$ {round(df.redshift, 1)}",
                    transform=axs[i].transAxes,
                    ha="left", va='top', fontsize=6)
        axs[i].plot(axs[i].get_xlim(),
                    [r.slope * axs[i].get_xlim()[0] + r.intercept,
                     r.slope * axs[i].get_xlim()[1] + r.intercept],
                    c="black", lw=0.75, ls="--")

    with open('data/auriga/simulation_data.json', 'r') as file:
        data = json.load(file)
    time = data["Original"]["Time_Gyr"]

    slopes = [np.nan] * 128
    pvalues = [np.nan] * 128
    for snapnum in range(50, 128):
        df = _get_data(snapnum, config)
        r = linregress(np.log10(df["Delta1200"]), np.log10(df["SFR_Msun/yr"]))
        slopes[snapnum] = r.slope
        pvalues[snapnum] = r.pvalue
        if snapnum in snapnums:
            ax.scatter(time[snapnum], slopes[snapnum], color="k",
                       s=10, facecolor="none", zorder=11, lw=0.75)
            ax.annotate(
                f"$z =$ {round(df.redshift, 1)}",
                xy=(time[snapnum], slopes[snapnum]), xycoords='data',
                xytext=(-20, 30), textcoords='offset points',
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


def plot_time_correlation_ssfr_vs_delta(config: dict) -> None:
    fig, ax = plt.subplots(figsize=(2.5, 2.5))

    ax.set_xlabel("Time [Gyr]", fontsize=8)
    ax.set_ylabel("Regression Slope", fontsize=8)
    ax.set_xlim(0, 14)
    ax.set_ylim(-1, 1.5)
    ax.set_xticks(ticks=[2, 4, 6, 8, 10, 12],
                  labels=["2", "4", "6", "8", "10", "12"],
                  fontsize=6)
    ax.set_yticks(ticks=[-1, -0.5, 0, 0.5, 1, 1.5],
                  labels=["$-1$", "$-0.5$", "0.0", "0.5", "1.0", "1.5"],
                  fontsize=6)

    ax1 = ax.inset_axes([1, 0, 1, 1/3])
    ax1.tick_params(axis="y", labelleft=False, labelright=True)
    ax1.set_xlabel(r"$\log_{10} \delta_{1200}$", fontsize=8)
    ax1.set_xlim(0.4, 1.2)
    ax1.set_ylim(-1.5, 0.5)
    ax1.set_xticks(ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1],
                   labels=["0.5", "0.6", "0.7", "0.8", "0.9", "1", "1.1"],
                   fontsize=6)
    ax1.set_yticks(ticks=[-1, 0],
                   labels=["$-1.0$", "0.0"],
                   fontsize=6)
    ax2 = ax.inset_axes([1, 1/3, 1, 1/3], sharex=ax1, sharey=ax1)
    ax2.tick_params(axis="x", labelbottom=False)
    ax2.tick_params(axis="y", labelleft=False, labelright=True, labelsize=6)
    ax2.set_ylabel(
        r"$\log_{10} \mathrm{sSFR} \, [\mathrm{Gyr}^{-1}]$",
        fontsize=8)
    ax2.yaxis.set_label_position("right")
    ax3 = ax.inset_axes([1, 2/3, 1, 1/3], sharex=ax1, sharey=ax1)
    ax3.tick_params(axis="x", labelbottom=False)
    ax3.tick_params(axis="y", labelleft=False, labelright=True, labelsize=6)

    axs = [ax1, ax2, ax3]
    snapnums = [61, 77, 95]

    for j, snapnum in enumerate(snapnums):
        i = len(snapnums) - j - 1
        df = _get_data(snapnum, config)
        for _, row in df.iterrows():
            ssfr = row["SFR_Msun/yr"] / row["Mstar_10^10Msun"] / 10
            if row["Galaxy"].startswith("Au"):
                axs[i].scatter(
                    np.log10(row["Delta1200"]), np.log10(ssfr),
                    s=12, facecolor=row["Colors"], edgecolors="none",
                    marker=row["Symbols"])
            else:
                axs[i].scatter(
                    np.log10(row["Delta1200"]), np.log10(ssfr),
                    s=12, color=row["Colors"], facecolors="none",
                    marker=row["Symbols"])
        ssfr = df["SFR_Msun/yr"] / df["Mstar_10^10Msun"] / 10
        r = linregress(np.log10(df["Delta1200"]), np.log10(ssfr))
        axs[i].text(0.025, 0.95, f"$z =$ {round(df.redshift, 1)}",
                    transform=axs[i].transAxes,
                    ha="left", va='top', fontsize=6)
        axs[i].plot(axs[i].get_xlim(),
                    [r.slope * axs[i].get_xlim()[0] + r.intercept,
                     r.slope * axs[i].get_xlim()[1] + r.intercept],
                    c="black", lw=0.75, ls="--")

    with open('data/auriga/simulation_data.json', 'r') as file:
        data = json.load(file)
    time = data["Original"]["Time_Gyr"]

    slopes = [np.nan] * 128
    pvalues = [np.nan] * 128
    for snapnum in range(50, 128):
        df = _get_data(snapnum, config)
        ssfr = df["SFR_Msun/yr"] / df["Mstar_10^10Msun"] / 10
        r = linregress(np.log10(df["Delta1200"]), np.log10(ssfr))
        slopes[snapnum] = r.slope
        pvalues[snapnum] = r.pvalue
        if snapnum in snapnums:
            ax.scatter(time[snapnum], slopes[snapnum], color="k",
                       s=10, facecolor="none", zorder=11, lw=0.75)
            ax.annotate(
                f"$z =$ {round(df.redshift, 1)}",
                xy=(time[snapnum], slopes[snapnum]), xycoords='data',
                xytext=(-15, -40), textcoords='offset points',
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

    plt.savefig("images/prop_correlation_ssfr_vs_delta.pdf")
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
    plot_time_correlation_sfr_vs_delta(config)
    plot_time_correlation_ssfr_vs_delta(config)
