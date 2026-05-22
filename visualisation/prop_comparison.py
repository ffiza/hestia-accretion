import json
import yaml
import argparse
import numpy as np
import pandas as pd
from typing import List, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr

from hestia.images import figure_setup
from hestia.settings import Settings


def _get_data(snapnum: int, config: dict) -> pd.DataFrame:
    galaxies = []

    sfr: List[float] = []
    m_star: List[float] = []
    m_gas: List[float] = []
    m_cold_gas: List[float] = []
    delta: List[float] = []
    r200: List[float] = []
    m200: List[float] = []

    for galaxy in range(1, 31):
        data = pd.read_csv("data/iza_et_al_2022/sfr.csv")
        sfr.append(data[f"SFR_Au{galaxy}_Msun/yr"].to_numpy()[snapnum - 1])
        data = np.loadtxt(f"data/auriga/au{galaxy}/baryon_mass.csv")
        m_star.append(data[snapnum, 1])
        m_gas.append(data[snapnum, 0])
        data = pd.read_csv(f"data/auriga/au{galaxy}/environment_evolution.csv")
        delta.append(data["Delta1200"].to_numpy()[snapnum])
        galaxies.append(f"Au{galaxy}")
        data = pd.read_csv("data/auriga/virial_properties.csv")
        r200.append(data[f"R200_Au{galaxy}_ckpc"].to_numpy()[snapnum])
        m200.append(
            data[f"M200_Au{galaxy}_1E10Msun"].to_numpy()[snapnum] * 1E10)

        data = pd.read_csv(
            "data/auriga/cold_gas_mass_evolution.csv",
            usecols=["Snapshot", "SubhaloGasMassUnder20000K_Msun",
                     "Simulation"])
        m_cold_gas.append(
            data.loc[
                (data["Snapshot"] == snapnum)
                & (data["Simulation"] == galaxy),
                "SubhaloGasMassUnder20000K_Msun"
            ].values[0]
        )

    for simulation in Settings.HIGH_RES_SIMULATIONS:
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
            data = pd.read_csv(
                f"data/hestia/r200_t/r200_t_{galaxy}_{simulation}.csv")
            m200.append(data["Mvir"].to_numpy()[snapnum])

            data = pd.read_csv(
                f"data/hestia/{galaxy}_M_SFR_t_Hestia{simulation}.csv",
                usecols=["SnapNo", "Mcold"])
            m_cold_gas.append(data.loc[
                (data["SnapNo"] == snapnum), "Mcold"].values[0])

    colors = ["#4d4d4d"] * 30
    for s in Settings.HIGH_RES_SIMULATIONS:
        for _ in Settings.GALAXIES:
            colors.append(Settings.SIMULATION_COLORS[s])

    symbols = ["o"] * 30
    for _ in Settings.HIGH_RES_SIMULATIONS:
        for g in Settings.GALAXIES:
            symbols.append(Settings.GALAXY_SYMBOLS[g])

    df = pd.DataFrame({
        "Galaxy": galaxies,
        "SFR_Msun/yr": sfr,
        "Mstar_10^10Msun": np.array(m_star, np.float64),
        "Mgas_10^10Msun": np.array(m_gas, np.float64),
        "Mcoldgas_10^10Msun": np.array(m_cold_gas, np.float64) / 1E10,
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
    df["VirialMass_Msun"] = m200
    df["logVirialMass_Msun"] = np.log10(df["VirialMass_Msun"])
    df["StellarMassFraction"] = df["Mstar_10^10Msun"] \
        / (df["Mstar_10^10Msun"] + df["Mcoldgas_10^10Msun"])
    df["Mstar/M200"] = df["Mstar_10^10Msun"] * 1e10 / df["VirialMass_Msun"]
    df["Mgas/M200"] = df["Mgas_10^10Msun"] * 1e10 / df["VirialMass_Msun"]
    df["ColdGasMassFraction"] = df["Mcoldgas_10^10Msun"] \
        / (df["Mstar_10^10Msun"] + df["Mcoldgas_10^10Msun"])

    with open('data/auriga/simulation_data.json', 'r') as file:
        data = json.load(file)
    df.time = data["Original"]["Time_Gyr"][snapnum]
    df.redshift = data["Original"]["Redshift"][snapnum]
    df.expansion_factor = data["Original"]["ExpansionFactor"][snapnum]
    return df


@dataclass
class FeatureData:
    name: str
    axis_limits: Tuple[float, float]
    axis_label: str
    axis_ticks: List[float]
    axis_tick_labels: List[str]


def plot_prop_comparison(
        config: dict,
        snapnum: int,
        features: List[FeatureData]) -> None:

    if len(features) != 6:
        raise ValueError(
            "This plot is optimized for 6 elements in `features`.")

    df = _get_data(snapnum, config)
    df_au = df[df["Galaxy"].str.contains("Au")]
    df_he = df[~df["Galaxy"].str.contains("Au")]

    feature_names = [feature.name for feature in features]
    axis_limits = [feature.axis_limits for feature in features]
    axis_labels = [feature.axis_label for feature in features]
    axis_ticks = [feature.axis_ticks for feature in features]
    axis_tick_labels = [feature.axis_tick_labels for feature in features]

    fig = plt.figure(
        figsize=(6, 6))
    gs = fig.add_gridspec(
        nrows=len(features) - 1,
        ncols=len(axis_limits) - 1,
        hspace=0,
        wspace=0)
    axs = np.array(
        gs.subplots(
            sharex=False,
            sharey=False))

    for i in range(len(feature_names)):
        f1 = feature_names[i]
        for j in range(i + 1, len(feature_names)):
            f2 = feature_names[j]
            ax = np.array(axs)[j - 1, i]
            ax.set_xlim(axis_limits[i])
            ax.set_ylim(axis_limits[j])
            ax.set_xticks(
                ticks=axis_ticks[i],
                labels=axis_tick_labels[i],
                fontsize=5,
                rotation=45)
            ax.set_yticks(
                ticks=axis_ticks[j],
                labels=axis_tick_labels[j],
                fontsize=5)
            ax.set_xlabel(axis_labels[i], fontsize=8)
            ax.set_ylabel(axis_labels[j], fontsize=8)
            ax.yaxis.set_label_coords(-0.4, 0.5)
            ax.xaxis.set_label_coords(0.5, -0.4)
            ax.scatter(
                df_au[f1].to_numpy(), df_au[f2].to_numpy(),
                s=5, edgecolor="none",
                facecolor=df_au["Colors"].values[0],
                marker=df_au["Symbols"].values[0], label="Auriga", zorder=10,
            )
            for _, row in df_he.iterrows():
                # This prefix prevents the name from appearing in the legend
                prefix = "_" if row["Galaxy"].startswith("i_") else ""
                ax.scatter(
                    row[f1], row[f2],
                    s=5, facecolors="none", marker=row["Symbols"],
                    edgecolor=row["Colors"], zorder=11,
                    label=prefix + r"$\texttt{" + f"{row['Galaxy']}" + "}$",
                )
            correlation = pearsonr(df[f1], df[f2])
            rho = correlation.__getattribute__("statistic")
            pvalue = correlation.__getattribute__("pvalue")
            color = "tab:green" if pvalue < 0.05 else "tab:red"
            stat_text = r"$r = $ " + f"{np.round(rho, 2)}" \
                if rho > 0 else r"$r = -$" + f"{np.abs(rho):.2f}"
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

    for ax in axs[np.triu_indices_from(axs, k=1)]:
        ax.axis('off')

    handles, labels = axs[1, 0].get_legend_handles_labels()
    axs[0, 1].legend(handles, labels, frameon=False, fontsize=5,
                     bbox_to_anchor=(0.5, 0.5), loc='center')

    plt.savefig(
        f"images/prop_comparison_snap{snapnum}_{config['RUN_CODE']}.pdf")
    plt.close(fig)


def plot_correlations_with_feature(
        y_features: List[FeatureData],
        x_feature: FeatureData,
        snapnum: int,
        config: dict,
        fig_name: str,
        ) -> None:
    df = _get_data(snapnum, config)
    df_au = df[df["Galaxy"].str.contains("Au")]
    df_he = df[~df["Galaxy"].str.contains("Au")]

    feature_names = [feature.name for feature in features]
    axis_limits = [feature.axis_limits for feature in features]
    axis_labels = [feature.axis_label for feature in features]
    axis_ticks = [feature.axis_ticks for feature in features]
    axis_tick_labels = [feature.axis_tick_labels for feature in features]

    fig = plt.figure(
        figsize=(6, 3))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        hspace=0,
        wspace=0.4)
    axs = np.array(gs.subplots(
        sharex=False,
        sharey=False))

    for i in range(len(feature_names)):
        feat = feature_names[i]
        ax = np.array(axs).flatten()[i]

        ax.set_xlim(x_feature.axis_limits)
        ax.set_ylim(axis_limits[i])
        ax.set_xticks(
            ticks=(x_feature.axis_ticks),
            labels=x_feature.axis_tick_labels,
            fontsize=5)
        ax.set_yticks(
            ticks=axis_ticks[i],
            labels=axis_tick_labels[i],
            fontsize=5)
        ax.yaxis.set_label_coords(-0.25, 0.5)
        ax.set_xlabel(
            x_feature.axis_label,
            fontsize=8)
        ax.set_ylabel(
            axis_labels[i].replace("\n", " "),
            fontsize=8)
        ax.scatter(
            df_au[x_feature.name].to_numpy(),
            df_au[feat].to_numpy(),
            s=8,
            edgecolor="none",
            facecolor=df_au["Colors"].values[0],
            marker=df_au["Symbols"].values[0],
            label="Auriga",
            zorder=10,
        )
        for _, row in df_he.iterrows():
            # This prefix prevents the name from appearing in the legend
            prefix = "_" if row["Galaxy"].startswith("i_") else ""
            ax.scatter(
                row[x_feature.name],
                row[feat],
                s=8,
                facecolors="none",
                marker=row["Symbols"],
                edgecolor=row["Colors"],
                zorder=11,
                label=prefix + r"$\texttt{" + f"{row['Galaxy']}" + "}$",
            )
        correlation = pearsonr(df[x_feature.name], df[feat])
        rho = correlation.__getattribute__("statistic")
        pvalue = correlation.__getattribute__("pvalue")
        color = "tab:green" if pvalue < 0.05 else "tab:red"
        stat_text = r"$r = $ " + f"{np.round(rho, 2)}" \
            if rho > 0 else r"$r = -$" + f"{np.abs(rho):.2f}"
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

    handles, labels = axs[1, 0].get_legend_handles_labels()
    axs[1, 0].legend(
        handles,
        labels,
        frameon=False,
        fontsize=3.5,
        loc='lower center',
        ncols=2)

    plt.savefig(f"images/{fig_name}.pdf")
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

    ax1 = ax.inset_axes([1, 0, 1, 1/4])
    ax1.tick_params(axis="y", labelleft=False, labelright=True)
    ax1.set_xlabel(r"$\log_{10} \delta_{1200}$", fontsize=8)
    ax1.set_xlim(0.4, 1.4)
    ax1.set_ylim(-0.4, 1.6)
    ax1.set_xticks(
        ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3],
        labels=["0.5", "0.6", "0.7", "0.8", "0.9", "1.0", "1.1", "1.2", "1.3"],
        fontsize=6)
    ax1.set_yticks(
        ticks=[0, 0.5, 1],
        labels=["0.0", "0.5", "1.0"],
        fontsize=6)
    ax2 = ax.inset_axes([1, 1/4, 1, 1/4], sharex=ax1, sharey=ax1)
    ax2.tick_params(axis="x", labelbottom=False)
    ax2.tick_params(axis="y", labelleft=False, labelright=True, labelsize=6)
    ax2.set_ylabel(
        r"$\log_{10} \mathrm{SFR} \, [\mathrm{M}_\odot \, \mathrm{yr}^{-1}]$",
        fontsize=8)
    ax2.yaxis.set_label_coords(1.2, 1)
    ax3 = ax.inset_axes([1, 2/4, 1, 1/4], sharex=ax1, sharey=ax1)
    ax3.tick_params(axis="x", labelbottom=False)
    ax3.tick_params(axis="y", labelleft=False, labelright=True, labelsize=6)
    ax4 = ax.inset_axes([1, 3/4, 1, 1/4], sharex=ax1, sharey=ax1)
    ax4.tick_params(axis="x", labelbottom=False)
    ax4.tick_params(axis="y", labelleft=False, labelright=True, labelsize=6)

    axs = [ax1, ax2, ax3, ax4]
    snapnums = [61, 77, 95, 127]

    for j, snapnum in enumerate(snapnums):
        i = len(snapnums) - j - 1
        df = _get_data(snapnum, config)
        for _, row in df.iterrows():
            if row["Galaxy"].startswith("Au"):
                axs[i].scatter(
                    np.log10(row["Delta1200"]), np.log10(row["SFR_Msun/yr"]),
                    s=5, facecolor=row["Colors"], edgecolors="none",
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
                xytext=(-25, -15), textcoords='offset points',
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
                  labels=["$-1.0$", "$-0.5$", "0.0", "0.5", "1.0", "1.5"],
                  fontsize=6)

    ax1 = ax.inset_axes([1, 0, 1, 1/4])
    ax1.tick_params(axis="y", labelleft=False, labelright=True)
    ax1.set_xlabel(r"$\log_{10} \delta_{1200}$", fontsize=8)
    ax1.set_xlim(0.4, 1.4)
    ax1.set_ylim(-2.5, 0.5)
    ax1.set_xticks(
        ticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3],
        labels=["0.5", "0.6", "0.7", "0.8", "0.9", "1.0", "1.1", "1.2", "1.3"],
        fontsize=6)
    ax1.set_yticks(ticks=[-2, -1, 0],
                   labels=["$-2$", "$-1$", "0"],
                   fontsize=6)
    ax2 = ax.inset_axes([1, 1/4, 1, 1/4], sharex=ax1, sharey=ax1)
    ax2.tick_params(axis="x", labelbottom=False)
    ax2.tick_params(axis="y", labelleft=False, labelright=True, labelsize=6)
    ax2.set_ylabel(
        r"$\log_{10} \mathrm{sSFR} \, [\mathrm{Gyr}^{-1}]$",
        fontsize=8)
    ax2.yaxis.set_label_coords(1.2, 1)
    ax3 = ax.inset_axes([1, 2/4, 1, 1/4], sharex=ax1, sharey=ax1)
    ax3.tick_params(axis="x", labelbottom=False)
    ax3.tick_params(axis="y", labelleft=False, labelright=True, labelsize=6)
    ax4 = ax.inset_axes([1, 3/4, 1, 1/4], sharex=ax1, sharey=ax1)
    ax4.tick_params(axis="x", labelbottom=False)
    ax4.tick_params(axis="y", labelleft=False, labelright=True, labelsize=6)

    axs = [ax1, ax2, ax3, ax4]
    snapnums = [61, 77, 95, 127]

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
                xytext=(-25, -35), textcoords='offset points',
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

    features: List[FeatureData] = [
        FeatureData(
            name="StellarMassFraction",
            axis_limits=(0.2, 1.0),
            axis_label=r"$M_\star / \left(M_\star + M_\mathrm{cold} \right)$",
            axis_ticks=[0.4, 0.6, 0.8],
            axis_tick_labels=["0.4", "0.6", "0.8"],
        ),
        FeatureData(
            name="logSFR_Msun/yr",
            axis_limits=(-0.5, 1.6),
            axis_label=r"$\log_{10} \mathrm{SFR}$" \
            + "\n" + r"$[\mathrm{M}_\odot \, \mathrm{yr}^{-1}]$",
            axis_ticks=[-0.1, 0.3, 0.7, 1.1],
            axis_tick_labels=["$-0.1$", "0.3", "0.7", "1.1"],
        ),
        FeatureData(
            name="logsSFR_Gyr^-1",
            axis_limits=(-2, -0.6),
            axis_label=r"$\log_{10} \mathrm{sSFR}$" \
                + "\n" + r"$[\mathrm{Gyr}^{-1}]$",
            axis_ticks=[-1.8, -1.4, -1],
            axis_tick_labels=["$-1.8$", "$-1.4$", "$-1.0$"],
        ),
        FeatureData(
            name="Mstar/M200",
            axis_limits=(0, 0.1),
            axis_label=r"$M_\star / M_{200}$",
            axis_ticks=[0.02, 0.04, 0.06, 0.08],
            axis_tick_labels=["0.02", "0.04", "0.06", "0.08"],
        ),
        FeatureData(
            name="Mgas/M200",
            axis_limits=(0.02, 0.1),
            axis_label=r"$M_\mathrm{gas} / M_{200}$",
            axis_ticks=[0.04, 0.06, 0.08],
            axis_tick_labels=["0.04", "0.06", "0.08"],
        ),
        FeatureData(
            name="ColdGasMassFraction",
            axis_limits=(0, 1.0),
            axis_label=r"$M_\mathrm{cg} / \left(M_\star + M_\mathrm{cold} \right)$",
            axis_ticks=[0.2, 0.4, 0.6, 0.8],
            axis_tick_labels=["0.2", "0.4", "0.6", "0.8"],
        ),
    ]
    plot_prop_comparison(
        features=features,
        snapnum=127,
        config=config,
        )

    x_feature = FeatureData(
            name="logDelta1200",
            axis_limits=(0.6, 1.5),
            axis_label=r"$\log_{10} \delta_{1200}$",
            axis_ticks=[0.8, 1.0, 1.2, 1.4],
            axis_tick_labels=["0.8", "1.0", "1.2", "1.4"],
            )
    snapnum = 127
    plot_correlations_with_feature(
        y_features=features,
        x_feature=x_feature,
        snapnum=snapnum,
        config=config,
        fig_name=f"prop_comparison_env_snap{snapnum}_{config['RUN_CODE']}")

    # plot_time_correlation_sfr_vs_delta(config)
    # plot_time_correlation_ssfr_vs_delta(config)
