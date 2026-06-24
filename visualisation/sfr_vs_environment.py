import yaml
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hestia.images import figure_setup
from hestia.settings import Settings


def _get_data(config: dict) -> pd.DataFrame:
    sfr = pd.read_csv("data/iza_et_al_2022/sfr.csv")
    dfs = []

    for simulation in range(1, 31):
        env = pd.read_csv(
            f"data/auriga/au{simulation}/environment_evolution.csv")
        dfs.append(pd.DataFrame(
            np.column_stack([
                sfr["Time_Gyr"],
                sfr[f"SFR_Au{simulation}_Msun/yr"],
                env["Delta1200"].to_numpy()[1:],
                env["Redshift"].to_numpy()[1:],
                # env["ExpansionFactor"].to_numpy()[1:],
                np.arange(1, len(sfr) + 1),
                ["Au"] * len(sfr),
                [simulation] * len(sfr),
                [np.nan] * len(sfr)]),
            columns=[
                "Time_Gyr",
                "SFR_Msun/yr",
                "Delta1200",
                "Redshift",
                # "ExpansionFactor",
                "Snapshot",
                "Suite",
                "Simulation",
                "Galaxy"]))

    for simulation in Settings.HIGH_RES_SIMULATIONS:
        for galaxy in Settings.GALAXIES:
            sfr = pd.read_csv(
                f"data/hestia/{galaxy}_M_SFR_t_Hestia{simulation}.csv",
                usecols=["SnapNo", "SFR"])
            time = pd.read_csv(
                f"data/hestia/r200_t/r200_t_{galaxy}_{simulation}.csv",
                usecols=["SnapshotNumber", "Time_Gyr", "Redshift",])
            time = time.set_index("SnapshotNumber")
            env = pd.read_csv(
                f"results/{simulation}_{galaxy}/"
                f"delta_1200_{config['RUN_CODE']}.csv",
                usecols=["SnapshotNumbers", "Delta"])
            env = env.set_index("SnapshotNumbers")
            df_ = sfr.join(time, how="left", lsuffix="_sfr",
                           rsuffix="_time", on="SnapNo")
            df_ = df_.join(env, how="left", lsuffix="_sfr",
                           rsuffix="_env", on="SnapNo")
            df_ = df_.rename(columns={
                "SnapNo": "Snapshot",
                "SFR": "SFR_Msun/yr",
                "Delta": "Delta1200",
            })
            df_["Suite"] = ["He"] * len(df_)
            df_["Simulation"] = [simulation] * len(df_)
            df_["Galaxy"] = [galaxy] * len(df_)
            dfs.append(df_)

    df = pd.concat(dfs, ignore_index=True)
    df["Delta1200"] = df["Delta1200"].astype(float)
    df["Time_Gyr"] = df["Time_Gyr"].astype(float)
    df["SFR_Msun/yr"] = df["SFR_Msun/yr"].astype(float)
    df["Redshift"] = df["Redshift"].astype(float)
    df["Snapshot"] = df["Snapshot"].astype(int)
    df["Simulation"] = df["Simulation"].astype(str)
    df["Galaxy"] = df["Galaxy"].astype(str)
    df["Suite"] = df["Suite"].astype(str)

    return df


def make_plot(df: pd.DataFrame, config: dict) -> None:
    fig = plt.figure(figsize=(5.0, 3.5))
    gs = fig.add_gridspec(nrows=2, ncols=3, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=False)

    for ax in axs.flatten():
        ax.set_axisbelow(True)
        ax.set_xlim(0, 25)
        ax.set_ylim(1, 50)
        ax.set_yscale("log")
        ax.set_xticks(ticks=[5, 10, 15, 20],
                      labels=["5", "10", "15", "20"],
                      fontsize=6)
        ax.set_yticks(ticks=[1, 3, 5, 10, 20],
                      labels=["1", "3", "5", "10", "20"],
                      fontsize=6)
        ax.set_ylabel(r'SFR [$\mathrm{M}_\odot \, \mathrm{yr}^{-1}$]',
                      fontsize=8)
        ax.set_xlabel(r'$\delta_{1200}$', fontsize=8)
        ax.label_outer()

    for j, simulation in enumerate(Settings.HIGH_RES_SIMULATIONS):
        for i, galaxy in enumerate(Settings.GALAXIES):
            axs[i, j].plot(
                df[
                    (df["Simulation"] == simulation)
                    & (df["Galaxy"] == galaxy)]["Delta1200"],
                df[
                    (df["Simulation"] == simulation)
                    & (df["Galaxy"] == galaxy)]["SFR_Msun/yr"],
                color=Settings.SIMULATION_COLORS[simulation], lw=0.75,
                zorder=10,
                label=r"$\texttt{" + f"{simulation}" + f"_{galaxy}" + "}$")
            axs[i, j].scatter(
                df[
                    (df["Simulation"] == simulation)
                    & (df["Galaxy"] == galaxy)]["Delta1200"].to_numpy()[-1],
                df[
                    (df["Simulation"] == simulation)
                    & (df["Galaxy"] == galaxy)]["SFR_Msun/yr"].to_numpy()[-1],
                color=Settings.SIMULATION_COLORS[simulation], s=3,
                zorder=10)
            axs[i, j].scatter(
                df[(df["Suite"] == "Au")]["Delta1200"],
                df[(df["Suite"] == "Au")]["SFR_Msun/yr"],
                color="#8a8a8a", s=0.5, zorder=9, alpha=0.25,
                label="Auriga")

    for ax in axs.flatten():
        ax.legend(loc="upper right", framealpha=0, fontsize=5, ncol=1)

    plt.savefig(f"images/sfr_vs_environment_{config['RUN_CODE']}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    figure_setup()

    # Get arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = yaml.safe_load(open(f"configs/{args.config}.yml"))

    df = _get_data(config)
    make_plot(df, config)
