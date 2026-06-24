import pandas as pd
import numpy as np

from hestia.settings import Settings


def get_data(config: dict) -> pd.DataFrame:
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
