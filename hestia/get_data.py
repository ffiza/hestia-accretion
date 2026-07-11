import argparse
import numpy as np
import pandas as pd

from hestia.settings import Settings


def get_data(config: dict) -> pd.DataFrame:
    sfr = pd.read_csv("data/iza_et_al_2022/sfr.csv")
    dfs = []

    for simulation in range(1, 31):
        env = pd.read_csv(
            f"data/auriga/au{simulation}/environment_evolution.csv")
        vir_props = pd.read_csv("data/auriga/virial_properties.csv")
        gas = pd.read_csv(
            "data/auriga/cold_gas_mass_evolution.csv",
            usecols=["SubhaloGasMassUnder20000K_Msun",
                     "Simulation", "Snapshot"])
        cold_gas_mass = np.nan * np.ones(128)
        cold_gas_mass[gas["Snapshot"][gas["Simulation"] == simulation].to_numpy()] = gas[
            "SubhaloGasMassUnder20000K_Msun"][
                gas["Simulation"] == simulation].to_numpy()
        baryon_mass = np.loadtxt(
            f"data/auriga/au{simulation}/baryon_mass.csv",
            delimiter=" ",
            skiprows=1,
            max_rows=128,
        )
        dfs.append(pd.DataFrame(
            np.column_stack([
                sfr["Time_Gyr"],
                sfr[f"SFR_Au{simulation}_Msun/yr"],
                env["Delta1200"].to_numpy()[1:],
                env["Redshift"].to_numpy()[1:],
                vir_props[f"M200_Au{simulation}_1E10Msun"].to_numpy()[1:],
                cold_gas_mass[1:],
                baryon_mass[1:, 1],
                baryon_mass[1:, 0],
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
                "M200_1E10Msun",
                "SubhaloGasMassUnder20000K_Msun",
                "StellarMass_1E10Msun",
                "GasMass_1E10Msun",
                # "ExpansionFactor",
                "Snapshot",
                "Suite",
                "Simulation",
                "Galaxy"]))

    for simulation in Settings.HIGH_RES_SIMULATIONS:
        for galaxy in Settings.GALAXIES:
            sfr = pd.read_csv(
                f"data/hestia/{galaxy}_M_SFR_t_Hestia{simulation}.csv",
                usecols=["SnapNo", "SFR", "Mstar", "Mcold", "Mgas"])
            time = pd.read_csv(
                f"data/hestia/r200_t/r200_t_{galaxy}_{simulation}.csv",
                usecols=["SnapshotNumber", "Time_Gyr", "Redshift", "Mvir"])
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
                "Mstar": "StellarMass_Msun",
                "Mcold": "SubhaloGasMassUnder20000K_Msun",
                "Mgas": "GasMass_Msun",
                "Delta": "Delta1200",
            })
            df_["Suite"] = ["He"] * len(df_)
            df_["Simulation"] = [simulation] * len(df_)
            df_["Galaxy"] = [galaxy] * len(df_)
            df_["StellarMass_1E10Msun"] = df_["StellarMass_Msun"] / 1E10
            df_["GasMass_1E10Msun"] = df_["GasMass_Msun"] / 1E10
            df_["M200_1E10Msun"] = df_["Mvir"] / 1E10
            df_ = df_.drop(
                columns=["Mvir", "StellarMass_Msun", "GasMass_Msun"])
            dfs.append(df_)

    df = pd.concat(dfs, ignore_index=True)
    df["Delta1200"] = df["Delta1200"].astype(float)
    df["Time_Gyr"] = df["Time_Gyr"].astype(float)
    df["SFR_Msun/yr"] = df["SFR_Msun/yr"].astype(float)
    df["M200_1E10Msun"] = df["M200_1E10Msun"].astype(float)
    df["SubhaloGasMassUnder20000K_1E10Msun"] = df[
        "SubhaloGasMassUnder20000K_Msun"].astype(float) / 1E10
    df["StellarMass_1E10Msun"] = df["StellarMass_1E10Msun"].astype(float)
    df["GasMass_1E10Msun"] = df["GasMass_1E10Msun"].astype(float)
    df = df.drop(
        columns=["SubhaloGasMassUnder20000K_Msun"])
    df["Redshift"] = df["Redshift"].astype(float)
    df["Snapshot"] = df["Snapshot"].astype(int)
    df["Simulation"] = df["Simulation"].astype(str)
    df["Galaxy"] = df["Galaxy"].astype(str)
    df["Suite"] = df["Suite"].astype(str)

    return df


if __name__ == "__main__":
    import yaml

    # Get arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = yaml.safe_load(open(f"configs/{args.config}.yml"))

    df = get_data(config)
    print(df.sample(n=20))
