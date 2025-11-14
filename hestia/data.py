import json
import yaml
import argparse
import warnings
import numpy as np
import pandas as pd

from hestia.tools import windowed_average
from hestia.accretion_region import (AccretionRegionType,
                                     get_accretion_region_suffix)
from hestia.settings import Settings


class HestiaData:
    @staticmethod
    def get_accretion(
            config: dict,
            accretion_region_type: AccretionRegionType) -> pd.DataFrame:
        window_length = config["TEMPORAL_AVERAGE_WINDOW_LENGTH"]

        suffix = get_accretion_region_suffix(accretion_region_type)

        df = pd.DataFrame()
        for s in Settings.SIMULATIONS:
            for g in Settings.GALAXIES:
                path = f"results/{s}_{g}/accretion_tracers" \
                       + f"{suffix}_{config['RUN_CODE']}.json"
                with open(path) as f:
                    data = json.load(f)
                    time = np.array(data["Times_Gyr"])
                    inflow_rate = np.array(data["InflowRate_Msun/yr"])
                    outflow_rate = np.array(data["OutflowRate_Msun/yr"])
                if "Time_Gyr" not in df:
                    df["Time_Gyr"] = time
                df[f"InflowRate_{s}_{g}_Msun/yr"] = inflow_rate
                df[f"OutflowRate_{s}_{g}_Msun/yr"] = outflow_rate
                df[f"InflowRateSmoothed_{s}_{g}_Msun/yr"] = windowed_average(
                    df["Time_Gyr"].to_numpy(),
                    df[f"InflowRate_{s}_{g}_Msun/yr"].to_numpy(),
                    window_length)
                df[f"OutflowRateSmoothed_{s}_{g}_Msun/yr"] = windowed_average(
                    df["Time_Gyr"].to_numpy(),
                    df[f"OutflowRate_{s}_{g}_Msun/yr"].to_numpy(),
                    window_length)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            sample_inflow = df[[
                f"InflowRate_{s}_{g}_Msun/yr"
                for s in Settings.SIMULATIONS for g in Settings.GALAXIES]]
            df["InflowRateMin_Msun/yr"] = sample_inflow.min(axis=1)
            df["InflowRateMax_Msun/yr"] = sample_inflow.max(axis=1)
            df["InflowRateMean_Msun/yr"] = np.nanmean(
                sample_inflow, axis=1)
            df["InflowRateStd_Msun/yr"] = np.nanstd(
                sample_inflow, axis=1)

            sample_outflow = df[[
                f"OutflowRate_{s}_{g}_Msun/yr"
                for s in Settings.SIMULATIONS for g in Settings.GALAXIES]]
            df["OutflowRateMin_Msun/yr"] = sample_outflow.min(axis=1)
            df["OutflowRateMax_Msun/yr"] = sample_outflow.max(axis=1)
            df["OutflowRateMean_Msun/yr"] = np.nanmean(
                sample_outflow, axis=1)
            df["OutflowRateStd_Msun/yr"] = np.nanstd(
                sample_outflow, axis=1)

            sample_inflow = df[[
                f"InflowRateSmoothed_{s}_{g}_Msun/yr"
                for s in Settings.SIMULATIONS for g in Settings.GALAXIES]]
            df["InflowRateSmoothedMin_Msun/yr"] = sample_inflow.min(axis=1)
            df["InflowRateSmoothedMax_Msun/yr"] = sample_inflow.max(axis=1)
            df["InflowRateSmoothedMean_Msun/yr"] = np.nanmean(
                sample_inflow, axis=1)
            df["InflowRateSmoothedStd_Msun/yr"] = np.nanstd(
                sample_inflow, axis=1)

            sample_outflow = df[[
                f"OutflowRateSmoothed_{s}_{g}_Msun/yr"
                for s in Settings.SIMULATIONS for g in Settings.GALAXIES]]
            df["OutflowRateSmoothedMin_Msun/yr"] = sample_outflow.min(axis=1)
            df["OutflowRateSmoothedMax_Msun/yr"] = sample_outflow.max(axis=1)
            df["OutflowRateSmoothedMean_Msun/yr"] = np.nanmean(
                sample_outflow, axis=1)
            df["OutflowRateSmoothedStd_Msun/yr"] = np.nanstd(
                sample_outflow, axis=1)

        df = df.dropna()

        return df


if __name__ == "__main__":
    # Get arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = yaml.safe_load(open(f"configs/{args.config}.yml"))

    print(HestiaData.get_accretion(config, AccretionRegionType.HALO))
    print(HestiaData.get_accretion(config, AccretionRegionType.STELLAR_DISC))
