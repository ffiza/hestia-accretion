import numpy as np
import pandas as pd

from hestia.tools import windowed_average
from hestia.accretion_region import AccretionRegionType


class AurigaData:
    RERUNS: list = [5, 6, 9, 13, 17, 23, 24, 26, 28]

    @staticmethod
    def get_accretion(
            config: dict,
            accretion_region_type: AccretionRegionType) -> pd.DataFrame:
        window_length = config["TEMPORAL_AVERAGE_WINDOW_LENGTH"]

        match accretion_region_type:
            case AccretionRegionType.HALO:
                df = pd.read_csv(
                    "data/iza_et_al_2022/"
                    "accretion_tracers_spherical_spacing_rvir.csv")
            case AccretionRegionType.STELLAR_DISC:
                df = pd.read_csv(
                    "data/iza_et_al_2022/accretion_rate_tracers.csv")
            case _:
                raise ValueError("Invalid accretion region type.")

        for i in AurigaData.RERUNS:
            inflow_rate = df[f"InflowRate_Au{i}_Msun/yr"].to_numpy()
            outflow_rate = df[f"OutflowRate_Au{i}_Msun/yr"].to_numpy()
            time = df["Time_Gyr"].to_numpy()
            df[f"InflowRateSmoothed_Au{i}_Msun/yr"] = windowed_average(
                time, inflow_rate, window_length)
            df[f"OutflowRateSmoothed_Au{i}_Msun/yr"] = windowed_average(
                time, outflow_rate, window_length)

        sample_inflow = df[[
            f"InflowRate_Au{i}_Msun/yr"
            for i in AurigaData.RERUNS]]
        df["InflowRateMin_Msun/yr"] = sample_inflow.min(axis=1)
        df["InflowRateMax_Msun/yr"] = sample_inflow.max(axis=1)
        df["InflowRateMean_Msun/yr"] = np.nanmean(
            sample_inflow, axis=1)
        df["InflowRateSmoothedStd_Msun/yr"] = np.nanstd(
            sample_inflow, axis=1)

        sample_outflow = df[[
            f"InflowRate_Au{i}_Msun/yr"
            for i in AurigaData.RERUNS]]
        df["OutflowRateMin_Msun/yr"] = sample_outflow.min(axis=1)
        df["OutflowRateMax_Msun/yr"] = sample_outflow.max(axis=1)
        df["OutflowRateMean_Msun/yr"] = np.nanmean(
            sample_outflow, axis=1)
        df["OutflowRateStd_Msun/yr"] = np.nanstd(
            sample_outflow, axis=1)

        sample_inflow = df[[
            f"InflowRateSmoothed_Au{i}_Msun/yr"
            for i in AurigaData.RERUNS]]
        df["InflowRateSmoothedMin_Msun/yr"] = sample_inflow.min(axis=1)
        df["InflowRateSmoothedMax_Msun/yr"] = sample_inflow.max(axis=1)
        df["InflowRateSmoothedMean_Msun/yr"] = np.nanmean(
            sample_inflow, axis=1)
        df["InflowRateSmoothedStd_Msun/yr"] = np.nanstd(
            sample_inflow, axis=1)

        sample_outflow = df[[
            f"InflowRateSmoothed_Au{i}_Msun/yr"
            for i in AurigaData.RERUNS]]
        df["OutflowRateSmoothedMin_Msun/yr"] = sample_outflow.min(axis=1)
        df["OutflowRateSmoothedMax_Msun/yr"] = sample_outflow.max(axis=1)
        df["OutflowRateSmoothedMean_Msun/yr"] = np.nanmean(
            sample_outflow, axis=1)
        df["OutflowRateSmoothedStd_Msun/yr"] = np.nanstd(
            sample_outflow, axis=1)

        df = df.dropna()
        return df
