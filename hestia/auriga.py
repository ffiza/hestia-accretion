import numpy as np
import pandas as pd

from hestia.tools import windowed_average
from hestia.accretion_region import AccretionRegionType


class AurigaData:
    RERUNS: list = [5, 6, 9, 13, 17, 23, 24, 26, 28]

    @staticmethod
    def _get_halo_accretion(config: dict) -> pd.DataFrame:
        window_length = config["TEMPORAL_AVERAGE_WINDOW_LENGTH"]
        df = pd.read_csv(
            "data/iza_et_al_2022/accretion_tracers_spherical_spacing_rvir.csv")

        for i in AurigaData.RERUNS:
            df[f"InflowRateSmoothed_Au{i}_Msun/yr"] = windowed_average(
                df["Time_Gyr"].to_numpy(),
                df[f"InflowRate_Au{i}_Msun/yr"].to_numpy(), window_length)
            df[f"OutflowRateSmoothed_Au{i}_Msun/yr"] = windowed_average(
                df["Time_Gyr"].to_numpy(),
                df[f"OutflowRate_Au{i}_Msun/yr"].to_numpy(), window_length)

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

    @staticmethod
    def _get_disc_accretion(config: dict) -> pd.DataFrame:
        window_length = config["TEMPORAL_AVERAGE_WINDOW_LENGTH"]
        df = pd.read_csv("data/iza_et_al_2022/accretion_rate_tracers.csv")

        for i in AurigaData.RERUNS:
            inflow_rate = df[f"InflowRate_Au{i}_Msun/yr"].to_numpy()
            outflow_rate = df[f"OutflowRate_Au{i}_Msun/yr"].to_numpy()
            time = df["Time_Gyr"].to_numpy()
            df[f"InflowRateSmoothed_Au{i}_Msun/yr"] = windowed_average(
                time, inflow_rate, window_length)
            df[f"OutflowRateSmoothed_Au{i}_Msun/yr"] = windowed_average(
                time, outflow_rate, window_length)

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

    @staticmethod
    def get_accretion(
            config: dict,
            accretion_region_type: AccretionRegionType) -> pd.DataFrame:

        match accretion_region_type:
            case AccretionRegionType.STELLAR_DISC:
                return AurigaData._get_disc_accretion(config)
            case AccretionRegionType.HALO:
                return AurigaData._get_halo_accretion(config)
