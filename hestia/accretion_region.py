import numpy as np
from abc import ABC
import pandas as pd
from enum import Enum


class AccretionRegionType(Enum):
    STELLAR_DISC = 1
    HALO = 2


class FilterType(Enum):
    IN = 1
    OUT = 2


class AccretionRegion(ABC):
    def __init__(self) -> None:
        pass

    def set_params(self, *args, **kwargs) -> None:
        pass

    def select(self, df: pd.DataFrame,
               filter_type: FilterType) -> np.ndarray | None:
        pass


class StellarDiscRegion(AccretionRegion):
    def __init__(self, radius: float, half_height: float) -> None:
        self.accretion_region_type = AccretionRegionType.STELLAR_DISC
        self.radius = radius
        self.half_height = half_height

    def set_params(self, radius: float, half_height: float) -> None:
        self.radius = radius
        self.half_height = half_height

    def select(self, df: pd.DataFrame, filter_type: FilterType) -> np.ndarray:
        if "CylindricalRadius_ckpc" not in df.columns:
            raise ValueError(
                "Feature `CylindricalRadius_ckpc` not found in dataframe.")
        if "zPosition_ckpc" not in df.columns:
            raise ValueError(
                "Feature `zPosition_ckpc` not found in dataframe.")

        is_selected = (
            df["CylindricalRadius_ckpc"].to_numpy() <= self.radius) \
            & (np.abs(df["zPosition_ckpc"].to_numpy()) <= self.half_height)
        if filter_type == FilterType.OUT:
            return ~is_selected
        return is_selected


class HaloRegion(AccretionRegion):
    def __init__(self, radius: float) -> None:
        self.accretion_region_type = AccretionRegionType.HALO
        self.radius = radius

    def set_params(self, radius: float) -> None:
        self.radius = radius

    def select(self, df: pd.DataFrame, filter_type: FilterType) -> np.ndarray:
        if "SphericalRadius_ckpc" not in df.columns:
            raise ValueError(
                "Feature `CylindricalRadius_ckpc` not found in dataframe.")

        is_selected = (
            df["SphericalRadius_ckpc"].to_numpy() <= self.radius)
        if filter_type == FilterType.OUT:
            return ~is_selected
        return is_selected


def get_accretion_region_suffix(accretion_region_type: AccretionRegionType
                                ) -> str:
    match accretion_region_type:
        case AccretionRegionType.STELLAR_DISC:
            return ""
        case AccretionRegionType.HALO:
            return "_halo"
        case _:
            raise ValueError("Invalid `AccretionRegionType`.")
