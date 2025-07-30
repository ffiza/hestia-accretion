import numpy
import numpy as np
import pandas as pd
import astropy
import yaml

import TrackGalaxy
from hestia.pca import PCA_matrix
from hestia.df_type import DFType

GLOBAL_CONFIG = yaml.safe_load(open("configs/global.yml"))


def _make_dataframe_cells(
        SimName: str, SnapNo: int, MW_or_M31: str,
        max_radius: float = 100.0) -> pd.DataFrame:
    """
    Loads a snapshot and returns a dataframe with the following columns:

    - `xPosition_ckpc`: x-coordinate of the particles in ckpc.
    - `yPosition_ckpc`: y-coordinate of the particles in ckpc.
    - `zPosition_ckpc`: z-coordinate of the particles in ckpc
    - `Mass_Msun`: mass of the particles in Msun.
    - `ParticleType`: type of the particle (0 for gas and 4 for stars).
    - `StellarBirthTime_Gyr`: time of birth of the star in Gyr. If the particle
      is not a star, this element should be set to `np.nan`.

    Parameters
    ----------
    SimName : str
        Simulation name from '17_11', '09_18' or '37_11'.
    SnapNo : int
        Snapshot number (z=0 corresponds to SnapNo=127)
    MW_or_M31 : str, optional
        Choose one of the two main galaxies from 'MW' or 'M31' to center the
        sphere that will be considered for the dataframe.
    output_dir : str, optional
        Directory where the pickle containing the df will be saved. By default
        "results/dataframes/".
    max_radius : float, optional
        Radius of the sphere required for the df in ckpc. By default 100.0.
    """

    print(f'Running make_dataframe for snapshot {SnapNo}...')

    # These numbers come from cross-correlating with
    # /z/nil/codes/HESTIA/FIND_LG/LGs_8192_GAL_FOR.txt andArepo's SUBFIND.
    if SimName == '17_11':
        # subhalo_number = 1 if MW_or_M31 == "MW" else 0
        SimulationDirectory = "/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/" \
            + "17_11/output_2x2.5Mpc/"
    elif SimName == '09_18':
        # subhalo_number = 3911 if MW_or_M31 == "MW" else 2608
        SimulationDirectory = "/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/" \
            + "09_18/output_2x2.5Mpc/"
    elif SimName == '37_11':
        # subhalo_number = 920 if MW_or_M31 == "MW" else 0
        SimulationDirectory = "/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/" \
            + "37_11/output_2x2.5Mpc/"
    else:
        raise ValueError("Invalid simulation name.")

    cosmo = astropy.cosmology.FlatLambdaCDM(
        H0=GLOBAL_CONFIG["HUBBLE_CONST"],
        Om0=GLOBAL_CONFIG["OMEGA_MATTER"] + GLOBAL_CONFIG["OMEGA_BARYONS"])

    T = TrackGalaxy.TrackGalaxy(numpy.array([SnapNo]),
                                SimName,
                                Dir=SimulationDirectory,
                                MultipleSnaps=True)
    SnapTime = T.SnapTimes[0]  # Scale factor
    Redshift = 1.0 / SnapTime - 1
    SnapTime_Gyr = cosmo.age(Redshift).value  # Gyr

    Gas_Attrs = T.GetParticles(
        SnapNo, Type=0, Attrs=['Coordinates',
                               'Masses',
                               'Velocities',
                               'ParticleIDs'])
    GasPos = 1000*Gas_Attrs['Coordinates'] \
        / GLOBAL_CONFIG["SMALL_HUBBLE_CONST"]  # ckpc
    GasMass = Gas_Attrs['Masses'] * 1e10 \
        / GLOBAL_CONFIG["SMALL_HUBBLE_CONST"]  # Msun
    GasVel = Gas_Attrs['Velocities']*numpy.sqrt(SnapTime)  # km/s
    GasIDs = Gas_Attrs['ParticleIDs']

    Star_Attrs = T.GetParticles(
        SnapNo, Type=4, Attrs=['Coordinates',
                               'Masses',
                               'ParticleIDs',
                               'GFM_StellarFormationTime'])
    StarPos = 1000*Star_Attrs['Coordinates'] \
        / GLOBAL_CONFIG["SMALL_HUBBLE_CONST"]  # ckpc
    StarMass = Star_Attrs['Masses'] * 1e10 \
        / GLOBAL_CONFIG["SMALL_HUBBLE_CONST"]  # Msun
    StarIDs = Star_Attrs['ParticleIDs']
    StarBirths = Star_Attrs['GFM_StellarFormationTime']
    StarBirths_z = 1/StarBirths - 1
    StarBirths_Gyr = cosmo.age(StarBirths_z).value

    DM_Attrs = T.GetParticles(
        SnapNo, Type=1, Attrs=['Coordinates',
                               'Masses',
                               'ParticleIDs'])
    DMPos = 1000*DM_Attrs['Coordinates'] \
        / GLOBAL_CONFIG['SMALL_HUBBLE_CONST']  # ckpc
    DMMass = DM_Attrs['Masses'] * 1e10 \
        / GLOBAL_CONFIG['SMALL_HUBBLE_CONST']  # Msun
    DMIDs = DM_Attrs['ParticleIDs']

    try:
        BH_Attrs = T.GetParticles(
            SnapNo, Type=5, Attrs=['Coordinates',
                                   'Masses',
                                   'ParticleIDs'])
    except KeyError:
        print('No BHs in this file.')
        BH_Attrs = None

    if BH_Attrs:
        BHPos = 1000*BH_Attrs['Coordinates'] \
            / GLOBAL_CONFIG['SMALL_HUBBLE_CONST']  # ckpc
        BHMass = BH_Attrs['Masses'] * 1e10 \
            / GLOBAL_CONFIG['SMALL_HUBBLE_CONST']  # Msun
        BHIDs = BH_Attrs['ParticleIDs']

    # Reading progenitor numbers calculated with T.TrackProgenitor() from TrackGalaxy.py
    Snaps, Tracked_Numbers_MW, Tracked_Numbers_M31 = np.loadtxt('/z/lbiaus/hestia-accretion/data/progenitor_lists/snaps_MWprogs_M31progs_{}.txt'.format(SimName))
    Snaps = Snaps.astype(int)
    Tracked_Numbers_MW = Tracked_Numbers_MW.astype(int)
    Tracked_Numbers_M31 = Tracked_Numbers_M31.astype(int)
    SubhaloNumberMW, SubhaloNumberM31 = Tracked_Numbers_MW[Snaps == SnapNo], Tracked_Numbers_M31[Snaps == SnapNo]

    # Read in subhaloes position and velocities:
    GroupCatalog = T.GetGroups(SnapNo, Attrs=['/Subhalo/SubhaloPos', '/Subhalo/SubhaloVel'])
    SubhaloPos = 1000*GroupCatalog['/Subhalo/SubhaloPos'] / GLOBAL_CONFIG["SMALL_HUBBLE_CONST"]  # ckpc
    SubhaloVel = GroupCatalog['/Subhalo/SubhaloVel'] * np.sqrt(SnapTime)  # km s^-1
    MW_pos, MW_vel = SubhaloPos[SubhaloNumberMW], SubhaloVel[SubhaloNumberMW]
    M31_pos, M31_vel = SubhaloPos[SubhaloNumberM31], SubhaloVel[SubhaloNumberM31]

    # We keep only particles within the chosen halo
    if MW_or_M31 == 'MW':
        GasPos -= MW_pos
        StarPos -= MW_pos
        DMPos -= MW_pos
        if BH_Attrs:
            BHPos -= MW_pos
        GasVel -= MW_vel
    elif MW_or_M31 == 'M31':
        GasPos -= M31_pos
        StarPos -= M31_pos
        DMPos -= M31_pos
        if BH_Attrs:
            BHPos -= M31_pos
        GasVel -= M31_vel

    # Keep only particles within RMAX:
    index_of_nearby_gas = numpy.where(
        GasPos[:, 0]**2 + GasPos[:, 1]**2 + GasPos[:, 2]**2 < max_radius**2)
    index_of_nearby_stars = numpy.where(
        StarPos[:, 0]**2 + StarPos[:, 1]**2 + StarPos[:, 2]**2 < max_radius**2)
    index_of_nearby_DM = numpy.where(
        DMPos[:, 0]**2 + DMPos[:, 1]**2 + DMPos[:, 2]**2 < max_radius**2)
    if BH_Attrs:
        index_of_nearby_BH = numpy.where(
            BHPos[:, 0]**2 + BHPos[:, 1]**2 + BHPos[:, 2]**2 < max_radius**2)
    
    GasPos = GasPos[index_of_nearby_gas]
    GasMass = GasMass[index_of_nearby_gas]
    GasIDs = GasIDs[index_of_nearby_gas]

    StarPos = StarPos[index_of_nearby_stars]
    StarMass = StarMass[index_of_nearby_stars]
    StarIDs = StarIDs[index_of_nearby_stars]
    StarBirths_Gyr = StarBirths_Gyr[index_of_nearby_stars]

    DMPos = DMPos[index_of_nearby_DM]
    DMMass = DMMass[index_of_nearby_DM]
    DMIDs = DMIDs[index_of_nearby_DM]

    if BH_Attrs:
        BHPos = BHPos[index_of_nearby_BH]
        BHMass = BHMass[index_of_nearby_BH]
        BHIDs = BHIDs[index_of_nearby_BH]

    # We align positions with gas disk:
    R = PCA_matrix(GasPos, GasVel, 15)
    GasPos = np.dot(GasPos, R)
    StarPos = np.dot(StarPos, R)
    DMPos = np.dot(DMPos, R)
    if BH_Attrs:
        BHPos = np.dot(BHPos, R)


    if BH_Attrs:
        AllPos = np.concatenate((GasPos, StarPos, DMPos, BHPos))
        AllMass = np.concatenate((GasMass, StarMass, DMMass, BHMass))
        AllIDs = np.concatenate((GasIDs, StarIDs, DMIDs, BHIDs))
        AllTypes = np.concatenate((np.zeros(np.size(GasIDs)),
                                4 * np.ones(np.size(StarIDs)),
                                1 * np.ones(np.size(DMIDs)),
                                5 * np.ones(np.size(BHIDs))))
        AllBirths = np.concatenate((np.nan*np.ones(np.size(GasIDs)),
                                    StarBirths_Gyr,
                                    np.nan*np.ones(np.size(DMIDs)),
                                    np.nan*np.ones(np.size(BHIDs))))
    else:
        AllPos = np.concatenate((GasPos, StarPos, DMPos))
        AllMass = np.concatenate((GasMass, StarMass, DMMass))
        AllIDs = np.concatenate((GasIDs, StarIDs, DMIDs))
        AllTypes = np.concatenate((np.zeros(np.size(GasIDs)),
                                4 * np.ones(np.size(StarIDs)),
                                1 * np.ones(np.size(DMIDs))))
        AllBirths = np.concatenate((np.nan*np.ones(np.size(GasIDs)),
                                    StarBirths_Gyr,
                                    np.nan*np.ones(np.size(DMIDs))))

    data_dict = {
        'xPosition_ckpc': AllPos[:, 0],
        'yPosition_ckpc': AllPos[:, 1],
        'zPosition_ckpc': AllPos[:, 2],
        'Mass_Msun': AllMass,
        'ParticleType': AllTypes,
        'StellarBirthTime_Gyr': AllBirths,
        'ParticleIDs': AllIDs
    }

    df = pd.DataFrame(data_dict)

    # Additional data as dataframe metadata
    df.expansion_factor = SnapTime
    df.time = SnapTime_Gyr
    df.redshift = Redshift
    df.snapshot_number = SnapNo

    return df


def _make_dataframe_tracers(
    SimName: str, SnapNo: int, MW_or_M31: str,
    max_radius: float = 100.0) -> pd.DataFrame:
    
    print(f'Running make_dataframe for snapshot {SnapNo}...')

    # These numbers come from cross-correlating with
    # /z/nil/codes/HESTIA/FIND_LG/LGs_8192_GAL_FOR.txt andArepo's SUBFIND.
    if SimName == '17_11':
        # subhalo_number = 1 if MW_or_M31 == "MW" else 0
        SimulationDirectory = "/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/" \
            + "17_11/output_2x2.5Mpc/"
    elif SimName == '09_18':
        # subhalo_number = 3911 if MW_or_M31 == "MW" else 2608
        SimulationDirectory = "/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/" \
            + "09_18/output_2x2.5Mpc/"
    elif SimName == '37_11':
        # subhalo_number = 920 if MW_or_M31 == "MW" else 0
        SimulationDirectory = "/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/" \
            + "37_11/output_2x2.5Mpc/"
    else:
        raise ValueError("Invalid simulation name.")

    cosmo = astropy.cosmology.FlatLambdaCDM(
        H0=GLOBAL_CONFIG["HUBBLE_CONST"],
        Om0=GLOBAL_CONFIG["OMEGA_MATTER"] + GLOBAL_CONFIG["OMEGA_BARYONS"])

    T = TrackGalaxy.TrackGalaxy(numpy.array([SnapNo]),
                                SimName,
                                Dir=SimulationDirectory,
                                MultipleSnaps=True)
    SnapTime = T.SnapTimes[0]  # Scale factor
    Redshift = 1.0 / SnapTime - 1
    SnapTime_Gyr = cosmo.age(Redshift).value  # Gyr

    Gas_Attrs = T.GetParticles(
        SnapNo, Type=0, Attrs=['Coordinates',
                               'ParticleIDs'])
    GasPos = 1000*Gas_Attrs['Coordinates'] \
        / GLOBAL_CONFIG["SMALL_HUBBLE_CONST"]  # ckpc
    GasIDs = Gas_Attrs['ParticleIDs']

    Star_Attrs = T.GetParticles(
        SnapNo, Type=4, Attrs=['Coordinates',
                               'ParticleIDs'])
    StarPos = 1000*Star_Attrs['Coordinates'] \
        / GLOBAL_CONFIG["SMALL_HUBBLE_CONST"]  # ckpc
    StarIDs = Star_Attrs['ParticleIDs']

    BH_Attrs = T.GetParticles(
        SnapNo, Type=5, Attrs=['Coordinates',
                               'ParticleIDs'])
    BHPos = 1000*BH_Attrs['Coordinates'] \
        / GLOBAL_CONFIG["SMALL_HUBBLE_CONST"]  # ckpc
    BHIDs = BH_Attrs['ParticleIDs']

    Tracer_Attrs = T.GetParticles(
        SnapNo, Type=6, Attrs=['ParentID',
                               'TracerID'])
    TracerID = Tracer_Attrs['TracerID']
    TracerParentID = Tracer_Attrs['ParentID']

    
    # Reading progenitor numbers calculated with T.TrackProgenitor() from TrackGalaxy.py
    Snaps, Tracked_Numbers_MW, Tracked_Numbers_M31 = np.loadtxt('/z/lbiaus/hestia-accretion/data/progenitor_lists/snaps_MWprogs_M31progs_{}.txt'.format(SimName))
    Snaps = Snaps.astype(int)
    Tracked_Numbers_MW = Tracked_Numbers_MW.astype(int)
    Tracked_Numbers_M31 = Tracked_Numbers_M31.astype(int)
    SubhaloNumberMW, SubhaloNumberM31 = Tracked_Numbers_MW[Snaps==SnapNo], Tracked_Numbers_M31[Snaps==SnapNo]

    # Read in subhaloes position and velocities:
    GroupCatalog = T.GetGroups(SnapNo, Attrs=['/Subhalo/SubhaloPos', '/Subhalo/SubhaloVel'])
    SubhaloPos = 1000*GroupCatalog['/Subhalo/SubhaloPos'] / GLOBAL_CONFIG["SMALL_HUBBLE_CONST"] # ckpc
    SubhaloVel = GroupCatalog['/Subhalo/SubhaloVel'] * np.sqrt(SnapTime) # km s^-1
    MW_pos, MW_vel = SubhaloPos[SubhaloNumberMW], SubhaloVel[SubhaloNumberMW]
    M31_pos, M31_vel = SubhaloPos[SubhaloNumberM31], SubhaloVel[SubhaloNumberM31]


    # We keep only particles within the chosen halo
    if MW_or_M31 == 'MW':
        GasPos -= MW_pos
        StarPos -= MW_pos
        BHPos -= MW_pos
    elif MW_or_M31 == 'M31':
        GasPos -= M31_pos
        StarPos -= M31_pos
        BHPos -= M31_pos

    # Build dictionaries for each type
    gas_id_to_index = {pID: ind for ind, pID in enumerate(GasIDs)}
    star_id_to_index = {pID: ind for ind, pID in enumerate(StarIDs)}
    bh_id_to_index = {pID: ind for ind, pID in enumerate(BHIDs)}

    # For each tracer particle, find the type and position
    matched_types = []
    matched_positions = []

    for pID in TracerParentID:
        if pID in gas_id_to_index:
            matched_types.append(0)
            matched_positions.append(GasPos[gas_id_to_index[pID]])
        elif pID in star_id_to_index:
            matched_types.append(4)
            matched_positions.append(StarPos[star_id_to_index[pID]])
        elif pID in bh_id_to_index:
            matched_types.append(5)
            matched_positions.append(BHPos[bh_id_to_index[pID]])

    matched_types, matched_positions = np.array(matched_types), np.array(matched_positions)

    index_nearby = np.linalg.norm(matched_positions, axis=1) < max_radius
    TracerID, matched_positions, matched_types = TracerID[index_nearby], matched_positions[index_nearby], matched_types[index_nearby]
    ID_sorting = np.argsort(TracerID)
    TracerID_sorted, matched_positions_sorted, matched_types_sorted = TracerID[ID_sorting], matched_positions[ID_sorting], matched_types[ID_sorting]

    data_dict = {
        'TracerID': TracerID_sorted,
        'xPosition_ckpc': matched_positions_sorted[:, 0],
        'yPosition_ckpc': matched_positions_sorted[:, 1],
        'zPosition_ckpc': matched_positions_sorted[:, 2],
        'ParentCellType': matched_types_sorted 
    }

    df = pd.DataFrame(data_dict)

    # Force data types
    df["TracerID"] = df["TracerID"].astype(np.uint64)
    df["ParentCellType"] = df["ParentCellType"].astype(np.uint8)

    # Additional data as dataframe metadata
    df.expansion_factor = SnapTime
    df.time = SnapTime_Gyr
    df.redshift = Redshift
    df.snapshot_number = SnapNo

    # Add target gas mass value to dataframe
    df.target_gas_mass = yaml.safe_load(
        open("data/hestia/target_gas_mass.yml"))[SimName] \
        * 1E10 / GLOBAL_CONFIG["SMALL_HUBBLE_CONST"]  # Msun

    return df


def make_dataframe(
        SimName: str, SnapNo: int, MW_or_M31: str,
        df_type: DFType, max_radius: float = 100.0) -> pd.DataFrame:
    """
    Loads a snapshot and returns a dataframe with data pertaining to cells
    if `df_type=DFType.CELLS` or to tracer particles if
    `df_tpye=DFType.TRACERS`.

    Parameters
    ----------
    SimName : str
        Simulation name from '17_11', '09_18' or '37_11'.
    SnapNo : int
        Snapshot number (z=0 corresponds to SnapNo=127)
    MW_or_M31 : str, optional
        Choose one of the two main galaxies from 'MW' or 'M31' to center the
        sphere that will be considered for the dataframe.
    output_dir : str, optional
        Directory where the pickle containing the df will be saved. By default
        "results/dataframes/".
    df_type : DFType
        `DFType.CELLS` to read cell data, `DFType.TRACERS` to read tracer
        particle data.
    max_radius : float, optional
        Radius of the sphere required for the df in ckpc. By default 100.0.
    """
    if df_type == DFType.CELLS:
        return _make_dataframe_cells(SimName, SnapNo, MW_or_M31, max_radius)
    if df_type == DFType.TRACERS:
        return _make_dataframe_tracers(SimName, SnapNo, MW_or_M31, max_radius)
    raise ValueError(
        f"{df_type} can only be `DFType.CELLS` or `DFType.TRACERS`.")
