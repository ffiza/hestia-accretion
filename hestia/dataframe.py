import numpy
import numpy as np
import pandas as pd
import astropy
import yaml

import TrackGalaxy
from hestia.pca import PCA_matrix

GLOBAL_CONFIG = yaml.safe_load(open("configs/global.yml"))


def make_dataframe(SimName: str, SnapNo: int, MW_or_M31: str = 'MW',
                   max_radius: float = 100.0):
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

    if SimName != "17_11":
        raise NotImplementedError("Only the mergers trees for simulation "
                                  "17-11 are available.")

    if MW_or_M31 not in ["MW", "M31"]:
        raise ValueError("Incorrect value for `MW_or_M31`. Can be `MW` "
                         "or `M31`.")

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
        /GLOBAL_CONFIG['SMALL_HUBBLE_CONST']  # Msun
    DMIDs = DM_Attrs['ParticleIDs']

    BH_Attrs = T.GetParticles(
        SnapNo, Type=5, Attrs=['Coordinates',
                               'Masses',
                               'ParticleIDs'])
    BHPos = 1000*BH_Attrs['Coordinates'] \
        / GLOBAL_CONFIG['SMALL_HUBBLE_CONST']  # ckpc
    BHMass = BH_Attrs['Masses'] * 1e10 \
        /GLOBAL_CONFIG['SMALL_HUBBLE_CONST']  # Msun
    BHIDs = BH_Attrs['ParticleIDs']

    # Subhalo center and velocity directly from the merger trees.
    # FIX: Currently directories for 17_11, change for other realisations.
    if SimName != "17_11":
        raise NotImplementedError("Check the following lines.")
    MergerTreeMW = np.loadtxt(
        "/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/17_11/MergerTrees/"
        "HESTIA_100Mpc_8192_17_11.127_halo_127000000000003.dat")
    MergerTreeM31 = np.loadtxt(
        "/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/17_11/MergerTrees/"
        "HESTIA_100Mpc_8192_17_11.127_halo_127000000000002.dat")
    MW_pos = MergerTreeMW[
        MergerTreeMW[:, 1] // 1000000000000 == SnapNo][0][6:9] \
        / GLOBAL_CONFIG["SMALL_HUBBLE_CONST"]
    MW_vel = MergerTreeMW[
        MergerTreeMW[:, 1] // 1000000000000 == SnapNo][0][9:12] \
        / GLOBAL_CONFIG["SMALL_HUBBLE_CONST"]
    M31_pos = MergerTreeM31[
        MergerTreeM31[:, 1] // 1000000000000 == SnapNo][0][6:9] \
        / GLOBAL_CONFIG["SMALL_HUBBLE_CONST"]
    M31_vel = MergerTreeM31[
        MergerTreeM31[:, 1] // 1000000000000 == SnapNo][0][9:12] \
        / GLOBAL_CONFIG["SMALL_HUBBLE_CONST"]

    # We keep only particles within the chosen halo
    if MW_or_M31 == 'MW':
        GasPos -= MW_pos
        StarPos -= MW_pos
        DMPos -= MW_pos
        BHPos -= MW_pos
        GasVel -= MW_vel
    elif MW_or_M31 == 'M31':
        GasPos -= M31_pos
        StarPos -= M31_pos
        DMPos -= M31_pos
        BHPos -= M31_pos
        GasVel -= M31_vel

    # Keep only particles within RMAX:
    index_of_nearby_gas = numpy.where(
        GasPos[:, 0]**2 + GasPos[:, 1]**2 + GasPos[:, 2]**2 < max_radius**2)
    index_of_nearby_stars = numpy.where(
        StarPos[:, 0]**2 + StarPos[:, 1]**2 + StarPos[:, 2]**2 < max_radius**2)
    index_of_nearby_DM = numpy.where(
        DMPos[:, 0]**2 + DMPos[:, 1]**2 + DMPos[:, 2]**2 < max_radius**2)
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

    BHPos = BHPos[index_of_nearby_BH]
    BHMass = BHMass[index_of_nearby_BH]
    BHIDs = BHIDs[index_of_nearby_BH]

    # We align positions with gas disk:
    R = PCA_matrix(GasPos, GasVel, 15)
    GasPos = np.dot(GasPos, R)
    StarPos = np.dot(StarPos, R)
    DMPos = np.dot(DMPos, R)
    BHPos = np.dot(BHPos, R)

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


if __name__ == "__main__":
    SimName = '17_11'
    SnapNo = 127
    output_dir = '../results/dataframes/'
    make_dataframe(SimName, SnapNo, output_dir=output_dir)
