import numpy as np
import healpy as hp
import yaml
import argparse
import pandas as pd
import json
from scipy.signal import savgol_filter

from hestia.dataframe import make_dataframe
from hestia.df_type import DFType
from hestia.tools import weighted_percentile
from hestia.settings import Settings
from hestia.cosmology import Cosmology

GLOBAL_CONFIG = yaml.safe_load(open("configs/global.yml"))



def radial_density_healpix(
    simulation,
    config,
    SnapNo=127,
    galaxy='MW',
    rmax=1000,
    nbins=100,
    nside=16,
    center=np.zeros(3),
    cone_angle=10,
    rmin=0.0
):
    """
    Perfil radial de densidad con HEALPix.
    Bins radiales lineales.
    Devuelve densidad media y std angular.
    Opción de restringir el promedio a un cono.

    Parameters
    ----------
    pos : (N,3) array
        Posiciones cartesianas
    mass : (N,) array
        Masas
    rmax : float
        Radio máximo
    nbins : int
        Número de bins radiales (lineales)
    nside : int
        NSIDE de HEALPix
    center : (3,) array
        Centro del halo
    cone_axis : (3,) array or None
        Eje del cono (vector unitario)
    cone_angle : float or None
        Ángulo de apertura del cono (radianes)
    rmin : float
        Radio mínimo (default: 0)

    Returns
    -------
    r : (nbins,) array
        Centros de los bins radiales
    rho_med : (nbins,) array
        Densidad mediana
    rho_p16 : (nbins,) array
        percentil 16
    rho_p84 : (nbins,) array
        percentil 84
    """


    data = {"SnapshotNumber": SnapNo,
            "radii": np.linspace(rmin, rmax, nbins),
            "rho_p16": [np.nan] * nbins,
            "rho_med": [np.nan] * nbins,
            "rho_p84": [np.nan] * nbins,
            "rho_med_cone": [np.nan] * nbins
            }
    
    df = make_dataframe(simulation, SnapNo, galaxy, config, DFType.CELLS, max_radius=rmax)

    pos = df[["xPosition_ckpc", "yPosition_ckpc", "zPosition_ckpc"]]
    mass = df[["Mass_Msun"]]
    is_gas = df["ParticleType"] == 0

    pos, mass = pos[is_gas], mass[is_gas]

    # Switch to AHF for centering:
    M31_MW_AHF_IDs = {
                    '09_18': [127000000000002, 127000000000003],
                    '17_11': [127000000000002, 127000000000003],
                    '37_11': [127000000000001, 127000000000002],
                    'i_09_10':       [127000000000003, 127000000000005], 
                    'i_09_16':       [127000000000003, 127000000000005], 
                    'i_09_17':       [127000000000002, 127000000000004], 
                    'i_09_18':       [127000000000003, 127000000000004], 
                    'i_09_19':       [127000000000002, 127000000000005], 
                    'i_17_10':       [127000000000002, 127000000000003], 
                    'i_17_11':       [127000000000002, 127000000000003], 
                    'i_17_13':       [127000000000002, 127000000000003], 
                    'i_17_14':       [127000000000002, 127000000000003], 
                    'i_37_11':       [127000000000002, 127000000000001], 
                    'i_37_12':       [127000000000001, 127000000000002], 
                    'i_37_16':       [127000000000001, 127000000000002], 
                    'i_37_17':       [127000000000001, 127000000000002]
                    }

    if simulation[0] == 'i':
        SimulationDirectory = '/store/clues/HESTIA/RE_SIMS/4096/GAL_FOR/{}/output'.format(simulation.lstrip('i_'))
        AHF_path = '/store/clues/HESTIA/RE_SIMS/4096/GAL_FOR/{}/AHF_output/'.format(simulation.lstrip('i_'))
    else:
        SimulationDirectory = '/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/{}/output_2x2.5Mpc'.format(simulation)
        AHF_path = '/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/{}/AHF_output_2x2.5Mpc/'.format(simulation)

    AHF_M31_ID = M31_MW_AHF_IDs[simulation][0]
    AHF_MW_ID = M31_MW_AHF_IDs[simulation][1]

    if simulation[0] == 'i':
        AHF_filename_M31 = 'HESTIA_100Mpc_4096_{}.127_halo_{}.dat'.format(simulation.lstrip('i_'), AHF_M31_ID)
        AHF_filename_MW = 'HESTIA_100Mpc_4096_{}.127_halo_{}.dat'.format(simulation.lstrip('i_'), AHF_MW_ID)
    else:
        AHF_filename_M31 = 'HESTIA_100Mpc_8192_{}.127_halo_{}.dat'.format(simulation, AHF_M31_ID)
        AHF_filename_MW = 'HESTIA_100Mpc_8192_{}.127_halo_{}.dat'.format(simulation, AHF_MW_ID)


    AHF_table_M31 = np.loadtxt(AHF_path + AHF_filename_M31)
    AHF_table_MW = np.loadtxt(AHF_path + AHF_filename_MW)
    MW_pos  = AHF_table_MW[ AHF_table_MW[:, 1] // 1000000000000 == SnapNo ][0][6:9] / Cosmology.SMALL_HUBBLE_CONST
    M31_pos = AHF_table_M31[ AHF_table_M31[:, 1] // 1000000000000 == SnapNo ][0][6:9] / Cosmology.SMALL_HUBBLE_CONST

    # pos -= MW_pos
    cone_axis = M31_pos - MW_pos

    # ------------------
    # bins radiales lineales
    # ------------------
    r_edges = np.linspace(rmin, rmax, nbins + 1)
    r = 0.5 * (r_edges[:-1] + r_edges[1:])

    # ------------------
    # posiciones relativas
    # ------------------
    x = pos - center
    x = x.to_numpy()
    rad = np.linalg.norm(x, axis=1)

    theta = np.arccos(x[:,2] / rad)
    phi   = np.arctan2(x[:,1], x[:,0]) % (2*np.pi)

    pix = hp.ang2pix(nside, theta, phi)

    Npix = hp.nside2npix(nside)
    rho_pix = np.full((Npix, nbins), np.nan)

    omega_pix = hp.nside2pixarea(nside)

    # ------------------
    # selección angular (cono)
    # ------------------
    vec_pix = np.array(
        hp.pix2vec(nside, np.arange(Npix))
    ).T

    if cone_axis is not None and cone_angle is not None:
        cone_axis = np.asarray(cone_axis)
        cone_axis /= np.linalg.norm(cone_axis)

        mu = vec_pix @ cone_axis
        pix_sel_cone = mu >= np.cos(np.deg2rad(cone_angle))
    else:
        pix_sel_cone = None

    pix_sel_all = np.ones(Npix, dtype=bool)


    # ------------------
    # masa por pixel y bin
    # ------------------
    for p in range(Npix):

        m_pix = pix == p
        if not np.any(m_pix):
            continue

        rp = rad[m_pix]
        mp = mass[m_pix]

        for i in range(nbins):
            sel = (rp >= r_edges[i]) & (rp < r_edges[i+1])
            if not np.any(sel):
                continue

            M = mp[sel].sum()
            V = (omega_pix / (4*np.pi)) * (4*np.pi/3) * (
                r_edges[i+1]**3 - r_edges[i]**3
            )

            rho_pix[p, i] = M / V



    rho_all = rho_pix[pix_sel_all]
    rho_cone = rho_pix[pix_sel_cone]

    rho_p16 = np.nanpercentile(rho_all, 16, axis=0)
    rho_med = np.nanpercentile(rho_all, 50, axis=0)
    rho_p84 = np.nanpercentile(rho_all, 84, axis=0)
    rho_med_cone = np.nanpercentile(rho_cone, 50, axis=0)

    # Add data to dictionary
    data["SnapshotNumber"] = SnapNo
    data["radii"] = r.tolist()
    data["rho_p16"] = rho_p16.tolist()
    data["rho_med"] = rho_med.tolist()
    data["rho_p84"] = rho_p84.tolist()
    data["rho_med_cone"] = rho_med_cone.tolist()

    # Save dictionary
    path = f"results/{simulation}_{galaxy}/" \
        + f"density_profile_{config['RUN_CODE']}.json"
    with open(path, "w") as f:
        json.dump(data, f)


    return r, rho_med, rho_p16, rho_p84, rho_med_cone



def main():
    # Get arguments from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = yaml.safe_load(open(f"configs/{args.config}.yml"))



    for simulation in Settings.SIMULATIONS:
        radial_density_healpix(simulation=simulation, galaxy="MW", config=config)

if __name__ == "__main__":
    main()
