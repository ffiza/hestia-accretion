import sys
import time
import numpy
import TrackGalaxy
import numpy as np
import pandas as pd
import astropy

from functions import PCA_matrix


h=0.677
OmegaLambda = .682
OmegaMatter = .27
OmegaBaryons = .048
cosmo = astropy.cosmology.FlatLambdaCDM(H0=h*100, Om0=OmegaMatter+OmegaBaryons)

SimName = 'Hestia17-11'
SnapNo = 127

if SimName == 'Hestia17-11':
    SubhaloNumberMW = 1#These numbers come from cross-correlating with /z/nil/codes/HESTIA/FIND_LG/LGs_8192_GAL_FOR.txt andArepo's SUBFIND.
    SubhaloNumberM31 = 0
    SimulationDirectory = '/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/17_11/output_2x2.5Mpc/'
elif SimName == 'Hestia09-18':
    SubhaloNumberMW = 3911
    SubhaloNumberM31 = 2608
    SimulationDirectory = '/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/09_18/output_2x2.5Mpc/'
elif SimName == 'Hestia37-11':
    SubhaloNumberMW = 920
    SubhaloNumberM31 = 0
    SimulationDirectory = '/store/clues/HESTIA/RE_SIMS/8192/GAL_FOR/37_11/output_2x2.5Mpc/'
else:
    print( 'SimName',SimName,'is not properly set, try again!. exiting.')
    sys.exit()


print( 'Initialising TrackGalaxy class')
T = TrackGalaxy.TrackGalaxy(numpy.array([SnapNo]),SimName,Dir = SimulationDirectory,MultipleSnaps=True) #Imports TrackGalaxy module from TrackGalaxy script
SnapTime = T.SnapTimes[0]#SnapTime is the scale factor
Redshift = 1.0/SnapTime-1
SnapTime_Gyr = cosmo.age(Redshift).value

#Read in position, SFR, stellar mass, gas mass, dark matter mass of all the galaxies (and subhalos) from the simulation
GroupCatalog = T.GetGroups(SnapNo,Attrs=['/Subhalo/SubhaloPos'])
#we get the subhalo center
SubhaloPos = 1000*GroupCatalog['/Subhalo/SubhaloPos']/h # in ckpc

pos_MW = SubhaloPos[SubhaloNumberMW]




print( 'We are now reading in gas cells')
tstartread = time.time()
Gas_Attrs = T.GetParticles(SnapNo, Type=0, Attrs=['Coordinates','Masses','Velocities','ParticleIDs'])
print( 'We finished reading data, it took (sec)',time.time()-tstartread)
GasPos = 1000*Gas_Attrs['Coordinates'] / h #ckpc
GasMass = Gas_Attrs['Masses'] * 1e10 / h # Msun
GasVel = Gas_Attrs['Velocities']*numpy.sqrt(SnapTime)#km/s
GasIDs = Gas_Attrs['ParticleIDs']

Star_Attrs = T.GetParticles(SnapNo, Type=4, Attrs=['Coordinates', 'Masses', 'ParticleIDs', 'GFM_StellarFormationTime'])
StarPos = 1000*Star_Attrs['Coordinates'] / h # ckpc
StarMass = Star_Attrs['Masses'] * 1e10 / h # Msun
StarIDs = Star_Attrs['ParticleIDs']
StarBirths = Star_Attrs['GFM_StellarFormationTime'] # Scalefactor
StarBirths_z = 1/StarBirths - 1
StarBirths_Gyr = cosmo.age(StarBirths_z).value

# We keep only particles within the MW halo
GasPos  -= pos_MW
StarPos -= pos_MW



RMAX = 100
index_of_nearby_gas   = numpy.where(GasPos[:,0]**2+GasPos[:,1]**2+GasPos[:,2]**2<RMAX**2)
index_of_nearby_stars = numpy.where(StarPos[:,0]**2+StarPos[:,1]**2+StarPos[:,2]**2<RMAX**2)


GasPos  = GasPos[index_of_nearby_gas]
GasMass  = GasMass[index_of_nearby_gas]
GasIDs = GasIDs[index_of_nearby_gas]

StarPos = StarPos[index_of_nearby_stars]
StarMass = StarMass[index_of_nearby_stars]
StarIDs = StarIDs[index_of_nearby_stars]
StarBirths_Gyr = StarBirths_Gyr[index_of_nearby_stars]


# We align positions with MW disk:
R = PCA_matrix(GasPos, GasVel, 15)
GasPos = np.dot(GasPos, R)
StarPos = np.dot(StarPos, R)

AllPos = np.concatenate((GasPos, StarPos))
AllMass = np.concatenate((GasMass, StarMass))
AllIDs = np.concatenate((GasIDs, StarIDs))
AllTypes = np.concatenate((np.zeros(np.size(GasIDs)), 4*np.ones(np.size(StarIDs))))
AllBirths = np.concatenate((np.nan*np.ones(np.size(GasIDs)), StarBirths_Gyr))

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
# df.to_csv('df_SnapNo{}_SnapTimeGyr{:.4f}_{}.csv'.format(SnapNo, SnapTime_Gyr, SimName), index=False)
df.to_pickle('df_SnapNo{}_SnapTimeGyr{:.4f}_{}.pkl'.format(SnapNo, SnapTime_Gyr, SimName))