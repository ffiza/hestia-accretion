<div align="center">
    <h1>Gas Accretion on Local Group Galaxy Simulations</h1>
</div>

<p align="center">
    <a href="https://www.python.org/"><img src="https://forthebadge.com/images/badges/made-with-python.svg"></a>
</p>

Code written for an ongoing analysis of the gas accretion rates onto the disc
of Local Groyp galaxies from [the Hestia project](https://hestia.aip.de/).

# Description of Configuration Variables

This sections contains a description of each configuration variable found in
`configs/`.


| Name | Type | Unit | Description |
|:-----|:----:|:----:|:------------|
| `RUN_CODE` | `str` | - | The code of this configuration. |
| `DISC_SIZE_SPHERICAL_CUT_CKPC` | `float` | $\mathrm{ckpc}$ | A spherical cut for the disc size calculation. All particles beyond this radius will be ignored when calculating the disc size.  |
| `DISC_SIZE_HEIGHT_CUT_CKPC` | `float` | $\mathrm{ckpc}$ | A height cut for the disc size calculation. All particles beyond this $z$ coordinate will be ignored when calculating the disc size. |
| `DISC_ENCLOSED_MASS_PERCENTILE` | `int` | - | The percentile to use when calculating the enclosed mass fraction for the disc size. For example, if `DISC_ENCLOSED_MASS_PERCENTILE = 90`, the radius and height of the disc will enclose 90% of the total stellar mass of the subhalo. |
| `VIRIAL_RADIUS_FRACTION` | `float` | $1$ | The fraction of the virial radius to use as disc size, if needed, for times smaller than `VIRIAL_RADIUS_TIME_THRESHOLD_GYR`. |
| `VIRIAL_RADIUS_TIME_THRESHOLD_GYR` | `float` | $\mathrm{Gyr}$ | The time before which the disc radius will be replaced by a fraction of the virial radius if needed. |
| `DISC_SIZE_SMOOTHING_WINDOW_LENGTH` | `int` | - | The window length to use in the Savitzky-Golay filter in the disc size calculation. |
| `DISC_SIZE_SMOOTHING_POLYORDER` | `int` | - | The polynomial order to use in the Savitzky-Golay filter in the disc size calculation. |
| `SMOOTHING_WINDOW_LENGTH` | `int` | - | The window length for the Savitzky-Golay filter for smoothing. |
| `SMOOTHING_POLYORDER` | `int` | - | The polynomial order for the Savitzky-Golay filter for smoothing. |
| `SUBHALO_VELOCITY_DISTANCE_CKPC` | `float` | $\mathrm{ckpc}$ | The spherical radius in ckpc to calculate the velocity of the main subhalo. |
| `ROTATION_MATRIX_DISTANCE_CKPC` | `float` | $\mathrm{ckpc}$ | The spherical radius to calculate the orientation matrix. |
| `REPOSITORY_NAME` | `str` | - | The name of the repository. |
| `EREBOS_SIMULATION_PATH` | `str` | - | The path of the snapshot files in the Erebos system. |
| `GAS_PARTICLE_TYPE` | `int` | - | The type of gas cells. |
| `DARK_MATTER_PARTICLE_TYPE` | `int` | - | The type of dark matter particles. |
| `STAR_PARTICLE_TYPE` | `int` | - | The type of star particles. |
| `BG_PARTICLE_TYPE` | `int` | - | The type of black hole particles. |
| `TRACER_PARTICLE_TYPE` | `int` | - | The type of tracer particles. |
| `SMALL_HUBBLE_CONST` | `float` | $1$ | The "small" Hubble constant. To get the Hubble constant, multiply by $100~\mathrm{km} ~ \mathrm{s}^{-1} ~ \mathrm{Mpc}^{-1}$. |
| `HUBBLE_CONST` | `float` | $\mathrm{km} ~ \mathrm{s}^{-1} ~ \mathrm{Mpc}^{-1}$ | The Hubble constant. |
| `OMEGA_LAMBDA` | `float` | $1$ | The dark energy density (relative to the critical density of the universe). |
| `OMEGA_MATTER` | `float` | $1$ | The matter density (relative to the critical density of the universe). |
| `OMEGA_BARYONS` | `float` | $1$ | The baryon density (relative to the critical density of the universe). |
| `OMEGA_0` | `float` | $1$ | The total matter density (relative to the critical density of the universe). |
| `FIRST_SNAPSHOT` | `int` | - | The first snapshot to analyze. Ignore all the previous snapshots. This is useful because the first snapshots may not have a main halo/subhalo properly identified. |
| `N_SNAPSHOTS`| `int` | - | The total number of snapshots. |
| `N_PROCESSES` | `int` | - | The number of processes for parallel calculations.