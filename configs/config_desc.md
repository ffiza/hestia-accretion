`RUN_SUFFIX: int`

The suffix of this run to append to output files.

`DISC_SIZE_TYPE: str`

The type of disc size calculation.

- `"fixed"`:    The radius and height of the disc are constant and fixed in configurations.
- `"static"`:   The radius and height of the disc are constant but calculated in code with parameters set in configurations at the present.
- `"dynamic"`:  The radius and height of the disc are calculated time-by-time, with parameters set in configurations.

`DISC_RADIUS_CKPC: float`

The radius of the disc in ckpc. Only used if `DISC_SIZE_TYPE = "fixed"`.

`DISC_HEIGHT_CKPC: float`

The height of the disc in ckpc. The z-coordinates for the disc are 
`|z| <= DISC_HEIGHT_CKPC / 2`. Only used if
`DISC_SIZE_TYPE = "fixed"`.

`DISC_SIZE_SPHERICAL_CUT_CKPC: float`

A spherical cut for the disc size calculation. Only used if
`DISC_SIZE_TYPE = "static"` or `DISC_SIZE_TYPE = "dynamic"`. Can be set
to `-1.0` to ignore any cuts. All particles beyond this radius will be
ignored when calculating the disc size. 

`DISC_SIZE_HEIGHT_CUT_CKPC: float`

A height cut for the disc size calculation. Only used if
`DISC_SIZE_TYPE = "static"` or `DISC_SIZE_TYPE = "dynamic"`. Can be set
to `-1.0` to ignore any cuts. All particles beyond this $z$ coordinate will
be ignored when calculating the disc size. 

`DISC_ENCLOSED_MASS_PERCENTILE: int`

The percentile to use when calculating the enclosed mass fraction for the
disc size. For example, if `DISC_ENCLOSED_MASS_PERCENTILE = 90`, the
radius and height of the disc will enclose 90% of the total stellar mass
of the subhalo.

`SMOOTHING_WINDOW_LENGTH: int`

The window length for the Savitzky-Golay filter for smoothing.

`SMOOTHING_POLYORDER: int`

The polynomial order for the Savitzky-Golay filter for smoothing.

`SUBHALO_VELOCITY_DISTANCE_CKPC: float`

The spherical radius in ckpc to calculate the velocity of the main subhalo.

`ROTATION_MATRIX_DISTANCE_CKPC: float`

The spherical radius in ckpc to calculate the orientation matrix.

`REPOSITORY_NAME: str`

The name of the repository.

`EREBOS_SIMULATION_PATH: str`

The path of the snapshot files in the Erebos system.

`GAS_PARTICLE_TYPE: int`

The type of gas cells.

`DARK_MATTER_PARTICLE_TYPE: int`

The type of dark matter particles.

`STAR_PARTICLE_TYPE: int`

The type of star particles.

`BG_PARTICLE_TYPE: int`

The type of black hole particles.

`TRACER_PARTICLE_TYPE: int`

The type of tracer particles.

`SMALL_HUBBLE_CONST: float`

The "small" Hubble constant. To get the Hubble constant, multiply by
100 km/s/Mpc.

`HUBBLE_CONST: float`

The Hubble constant in km/s/Mpc.

`OMEGA_LAMBDA: float`

The dark energy density (relative to the critical density of the 
universe).

`OMEGA_MATTER: float`

The matter density (relative to the critical density of the 
universe).

`OMEGA_BARYONS: float`

The baryon density (relative to the critical density of the 
universe).

`OMEGA_0: float`

The total matter density (relative to the critical density of the 
universe).

`FIRST_SNAPSHOT: int`

The first snapshot to analyze. Ignore all the previous snapshots. This
is useful because the first snapshots may not have a main halo/subhalo
properly identified.

`N_SNAPSHOTS: int`

The total number of snapshots.

`N_PROCESSES: int`

The number of processes for parallel calculations.