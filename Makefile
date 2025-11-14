SHELL := /bin/bash

.PHONY: all create-plots

all: create-plots

create-plots:
	python ./visualisation/disc_size.py --config 01
	python ./visualisation/accretion_cells.py --config 01
	python ./visualisation/accretion_tracers.py --config 01
	python ./visualisation/environment.py --config 01
	python ./visualisation/prop_comparison.py --config 01
	python ./visualisation/time_series.py --config 01
