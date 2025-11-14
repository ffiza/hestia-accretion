SHELL := /bin/bash

CONFIG ?= 01

.PHONY: all create-plots

all: create-plots

create-plots:
	python ./visualisation/disc_size.py --config $(CONFIG)
	python ./visualisation/accretion_cells.py --config $(CONFIG)
	python ./visualisation/accretion_tracers.py --config $(CONFIG)
	python ./visualisation/environment.py --config $(CONFIG)
	python ./visualisation/prop_comparison.py --config $(CONFIG)
	python ./visualisation/time_series.py --config $(CONFIG)
