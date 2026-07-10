#!/bin/bash

# This export reserves for kine computations 90 percent of available GPU space
export XLA_PYTHON_CLIENT_MEM_FRACTION=.9

# taskset --cpu-list 0 is only required to restrict the CPU core.
# You may not need it.

# Single dataset example
taskset --cpu-list 0 python example_static_imaging.py -obs ../data/eht_m87_2017-04-10.uvfits -yml ../parameters/params_static_imaging.yml

# Multiple datasets example
taskset --cpu-list 0 python example_multiepoch_imaging.py -obs ../data/ -yml ../parameters/params_multiepoch_imaging.yml