#!/bin/bash

# This export reserves for kine computations 90 percent of available GPU space
export XLA_PYTHON_CLIENT_MEM_FRACTION=.9

# taskset --cpu-list 0 is only required to restrict the CPU core.
# You may not need it.
taskset --cpu-list 0 python dynamic_imaging_example.py -obs ../data/some_dataset.uvfits -yml ./dynamic_imaging_params.yml