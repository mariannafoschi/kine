#!/bin/bash

# Single dataset example
python example_static_imaging.py -obs ../data/eht_m87_2017-04-10.uvfits -yml ../parameters/params_static_imaging.yml

# Multiple datasets example
python example_multiepoch_imaging.py -obs ../data/ -yml ../parameters/params_multiepoch_imaging.yml