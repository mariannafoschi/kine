#!/bin/bash

# Run from the scripts/ folder. The results of each run are saved in the folder
# given by -out, which is created if it does not exist.


# Single dataset example
python example_static_imaging.py -obs ../data/eht_m87_2017-04-10.uvfits -yml ../parameters/params_static_imaging.yml -out ../out/static_m87_2017-04-10

# Multiple datasets example
python example_multiepoch_imaging.py -obs ../data/ -yml ../parameters/params_multiepoch_imaging.yml -out ../out/multiepoch_m87_2017
