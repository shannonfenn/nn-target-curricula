#!/bin/bash
#
#PBS -l select=1:ncpus=2:mem=4GB

nn_run_experiments.py ${EXP_FILE}
