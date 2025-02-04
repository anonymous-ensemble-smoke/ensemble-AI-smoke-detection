#!/bin/bash

for seed in {1..12}; do
    sbatch --export=EXP_NUM=1,SEED=$seed --output=logs/exp1_seed$seed.log --job-name=exp1_seed$seed run_model.script
done
