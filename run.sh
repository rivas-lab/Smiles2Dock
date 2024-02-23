#!/bin/bash

input_dir="./input"
output_dir="./slurm" 

for sub_dir in ${input_dir}/*; do
    if [ -d "${sub_dir}" ]; then
        # Extract the name of the sub-directory
        sub_dir_name=$(basename "${sub_dir}")
        # Specify the output file path, incorporating the sub-directory name
        sbatch -t 24:00:00 -p normal,mrivas,owners -N 1 --mem=16Gb --cpus-per-task=4 --output="${output_dir}/${sub_dir_name}_slurm-%j.out" --wrap="python run.py ${sub_dir}"
    fi
done