#!/bin/bash

input_dir="./input"

for sub_dir in ${input_dir}/*; do
    if [ -d "${sub_dir}" ]; then
        sbatch -t 24:00:00 -p normal,mrivas,owners -N 1 --mem=4Gb --wrap="python run.py ${sub_dir}"
    fi
done