#!/bin/bash

# Check if a protein name is passed as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 protein_name"
    exit 1
fi

# Assign the first argument to a variable
protein_name="$1"

input_dir="./input"
output_dir="./slurm/${protein_name}"

# Create the output directory if it doesn't exist
mkdir -p "${output_dir}"

# Iterate over subdirectories in the input directory in reverse order
#for sub_dir in $(ls -d ${input_dir}/* | sort -r); do
for sub_dir in ${input_dir}/*; do
    if [ -d "${sub_dir}" ]; then
        # Extract the name of the sub-directory
        sub_dir_name=$(basename "${sub_dir}")
        # Specify the output file path, incorporating the sub-directory name
        sbatch -t 24:00:00 -p normal,mrivas,owners -N 1 --mem=16Gb --cpus-per-task=4 --output="${output_dir}/${sub_dir_name}_slurm-%j.out" --wrap="python run.py ${sub_dir} ${protein_name}"
    fi
done
