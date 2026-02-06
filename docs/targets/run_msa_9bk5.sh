#!/bin/sh
#SBATCH -J msa_9bk5
#SBATCH -p normal.q
#SBATCH --nodes=1
#SBATCH -c 6
#SBATCH -o ./log_msa_9bk5.log
#SBATCH -e ./log_msa_9bk5.log
#SBATCH --nice=1000
af3 --norun_inference --json_path=./target_9bk5.json --output_dir=./
