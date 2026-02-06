#!/bin/sh
#SBATCH -J msa_pep1
#SBATCH -p normal.q
#SBATCH --nodes=1
#SBATCH -c 6
#SBATCH -o ./log_msa_pep1.log
#SBATCH -e ./log_msa_pep1.log
#SBATCH --nice=1000
af3 --norun_inference --json_path=./PepBinder1_msa.json --output_dir=./
