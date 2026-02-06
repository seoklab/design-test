#!/bin/sh
#SBATCH -J msa_pep2
#SBATCH -p normal.q
#SBATCH --nodes=1
#SBATCH -c 6
#SBATCH -o ./log_msa_pep2.log
#SBATCH -e ./log_msa_pep2.log
#SBATCH --nice=1000
af3 --norun_inference --json_path=./PepBinder2_msa.json --output_dir=./
