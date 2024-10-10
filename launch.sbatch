#!/bin/bash
#SBATCH --mail-user=mailto:{xjyin@pku.edu.cn}
#SBATCH --mail-type=ALL
#SBATCH --time=2-00:00:00
#SBATCH --nodes 8
#SBATCH --gpus-per-node=8
#SBATCH --job-name rdclm_2b_01
#SBATCH --partition compute

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

echo $MASTER_ADDR

# bash neo/scripts/rdclm_2b_01.sh
bash neo/scripts/rdclm_7b_01.sh