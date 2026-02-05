#!/bin/bash
#SBATCH --job-name=propr
#SBATCH --partition=gpu
# pick ONE of these gres lines based on where you want to run:
#SBATCH --gres=gpu:1g.20gb:1              # MIG 20GB slice on genoa64-09
# #SBATCH --gres=gpu:nvidia_rtx_a4000:1   # full A4000 (16GB) on gpu02
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=125G
# logs per array task:
#SBATCH --output=/users/cn/caraiz/propr_new/logs/%x_%A_%a.out
#SBATCH --error=/users/cn/caraiz/propr_new/logs/%x_%A_%a.err
#SBATCH --array=1-150%10   # adjust max parallel tasks with %N

set -euo pipefail

# --- map array index -> gene name ---
mapfile -t GENES < <(awk 'NF && $0 !~ /^#/' /users/cn/caraiz/propr_new/data/genes.txt)
idx=$(( SLURM_ARRAY_TASK_ID - 1 ))
if (( idx < 0 || idx >= ${#GENES[@]} )); then
  echo "No gene for task $SLURM_ARRAY_TASK_ID"; exit 0
fi
export TARGET_GENE="${GENES[$idx]}"
#export TARGET_GENE="CASP3"

echo "$(date) task=$SLURM_ARRAY_TASK_ID host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES gene=$TARGET_GENE"

# --- env for your R run ---
module load R
module load CUDA/12.4.0
export R_LIBS_USER="$HOME/Rlibs/4.4"
export RETICULATE_PYTHON="/users/cn/caraiz/VCC/.venv/bin/python"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

# Run your script (reads TARGET_GENE via Sys.getenv)
Rscript /users/cn/caraiz/propr_new/propr_single.R
