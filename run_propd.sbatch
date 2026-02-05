#!/bin/bash
#
#SBATCH --job-name=propr_single
#SBATCH --output=/users/cn/caraiz/propr_new/logs/propr_single%j.out  # stdout log (%j = job ID)
#SBATCH --error=/users/cn/caraiz/propr_new/logs/propr_single%j.err   # stderr log
#SBATCH --time=02:00:00                # max runtime (HH:MM:SS)
#SBATCH --mem=200G                      # total memory
#SBATCH --partition=gpu           # queue/partition name
#SBATCH --gres=gpu:1g.20gb:1
#SBATCH --cpus-per-task=80
#SBATCH --array=1-1






### === DRY‑RUN SWITCH ===
# Set to "true" for a dry run (no R script execution),
# or "false" to actually run.
DRY_RUN=false

GENES=(
    ACAT2
)

TOTAL_GENES=${#GENES[@]}
echo "Total genes: $TOTAL_GENES"
N_CHUNKS=$(( (TOTAL_GENES + 3) / 4 ))
echo "Total chunks: $N_CHUNKS"



gene_idx=$(( SLURM_ARRAY_TASK_ID - 1))
echo "Processing gene index: $gene_idx"

if (( gene_idx >= TOTAL_GENES )); then
  echo "[DRY_RUN=$DRY_RUN] Proc $SLURM_ARRAY_TASK_ID: no gene → exit"
  exit 0
fi

TARGET_GENE="${GENES[$gene_idx]}"
export TARGET_GENE
export RETICULATE_PYTHON="/users/cn/caraiz/VCC/.venv/bin/python"

echo "=== Proc $SLURM_ARRAY_TASK_ID ==="
echo " GPU: $CUDA_VISIBLE_DEVICES   Gene: $TARGET_GENE"
echo $RETICULATE_PYTHON

module load R
module load CUDA/12.4.0
# (optional) load any required modules:
module load Python/3.12.4-GCCcore-13.2.0

Rscript propr_single.R