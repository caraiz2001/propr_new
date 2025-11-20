#!/bin/bash
#SBATCH --job-name=pair2gene
#SBATCH --output=/users/cn/caraiz/VCC/logs/%x_%A_%a.out
#SBATCH --error=/users/cn/caraiz/VCC/logs/%x_%A_%a.err
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=genoa64           # queue/partition name
#SBATCH --array=0-1   # adjust max parallel tasks with %N


set -euo pipefail

# --- map array index -> gene name ---
mapfile -t GENES < <(awk 'NF && $0 !~ /^#/' /users/cn/caraiz/propr_new/data/missing_genes.txt)
#mapfile -t GENES < <(awk 'NF && $0 !~ /^#/' /db/VCC/vcc_data/validation_genes.txt)
idx=$(( SLURM_ARRAY_TASK_ID - 1 ))
if (( idx < 0 || idx >= ${#GENES[@]} )); then
  echo "No gene for task $SLURM_ARRAY_TASK_ID"; exit 0
fi
export TARGET_GENE="${GENES[$idx]}"

# (optional) load any required modules:
module load Python/3.12.4-GCCcore-13.2.0

echo "$(date) start gene=$TARGET_GENE on $(hostname)"

# Run the python script
uv run pairwise_to_genewise.py