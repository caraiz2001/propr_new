#!/bin/bash
#SBATCH --job-name=gsea_pair2gene
#SBATCH --output=/users/cn/caraiz/propr_new/logs/%x_%A_%a.out
#SBATCH --error=/users/cn/caraiz/propr_new/logs/%x_%A_%a.err
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=genoa64           # queue/partition name
#SBATCH --array=1-150%25               # array job with 1000 tasks


set -euo pipefail

mapfile -t GENES < <(awk 'NF && $0 !~ /^#/' /users/cn/caraiz/propr_new/data/genes_sorted.txt)
#mapfile -t GENES < <(awk 'NF && $0 !~ /^#/' /db/VCC/vcc_data/validation_genes.txt)
idx=$(( SLURM_ARRAY_TASK_ID - 1 ))
if (( idx < 0 || idx >= ${#GENES[@]} )); then
  echo "No gene for task $SLURM_ARRAY_TASK_ID"; exit 0
fi
export TARGET_GENE="${GENES[$idx]}"
#export TARGET_GENE="MED12"

# (optional) load any required modules:
module load Python/3.12.4-GCCcore-13.2.0

echo "$(date) start gene=$TARGET_GENE on $(hostname)"

# Run the python script
uv run gsea_pair2gene.py