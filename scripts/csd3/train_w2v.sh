#!/bin/bash
# SLURM array job: train Word2Vec on one skipgrams file per array task.
#
# Submit from CSD3 after generating the skipgrams manifest:
#   python scripts/csd3/list_skipgrams.py data/models --runs 5 --incomplete-only > manifest.txt
#   sbatch --array=1-$(wc -l < manifest.txt) scripts/csd3/train_w2v.sh manifest.txt
#
# Or train a single index to test:
#   sbatch --array=1 scripts/csd3/train_w2v.sh manifest.txt
#
# Each task reads the Nth line from manifest.txt (N = $SLURM_ARRAY_TASK_ID)
# and runs `abstraction train-model <skipgrams> --runs N`. train-model is
# idempotent — it skips runs whose model.bin already exists.

#SBATCH -J abs-w2v
#SBATCH -A HEUSER-SL3-CPU
#SBATCH -p icelake                    # 76 cores/node; use icelake-himem if OOM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=ryan.heuser@gmail.com
#SBATCH --output=logs/%x-%A_%a.out    # %A=job id, %a=array task id

set -euo pipefail

MANIFEST="${1:?usage: sbatch --array=1-N $0 manifest.txt}"
RUNS="${RUNS:-5}"
WORKERS="${SLURM_CPUS_PER_TASK:-32}"

mkdir -p logs

# Pick the Nth skipgrams file from the manifest
SKIPGRAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
if [[ -z "$SKIPGRAMS" ]]; then
    echo "error: no line $SLURM_ARRAY_TASK_ID in $MANIFEST" >&2
    exit 1
fi

echo "[$(date -Iseconds)] task=$SLURM_ARRAY_TASK_ID host=$(hostname) file=$SKIPGRAMS"

cd "${PROJECT_DIR:-/rds/user/$USER/hpc-work/abstraction}"

# Activate env (adjust if you use conda or a different venv path)
source .venv/bin/activate

# Train. gensim uses $WORKERS threads; train-model skips runs with existing model.bin.
abstraction train-model "$SKIPGRAMS" --runs "$RUNS" --workers "$WORKERS" -v

echo "[$(date -Iseconds)] done task=$SLURM_ARRAY_TASK_ID"
