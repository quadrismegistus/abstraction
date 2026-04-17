#!/bin/bash
# SLURM job: generate skipgrams from a corpus (pre-step before train_w2v.sh).
#
# Usage:
#   sbatch scripts/csd3/train_skipgrams.sh BLBooks
#   sbatch --export=FAST=1 scripts/csd3/train_skipgrams.sh BLBooks    # use --fast
#
# Note: this requires the corpus's freqs/ directory to be on CSD3 under
# $LLTK_DATA/corpora/<corpus>/. For most abstraction workflows, generate
# skipgrams locally (where the corpora live) and rsync them up — that's
# usually faster than shipping raw corpora to CSD3.

#SBATCH -J abs-skip
#SBATCH -A CHANGE_ME-SL3-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=ryan.heuser@gmail.com
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

CORPUS="${1:?usage: sbatch $0 <CorpusName>}"
WORKERS="${SLURM_CPUS_PER_TASK:-16}"
FAST_FLAG=""
[[ "${FAST:-0}" == "1" ]] && FAST_FLAG="--fast"

mkdir -p logs
cd "${PROJECT_DIR:-/rds/user/$USER/hpc-work/abstraction}"
source .venv/bin/activate

echo "[$(date -Iseconds)] corpus=$CORPUS workers=$WORKERS fast=${FAST:-0}"

abstraction train-skipgrams "$CORPUS" --workers "$WORKERS" --output-dir data/models $FAST_FLAG

echo "[$(date -Iseconds)] done"
