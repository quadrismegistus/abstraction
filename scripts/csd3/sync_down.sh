#!/bin/bash
# Pull trained models (run_*/model.bin and friends) from CSD3 → local.
#
# Usage:
#   scripts/csd3/sync_down.sh                    # pulls data/models/
#   scripts/csd3/sync_down.sh data/models_fr     # pulls French models
#
# Env:
#   CSD3_USER=rj416           your CRSid
#   CSD3_HOST=login.hpc.cam.ac.uk
#   CSD3_DIR=/rds/user/$CSD3_USER/hpc-work/abstraction

set -euo pipefail

LOCAL_DIR="${1:-data/models}"
CSD3_USER="${CSD3_USER:-rj416}"
CSD3_HOST="${CSD3_HOST:-login.hpc.cam.ac.uk}"
CSD3_DIR="${CSD3_DIR:-/rds/user/$CSD3_USER/hpc-work/abstraction}"

mkdir -p "$LOCAL_DIR"

echo "pulling $CSD3_USER@$CSD3_HOST:$CSD3_DIR/$LOCAL_DIR → $LOCAL_DIR"
rsync -avP --partial --human-readable \
    --include='*/' \
    --include='run_*/**' \
    --exclude='skipgrams.txt.gz' \
    --exclude='*' \
    "$CSD3_USER@$CSD3_HOST:$CSD3_DIR/$LOCAL_DIR/" \
    "$LOCAL_DIR/"
