#!/bin/bash
# Push skipgrams (and optionally models dir) from local → CSD3.
#
# Typical use: after generating skipgrams locally, push just the .txt.gz files
# (excludes already-trained run_* dirs to avoid overwriting in-progress work).
#
# Usage:
#   scripts/csd3/sync_up.sh                            # pushes data/models/
#   scripts/csd3/sync_up.sh data/models_fr             # pushes French models
#   scripts/csd3/sync_up.sh data/models --include-runs # also pushes run_* dirs
#
# Env:
#   CSD3_USER=rj416           your CRSid
#   CSD3_HOST=login.hpc.cam.ac.uk
#   CSD3_DIR=/rds/user/$CSD3_USER/hpc-work/abstraction

set -euo pipefail

LOCAL_DIR="${1:-data/models}"
INCLUDE_RUNS=0
[[ "${2:-}" == "--include-runs" ]] && INCLUDE_RUNS=1

CSD3_USER="${CSD3_USER:-rj416}"
CSD3_HOST="${CSD3_HOST:-login.hpc.cam.ac.uk}"
CSD3_DIR="${CSD3_DIR:-/rds/user/$CSD3_USER/hpc-work/abstraction}"

if [[ ! -d "$LOCAL_DIR" ]]; then
    echo "error: $LOCAL_DIR does not exist" >&2
    exit 1
fi

EXCLUDES=()
if [[ "$INCLUDE_RUNS" == "0" ]]; then
    EXCLUDES+=(--exclude 'run_*/')
fi

echo "pushing $LOCAL_DIR → $CSD3_USER@$CSD3_HOST:$CSD3_DIR/$LOCAL_DIR"
rsync -avP --partial --human-readable \
    "${EXCLUDES[@]}" \
    "$LOCAL_DIR/" \
    "$CSD3_USER@$CSD3_HOST:$CSD3_DIR/$LOCAL_DIR/"
