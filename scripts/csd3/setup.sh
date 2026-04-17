#!/bin/bash
# One-time setup script to run on CSD3 after first SSH login.
#
# Run on CSD3:
#   scp scripts/csd3/setup.sh rj416@login.hpc.cam.ac.uk:/tmp/
#   ssh rj416@login.hpc.cam.ac.uk
#   bash /tmp/setup.sh
#
# What it does:
#   1. Clone abstraction repo into hpc-work
#   2. Load python/3.10 module
#   3. Create .venv and pip install -e .
#   4. Print next steps

set -euo pipefail

HPC_WORK="/rds/user/$USER/hpc-work"
PROJECT_DIR="$HPC_WORK/abstraction"
REPO_URL="${REPO_URL:-https://github.com/quadrismegistus/abstraction.git}"  # edit if different

echo "== CSD3 setup for abstraction =="
echo "User:    $USER"
echo "Host:    $(hostname)"
echo "Target:  $PROJECT_DIR"
echo

# 1. Check quota / budget
echo "-- Quota --"
quota -s 2>/dev/null || echo "(quota command not available)"
echo
echo "-- CPU budget --"
mybalance 2>/dev/null || echo "(mybalance not available; ask the helpdesk if this is missing)"
echo

# 2. Load python module
echo "-- Loading modules --"
module load python/3.10
python --version
echo

# 3. Clone repo
if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "-- Cloning repo --"
    mkdir -p "$HPC_WORK"
    cd "$HPC_WORK"
    git clone "$REPO_URL" abstraction
else
    echo "-- Repo already present, pulling latest --"
    cd "$PROJECT_DIR"
    git pull --ff-only
fi
cd "$PROJECT_DIR"

# 4. Create venv and install
if [[ ! -d .venv ]]; then
    echo "-- Creating venv --"
    python -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -e .

# 5. Create logs dir, data dir
mkdir -p logs data/models data/models_fr data/models_de

echo
echo "== Setup complete =="
echo
echo "Next steps:"
echo "  1. Note your CPU-account code from \`mybalance\` above (e.g. RJ416-SL3-CPU)"
echo "  2. Edit scripts/csd3/train_w2v.sh line with '#SBATCH -A' to set your account"
echo "  3. From your Mac, push skipgrams: ./scripts/csd3/sync_up.sh data/models"
echo "  4. On CSD3, submit jobs: see scripts/csd3/CSD3.md"
