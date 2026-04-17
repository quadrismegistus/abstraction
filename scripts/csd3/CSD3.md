# CSD3 cheat sheet (abstraction)

Everything you need to train W2V on Cambridge HPC without befriending SLURM.

## 0. One-time setup

```bash
# From your Mac, copy the setup script up and run it on CSD3
scp scripts/csd3/setup.sh rj416@login.hpc.cam.ac.uk:/tmp/
ssh rj416@login.hpc.cam.ac.uk
bash /tmp/setup.sh
```

The setup script clones the repo into `/rds/user/rj416/hpc-work/abstraction`, creates a venv, and installs the package. It also prints your quota and CPU-account budget — **note the account code** (something like `RJ416-SL3-CPU`) and edit it into the `#SBATCH -A` line of `train_w2v.sh` and `train_skipgrams.sh`.

SSH key tip — run on your Mac first so you're not typing passwords forever:
```bash
ssh-copy-id rj416@login.hpc.cam.ac.uk
```

## 1. Push skipgrams up

Generate skipgrams locally (where corpora live), push the .txt.gz files up:

```bash
# From Mac, from the abstraction/ dir
./scripts/csd3/sync_up.sh data/models        # English
./scripts/csd3/sync_up.sh data/models_fr     # French
./scripts/csd3/sync_up.sh data/models_de     # German
```

By default this excludes already-trained `run_*/` dirs. The skipgram `.txt.gz` files are GBs, not TBs — fine over home internet.

## 2. Submit training jobs on CSD3

```bash
ssh rj416@login.hpc.cam.ac.uk
cd /rds/user/rj416/hpc-work/abstraction

# Build a manifest of skipgrams files that still need runs
python scripts/csd3/list_skipgrams.py data/models --runs 5 --incomplete-only > manifest.txt
wc -l manifest.txt           # how many array tasks you'll submit

# Submit as an array. Each task trains 5 runs for one skipgrams file.
sbatch --array=1-$(wc -l < manifest.txt) scripts/csd3/train_w2v.sh manifest.txt
```

To test with one task first:
```bash
sbatch --array=1 scripts/csd3/train_w2v.sh manifest.txt
```

## 3. Watch it

```bash
squeue -u $USER                    # pending + running jobs
sacct -u $USER -X -S today         # today's job history with exit codes
tail -f logs/abs-w2v-*_1.out       # stream output of task 1
scancel <jobid>                    # kill a job
scancel -u $USER                   # kill all your jobs (use with care)
```

## 4. Pull trained models back

```bash
# From Mac
./scripts/csd3/sync_down.sh data/models
./scripts/csd3/sync_down.sh data/models_fr
```

Then locally regenerate vecnorms:
```bash
abstraction gen-vecnorms                   # English
# For French/German: use gen_vecnorms_fr() / gen_vecnorms_de() in Python
```

## 5. Partition cheat sheet

| Partition | Cores/node | RAM/node | Use when |
|---|---|---|---|
| `icelake` | 76 | 256 GB | default CPU work |
| `icelake-himem` | 76 | 512 GB | skipgram OOM (e.g. German C19 at 788M lines) |
| `sapphire` | 112 | 512 GB | newer, sometimes shorter queue |
| `cclake` | 56 | 192 GB | smaller jobs, often idle |
| `ampere` | — | — + A100 | GPU (not useful for gensim W2V) |

Switch by editing `#SBATCH -p icelake` to the partition you want, or override at submit:
```bash
sbatch -p icelake-himem --array=1-N scripts/csd3/train_w2v.sh manifest.txt
```

## 6. Keep queue times short

- **`--time`**: be honest but tight. A 4-hour job often starts in minutes; a 36-hour one can wait days. Benchmark locally, pad 50%.
- **`--cpus-per-task`**: 32 is a good default. Requesting a whole node (76) can queue longer for marginal speedup.
- **Array tasks** run in parallel when slots open, so many small tasks finish faster than one huge job.

## 7. Storage

| Path | Size | Use for |
|---|---|---|
| `/home/rj416` | ~40 GB, backed up | code, dotfiles |
| `/rds/user/rj416/hpc-work` | ~1 TB | **all data, skipgrams, models, outputs** |
| `/rds/project/<code>` | project quota | shared data if you have a project |

Never do heavy I/O from `/home`.

## 8. Common annoyances

- **Job stuck pending, reason `(QOSMaxCpuPerUserLimit)` or similar** — you're over concurrent-use limits. Reduce `--cpus-per-task` or stagger submissions.
- **`sbatch: error: Batch job submission failed: Invalid account or account/partition combination`** — you haven't edited the `-A` line. Run `mybalance` to see your account.
- **`Transport endpoint is not connected` on RDS** — filesystem blip. Retry.
- **SSH disconnects during long jobs** — use `tmux` on CSD3 if you want a persistent shell; jobs keep running regardless of your SSH session.

## 9. Day-to-day commands summary

```bash
# On Mac
./scripts/csd3/sync_up.sh data/models        # push skipgrams
./scripts/csd3/sync_down.sh data/models      # pull trained models

# On CSD3
python scripts/csd3/list_skipgrams.py data/models --incomplete-only > manifest.txt
sbatch --array=1-$(wc -l < manifest.txt) scripts/csd3/train_w2v.sh manifest.txt
squeue -u $USER
sacct -u $USER -X -S today
```
