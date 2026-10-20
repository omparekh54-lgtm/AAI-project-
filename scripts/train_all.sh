#!/bin/bash
# Reproduces every learned model used in the benchmark (single CPU core: roughly one hour).
set -e
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
EP=${1:-120}
python -m backend.train --mode hybrid --seed 42 --episodes $EP
python -m backend.train --mode full   --seed 42 --episodes $EP
python -m backend.train_vdn --seed 42 --episodes $EP
for s in 43 44; do
  python -m backend.train --mode hybrid --seed $s --episodes $EP
  python -m backend.train --mode full   --seed $s --episodes $EP
done
