#!/bin/bash
#SBATCH --job-name=fl_dp
#SBATCH --partition=gpuqs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /home/017033874/collusion-defense

module purge
module load miniconda3/3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate FL_DP

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "Arguments: $*"

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("GPU allocated but unavailable to PyTorch")

print("GPU:", torch.cuda.get_device_name(0))
PY

python - <<'PY'
from pathlib import Path
import sys

required = Path("data")
if not required.exists():
    print(f"Dataset directory is missing: {required.resolve()}", file=sys.stderr)
    sys.exit(1)

print("Dataset root:", required.resolve())
PY

srun python -u BD_Attack_Collusion_ProbeGroup.py "$@"
