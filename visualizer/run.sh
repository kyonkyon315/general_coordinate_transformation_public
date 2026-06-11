#!/bin/bash
#SBATCH -p gr20001a
#SBATCH -t 00:30:00
#SBATCH -o py_out.txt
#SBATCH -e py_err.txt
#SBATCH --rsc p=1:t=1:c=1:m=4G

set -e
set -x   # デバッグ表示（不要なら消してOK）

echo "================ Slurm Info ================"
echo "DATE      = $(date --iso-8601=seconds)"
echo "JOB_ID    = $SLURM_JOB_ID"
echo "NNODES    = $SLURM_JOB_NUM_NODES"
echo "NTASKS    = $SLURM_NTASKS"
echo "CPUS/TASK = $SLURM_CPUS_PER_TASK"
echo "============================================"

# ---- Intel環境（MKLなどのため）----
module purge
module load PrgEnvIntel/2023

echo "LD_LIBRARY_PATH = $LD_LIBRARY_PATH"

# ---- Python仮想環境 ----
source ~/pyenv/bin/activate

echo "Python path:"
which python3
python3 --version

# ---- 並列暴走防止 ----
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ---- Python側が読む並列数 ----
export NPROC=$SLURM_CPUS_PER_TASK

echo "NPROC = $NPROC"

echo "Start Python job"
#srun -n 1 python3 whistler_wave.py
srun -n 1 python3 calc_growth_rate.py
echo "Finished"
