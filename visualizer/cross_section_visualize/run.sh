#!/bin/bash
#============ Slurm Options ===========
#SBATCH -p gr20001a
#SBATCH -t 08:00:00
#SBATCH -o out.txt
#SBATCH -e err.txt
#SBATCH --rsc p=1:t=1:c=1:m=1G

# エラーが出たら即終了
set -e

#============ Shell Script ============
# 1. 環境初期化
. /usr/share/Modules/init/bash

module use /opt/system/app/env/intel/Compiler
module use /opt/system/app/env/intel/MPI
module purge
module load intel
module load intelmpi

echo "Compiler:"
mpiicpx --version
echo "Current dir:"
pwd

# 2. コンパイル（C++20対応 & includeパス追加）
#mpiicpx -O3 -std=c++20 -fp-model=precise \
mpiicpx -O3 -std=c++20 \
-Wl,-rpath=${I_MPI_ROOT}/libfabric/lib \
#calc_anisotropy.cpp -o vlasov 
whistler_super.cpp -o vlasov 

echo "Compile finished."

# 高速通信を明示
export I_MPI_FABRICS=shm:ofi

# OFIプロバイダ自動選択
unset I_MPI_OFI_PROVIDER

# デバッグ表示（確認用）
#export I_MPI_DEBUG=5

# 3. 実行
srun --mpi=pmi2 -n 1 ./vlasov

