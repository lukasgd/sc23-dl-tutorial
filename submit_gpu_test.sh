#!/bin/bash
#SBATCH -p nvgpu
#SBATCH -A csstaff
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -J gpu-state
#SBATCH -o logs/%x.%j.out

environment=$(realpath env/ngc-fcn-24.01.toml)

srun -ul --environment=${environment} bash -c "
    echo \"CUDA devices on \$(hostname)/\${SLURM_LOCALID:-0}: \${CUDA_VISIBLE_DEVICES:-0}\"
    python gpu_state_test/gpu_state_test.py
    "
