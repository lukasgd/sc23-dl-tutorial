#!/bin/bash 
#SBATCH -A csstaff
#SBATCH --ntasks-per-node 4
##SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH -J vit-era5-mp
#SBATCH -o logs/%x-%j.out

environment=$(realpath env/ngc-sc22-dl-tutorial-24.05.toml)

DATADIR=/mchstor2/scratch/cscs/lukasd/tutorials/sc23_data
LOGDIR=logs
mkdir -p ${LOGDIR}
args="--expdir ${LOGDIR} --datadir ${DATADIR} ${@}"
#args="--config=mp --row_parallel_size=4"

export HDF5_USE_FILE_LOCKING=FALSE

# Profiling
if [ "${ENABLE_PROFILING:-0}" -eq 1 ]; then
    echo "Enabling profiling..."
    NSYS_ARGS="--trace=cuda,cublas,nvtx --cuda-graph-trace=node --kill none -c cudaProfilerApi -f true"
    NSYS_OUTPUT=${LOGDIR}/${PROFILE_OUTPUT:-"profile"}
    export PROFILE_CMD="nsys profile $NSYS_ARGS -o $NSYS_OUTPUT"
fi

export MASTER_ADDR=$(hostname)

# Debugging (single rank, controlled by DEBUG_RANK, defaults to rank 0)
if [ "${ENABLE_DEBUGGING:-0}" -eq 1 ]; then
    echo "Enabling debugging..."
    ENROOT_ENTRYPOINT="env/enroot-entrypoint.sh"
    if [ "${DEBUG_RANK:-0}" -ge "$SLURM_NTASKS" ]; then
        echo "DEBUG_RANK = ${DEBUG_RANK:-0} is not a valid rank (#ranks = $SLURM_NTASKS), exiting..."
        exit 1
    fi
else
    ENROOT_ENTRYPOINT=""
fi

# if cuda graphs, use train_mp_graphs.py
set -x
# Launching training with tensorboard on rank 0
srun -ul --environment=${environment} ${ENROOT_ENTRYPOINT} \
    bash -c "
#     if [ \"\${SLURM_PROCID:-0}\" -eq 0 ]; then
#         echo \"Launching tensorboard...
# Use port-forwarding on your client machine with
#   ssh -N -L 6006:localhost:6006 \$(hostname)
# and open the browser at http://localhost:6006/.\"
#         tensorboard --logdir ${LOGDIR} --port 6006 &
#     fi
    source export_DDP_vars.sh
    if [ "${ENABLE_DEBUGGING:-0}" -eq 1 ] && [ \"\${SLURM_PROCID:-0}\" -eq ${DEBUG_RANK:-0} ]; then
        echo \"Running training script with debugpy on \$(hostname)\"
        DEBUG_CMD=\"-m debugpy --listen 5678 --wait-for-client\"
    else
        DEBUG_CMD=\"\"
    fi
    CUDA_VISIBLE_DEVICES=\${SLURM_LOCALID} ${PROFILE_CMD} python \${DEBUG_CMD} train_mp.py ${args}
    "
