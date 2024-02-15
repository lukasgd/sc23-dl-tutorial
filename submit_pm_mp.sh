#!/bin/bash 
#SBATCH -p nvgpu
#SBATCH -A csstaff
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH -J vit-era5-mp
#SBATCH -o logs/%x-%j.out

environment=$(realpath env/ngc-fcn-24.01.toml)

DATADIR=/iopsstor/scratch/cscs/lukasd/ds/tutorials/sc23_data
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

# if cuda graphs, use train_mp_graphs.py
set -x
# Launching training with tensorboard on rank 0
srun -ul --environment=${environment} \
    bash -c "
if [ \"\${SLURM_PROCID:-0}\" -eq 0 ]; then
        echo \"Launching tensorboard...
Use port-forwarding on your client machine with
  ssh -N -L 6006:localhost:6006 \$(hostname)
and open the browser at http://localhost:6006/.\"
        tensorboard --logdir ${LOGDIR} --port 6006 &
    fi
    echo \"\${SLURM_PROCID:-0}: Running training script on \$(hostname)\"
    source export_DDP_vars.sh
    ${PROFILE_CMD} python train_mp.py ${args}
    "
