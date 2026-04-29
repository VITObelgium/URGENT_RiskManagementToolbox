#!/bin/bash
#SBATCH --job-name=rmt_run
#SBATCH --output=rmt_%j.out
#SBATCH --error=rmt_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00

# Go to your repo directory
cd $SLURM_SUBMIT_DIR || exit 1

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TMPDIR=${TMPDIR:-/tmp}

if [ -d "$TMPDIR" ]; then
  cp examples/complex_model.json "$TMPDIR"/
  cp examples/complex_model.zip "$TMPDIR"/
  CFG="$TMPDIR/complex_model.json"
  MODEL="$TMPDIR/complex_model.zip"
else
  CFG="/opt/rmt/examples/complex_model.json"
  MODEL="/opt/rmt/examples/complex_model.zip"
fi

echo "Running on node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory requested: 16G"
echo "Using config: $CFG"
echo "Using model: $MODEL"
echo "TMPDIR: $TMPDIR"

srun --cpu-bind=cores apptainer exec \
  --bind "$PWD:/opt/rmt" \
  --bind "$TMPDIR:$TMPDIR" \
  rmt.sif \
  bash -lc "cd /opt/rmt && \
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TMPDIR=$TMPDIR && \
  /root/.pixi/bin/pixi run python src/main.py \
    --config-file \"$CFG\" \
    --model-file \"$MODEL\""