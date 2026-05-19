#!/bin/bash
#SBATCH --job-name=rmt_run_geomech
#SBATCH --output=rmt_%j.out
#SBATCH --error=rmt_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --mail-type END,FAIL 
#SBATCH --mail-user=justin.pogacnik@vito.be

# Go to your repo directory
cd $SLURM_SUBMIT_DIR || exit 1

export PROJECT_ROOT="/home/pogacnij/DEVELOPER/URGENT_RiskManagementToolbox"
export FEM_CACHE_DIR="$PROJECT_ROOT/log/fem_cache"

mkdir -p "$FEM_CACHE_DIR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TMPDIR=${TMPDIR:-/tmp}

mkdir -p "$FEM_CACHE_DIR"

if [ -d "$TMPDIR" ]; then
  cp examples/complex_model_geomech.json "$TMPDIR"/
  cp examples/complex_model_geomech.zip "$TMPDIR"/
  CFG="$TMPDIR/complex_model_geomech.json"
  MODEL="$TMPDIR/complex_model_geomech.zip"
else
  CFG="/opt/rmt/examples/complex_model_geomech.json"
  MODEL="/opt/rmt/examples/complex_model_geomech.zip"
fi

echo "Running on node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory requested: 24G"
echo "Using config: $CFG"
echo "Using model: $MODEL"
echo "TMPDIR: $TMPDIR"

apptainer exec \
  --bind "$PWD:/opt/rmt" \
  --bind "$PROJECT_ROOT:$PROJECT_ROOT" \
  --bind "$TMPDIR:$TMPDIR" \
  rmt.sif \
  bash -lc "cd /opt/rmt && \
  export PROJECT_ROOT=$PROJECT_ROOT FEM_CACHE_DIR=$FEM_CACHE_DIR && \
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TMPDIR=$TMPDIR && \
  /root/.pixi/bin/pixi run python src/main.py \
    --config-file \"$CFG\" \
    --model-file \"$MODEL\""