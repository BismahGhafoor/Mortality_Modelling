#!/bin/bash
# This is template_smoking_job.sbatch

#SBATCH --job-name=aurum_filter         # Name for each small batch job
#SBATCH --array=__ARRAY_RANGE_AND_THROTTLE__  # << THIS IS A PLACEHOLDER the wrapper will replace
#SBATCH --time=01:00:00                      # Time PER TASK in this batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=/scratch/alice/b/bg205/01_03_AURUM/logs/filtered_chunks/aurumfilter_batch_%A_%a.out
#SBATCH --error=/scratch/alice/b/bg205/01_03_AURUM/logs/filtered_chunks/aurumfilter_batch_%A_%a.err

# --- IMPORTANT: Load your Python environment ---
module purge

# --- Define where your Python script and data are on scratch ---
SCRATCH_WORK_DIR="/scratch/alice/b/bg205/01_03_AURUM"
PYTHON_PROCESSING_SCRIPT="/scratch/alice/b/bg205/01_03_AURUM/2.py"

# --- Go to the working directory on scratch ---
cd "${SCRATCH_WORK_DIR}" || exit 1 # Exit if 'cd' fails

# --- Echo some info for logging ---
echo "Slurm Array Job ID (for this batch): $SLURM_ARRAY_JOB_ID"
echo "Slurm Array Task ID (within this batch): $SLURM_ARRAY_TASK_ID"
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Executing Python script: python ${PYTHON_PROCESSING_SCRIPT} $SLURM_ARRAY_TASK_ID"

# --- Run your Python script ---
# Your Python script (process_zip_chunk.py) takes the SLURM_ARRAY_TASK_ID
# as an argument to know which ZIP file (from 0 to 1046) it needs to process.
python "${PYTHON_PROCESSING_SCRIPT}" $SLURM_ARRAY_TASK_ID

echo "Python script for Task $SLURM_ARRAY_TASK_ID finished."
