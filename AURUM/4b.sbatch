#!/bin/bash
#SBATCH --job-name=enriched_job
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/alice/b/bg205/01_03_AURUM/logs/enricheddf_%j.out
#SBATCH --error=/scratch/alice/b/bg205/01_03_AURUM/logs/enricheddf_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bg205@leicester.ac.uk

set -euo pipefail

# Load whichever your cluster uses:
module purge # or: module load Anaconda3; source activate <env-name>

LOG_DIR="/scratch/alice/b/bg205/01_03_AURUM/logs"
SCRATCH_WORK_DIR="/scratch/alice/b/bg205/01_03_AURUM"
#PYTHON_PROCESSING_SCRIPT="/scratch/alice/b/bg205/DataCleaning_Aurum_v2/4_optimised.py"
PYTHON_PROCESSING_SCRIPT="/scratch/alice/b/bg205/01_03_AURUM/4.py"

mkdir -p "$LOG_DIR"
cd "$SCRATCH_WORK_DIR"

echo "Host: $(hostname)"
echo "CWD:  $(pwd)"
echo "Python: $(which python)"
python --version
echo "Running: python ${PYTHON_PROCESSING_SCRIPT}"

export PYTHONUNBUFFERED=1

/usr/bin/time -v python -u "${PYTHON_PROCESSING_SCRIPT}"

echo "Done."
