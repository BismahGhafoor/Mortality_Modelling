#!/bin/bash
# This is run_smoking_in_batches.sh

# --- Configuration ---
TOTAL_TASKS_TO_PROCESS=1047 # This means array indices from 0 to 1046

DECLARED_BATCH_SIZE=10

RUNTIME_CONCURRENT_TASK_LIMIT=16

# Path to your Slurm job template script
#SBATCH_TEMPLATE_FILE="template_smoking_job.sbatch" # Assumed to be in the same directory
SBATCH_TEMPLATE_FILE="/scratch/alice/b/bg205/01_03_AURUM/2b.sbatch" # Assumed to be in the same directory
# --- End Configuration ---

# Simple check to ensure the template file exists
if [ ! -f "${SBATCH_TEMPLATE_FILE}" ]; then
    echo "ERROR: The Slurm template file '${SBATCH_TEMPLATE_FILE}' was not found."
    echo "Please make sure it's in the same directory as this script."
    exit 1
fi
# Simple check for the placeholder in the template
if ! grep -q "__ARRAY_RANGE_AND_THROTTLE__" "${SBATCH_TEMPLATE_FILE}"; then
    echo "ERROR: The placeholder '__ARRAY_RANGE_AND_THROTTLE__' was not found in '${SBATCH_TEMPLATE_FILE}'."
    echo "Please check your template file."
    exit 1
fi

echo "Starting the process of submitting jobs in batches..."
echo "Total tasks to process: ${TOTAL_TASKS_TO_PROCESS} (indices 0 to $((TOTAL_TASKS_TO_PROCESS - 1)))"
echo "Tasks declared per Slurm batch job: ${DECLARED_BATCH_SIZE}"
echo "Max concurrent tasks running within each batch: ${RUNTIME_CONCURRENT_TASK_LIMIT}"
echo "-------------------------------------------------------------------"

current_overall_task_start_index=0 # This is the global task index (0 to 1046)
batch_number=0

# Loop until all global tasks are covered
while [ $current_overall_task_start_index -lt $TOTAL_TASKS_TO_PROCESS ]; do

    # Calculate the end index for the tasks in THIS specific batch
    # e.g., if start is 0 and batch_size is 16, end is 15
    # e.g., if start is 16 and batch_size is 16, end is 31
    current_batch_end_index=$((current_overall_task_start_index + DECLARED_BATCH_SIZE - 1))

    # Make sure the end index doesn't go past the total number of tasks
    if [ $current_batch_end_index -ge $TOTAL_TASKS_TO_PROCESS ]; then
        current_batch_end_index=$((TOTAL_TASKS_TO_PROCESS - 1))
    fi

    # This is the string that will go into the #SBATCH --array line
    # e.g., "0-15%16"
    # The SLURM_ARRAY_TASK_ID in your python script will still correctly receive values
    # from current_overall_task_start_index to current_batch_end_index.
    array_directive_for_this_batch="${current_overall_task_start_index}-${current_batch_end_index}%${RUNTIME_CONCURRENT_TASK_LIMIT}"

    echo "Preparing Batch ${batch_number}:"
    echo "  Tasks covered in this batch (global indices): ${current_overall_task_start_index} to ${current_batch_end_index}"
    echo "  Slurm --array directive for this batch: ${array_directive_for_this_batch}"

    # Create a temporary, specific sbatch script for this batch
    # by replacing the placeholder in the template.
    temp_sbatch_filename="temp_sbatch_for_batch_${batch_number}.sbatch"
    sed "s/__ARRAY_RANGE_AND_THROTTLE__/${array_directive_for_this_batch}/" "${SBATCH_TEMPLATE_FILE}" > "${temp_sbatch_filename}"

    echo "Submitting Batch ${batch_number} to Slurm and WAITING for it to complete..."
    
    # Submit the temporary sbatch script and WAIT for it to finish
    sbatch --wait "${temp_sbatch_filename}"
    sbatch_exit_code=$? # Get the exit code of the sbatch command itself

    # Clean up the temporary sbatch script
    rm -f "${temp_sbatch_filename}"

    if [ $sbatch_exit_code -ne 0 ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "ERROR: sbatch submission for Batch ${batch_number} (tasks ${current_overall_task_start_index}-${current_batch_end_index}) FAILED."
        echo "sbatch command exited with code: ${sbatch_exit_code}"
        echo "The wrapper script will now stop. Please check Slurm error messages."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit 1 # Stop the wrapper if sbatch submission fails
    else
        # Note: sbatch --wait returning 0 just means the job completed, not necessarily that all tasks succeeded.
        # You'll need to check the actual logs in /home/b/bg205/smoking_run/logs/
        echo "Batch ${batch_number} (tasks ${current_overall_task_start_index}-${current_batch_end_index}) has completed according to Slurm."
        echo "-------------------------------------------------------------------"
    fi

    # Set the start index for the NEXT batch
    current_overall_task_start_index=$((current_batch_end_index + 1))
    batch_number=$((batch_number + 1))
done

echo "All batches have been submitted and processed."
echo "Please check your output files and logs in /scratch/alice/b/bg205/01_03_AURUM/logs"
