#!/bin/sh
#SBATCH --account="ok-iot"      # E.g. "ie-idi" if you belong to IDI
#SBATCH --job-name="gpu_optimized_simulation"
#SBATCH --time=3-00:00:00         # format: D-HH:MM:SS
#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=16G                 # Asking for 16GB RAM
#SBATCH --nodes=1
#SBATCH --output=output.txt      # Specifying 'stdout'
#SBATCH --error=output.err        # Specifying 'stderr'

#SBATCH --mail-user=torbjsme@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge

# Running your python file
module load Python/3.11.3-GCCcore-12.3.0
module load foss/2024a
source myenv/bin/activate
pip install -r requirements.txt
pip install qiskit-aer-gpu-cu11
echo "=== GPUs allocated to this job ===" > gpu_check_$SLURM_JOB_ID}.txt
nvidia-smi -L >> gpu_check_$SLURM_JOB_ID}.txt
python driver.py
