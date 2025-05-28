#!/bin/sh
#SBATCH --account="ok-iot"      # E.g. "ie-idi" if you belong to IDI
#SBATCH --job-name="foerste_jobb_cpu"
#SBATCH --time=0-01:15:00         # format: D-HH:MM:SS
#SBATCH --partition=CPUQ          # Asking for a GPU
#SBATCH --mem=16G                 # Asking for 16GB RAM
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
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
module load Python/3.10.8-GCCcore-12.2.0
source myenv/bin/activate
python papergraphs.py