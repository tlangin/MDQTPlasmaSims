#!/bin/bash
#SBATCH --job-name=QuadTestRandom
#SBATCH --array=1-99
#SBATCH --partition=commons
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --mail-user=tkl1@rice.edu
#SBATCH --mail-type=ALL

echo "My job ran on:"
echo $SLURM_NODELIST
if [[ -d $SHARED_SCRATCH/$USER && -w $SHARED_SCRATCH/$USER ]]
then
    srun tagQuad $SLURM_ARRAY_TASK_ID
fi
