#!/bin/bash -l

#SBATCH --array=1,2%1
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --partition=all
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
#SBATCH -o ./log/%j.out
#SBATCH -e ./log/%j.err

. /etc/profile.d/modules.sh
module load nvhpc
# export OMP_NUM_THREADS=12
export PGI_ACC_TIME=1

base_size="0.01"

# use awk to format the floating value in scientific notation
DX=$(awk -v t="$SLURM_ARRAY_TASK_ID" -v base="$base_size" 'BEGIN{printf "%.8e", base / t}')

# Replace the dx_dy_dz line (commented or not) and write to task-specific file
sed -E "s|^([[:space:]]*#?[[:space:]]*dx_dy_dz[[:space:]]*:\s*).*|\1 ${DX} ${DX} ${DX}|I" test.master > test.in

echo "Running gprMax with input: test.in (dx set to ${DX})"
# python -m gprMax test.in -gpu
python -m gprMax test.in

# run conversion script (keeps original behavior)
python convert.py ${SLURM_ARRAY_TASK_ID}