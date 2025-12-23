#!/bin/bash
#PBS -j oe
#PBS -q calc-lm
#PBS -N gprMax
#PBS -l select=1:ncpus=96:mpiprocs=1
#PBS -l walltime=72:00:00
#PBS -o ./log/
#PBS -e ./log/

###########################
### PBS-JOBRUN-OPTION
###########################
### PBS -q calc-lm                 # queue (1 node with 96 cores, 12TB RAM)
### PBS -N jobname                 # jobname
### PBS -j oe                      # join outfile and errfile
### PBS -J 1-8                     # array job range
### PBS -l select=1:ncpus=96:mpiprocs=1  # calc-lm: 1 node, 96 CPUs (4x24 cores)
### PBS -l walltime=72:00:00       # Time resources


###########################
### Pre
###########################
cd ${PBS_O_WORKDIR}

# Load user environment
source /home/kusumoto/.bashrc

conda activate gprMax

echo "(`date`) Starting job on node: $(hostname)"

###########################
### Execute program
###########################
base_size="0.01"

for i in {1..8}; do
    echo "=========================================="
    echo "(`date`) Starting task ${i}/8"
    echo "=========================================="
    
    DX=$(awk -v t="${i}" -v base="${base_size}" 'BEGIN{printf "%.8e", base / t}')
    
    sed -E "s|^([[:space:]]*#?[[:space:]]*dx_dy_dz[[:space:]]*:\s*).*|\1 ${DX} ${DX} ${DX}|I" test.master > test.in
    
    echo "(`date`) Running gprMax with input: test.in (dx set to ${DX})"
    python -m gprMax test.in
    
    echo "(`date`) Running conversion script"
    python convert.py ${i}
    
    echo "(`date`) Completed task ${i}"
done

echo "(`date`) All tasks completed."