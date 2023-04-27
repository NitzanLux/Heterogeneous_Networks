#!/bin/bash
# Write output as following (%j is JOB_ID)
#SBATCH -o /ems/elsc-labs/segev-i/nitzan.luxembourg/projects/dendritic_tree/ArtificialNeuron/cluster_logs/output-%j.out
#SBATCH -e /ems/elsc-labs/segev-i/nitzan.luxembourg/projects/dendritic_tree/ArtificialNeuron/cluster_logs/error-%j.err
# Ask for one CPU, one GPU, enter the GPU queue, and limit run to 1 days
#SBATCH -c 1
#SBATCH -t 1-0
#SBATCH -p ss-gpu.q,gpu.q
#SBATCH --gres=gpu:1
# check if script is started via SLURM or bash
# if with SLURM: there variable '$SLURM_JOB_ID' will exist
# `if [ -n $SLURM_JOB_ID ]` checks if $SLURM_JOB_ID is not an empty string
if [ -n $SLURM_JOB_ID ]; then
# check the original location through scontrol and $SLURM_JOB_ID
SCRIPT_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
else
# otherwise: started with bash. Get the real location.
SCRIPT_PATH=$(realpath $0)
fi
# get script's path to allow running from any folder without errors
path=$(dirname $SCRIPT_PATH)

# put your script here - example script is sitting with this bash script
python3 -m utils.clear_by_name_script -re "$1"

