#!/bin/bash
##################
# slurm settings #
##################
# where to put stdout / stderr
#SBATCH --output=/users/cn/caraiz/logs/%x_%j.out
#SBATCH --error=/users/cn/caraiz/logs/%x_%j.err
# time limit in minutes
#SBATCH --time=06:00:00
# queue
#SBATCH --qos=short
# memory (MB)
#SBATCH --mem=64G
# job name
#SBATCH --job-name cristina_jupyter_server
# make bash behave more robustly
#################
# start message #
#################
start_epoch=`date +%s`
echo [$(date +"%Y-%m-%d %H:%M:%S")] starting on $(hostname)
###################
# set environment #
###################
module load Python/3.12.4-GCCcore-13.2.0
env | grep LD_LIBRARY_PATH > /users/cn/caraiz/propr_new/.env
source /users/cn/caraiz/propr_new/.venv/bin/activate
#################################################
# run jupyter in the backgound without trying to open a browser
#################################################
sleep 1000000000000000000
