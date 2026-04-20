#!/bin/bash
#SBATCH --job-name=Finetune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|l40
#SBATCH --qos=research-deadline
#SBATCH --output=Finetune-%j.out
#SBATCH --error=Finetune-%j.err

source myenv/bin/activate

# Get the IP address of the node instead of the hostname
NODE_IP=$(hostname -I | awk '{print $1}')

# Start Jupyter notebook on the node's IP
python3 -m jupyter notebook --no-browser --ip=$NODE_IP --port=8805
