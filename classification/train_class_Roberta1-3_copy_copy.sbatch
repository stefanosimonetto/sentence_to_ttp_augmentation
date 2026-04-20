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

source /etc/profile.d/modules.sh
module load nvidia/cuda-12

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HTTP_PROXY=http://proxy.utwente.nl:3128
export HTTPS_PROXY=http://proxy.utwente.nl:3128
export http_proxy=http://proxy.utwente.nl:3128
export https_proxy=http://proxy.utwente.nl:3128
source myenv/bin/activate
python train_multilabel.py OUTFILE_copy_2.json 'cuda' 10 12