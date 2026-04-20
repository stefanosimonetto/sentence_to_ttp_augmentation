#!/bin/bash
#SBATCH --job-name=Finetune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40|a100|l40
#SBATCH --output=Finetune-%j.out
#SBATCH --error=Finetune-%j.err

source /etc/profile.d/modules.sh
module load nvidia/cuda-12

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source myenv/bin/activate
python test_labeled_conf_matrix.py ./fine_tuned_conf_matrix tram_bosch_test_conf_matrix.txt 'cuda:0'

# python test_labeled.py ./fine_tuned/multi_label/ tram_bosch_test_best_model_augmented.txt 'cuda:0'