#!/bin/bash
#SBATCH -J vLLM_setup
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
# Load ENV variables
#SBATCH --qos=research-deadline

#SBATCH --output=gcn/securebert_feat/ccc_feat.out
#SBATCH --error=gcn/securebert_feat/ccc_feat.err


export HTTP_PROXY=http://proxy.utwente.nl:3128
export HTTPS_PROXY=http://proxy.utwente.nl:3128

# Load ENV
source /etc/profile.d/modules.sh
module load nvidia/cuda-11.8

source /home/simonettos/homhetgnn/gpuvenv/bin/activate

export PYTHONPATH=$PYTHONPATH:/home/simonettos/homhetgnn/HGB/LP/benchmark/methods/baseline:/home/simonettos/homhetgnn/HGB/LP/benchmark:/home/simonettos/homhetgnn/HGB/LP/benchmark/methods/GNN:/home/simonettos/homhetgnn/HGB/LP/benchmark/methods/GNN/utils:/home/simonettos/homhetgnn/HGB/LP/benchmark/methods/GNN/models

# === CONFIGURATION ===
SCRIPT_PATH="/home/simonettos/homhetgnn/HGB/LP/benchmark/methods/GNN/homoGNN.py"
MODEL="GCN"
# List of datasets
DATASETS=(
    "amazon_style_ccc_features_securebert_fold_1"
    "amazon_style_ccc_features_securebert_fold_2"
    "amazon_style_ccc_features_securebert_fold_3"
    "amazon_style_ccc_features_securebert_fold_4"
    "amazon_style_ccc_features_securebert_fold_5"
)
# Number of repetitions
REPEATS=6

echo "Starting $MODEL batch job with $REPEATS repetitions..."

for i in $(seq 1 $REPEATS); do
    echo "=== Repetition $i/$REPEATS ==="
    for data in "${DATASETS[@]}"; do
        echo "--------------------------------------"
        echo "Running model: $MODEL"
        echo "Dataset: $data"
        echo "Repetition: $i"
        echo "--------------------------------------"
        python "$SCRIPT_PATH" --data "$data" --model "$MODEL" --feat_type=1
        echo "Completed: $data (rep $i)"
        echo
    done
done

echo "All $((REPEATS * ${#DATASETS[@]})) jobs completed."

#GAT
# python /home/simonettos/homhetgnn/HGB/LP/benchmark/methods/GNN/homoGNN.py --data $data --model GAT
# echo "GAT"
# echo $data


# SimpleHGN
# cd /home/simonettos/homhetgnn/HGB/LP/benchmark/methods/baseline
# python run_dist.py --dataset $data
# echo "SimpleHGN"
# echo $data

# python run_new.py --dataset LastFM
# python run_dist.py --dataset amazon


# GATNE
# cd /home/simonettos/homhetgnn/HGB/LP/benchmark/methods/GATNE
# python main_pytorch.py --data $data --schema "0-1-0,1-0-1;;1-2-1,2-1-2"  --eval-type 3,6 #cct
# python main_pytorch.py --data $data --schema "0-2-0,2-0-2;;1-2-1,2-1-2;;1-3-1,3-1-3"  --eval-type 3,7 #ccct
# python main_pytorch.py --data $data --schema "0-2-0,2-0-2;;1-2-1,2-1-2;;1-3-1,3-1-3;;4-3-4,3-4-3"  --eval-type 3,7 #ccctt
# python main_pytorch.py --data $data --schema "0-1-0,1-0-1;;0-2-0,2-0-2;;3-2-3,2-3-2"  --eval-type 3,6 #cctt
# echo "GATNE"
# echo $data
# python main_pytorch.py --data LastFM --schema "0-1-0,1-0-1;;1-2-1,2-1-2" --eval-type 0

# HetGNN
# cd /home/simonettos/homhetgnn/HGB/LP/benchmark/methods/HetGNN
# python main.py --data $data 
# python do_LP.py --data $data
# echo "HetGNN"
# echo $data

# python main.py --data amazon 
# python main.py --data amazon_style_cct_restricted
# python main.py --data LastFM 
# python do_LP.py --data amazon
# python do_LP.py --data amazon_style_cct_restricted
# python do_LP.py --data LastFM


# HGT
# cd /home/simonettos/homhetgnn/HGB/LP/benchmark/methods/HGT
# CUDA_VISIBLE_DEVICES=0 python link_predict.py --device 0 --use_norm True --dataset $data
# echo "HGT"
# echo $data
# CUDA_VISIBLE_DEVICES=0 python link_predict.py --device 0 --use_norm True --dataset amazon
# CUDA_VISIBLE_DEVICES=0 python link_predict.py --device 0 --use_norm True  --dataset LastFM
# CUDA_VISIBLE_DEVICES=0 python link_predict.py --device 0 --use_norm True --dataset amazon_style_cct


# RGCN
# cd /home/simonettos/homhetgnn/HGB/LP/benchmark/methods/RGCN
# python link_predict.py --dataset=$data  --hidden-dim=60
# echo "RGCN"
# echo $data
# python link_predict.py --dataset=LastFM --hidden-dim=60
# python link_predict.py --dataset=amazon_style_cct --hidden-dim=60
# python link_predict.py --dataset=amazon_style_dataset_nofeats --hidden-dim=60


echo "Setup complete"


