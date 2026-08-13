#!/bin/bash
#SBATCH --job-name=Finetune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=Finetune-%j.out
#SBATCH --error=Finetune-%j.err

source /etc/profile.d/modules.sh
module load nvidia/cuda-12

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1

# Proxy (set BEFORE any python that needs internet)
export HTTP_PROXY="http://proxy.utwente.nl:3128"
export HTTPS_PROXY="http://proxy.utwente.nl:3128"
export http_proxy="http://proxy.utwente.nl:3128"
export https_proxy="http://proxy.utwente.nl:3128"

# NLTK data path (so runtime lookup works)
export NLTK_DATA="/home/simonettos/nltk_data"

# Activate venv
source /home/simonettos/thijs/classification/classification/myenv/bin/activate

# (Optional) download once — ideally do this on a login node, not every job
python - <<'PY'
import os
import nltk

print("HTTP_PROXY:", os.environ.get("HTTP_PROXY"))
print("NLTK_DATA:", os.environ.get("NLTK_DATA"))

nltk.download("wordnet", download_dir=os.environ["NLTK_DATA"])
nltk.download("omw-1.4", download_dir=os.environ["NLTK_DATA"])
PY

# Run your script
python /home/simonettos/thijs/smote_eda_augm/eda_bt_bosch.py
# python /home/simonettos/thijs/data_augmentatio_stefano/ttpxhunter/reproduce.py