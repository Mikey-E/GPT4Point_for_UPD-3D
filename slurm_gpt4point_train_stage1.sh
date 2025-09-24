#!/bin/bash
#SBATCH --account=3dllms
##SBATCH --partition=mb-a30
#SBATCH --partition=mb-l40s
#SBATCH --job-name=gpt4point_train_stage1
#SBATCH --output=./slurm_logs/gpt4point_train_stage1_%j.out
#SBATCH --error=./slurm_logs/gpt4point_train_stage1_%j.out
#SBATCH --gres=gpu:4
#SBATCH --mem=16G
#SBATCH --time=24:00:00

#This ensures "conda activate <env>" works in non-interactive shells.
#(running "conda init" every time won't work.)
if [ -n "$CONDA_INSTALL_PATH" ]; then
    CONDA_SH=$CONDA_INSTALL_PATH/etc/profile.d/conda.sh
    if [ ! -e "$CONDA_SH" ]; then
        echo "ERROR: $CONDA_SH does not exist."
        exit 1
    fi
    source "$CONDA_SH"
else
    CONDA_SH=/project/3dllms/melgin/conda/etc/profile.d/conda.sh
    echo "WARNING: CONDA_INSTALL_PATH is not set. Trying $CONDA_SH"
    if [ ! -e "$CONDA_SH" ]; then
        echo "ERROR: $CONDA_SH does not exist."
        exit 1
    fi
    source "$CONDA_SH"
fi
# Now the activation should work
conda activate gpt4point

python -m torch.distributed.run --master_port=32339 --nproc_per_node=4 train.py --cfg-path lavis/projects/gpt4point/train/pretrain_stage1_cap3d.yaml