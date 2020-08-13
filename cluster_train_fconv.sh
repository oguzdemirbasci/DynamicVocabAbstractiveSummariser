#!/usr/bin/env bash
#SBATCH -o /home/%u/slurm_logs/slurm-%A.out
#SBATCH -e /home/%u/slurm_logs/slurm-%A.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 2	  # tasks requested
#SBATCH --gres=gpu:2  # use 4 GPU
#SBATCH --mem=11400  # memory in Mb
#SBATCH -t 8:00:00  # time requested in hour:minute:seconds
#SBATCH --job-name=8fconv_train
#SBATCH --partition=CDT_Compute

set -e # fail fast
echo ${USER}

# Activate Conda
source /home/s1965737/miniconda/bin/activate dvoc

python -c "import torch; print(torch.__version__)"
python -c "import numpy; print(numpy.version.version)"

echo "I'm running on ${SLURM_JOB_NODELIST}"
dt=$(date '+%d_%m_%y__%H_%M');
echo ${dt}


# Env variables
export STUDENT_ID=${USER}
if [ -d "/disk/scratch/" ]; then
    export SCRATCH_HOME="/disk/scratch/${STUDENT_ID}" 
fi
if [ -d "/disk/scratch1/" ]; then
    export SCRATCH_HOME="/disk/scratch1/${STUDENT_ID}"
fi
if [ -d "/disk/scratch2/" ]; then
    export SCRATCH_HOME="/disk/scratch2/${STUDENT_ID}"
fi
if [ -d "/disk/scratch_big/" ]; then
    export SCRATCH_HOME="/disk/scratch_big/${STUDENT_ID}"
fi


export CLUSTER_HOME="/home/${STUDENT_ID}"
export EXP_ROOT="${CLUSTER_HOME}/thesis/fairseq"
export EXEC_ROOT="${EXP_ROOT}"
export EXEC_DATA="${EXP_ROOT}/data-bin"

MODEL_NAME=fconv_dvoc

MODEL_PATH=$SCRATCH_HOME/checkpoints/$MODEL_NAME
mkdir -p "${MODEL_PATH}"
LOG_PATH=$SCRATCH_HOME/logs
mkdir -p "${LOG_PATH}"
cd $EXP_ROOT/


CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py $EXEC_DATA \
    --num-workers 10 \
    --arch $MODEL_NAME \
    --lr 0.25 --clip-norm 0.1 --dropout 0.2 \
    --task dvoc_summarisation \
    --skip-invalid-size-inputs-valid-test \
    --max-target-positions 200 --max-source-positions 800 \
    --max-tokens 1000 --update-freq 4 \
    --save-dir $SAVEDIR \
    --optimizer nag --criterion cross_entropy \
    --truncate-source --truncate-target


echo "============"
echo "training finished successfully"

rsync -avuzhP "$MODEL_PATH" "${EXP_ROOT}/checkpoints/" # Copy output onto headnode
rsync -avuzhP "${LOG_PATH}/${MODEL_NAME}.log" "${EXP_ROOT}/logs/" # Copy output onto headnode

echo "deleting local files"
rm -rf ${MODEL_PATH}
rm "${LOG_PATH}/${MODEL_NAME}.log"
echo "============"
echo "job finished successfully"