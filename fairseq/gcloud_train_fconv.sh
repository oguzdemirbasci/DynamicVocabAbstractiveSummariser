#!/bin/sh
# train script

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate abs

SAVEDIR='./checkpoints/fconv'

# CUDA_LAUNCH_BLOCKING=1 python train.py ./data-bin --num-workers 12 --arch fconv_dvoc --lr 0.25 --clip-norm 0.1 --dropout 0.2 \
#       --task dvoc_summarisation --skip-invalid-size-inputs-valid-test --max-target-positions 200 --max-source-positions 500 \
#       --max-tokens 1000 --update-freq 4 --save-dir $SAVEDIR --keep-last-epochs 5 \
#       --optimizer nag --criterion cross_entropy --truncate-source --truncate-target \
#       --no-progress-bar --log-interval 100 2>&1 | tee $SAVEDIR/training.log


python /home/oguz/stop_instance.py