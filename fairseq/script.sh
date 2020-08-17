#!/bin/sh
# train script

# DATAPATH='../WikiCatSum/film_tok_min5_L7.5k'
# python preprocess.py -s src -t tgt \
#     --trainpref $DATAPATH/train --validpref $DATAPATH/valid --testpref $DATAPATH/test \
#     --train_dv_path $DATAPATH/train --valid_dv_path $DATAPATH/valid --test_dv_path $DATAPATH/test \
#     --nwordstgt 50000 --nwordssrc 50000 --workers 10


CUDA_LAUNCH_BLOCKING=1 python train.py ./data-bin --num-workers 10 --arch fconv_dvoc --lr 0.25 --clip-norm 0.1 --dropout 0.2 \
      --task dvoc_summarisation --skip-invalid-size-inputs-valid-test --max-target-positions 200 --max-source-positions 800 \
      --max-tokens 1000 --update-freq 4 --save-dir /media/oguz/Storage/thesis/checkpoints/dvocsum  --keep-best-checkpoints 5 \
      --optimizer nag --criterion small_softmax --enable-dvoc --dvoc-K 3000 --truncate-source --truncate-target

SAVEDIR='/media/oguz/Storage/thesis/checkpoints/fconv_dvoc'
RESULTDIR='/media/oguz/Storage/thesis/results/fconv_dvoc'

CUDA_LAUNCH_BLOCKING=1 python train.py ./data-bin --num-workers 12 --arch fconv_dvoc --lr 0.25 --clip-norm 0.1 --dropout 0.2 \
      --task dvoc_summarisation --skip-invalid-size-inputs-valid-test --max-target-positions 200 --max-source-positions 500 \
      --max-tokens 1000 --update-freq 4 --save-dir $SAVEDIR --keep-last-epochs 5 --max-epoch 25 \
      --optimizer nag --criterion cross_entropy --truncate-source --truncate-target \
      --no-progress-bar --log-interval 100 2>&1 | tee $SAVEDIR/training.log

CUDA_LAUNCH_BLOCKING=1 python train.py ./data-bin --num-workers 12 --arch fconv_dvoc --lr 0.25 --clip-norm 0.1 --dropout 0.2 \
      --task dvoc_summarisation --skip-invalid-size-inputs-valid-test --max-target-positions 200 --max-source-positions 500 \
      --max-tokens 1000 --update-freq 4 --save-dir $SAVEDIR --keep-last-epochs 5 --max-epoch 25 \
      --optimizer nag --criterion small_softmax --truncate-source --truncate-target \
      --enable-dvoc --dvoc-K 3000 \
      --no-progress-bar --log-interval 100 2>&1 | tee $SAVEDIR/training.log


CUDA_LAUNCH_BLOCKING=1 python generate.py ./data-bin --path $SAVEDIR/checkpoint_best.pt --num-workers 10 \
      --task dvoc_summarisation --skip-invalid-size-inputs-valid-test --optimizer nag \
      --max-target-positions 200 --max-source-positions 500 --max-tokens 4000 \
      --beam 5 --gen-subset valid --compute-rouge --results-path $RESULTDIR \
      --enable-dvoc --dvoc-K 3000 --truncate-source --truncate-target


CUDA_LAUNCH_BLOCKING=1 python generate.py ./data-bin --path $SAVEDIR/checkpoint_best.pt --num-workers 10 \
      --task dvoc_summarisation --skip-invalid-size-inputs-valid-test --optimizer nag \
      --max-target-positions 200 --max-source-positions 500 --max-tokens 4000 \
      --beam 5 --gen-subset valid --compute-rouge --results-path $RESULTDIR \
      --truncate-source --truncate-target