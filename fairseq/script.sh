#!/bin/sh
# train script

# DATAPATH='../WikiCatSum/film_tok_min5_L7.5k'
# python preprocess.py -s src -t tgt \
#     --trainpref $DATAPATH/train --validpref $DATAPATH/valid --testpref $DATAPATH/test \
#     --train_dv_path $DATAPATH/train --valid_dv_path $DATAPATH/valid --test_dv_path $DATAPATH/test \
#     --nwordstgt 50000 --nwordssrc 50000 --workers 10


CUDA_LAUNCH_BLOCKING=1 python train.py ./data-bin --num-workers 10 --arch fconv_dvoc --lr 0.25 --clip-norm 0.1 --dropout 0.2 \
      --task dvoc_summarisation --skip-invalid-size-inputs-valid-test --max-target-positions 200 --max-source-positions 800 \
      --max-tokens 2000 --update-freq 2 --save-dir /media/oguz/Storage/thesis/checkpoints/dvocsum  --keep-best-checkpoints 5 \
      --criterion small_softmax --enable-dvoc --dvoc-K 3000 --truncate-source --truncate-target

CUDA_LAUNCH_BLOCKING=1 python generate.py ./data-bin --path /media/oguz/Storage/thesis/checkpoints/dvocsum/checkpoint_best.pt --num-workers 10 \
      --task dvoc_summarisation --model fconv_dvoc --skip-invalid-size-inputs-valid-test --optimizer nag \
      --max-target-positions 200 --max-source-positions 800 --max-tokens 2000 --update-freq 2 \
      --beam 5 --gen-subset valid --compute-rouge --results-path /media/oguz/Storage/thesis/results/dvocsum/ --cpu
