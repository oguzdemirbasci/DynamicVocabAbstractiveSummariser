#!/bin/sh
# train script

# DATAPATH='/home/oguz/Documents/thesis/WikiCatSum/film_tok_min5_L7.5k'
# python preprocess.py -s src -t tgt \
#     --trainpref $DATAPATH/train --validpref $DATAPATH/valid --testpref $DATAPATH/test \
#     --train_dv_path $DATAPATH/train --valid_dv_path $DATAPATH/valid --test_dv_path $DATAPATH/test \
#     --nwordstgt 50000 --nwordssrc 50000 --workers 10


# CUDA_LAUNCH_BLOCKING=1 python train.py ./data-bin --num-workers 10 \
#     --arch fconv_dvoc --clip-norm 0.1 --dropout 0.2 \
#     --task dvoc_summarisation --skip-invalid-size-inputs-valid-test --optimizer sgd --lr 0.08 \
#     --max-target-positions 200 --max-source-positions 800 --max-tokens 8000 --max-sentences 100 \
#     --save-dir /media/oguz/Storage/thesis/checkpoints/dvocsum --criterion small_softmax --enable-dvoc \
#     --dvoc-K 3000 --truncate-source --truncate-target

CUDA_LAUNCH_BLOCKING=1 python generate.py ./data-bin --path /media/oguz/Storage/thesis/checkpoints/dvocsum/checkpoint_last.pt --num-workers 10 \
      --task dvoc_summarisation --skip-invalid-size-inputs-valid-test --optimizer adagrad \
      --max-target-positions 15 --max-source-positions 800 --max-tokens 8000 \
      --batch-size 5 --beam 5 --gen-subset valid
      --cpu
