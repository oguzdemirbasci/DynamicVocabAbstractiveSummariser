from data import Corpus
from model import Embedding
from model import EncDec
from model import VocGenerator
import utils

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import random
import math
import os
import time
import sys

import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Pre-training machine translation model with or without vocabulary prediction')

parser.add_argument('--seed', type = int, default = 1,
                    help='Random seed')
parser.add_argument('--gpu', type = int, default = 0,
                    help='GPU id')

parser.add_argument('--test_source', type = str, required = True,
                    help = 'File path to test data (source sentences)')
parser.add_argument('--test_source_orig', type = str, default = None,
                    help = 'File path to case-sensitive test data, if applicable (source sentences)')
parser.add_argument('--test_target', type = str, required = True,
                    help = 'File path to test data (target sentences)')

parser.add_argument('--model_vocgen', type = str, default = './params/vocgen.bin',
                    help = 'File name for loading model parameters of trained vocabulary predictor')

parser.add_argument('--trans_file', type = str, default = './trans.txt',
                    help = 'Temporary file to output model translations of development data')
parser.add_argument('--gold_file', type = str, default = './gold.txt',
                    help = 'Temporary file to output gold-standard translations of development data')
parser.add_argument('--bleu_file', type = str, default = './bleu.txt',
                    help = 'Temporary file to output BLEU score')

parser.add_argument('--fs', type = int, default = '2',
                    help = 'Minimum word frequency to construct source vocabulary')
parser.add_argument('--ft', type = int, default = '2',
                    help = 'Minimum word frequency to construct target vocabulary')
parser.add_argument('--mlen', type = int, default = '100',
                    help = 'Maximum length of sentences in training data')

parser.add_argument('--K', type = int, default = '1000',
                    help = 'Small vocabulary size for NMT model training (Full softmax if K <= 0 or K > target vocabulary size)')
parser.add_argument('--dim_vocgen', type = int, default = '256',
                    help = 'Dimensionality for embeddings and hidden states of vocabulary predictor')
parser.add_argument('--dim_nmt', type = int, default = '256',
                    help = 'Dimensionality for embeddings and hidden states of NMT model')
parser.add_argument('--layers', type = int, default = '1',
                    help = 'Number of LSTM layers (currently, 1 or 2)')
parser.add_argument('--mepoch', type = int, default = '20',
                    help = 'Maximum number of training epochs')
parser.add_argument('--lr', type = float, default = '1.0',
                    help = 'Learning rate for SGD')
parser.add_argument('--momentum', type = float, default = '0.75',
                    help = 'Momentum rate for SGD')
parser.add_argument('--lrd', type = float, default = '0.5',
                    help = 'Learning rate decay for SGD')
parser.add_argument('--dp', type = float, default = '0.2',
                    help = 'Dropout rate for NMT model')
parser.add_argument('--wd', type = float, default = '1.0e-06',
                    help = 'Weight decay rate for internal weight matrices')
parser.add_argument('--clip', type = float, default = '1.0',
                    help = 'Clipping value for gradient norm')

args = parser.parse_args()
print(args)

sourceTestFile = args.test_source
sourceOrigTestFile = (sourceTestFile if args.test_source_orig is None else args.test_source_orig)
targetTestFile = args.test_target

vocGenFile = args.model_vocgen

transFile = args.trans_file
goldFile = args.gold_file
bleuFile = args.bleu_file

minFreqSource = args.fs
minFreqTarget = args.ft
hiddenDim = args.dim_nmt
decay = args.lrd
gradClip = args.clip
dropoutRate = args.dp
numLayers = args.layers
    
maxLen = args.mlen
maxEpoch = args.mepoch
decayStart = 5

sourceEmbedDim = hiddenDim
targetEmbedDim = hiddenDim

vocGenHiddenDim = args.dim_vocgen

learningRate = args.lr
momentumRate = args.momentum

gpuId = args.gpu
seed = args.seed

weightDecay = args.wd

K = args.K

torch.set_num_threads(1)

torch.manual_seed(seed)
random.seed(seed)
torch.cuda.set_device(gpuId)
torch.cuda.manual_seed(seed)

device = torch.device('cuda:'+str(gpuId))
cpu = torch.device('cpu')

corpus = Corpus(sourceTestFile, sourceOrigTestFile, targetTestFile,
                sourceTestFile, sourceOrigTestFile, targetTestFile,
                minFreqSource, minFreqTarget, maxLen)

batchSize = len(corpus.devData)

vocGen = VocGenerator(vocGenHiddenDim, 12772, 226658)
vocGen.load_state_dict(torch.load(vocGenFile))
vocGen.cuda()
vocGen.eval()

batchListDev = utils.buildBatchList(len(corpus.devData), batchSize)

recall =  0.0
totalTokenCount = 0
for batch in batchListDev:
    batchSize = batch[1]-batch[0]+1
    batchInputSource, lengthsSource, batchInputTarget, batchTarget, lengthsTarget, tokenCount, batchData, maxTargetLen = corpus.processBatchInfoNMT(batch, train = False, volatile = True, device = device)

    targetVocGen, inputVocGen = corpus.processBatchInfoVocGen(batchData, smoothing = False, volatile = True, device = device)
    outputVocGen = vocGen(inputVocGen)
    
    tmp = outputVocGen.data
    val, output_list = torch.topk(tmp, k = K)
    
    output_list = output_list.to(cpu)

    recallTmp = utils.evalVocGen(batchSize, output_list.numpy(), batchInputTarget.data.numpy(), lengthsTarget, corpus.targetVoc.eosIndex)
    recall += recallTmp

    totalTokenCount += tokenCount

devRecall = 100.0*recall/totalTokenCount

print()
print("Recall of voc predictor:", devRecall, '(%)')

