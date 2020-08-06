from data import Corpus
from model import Embedding
from model import EncDec
from model import VocGenerator
import utils

from joblib import Parallel, delayed, parallel_backend

import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import argparse

parser = argparse.ArgumentParser(description='Pre-training machine translation model with or without vocabulary prediction')

parser.add_argument('--seed', type = int, default = 1,
                    help='Random seed')
parser.add_argument('--gpu', type = int, default = 0,
                    help='GPU id')

parser.add_argument('--train_source', type = str, required = True,
                    help = 'File path to training data (source sentences)')
parser.add_argument('--train_source_orig', type = str, default = None,
                    help = 'File path to case-sensitive training data, if applicable (source sentences)')
parser.add_argument('--train_target', type = str, required = True,
                    help = 'File path to training data (target sentences)')
parser.add_argument('--train_pickle', type = str, default = '',
                    help = 'File path to preprocessed train dataset')
parser.add_argument('--train_vocab_file_name', type = str, default = 'train.vocab',
                    help = 'File name to train small vocabularies')

parser.add_argument('--dev_source', type = str, required = True,
                    help = 'File path to validation data (source sentences)')
parser.add_argument('--dev_source_orig', type = str, default = None,
                    help = 'File path to case-sensitive validation data, if applicable (source sentences)')
parser.add_argument('--dev_target', type = str, required = True,
                    help = 'File path to validation data (target sentences)')
parser.add_argument('--dev_pickle', type = str, default = '',
                    help = 'File path to preprocessed train dataset')
parser.add_argument('--dev_vocab_file_name', type = str, default = 'valid.vocab',
                    help = 'File name to validain small vocabularies')

parser.add_argument('--test_source', type = str, required = True,
                    help = 'File path to test data (source sentences)')
parser.add_argument('--test_source_orig', type = str, default = None,
                    help = 'File path to case-sensitive test data, if applicable (source sentences)')
parser.add_argument('--test_target', type = str, required = True,
                    help = 'File path to test data (target sentences)')
parser.add_argument('--test_pickle', type = str, default = '',
                    help = 'File path to preprocessed test dataset')
parser.add_argument('--test_vocab_file_name', type = str, default = 'test.vocab',
                    help = 'File name to test small vocabularies')

parser.add_argument('--model_vocgen', type = str, default = './params/vocgen.bin',
                    help = 'File name for loading model parameters of trained vocabulary predictor')
parser.add_argument('--K', type = int, default = '1000',
                    help = 'Small vocabulary size for NMT model training (Full softmax if K <= 0 or K > target vocabulary size)')
parser.add_argument('--dim_vocgen', type = int, default = '256',
                    help = 'Dimensionality for embeddings and hidden states of vocabulary predictor')
parser.add_argument('--bs', type = int, default = '128',
                    help = 'Batch size')
parser.add_argument('--fs', type = int, default = '2',
                    help = 'Minimum word frequency to construct source vocabulary')
parser.add_argument('--ft', type = int, default = '2',
                    help = 'Minimum word frequency to construct target vocabulary')
parser.add_argument('--mlen', type = int, default = '100',
                    help = 'Maximum length of sentences in training data')


args = parser.parse_args()
print(args)

sourceTrainFile = args.train_source
sourceOrigTrainFile = (sourceTrainFile if args.train_source_orig is None else args.train_source_orig)
targetTrainFile = args.train_target
train_vocab_file = args.train_vocab_file_name
trainPickle = args.train_pickle

sourceDevFile = args.dev_source
sourceOrigDevFile = (sourceDevFile if args.dev_source_orig is None else args.dev_source_orig)
targetDevFile = args.dev_target
dev_vocab_file = args.dev_vocab_file_name
devPickle = args.dev_pickle

sourceTestFile = args.test_source
sourceOrigTestFile = (sourceTestFile if args.test_source_orig is None else args.test_source_orig)
targetTestFile = args.test_target
test_vocab_file = args.test_vocab_file_name
testPickle = args.test_pickle

vocGenFile = args.model_vocgen
batchSize = args.bs
K = args.K
vocGenHiddenDim = args.dim_vocgen

minFreqSource = args.fs
minFreqTarget = args.ft
maxLen = args.mlen

gpuId = args.gpu
seed = args.seed


torch.set_num_threads(1)

torch.manual_seed(seed)
random.seed(seed)
torch.cuda.set_device(gpuId)
torch.cuda.manual_seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

corpus = Corpus(sourceTrainFile = sourceTrainFile, sourceOrigTrainFile = sourceOrigTrainFile, targetTrainFile = targetTrainFile,
                sourceDevFile = sourceDevFile, sourceOrigDevFile = sourceOrigDevFile, targetDevFile = targetDevFile,
                sourceTestFile = sourceTestFile, sourceOrigTestFile = sourceOrigTestFile, targetTestFile = targetTestFile,
                minFreqSource = minFreqSource, minFreqTarget = minFreqTarget, maxTokenLen = maxLen,
                trainPickle = trainPickle, devPickle = devPickle, testPickle = testPickle)

vocGen = VocGenerator(vocGenHiddenDim, corpus.targetVoc.size(), corpus.sourceVoc.size())
vocGen.load_state_dict(torch.load(vocGenFile))
vocGen.to(device)
vocGen.cuda()
vocGen.eval()

def predict_smallvoc(corpus, vocGen, batchSize, K, split = 'train'):
    if split == 'train':
        vocab_file = train_vocab_file
        data = corpus.trainData
    elif split == 'valid':
        vocab_file = dev_vocab_file
        data = corpus.devData 
    elif split == 'test':
        vocab_file = test_vocab_file
        data = corpus.testData

    tokens = corpus.targetVoc.tokenList
    print('Pre-computing small vocabularies for ' + split + ' set (requires CPU memory)...')

    def batch(iterable, n = 1):
        current_batch = []
        for item in iterable:
            current_batch.append(item)
            if len(current_batch) == n:
                yield current_batch
                current_batch = []
        if current_batch:
            yield current_batch

    for d in batch(data, batchSize):
        targetVocGen, inputVocGen = corpus.processBatchInfoVocGen(d, smoothing = False, device = device)
        outputVocGen = vocGen(inputVocGen)

        tmp = F.sigmoid(outputVocGen.data).data
        val, output_list = torch.topk(tmp, k = K)
        output_list = output_list.cpu()
        output_list = output_list.numpy()
        
        save_vocab(output_list, vocab_file, tokens)

    # for batch in batchList:
    #     batchSize = batch[1]-batch[0]+1

    #     batchInputSource, lengthsSource, batchInputTarget, batchTarget, lengthsTarget, tokenCount, batchData, maxTargetLen = corpus.processBatchInfoNMT(batch, train = False, volatile = True, device=device)

    #     targetVocGen, inputVocGen = corpus.processBatchInfoVocGen(batchData, smoothing = False, device = device)
    #     outputVocGen = vocGen(inputVocGen)

    #     tmp = F.sigmoid(outputVocGen.data).data
    #     val, output_list = torch.topk(tmp, k = K)
    #     output_list = output_list.cpu()
    #     output_list = output_list.numpy()
        
    #     save_vocab(output_list, vocab_file, tokens)

def save_vocab(output_list, filename, tokens):
    path = './vocab/' + filename
    def find_vocabs(output_list, tokens):
        return [' '.join([tokens[x].str for x in output]) + "\n" for output in output_list]

    with open(path, 'a') as fw:
        fw.writelines(find_vocabs(output_list, tokens))

predict_smallvoc(corpus = corpus, vocGen = vocGen, batchSize = batchSize, K = K, split = 'train')
predict_smallvoc(corpus = corpus, vocGen = vocGen, batchSize = batchSize, K = K, split = 'valid')
predict_smallvoc(corpus = corpus, vocGen = vocGen, batchSize = batchSize, K = K, split = 'test')



# save_vocab(corpus.trainData, train_vocab_file, tokens)
# save_vocab(corpus.devData, dev_vocab_file, tokens)
# save_vocab(corpus.testData, test_vocab_file, tokens)
