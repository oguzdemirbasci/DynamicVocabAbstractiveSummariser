from data import Corpus
from model import VocGenerator
import utils

import torch
import torch.nn as nn
import torch.optim as optim

import random
import time
import argparse
import os
import logging

parser = argparse.ArgumentParser(description='Learning vocabulary predictor for machine translation')

parser.add_argument('--seed', type = int, default = 1,
                    help='Random seed')
parser.add_argument('--gpu', type = int, default = 0,
                    help='GPU id')

parser.add_argument('--train_source', type = str, required = True,
                    help = 'File path to training data (source sentences)')
parser.add_argument('--train_target', type = str, required = True,
                    help = 'File path to training data (target sentences)')
parser.add_argument('--train_pickle', type = str, default = '',
                    help = 'File path to preprocessed train dataset')

parser.add_argument('--dev_source', type = str, required = True,
                    help = 'File path to development data (source sentences)')
parser.add_argument('--dev_target', type = str, required = True,
                    help = 'File path to development data (target sentences)')
parser.add_argument('--dev_pickle', type = str, default = '',
                    help = 'File path to preprocessed development dataset')

parser.add_argument('--data_cat', type=str, default='film', 
                    help='WikiCatSum categories: Animal, Film or Company')

parser.add_argument('--model', type = str, default = 'vocgen.bin',
                    help = 'File name for saving model parameters')

parser.add_argument('--fs', type = int, default = '2',
                    help = 'Minimum word frequency to construct source vocabulary')
parser.add_argument('--ft', type = int, default = '2',
                    help = 'Minimum word frequency to construct target vocabulary')
parser.add_argument('--mlen', type = int, default = '10000',
                    help = 'Maximum length of sentences in training data')

parser.add_argument('--K', type = int, default = '1000',
                    help = 'Small vocabulary size for evaluating recall')
parser.add_argument('--dim', type = int, default = '256',
                    help = 'Dimensionality for embeddings and hidden states')
parser.add_argument('--mepoch', type = int, default = '20',
                    help = 'Maximum number of training epochs')
parser.add_argument('--lr', type = float, default = '8.0e-02',
                    help = 'Learning rate for AdaGrad')
parser.add_argument('--lrd', type = float, default = '0.5',
                    help = 'Learning rate decay for AdaGrad')
parser.add_argument('--bs', type = int, default = '128',
                    help = 'Batch size')
parser.add_argument('--dp', type = float, default = '0.4',
                    help = 'Dropout rate for residual block')
parser.add_argument('--wd', type = float, default = '1.0e-06',
                    help = 'Weight decay rate for internal weight matrices')
parser.add_argument('--clip', type = float, default = '1.0',
                    help = 'Clipping value for gradient norm')

args = parser.parse_args()
print(args)

sourceDevFile = args.dev_source
sourceOrigDevFile = sourceDevFile
targetDevFile = args.dev_target

sourceTrainFile = args.train_source
sourceOrigTrainFile = sourceTrainFile
targetTrainFile = args.train_target

trainPickle = args.train_pickle
devPickle = args.dev_pickle

last_path = "/"

if "bow" == sourceTrainFile[-7:-4]:
    last_path = "_bow/"
elif "raw" == sourceOrigTrainFile[-7:-4]:
    last_path = "_raw/"

file_path = './params/' + args.data_cat + "/outputs" + last_path

if not os.path.exists(file_path):
    os.makedirs(file_path)

dirs = os.listdir(file_path)

if not dirs:
    exp_no = 0 
else:
    exps = [int(x[3:]) for x in os.listdir(file_path)]
    exp_no = max(exps) + 1
    

file_path = file_path + "exp" + str(exp_no) + "/"

vocGenFile = file_path + args.model

minFreqSource = args.fs
minFreqTarget = args.ft
decay = args.lrd
gradClip = args.clip
dropoutRate = args.dp

maxLen = args.mlen
maxEpoch = args.mepoch

vocGenHiddenDim = args.dim

batchSize = args.bs

learningRateVocGen = args.lr

gpuId = args.gpu
seed = args.seed
######
device = torch.device('cuda:'+str(gpuId))
cpu = torch.device('cpu')
######
weightDecay = args.wd

K = args.K

os.makedirs(file_path)
logging.basicConfig(level=logging.INFO, filename=file_path+"log"+str(exp_no)+".log", format='%(message)s')

torch.set_num_threads(1)

torch.manual_seed(seed)
random.seed(seed)
torch.cuda.set_device(gpuId)
torch.cuda.manual_seed(seed)

corpus = Corpus(sourceTrainFile, sourceOrigTrainFile, targetTrainFile,
                sourceDevFile, sourceOrigDevFile, targetDevFile,
                minFreqSource, minFreqTarget, maxLen, trainPickle, devPickle)

print('Source vocabulary size: '+str(corpus.sourceVoc.size()))
print('Target vocabulary size: '+str(corpus.targetVoc.size()))
print()
print('# of training sentences: '+str(len(corpus.trainData)))
print('# of develop sentences:  '+str(len(corpus.devData)))
print('Random seed: ', str(seed))
print('K = ', K)
print()

logging.info('K:' + str(K) + ', dim:' + str(vocGenHiddenDim) + ', mepoch:' + str(maxEpoch) + ', lr:' + str(learningRateVocGen) + ', lrd:' + str(decay) + 
                ', bs:' + str(batchSize) + ', dp:' + str(dropoutRate) + ', wd:' + str(weightDecay) + ', clip:' + str(gradClip))
logging.info('')
logging.info("train src: " + sourceTrainFile.rsplit('/', 1)[-1] + ", train tgt: " + targetTrainFile.rsplit('/', 1)[-1])
logging.info("dev src: " + sourceDevFile.rsplit('/', 1)[-1] + ", dev tgt: " + targetDevFile.rsplit('/', 1)[-1])
logging.info('')

vocGen = VocGenerator(vocGenHiddenDim, corpus.targetVoc.size(), corpus.sourceVoc.size(), dropoutRate)
vocGen.to(device)

batchListTrain = utils.buildBatchList(len(corpus.trainData), batchSize)
batchListDev = utils.buildBatchList(len(corpus.devData), batchSize)

withoutWeightDecay = []
withWeightDecay = []

for name, param in vocGen.named_parameters():
    if 'bias' in name or 'softmax' in name or 'Embedding' in name:
        withoutWeightDecay += [param]
    else:
        withWeightDecay += [param]
optParams = [{'params': withWeightDecay, 'weight_decay': weightDecay},
             {'params': withoutWeightDecay, 'weight_decay': 0.0}]

totalParamsVocGen = withoutWeightDecay+withWeightDecay
optVocGen = optim.Adagrad(optParams, lr = learningRateVocGen)

bestDevRecall = -1.0
prevDevRecall = -1.0

i_epoch = 1
best_epoc = 0
for epoch in range(maxEpoch):
    batchProcessed = 0
    
    print('--- Epoch ' + str(epoch+1))

    random.shuffle(corpus.trainData)

    vocGen.train()

    for batch in batchListTrain:
        print('\r', end = '')
        
        batchSize = batch[1]-batch[0]+1

        batchInputSource, lengthsSource, batchInputTarget, batchTarget, lengthsTarget, tokenCount, batchData, maxTargetLen = corpus.processBatchInfoNMT(batch, train = True, device=device)
        
        targetVocGen, inputVocGen = corpus.processBatchInfoVocGen(batchData, device=device)
        outputVocGen = vocGen(inputVocGen)
        
        lossVocGen = vocGen.computeLoss(outputVocGen, targetVocGen)
        lossVocGen /= batchSize

        optVocGen.zero_grad()
        lossVocGen.backward()
        nn.utils.clip_grad_norm(totalParamsVocGen, gradClip)
        optVocGen.step()

        batchProcessed += 1
        if batchProcessed == len(batchListTrain)//2 or batchProcessed == len(batchListTrain):
            totalTokenCount = 0.0

            vocGen.eval()
            
            recall =  0.0

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
            vocGen.train()

            if devRecall < prevDevRecall:
                print('learning rate -> ' + str(learningRateVocGen*decay))
                learningRateVocGen *= decay
                
                for paramGroup in optVocGen.param_groups:
                    paramGroup['lr'] = learningRateVocGen

            elif devRecall >= bestDevRecall:
                bestDevRecall = devRecall
                best_epoc = i_epoch

                stateDict = vocGen.state_dict()
                for elem in stateDict:
                    stateDict[elem] = stateDict[elem].cpu()
                torch.save(stateDict, vocGenFile)

            prevDevRecall = devRecall
        
    i_epoch = i_epoch + 1

logging.info("--- Epoch " + str(best_epoc) + ' (Max Recall)')
logging.info('')
logging.info("Recall of voc predictor:" + str(bestDevRecall) + '(%)')
logging.info('')
logging.info('')
logging.info("--- Epoch 20")
logging.info('')
logging.info("Recall of voc predictor:" + str(devRecall) + '(%)')           
