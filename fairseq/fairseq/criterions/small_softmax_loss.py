# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch.nn.functional as F

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('small_softmax')
class SmallSoftmaxCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input']) # net_output = (decoder_output, avg_attn_scores)
        decoder_output = net_output[0].view(-1, net_output[0].size(-1)) # reshape decoder_output
        target = sample['target_dvoc_indices'].view(-1) # get and reshape target
        loss = model.decoder.dvoc_layer.computeLoss(decoder_output, target) # compute loss
        if torch.isnan(loss):
            print(decoder_output.data) 
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        loss /= sample_size
        logging_output = {
            'loss': loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def validation_forward(self, model, sample, reduce=True, greedyProb=1.0):
        """Compute the loss for the given sample for validation step.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        encoder_out = model.encoder(sample['net_input']['src_tokens'], sample['net_input']['src_lengths'])
        dvoc = sample['net_input']['dynamic_vocab']
        model.decoder.dvoc_layer.setSubset(dvoc)
        decoder_out, avg_attn_scores = model.decoder(sample['net_input']['prev_output_tokens'], encoder_out=encoder_out, dynamic_vocab=None)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        
        rnd = torch.FloatTensor(1).uniform_(0.0, 1.0)[0]
            
        def convert_back(sampled_index):
            for i, s in enumerate(sampledIndex):
                sampledIndex[i] = dvoc[i, s]
            return sampledIndex.data
            
        if rnd <= greedyProb:
            maxProb, sampledIndex = torch.max(decoder_out, dim = 1)
            #convert back to target dictionary indices
            sampledIndex = convert_back(sampledIndex)
        else:
            decoder_out = F.softmax(decoder_out, dim = 1)
            sampledIndex = torch.multinomial(decoder_out.data, num_samples = 1).squeeze(1)
            #convert back to target dictionary indices
            sampledIndex = convert_back(sampledIndex)

        decoder_out = decoder_out.view(-1, decoder_out.size(-1)) # reshape decoder_output
        target = sample['target_dvoc_indices'].view(-1)
        loss = model.decoder.dvoc_layer.computeLoss(decoder_out, target)
        loss /= sample_size

        logging_output = {
            'loss': loss,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'sampled_indices': sampledIndex,
            'target_indices': target,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        if sample_size != ntokens:
            metrics.log_scalar('nll_loss', loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        else:
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
