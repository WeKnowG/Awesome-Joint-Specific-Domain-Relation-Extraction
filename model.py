import collections
import logging
import json
import math
import os
import random
import pickle
from tqdm import tqdm, trange
import re

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch import nn
from torch.nn import NLLLoss
from torch.nn import functional as F

from Config import config_set
from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.modeling import PreTrainedBertModel,BertModel,BertConfig
from pytorch_pretrained_bert.optimization import BertAdam
from data import read_examples, convert_examples_to_features, write_predictions

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class BERT_SPO(PreTrainedBertModel):

    def __init__(self, config):
        super(BERT_SPO, self).__init__(config)
        self.bert = BertModel(config)
        self.subject_head_layer = nn.Linear(config.hidden_size, 1)
        self.subject_tail_layer = nn.Linear(config.hidden_size, 1)
        self.object_head_layer = nn.Linear(config.hidden_size, config.rel_nums)
        self.object_tail_layer = nn.Linear(config.hidden_size, config.rel_nums)

    def gather_info(self, input_tensor, positions):
        batch_size, seq_len, hidden_size = input_tensor.size()
        flat_offsets = torch.linspace(0, batch_size-1, steps=batch_size).long().view(-1, 1)*seq_len
        flat_positions = positions.long() + flat_offsets
        flat_positions = flat_positions.view(-1)
        flat_seq_tensor = input_tensor.view(batch_size*seq_len, hidden_size)
        output_tensor = torch.index_select(flat_seq_tensor, 0, flat_positions).view(batch_size, -1, hidden_size)
        return output_tensor

    def forward(self,
                input_ids,
                token_type_ids,
                attention_mask,
                positions=None,  # [batch_size, 2]
                gold_sub_heads=None,   # [batch_szie, seq_len]
                gold_sub_tails=None,   # [batch_szie, seq_len]
                gold_obj_heads=None,   # [batch, seq_len, num_rels]
                gold_obj_tails=None,   # [batch, seq_len, num_rels]
                is_train=True):  

        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        mask = attention_mask.view(-1)
        loss_fuc = nn.BCELoss(reduce=False)
        
        # subject_logits
        pred_sub_heads = F.sigmoid(self.subject_head_layer(sequence_output))  # [batch_size, seq_len, 1]
        pred_sub_tails = F.sigmoid(self.subject_tail_layer(sequence_output))

        sub_head_logits = pred_sub_heads.view(-1, 1)
        sub_tail_logits = pred_sub_tails.view(-1, 1)

        if positions is not None:
            # get entity span 
            V_subject = torch.mean(self.gather_info(sequence_output, positions), 1, keep_dim=True)  # [batch_size, 1, hidden_size]
            tokens_feature = sequence_output + V_subject  # [batch_size, seq_len, hidden_size]

            # obj_logits
            pred_obj_heads = F.sigmoid(self.object_head_layer(tokens_feature))   # [batch_size, seq, rel_nums]
            pred_obj_tails = F.sigmoid(self.object_tail_layer(tokens_feature))   # [batch_size, seq, rel_nums]

        if is_train:
            # loss1 for subject
            sub_head_golds = gold_sub_heads.unsqueeze(-1).view(-1, 1)
            sub_tail_golds = gold_sub_tails.unsqueeze(-1).view(-1, 1)

            loss1_head = loss_fuc(sub_head_logits, sub_head_golds).squeeze()  # [batch_size*seq_len]
            loss1_head = torch.sum(loss1_head * mask) / torch.sum(mask)

            loss1_tail = loss_fuc(sub_tail_logits, sub_tail_golds).squeeze()  # [batch_size*seq_len]
            loss1_tail = torch.sum(loss1_tail * mask) / torch.sum(mask)

            loss1 = loss1_head + loss1_tail

            # loss2 for object
            loss2_head = loss_fuc(pred_obj_heads, gold_obj_heads).view(-1, pred_obj_heads.shape[-1])
            loss2_head = torch.sum(loss2_head * mask) / torch.sum(mask)

            loss2_tail = loss_fuc(pred_obj_tails, gold_obj_tails).view(-1, pred_obj_tails.shape[-1])
            loss2_tail = torch.sum(loss2_tail * mask) / torch.sum(mask)

            loss2 = loss2_head + loss2_tail

            total_loss = loss1 + loss2

            return total_loss
        else:
            if positions is None:
                return pred_sub_heads, pred_sub_tails
            else:
                return pred_obj_heads, pred_obj_tails