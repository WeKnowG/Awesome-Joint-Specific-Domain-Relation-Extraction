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

from config import DefaultConfig
import data
from models import HBT_model
from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.modeling import BertConfig
from pytorch_pretrained_bert.optimization import BertAdam



logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def train(**kwargs):
    '''
    训练
    '''
    pass


def val(model, dataloader):
    '''
    计算模型在验证集上的准确率等信息，用以辅助训练
    '''
    pass


def test(**kwargs):
    '''
    测试（inference）
    '''
    pass


def help():
    '''
    打印帮助的信息
    '''
    print('help')


# if __name__ == "__main__":
    # import fire
    # fire.Fire()
if __name__ == "__main__":
    train_path = 'data/train_triples.json'
    dev_path = 'data/dev_triples.json'
    test_path = 'data/test_triples.json'
    # test_path = 'data/' + dataset + '/test_split_by_num/test_triples_5.json' # ['1','2','3','4','5']
    # test_path = 'data/' + dataset + '/test_split_by_type/test_triples_seo.json' # ['normal', 'seo', 'epo']
    # test_path = 'data/' + dataset + '/test_triples.json' # overall test
    rel_dict_path = 'data/rel2id.json'

    tokenizer = BertTokenizer(vocab_file='/data2/nianxw/joint_entity_relation_extraction/Awesome-Joint-Specific-Domain-Relation-Extraction/checkpoints/cased_L-12_H-768_A-12/vocab.txt', do_lower_case=True)
    train_data, dev_data, test_data, id2rel, rel2id, num_rels = data.load_data(train_path, dev_path, test_path, rel_dict_path)
    data_manager = data.SPO(train_data, tokenizer, rel2id, num_rels)