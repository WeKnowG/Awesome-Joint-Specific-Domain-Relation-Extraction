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
from models import HBT
from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from pytorch_pretrained_bert.modeling import BertConfig
from pytorch_pretrained_bert.optimization import BertAdam

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


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


if __name__ == "__main__":
    config = DefaultConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = config.train_data_root
    # dev_path = config.dev_data_root
    # test_path = config.test_data_root
    # test_path = 'data/' + dataset + '/test_split_by_num/test_triples_5.json' # ['1','2','3','4','5']
    # test_path = 'data/' + dataset + '/test_split_by_type/test_triples_seo.json' # ['normal', 'seo', 'epo']
    # test_path = 'data/' + dataset + '/test_triples.json' # overall test
    rel_dict_path = config.rel_data

    bert_config = BertConfig.from_json_file(config.bert_config_file)
    tokenizer = BertTokenizer(vocab_file=config.bert_vocab_file, do_lower_case=True)
    id2rel, rel2id, num_rels = data.load_rel(rel_dict_path)

    if config.train:
        if os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)
        if os.path.exists('./data/train_file.pkl'):
            train_data = pickle.load(open("./data/train_file.pkl", mode='rb'))
        else:
            train_data = data.load_data(train_path, tokenizer, rel2id, num_rels)
            pickle.dump(train_data, open("./data/train_file.pkl", mode='wb'))

        data_manager = data.SPO(train_data)
        train_sampler = RandomSampler(data_manager)
        train_data_loader = DataLoader(data_manager, sampler=train_sampler, batch_size=config.batch_size, drop_last=True)
        num_train_steps = int(len(data_manager) / config.batch_size) * config.max_epoch
        model = HBT(bert_config, config)

        if config.bert_pretrained_model is not None:
            logger.info('load bert weight')
            model.bert.from_pretrained(config.bert_pretrained_model)

        model.to(device)

        param_optimizer = list(model.named_parameters())

        # hack to remove pooler, which is not used
        # thus it produce None grad that break apex
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=config.lr,
                             warmup=config.warmup_proportion,
                             t_total=num_train_steps)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(data_manager))
        logger.info("  Num Epochs = %d", config.max_epoch)
        logger.info("  Total train batch size = %d", config.batch_size)
        logger.info("  Total optimization steps = %d", num_train_steps)
        logger.info("  Logging steps = %d", config.print_freq)
        logger.info("  Save steps = %d", config.save_freq)

        global_step = 0
        model.train()
        for _ in range(config.max_epoch):
            model.zero_grad()
            epoch_itorator = tqdm(train_data_loader, disable=None)
            for step, batch in enumerate(epoch_itorator):
                batch = tuple(t.to(device) for t in batch)
                input_ids, segment_ids, input_masks, sub_positions, sub_heads, sub_tails, obj_heads, obj_tails = batch
                loss1, loss2 = model(input_ids, segment_ids, input_masks, sub_positions, sub_heads, sub_tails, obj_heads, obj_tails)
                loss = loss1 + loss2
                loss.backward()
                lr_this_step = config.lr * warmup_linear(global_step/num_train_steps, config.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if (step+1) % config.print_freq == 0:
                    logger.info("epoch : {} step: {} #### loss1: {}  loss2: {}".format(_, step, loss1.cpu().item(), loss2.cpu().item()))
                if (global_step + 1) % config.save_freq == 0:
                    # Save a trained model
                    model_name = "pytorch_model_%d.bin" % (global_step + 1)
                    output_model_file = os.path.join(config.output_dir, model_name)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    torch.save(model_to_save.state_dict(), output_model_file)
        model_name = "pytorch_model_%d.bin" % (global_step + 1)
        output_model_file = os.path.join(config.output_dir, model_name)
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        torch.save(model_to_save.state_dict(), output_model_file)
    
    if config.dev:
        model_state_dict = torch.load(config.output_model_file)
        model = HBT(bert_config, config)
