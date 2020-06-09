import logging
import os
import pickle
import json
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from config import DefaultConfig
import data
import utils
from models import bert, sub_model, obj_model
from pytorch_pretrained_bert.modeling import BertConfig
from pytorch_pretrained_bert.optimization import BertAdam

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x


def train(config, bert_config, train_path, dev_path, rel2id, id2rel, tokenizer):
    if os.path.exists(config.output_dir) is False:
        os.makedirs(config.output_dir, exist_ok=True)
    if os.path.exists('./data/train_file.pkl'):
        train_data = pickle.load(open("./data/train_file.pkl", mode='rb'))
    else:
        train_data = data.load_data(train_path, tokenizer, rel2id, num_rels)
        pickle.dump(train_data, open("./data/train_file.pkl", mode='wb'))
    dev_data = json.load(open(dev_path))
    for sent in dev_data:
        data.to_tuple(sent)
    data_manager = data.SPO(train_data)
    train_sampler = RandomSampler(data_manager)
    train_data_loader = DataLoader(data_manager, sampler=train_sampler, batch_size=config.batch_size, drop_last=True)
    num_train_steps = int(len(data_manager) / config.batch_size) * config.max_epoch

    if config.bert_pretrained_model is not None:
        logger.info('load bert weight')
        bert_model = bert.from_pretrained(config.bert_pretrained_model, bert_config=bert_config)
    else:
        logger.info('random initialize bert model')
        bert_model = bert(bert_config)
        bert_model.apply(bert_model.init_bert_weights)
    bert_model.to(device)
    submodel = sub_model(config).to(device)
    objmodel = obj_model(config).to(device)

    loss_fuc = nn.BCELoss(reduction='none')
    # # hack to remove pooler, which is not used
    # # thus it produce None grad that break apex
    # param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    #     ]
    params = list(bert_model.parameters()) + list(submodel.parameters()) + list(objmodel.parameters())
    optimizer = BertAdam(params,
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
    bert_model.train()
    submodel.train()
    objmodel.train()

    for _ in range(config.max_epoch):
        optimizer.zero_grad()
        epoch_itorator = tqdm(train_data_loader, disable=None)
        for step, batch in enumerate(epoch_itorator):
            batch = tuple(t.to(device) for t in batch)
            input_ids, segment_ids, input_masks, sub_positions, sub_heads, sub_tails, obj_heads, obj_tails = batch

            bert_output = bert_model(input_ids, segment_ids, input_masks)
            pred_sub_heads, pred_sub_tails = submodel(bert_output)  # [batch_size, seq_len, 1]
            pred_obj_heads, pred_obj_tails = objmodel(bert_output, sub_positions)

            # 计算loss
            mask = input_masks.view(-1)

            # loss1
            sub_heads = sub_heads.unsqueeze(-1)  # [batch_szie, seq_len, 1]
            sub_tails = sub_tails.unsqueeze(-1)

            loss1_head = loss_fuc(pred_sub_heads, sub_heads).view(-1)
            loss1_head = torch.sum(loss1_head*mask) / torch.sum(mask)

            loss1_tail = loss_fuc(pred_sub_tails, sub_tails).view(-1)
            loss1_tail = torch.sum(loss1_tail*mask) / torch.sum(mask)

            loss1 = loss1_head + loss1_tail

            # loss2
            loss2_head = loss_fuc(pred_obj_heads, obj_heads).view(-1, obj_heads.shape[-1])
            loss2_head = torch.sum(loss2_head * mask.unsqueeze(-1)) / torch.sum(mask)

            loss2_tail = loss_fuc(pred_obj_tails, obj_tails).view(-1, obj_tails.shape[-1])
            loss2_tail = torch.sum(loss2_tail * mask.unsqueeze(-1)) / torch.sum(mask)

            loss2 = loss2_head + loss2_tail

            # optimize
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            if (step+1) % config.print_freq == 0:
                logger.info("epoch : {} step: {} #### loss1: {}  loss2: {}".format(_, step + 1, loss1.cpu().item(), loss2.cpu().item()))

            if (step + 1) % config.eval_freq == 0:
                logger.info("***** Running evaluating *****")
                with torch.no_grad():
                    bert_model.eval()
                    submodel.eval()
                    objmodel.eval()
                    P, R, F1 = utils.metric(bert_model, submodel, objmodel, dev_data, id2rel, tokenizer)
                    logger.info(f'precision:{P}\nrecall:{R}\nF1:{F1}')
                bert_model.train()
                submodel.train()
                objmodel.train()

            if (global_step + 1) % config.save_freq == 0:
                # Save a trained model
                model_name = "pytorch_model_%d" % (global_step + 1)
                output_model_file = os.path.join(config.output_dir, model_name)
                state = {
                    'bert_state_dict': bert_model.state_dict(),
                    'subject_state_dict': submodel.state_dict(),
                    'object_state_dict': objmodel.state_dict(),
                }
                torch.save(state, output_model_file)

    model_name = "pytorch_model_last"
    output_model_file = os.path.join(config.output_dir, model_name)
    state = {
        'bert_state_dict': bert_model.state_dict(),
        'subject_state_dict': submodel.state_dict(),
        'object_state_dict': objmodel.state_dict(),
    }
    torch.save(state, output_model_file)


def dev(config, bert_config, dev_path, id2rel, tokenizer, output_path=None):
    dev_data = json.load(open(data_path))
    bert_model = bert(bert_config)
    submodel = sub_model(config)
    objmodel = obj_model(config)

    state = torch.load(os.path.join(config.output_dir, "pytorch_model_last"))
    bert_model.load_state_dict(state['bert_state_dict'])
    submodel.load_state_dict(state['subject_state_dict'])
    objmodel.load_state_dict(state['object_state_dict'])

    precision, recall, f1 = utils.metric(bert, submodel, objmodel, dev_data, id2rel, tokenizer, output_path=output_path)
    logger.info('precision: %.4f' % precision)
    logger.info('recall: %.4f' % recall)
    logger.info('F1: %.4f' % f1)


if __name__ == "__main__":
    config = DefaultConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = config.train_data_root
    dev_path = config.dev_data_root
    # test_path = config.test_data_root
    # test_path = 'data/' + dataset + '/test_split_by_num/test_triples_5.json' # ['1','2','3','4','5']
    # test_path = 'data/' + dataset + '/test_split_by_type/test_triples_seo.json' # ['normal', 'seo', 'epo']
    # test_path = 'data/' + dataset + '/test_triples.json' # overall test
    rel_dict_path = config.rel_data

    bert_config = BertConfig.from_json_file(config.bert_config_file)
    tokenizer = utils.BERT_Tokenizer(vocab_file=config.bert_vocab_file, do_lower_case=False)
    id2rel, rel2id, num_rels = data.load_rel(rel_dict_path)

    if config.train:
        train(config, bert_config, train_path, dev_path, rel2id, id2rel, tokenizer)

    # if config.dev:
    #     dev(config, bert_config, dev_path, id2rel, tokenizer)
