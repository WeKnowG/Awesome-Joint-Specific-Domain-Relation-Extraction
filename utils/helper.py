#! -*- coding:utf-8 -*-
import torch
import logging
import numpy as np
from tqdm import tqdm
import json
import unicodedata
from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BertTokenizer


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

BERT_MAX_LEN = 200


class BERT_Tokenizer(BertTokenizer):
    def _tokenize(self, text):
        spaced = ''
        for ch in text:
            if ord(ch) == 0 or ord(ch) == 0xfffd or self._is_control(ch):
                continue
            else:
                spaced += ch
        split_tokens = []
        for word in spaced.strip().split():
            split_tokens += self.wordpiece_tokenizer.tokenize(word)
            split_tokens.append('[unused1]')      
        return split_tokens

    def _is_control(self, ch):
        return unicodedata.category(ch) in ('Cc', 'Cf')


def extract_items(bert, subject_model, object_model, tokenizer, text_in, id2rel, h_bar=0.5, t_bar=0.5):
    text = tokenizer.tokenize(text_in)
    if len(text) > BERT_MAX_LEN - 2:
        text = text[: BERT_MAX_LEN - 2]
    tokens = ["[CLS]"]
    segment_ids = [0]
    for word in text:
        tokens.append(word)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1]*len(input_ids)
    token_ids, segment_ids, input_mask = np.array([input_ids]), np.array([segment_ids]), np.array([input_mask])   # [1, seq_len]

    token_ids = torch.tensor(token_ids).long().cuda()
    segment_ids = torch.tensor(segment_ids).long().cuda()
    input_mask = torch.tensor(input_mask).float().cuda()

    bert_output = bert(token_ids, segment_ids, input_mask)
    sub_heads_logits, sub_tails_logits = subject_model(bert_output)  # [1, seq_len]
    sub_heads_logits = sub_heads_logits.cpu().numpy()
    sub_tails_logits = sub_tails_logits.cpu().numpy()
    sub_heads, sub_tails = np.where(sub_heads_logits[0] > h_bar)[0], np.where(sub_tails_logits[0] > t_bar)[0]  # 返回的是索引
    subjects = []
    for sub_head in sub_heads:
        sub_tail = sub_tails[sub_tails >= sub_head]  # 对应位置比较，仅返回大的值
        if len(sub_tail) > 0:
            sub_tail = sub_tail[0]
            subject = tokens[sub_head: sub_tail]
            subjects.append((subject, sub_head, sub_tail))  # 获取头实体信息
    logger.info('*********lala***********')
    if subjects:
        logger.info('**********************')
        logger.info(len(subjects))
        triple_list = []
        positions = torch.tensor(np.array([sub[1:] for sub in subjects])).cuda()  # [len(subjects), 2]
        obj_heads_logits, obj_tails_logits = object_model(bert_output, positions)  # [len(subjects), seq_len, num_rels]
        obj_heads_logits = obj_heads_logits.cpu().numpy()
        obj_tails_logits = obj_tails_logits.cpu().numpy()

        for i, subject in enumerate(subjects):
            logger.info(i)
            sub = subject[0]
            sub = ''.join([i.lstrip("##") for i in sub])
            sub = ' '.join(sub.split('[unused1]'))  # 还原头部实体词
            obj_heads, obj_tails = np.where(obj_heads_logits[i] > h_bar), np.where(obj_tails_logits[i] > t_bar)
            for obj_head, rel_head in zip(*obj_heads):
                for obj_tail, rel_tail in zip(*obj_tails):
                    if obj_head <= obj_tail and rel_head == rel_tail:
                        rel = id2rel[rel_head]
                        obj = tokens[obj_head: obj_tail]
                        obj = ''.join([i.lstrip("##") for i in obj])
                        obj = ' '.join(obj.split('[unused1]'))
                        triple_list.append((sub, rel, obj))
                        break
        triple_set = set()
        for s, r, o in triple_list:
            triple_set.add((s, r, o))
        return list(triple_set)
    else:
        return []


def partial_match(pred_set, gold_set):
    pred = {(i[0].split(' ')[0] if len(i[0].split(' ')) > 0 else i[0], i[1], i[2].split(' ')[0] if len(i[2].split(' ')) > 0 else i[2]) for i in pred_set}
    gold = {(i[0].split(' ')[0] if len(i[0].split(' ')) > 0 else i[0], i[1], i[2].split(' ')[0] if len(i[2].split(' ')) > 0 else i[2]) for i in gold_set}
    return pred, gold


def metric(bert, subject_model, object_model, eval_data, id2rel, tokenizer, exact_match=False, output_path=None):
    if output_path:
        F = open(output_path, 'w')
    orders = ['subject', 'relation', 'object'] 
    correct_num, predict_num, gold_num = 1e-10, 1e-10, 1e-10
    for line in tqdm(iter(eval_data)):
        Pred_triples = set(extract_items(bert, subject_model, object_model, tokenizer, line['text'], id2rel))
        Gold_triples = set(line['triple_list'])

        Pred_triples_eval, Gold_triples_eval = partial_match(Pred_triples, Gold_triples) if not exact_match else (Pred_triples, Gold_triples)        

        correct_num += len(Pred_triples_eval & Gold_triples_eval)
        predict_num += len(Pred_triples_eval)
        gold_num += len(Gold_triples_eval)

        if output_path:
            result = json.dumps({
                'text': line['text'],
                'triple_list_gold': [
                    dict(zip(orders, triple)) for triple in Gold_triples
                ],
                'triple_list_pred': [
                    dict(zip(orders, triple)) for triple in Pred_triples
                ],
                'new': [
                    dict(zip(orders, triple)) for triple in Pred_triples - Gold_triples
                ],
                'lack': [
                    dict(zip(orders, triple)) for triple in Gold_triples - Pred_triples
                ]
            }, ensure_ascii=False, indent=4)
            F.write(result + '\n')
    if output_path:
        F.close()

    precision = correct_num / predict_num
    recall = correct_num / gold_num
    f1_score = 2 * precision * recall / (precision + recall)

    # print(f'correct_num:{correct_num}\npredict_num:{predict_num}\ngold_num:{gold_num}')
    return precision, recall, f1_score