import os
import numpy as np
import json
from tqdm import tqdm
from random import choice
import torch
from torch.utils import data

BERT_MAX_LEN = 200
RANDOM_SEED = 2019
cur_path = os.path.dirname(os.path.abspath(__file__))


def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


def to_tuple(sent):
    triple_list = []
    for triple in sent['triple_list']:
        triple_list.append(tuple(triple))
    sent['triple_list'] = triple_list


def seq_padding(seq, padding=0):
    return np.array(np.concatenate([seq, [padding] * (BERT_MAX_LEN - len(seq))]) if len(seq) < BERT_MAX_LEN else seq)


def load_data(data_path, tokenizer, rel2id, num_rels, maxlen=100):
    data = json.load(open(data_path))
    random_order = list(range(len(data)))
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(random_order)
    data = [data[i] for i in random_order]
    for sent in data:
        to_tuple(sent)
    data = process_data(data, tokenizer, rel2id, num_rels, maxlen)
    return data


def load_rel(rel_dict_path):
    id2rel, rel2id = json.load(open(rel_dict_path))
    id2rel = {int(i): j for i, j in id2rel.items()}
    num_rels = len(id2rel)
    return id2rel, rel2id, num_rels


class Example:
    def __init__(self, input_ids, segment_ids, input_masks, 
                 sub_heads=None, sub_tails=None, 
                 sub_positions=None, obj_heads=None, obj_tails=None):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.input_masks = input_masks
        self.sub_heads = sub_heads
        self.sub_tails = sub_tails
        self.sub_positions = sub_positions
        self.obj_heads = obj_heads
        self.obj_tails = obj_tails


def process_data(data, tokenizer, rel2id, num_rels, maxlen=100):
    idxs = list(range(len(data)))
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(idxs)

    examples = []

    for idx in tqdm(idxs):
        line = data[idx]
        text = ' '.join(line['text'].split()[:maxlen])
        triples = line["triple_list"]

        text = tokenizer.tokenize(text)
        if len(text) > BERT_MAX_LEN - 2:
            text = text[: BERT_MAX_LEN - 2]

        tokens = ["[CLS]"]
        segments = [0]
        for word in text:
            tokens.append(word)
            segments.append(0)
        tokens.append("[SEP]")
        segments.append(0)

        s2ro_map = {}
        for triple in triples:
            triple = (tokenizer.tokenize(triple[0]), triple[1], tokenizer.tokenize(triple[2]))
            sub_head_idx = find_head_idx(tokens, triple[0])
            obj_head_idx = find_head_idx(tokens, triple[2])
            if sub_head_idx != -1 and obj_head_idx != -1:
                sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1)
                if sub not in s2ro_map:
                    s2ro_map[sub] = []
                s2ro_map[sub].append((obj_head_idx,
                                      obj_head_idx + len(triple[2]) - 1,
                                      rel2id[triple[1]]))

        if s2ro_map:
            text_len = len(tokens)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            segment_ids = segments
            input_masks = [1] * text_len
            sub_heads = np.zeros(text_len)
            sub_tails = np.zeros(text_len)
            for s in s2ro_map:
                sub_heads[s[0]] = 1
                sub_tails[s[1]] = 1
            sub_positions = choice(list(s2ro_map.keys()))
            obj_heads = np.zeros((text_len, num_rels))
            obj_tails = np.zeros((text_len, num_rels))
            for ro in s2ro_map.get(tuple(sub_positions), []):
                obj_heads[ro[0]][ro[2]] = 1
                obj_tails[ro[1]][ro[2]] = 1
            # pad
            while len(input_ids) < BERT_MAX_LEN:
                input_ids.append(0)
                segment_ids.append(0)
                input_masks.append(0)
            sub_heads = seq_padding(sub_heads)
            sub_tails = seq_padding(sub_tails)
            obj_heads = seq_padding(obj_heads, np.zeros(num_rels))
            obj_tails = seq_padding(obj_tails, np.zeros(num_rels))

            example = Example(
                input_ids=input_ids,
                segment_ids=segment_ids,
                input_masks=input_masks,
                sub_heads=sub_heads,
                sub_tails=sub_tails,
                sub_positions=sub_positions,
                obj_heads=obj_heads,
                obj_tails=obj_tails
            )

            examples.append(example)
    return examples


class SPO(data.Dataset):

    def __init__(self, data, ):
        self.data = data

    def __getitem__(self, index):
        example = self.data[index]
        input_ids = torch.tensor(example.input_ids, dtype=torch.long)
        segment_ids = torch.tensor(example.segment_ids, dtype=torch.long)
        input_masks = torch.tensor(example.input_masks, dtype=torch.float)
        sub_heads = torch.tensor(example.sub_heads, dtype=torch.float)
        sub_tails = torch.tensor(example.sub_tails, dtype=torch.float)
        sub_positions = torch.tensor(example.sub_positions, dtype=torch.long)
        obj_heads = torch.tensor(example.obj_heads, dtype=torch.float)
        obj_tails = torch.tensor(example.obj_tails, dtype=torch.float)
        return input_ids, segment_ids, input_masks, sub_positions, sub_heads, sub_tails, obj_heads, obj_tails

    def __len__(self):
        return len(self.data)
