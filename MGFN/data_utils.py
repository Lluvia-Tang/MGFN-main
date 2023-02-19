import os
import sys

from pygments import lex
sys.path.append(r'../LAL-Parser/src_joint')
import re
import json
import pickle
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer
from torch.utils.data import Dataset

def ParseData(data_path):
    with open(data_path) as infile:
        all_data = []
        data = json.load(infile)
        for d in data:
            for aspect in d['aspects']:
                text_list = list(d['token'])
                tok = list(d['token'])       # word token
                length = len(tok)            # real length
                tok = [t.lower() for t in tok]
                tok = ' '.join(tok)
                asp = list(aspect['term'])   # aspect
                asp = [a.lower() for a in asp]
                asp = ' '.join(asp)
                label = aspect['polarity']   # label
                pos = list(d['pos'])         # pos_tag 
                head = list(d['head'])       # head
                deprel = list(d['deprel'])   # deprel
                fro = aspect['from'] # 0 for 16
                to = aspect['to']
                # position
                aspect_post = [aspect['from'], aspect['to']] 
                post = [i-aspect['from'] for i in range(aspect['from'])] \
                       +[0 for _ in range(aspect['from'], aspect['to'])] \
                       +[i-aspect['to']+1 for i in range(aspect['to'], length)]
                # aspect mask
                if len(asp) == 0:
                    mask = [1 for _ in range(length)]    # for rest16
                else:
                    mask = [0 for _ in range(aspect['from'])] \
                       +[1 for _ in range(aspect['from'], aspect['to'])] \
                       +[0 for _ in range(aspect['to'], length)]
                       
                sample = {'text': tok, 'aspect': asp, 'pos': pos, 'post': post, 'head': head,'fro': fro, 'end': to,\
                          'deprel': deprel, 'length': length, 'label': label, 'mask': mask, \
                          'aspect_post': aspect_post, 'text_list': text_list}
                all_data.append(sample)

    return all_data


class Tokenizer4BertGCN:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
    def tokenize(self, s):
        return self.tokenizer.tokenize(s)
    def convert_tokens_to_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)


class ABSAGCNData(Dataset):
    def __init__(self, fname, tokenizer, opt, dep_vocab):
        self.data = []
        parse = ParseData
        polarity_dict = {'positive':0, 'negative':1, 'neutral':2}
        for obj in tqdm(parse(fname), total=len(parse(fname)), desc="Training examples"):
            polarity = polarity_dict[obj['label']]
            term_start = obj['fro']
            term_end = obj['end']
            head = obj['head'] 
            term = obj['aspect']
            flag = len(term)  # 是否有aspect
            text_list = obj['text_list']
            left, term, right = text_list[: term_start], text_list[term_start: term_end], text_list[term_end: ]
            
            left_tokens, term_tokens, right_tokens = [], [], []
            left_tok2ori_map, term_tok2ori_map, right_tok2ori_map = [], [], [] #子词到原idx的映射

            for ori_i, w in enumerate(left):
                for t in tokenizer.tokenize(w):
                    left_tokens.append(t)                   # * ['expand', '##able', 'highly', 'like', '##ing']
                    left_tok2ori_map.append(ori_i)          # * [0, 0, 1, 2, 2]
            asp_start = len(left_tokens)  #新的idx
            offset = len(left) 
            for ori_i, w in enumerate(term):        
                for t in tokenizer.tokenize(w):
                    term_tokens.append(t)
                    # term_tok2ori_map.append(ori_i)
                    term_tok2ori_map.append(ori_i + offset)
            asp_end = asp_start + len(term_tokens)
            offset += len(term) 
            for ori_i, w in enumerate(right):
                for t in tokenizer.tokenize(w):
                    right_tokens.append(t)
                    right_tok2ori_map.append(ori_i+offset)

            while len(left_tokens) + len(right_tokens) > tokenizer.max_seq_len-2*len(term_tokens) - 3:
                if len(left_tokens) > len(right_tokens):
                    left_tokens.pop(0)
                    left_tok2ori_map.pop(0)
                else:
                    right_tokens.pop()
                    right_tok2ori_map.pop()
                    
            bert_tokens = left_tokens + term_tokens + right_tokens
            tok2ori_map = left_tok2ori_map + term_tok2ori_map + right_tok2ori_map
            truncate_tok_len = len(bert_tokens) #分词后len
            
            context_asp_ids = [tokenizer.cls_token_id]+tokenizer.convert_tokens_to_ids(
                bert_tokens)+[tokenizer.sep_token_id]+tokenizer.convert_tokens_to_ids(term_tokens)+[tokenizer.sep_token_id]
            context_asp_len = len(context_asp_ids) #总长度
            paddings = [0] * (tokenizer.max_seq_len - context_asp_len)
            context_len = len(bert_tokens)  # word长度
            context_asp_seg_ids = [0] * (1 + context_len + 1) + [1] * (len(term_tokens) + 1) + paddings
            src_mask = [0] + [1] * context_len + [0] * (opt.max_length - context_len - 1)
            aspect_mask = [0] + [0] * asp_start + [1] * (asp_end - asp_start)
            if flag == 0:
                aspect_mask = [0] + [1] * asp_start + [1] * (asp_end - asp_start)
            aspect_mask = aspect_mask + (opt.max_length - len(aspect_mask)) * [0]
            asp_start = asp_start + 1
            asp_end = asp_end + 1
            context_asp_attention_mask = [1] * context_asp_len + paddings
            context_asp_ids += paddings
            context_asp_ids = np.asarray(context_asp_ids, dtype='int64')
            context_asp_seg_ids = np.asarray(context_asp_seg_ids, dtype='int64')
            context_asp_attention_mask = np.asarray(context_asp_attention_mask, dtype='int64')
            src_mask = np.asarray(src_mask, dtype='int64')
            aspect_mask = np.asarray(aspect_mask, dtype='int64')
            
            tok_adj = np.zeros(truncate_tok_len, dtype='int64')
            ori_tag = [dep_vocab.stoi.get(t, dep_vocab.unk_index) for t in obj['deprel']] # no-reshape
            tok_tag = np.zeros(context_len, dtype='int64')
            for i in range(context_len):     
                tok_tag[i] = ori_tag[tok2ori_map[i]]
            # pad adj
            context_asp_tag = np.zeros(tokenizer.max_seq_len).astype('int64')
            pad_adj = np.zeros(context_asp_len).astype('int64')
            pad_adj[1:context_len + 1] = tok_tag
            context_asp_tag[:context_asp_len] = pad_adj
            
            #lex 
            text_list = [x.lower() for x in obj["text_list"]]
            lexicon_vector = [opt.lexicons.get(token, 0)
                                for token in text_list]
            assert len(text_list) == len(lexicon_vector)
            tok_adj = np.zeros(context_len, dtype='float32')
            for i in range(context_len):     
                tok_adj[i] = lexicon_vector[tok2ori_map[i]]
            context_lex = np.zeros(tokenizer.max_seq_len).astype('float32')
            pad_lex = np.zeros(context_asp_len).astype('float32')
            pad_lex[1:context_len + 1] = tok_adj
            context_lex[:context_asp_len] = pad_lex
            
            #adpt head (no-reshape)
            head_new = np.zeros(context_len, dtype='int64')
            for i in range(context_len):     
                head_new[i] = head[tok2ori_map[i]]
            # pad head
            head_pad = np.zeros(tokenizer.max_seq_len).astype('int64')
            pad_adj = np.zeros(context_asp_len).astype('int64')
            pad_adj[1:context_len+1] = head_new
            head_pad[:context_asp_len] = pad_adj
            
            data = {
                'text_bert_indices': context_asp_ids,
                'bert_segments_ids': context_asp_seg_ids,
                'attention_mask': context_asp_attention_mask,
                'head': head_pad,
                'ori_tag':context_asp_tag, # no-reshape 
                'src_mask': src_mask,
                'aspect_mask': aspect_mask, #root_label
                'polarity': polarity,
                'lex':context_lex,
            }
            self.data.append(data)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    
def build_senticNet():
    file_path = ['./dataset/opinion_lexicon/SenticNet/negative.txt',
                 './dataset/opinion_lexicon/SenticNet/positive.txt']
    datalist1 = [x.strip().split('\t') for x in open(file_path[0]).readlines()]
    datalist2 = [x.strip().split('\t') for x in open(file_path[1]).readlines()]
    data_list = datalist1 + datalist2
    lexicon_dict = {}
    for key, val in data_list:
        lexicon_dict[key] = abs(float(val))
        # lexicon_dict[key] = float(val)
    return lexicon_dict
    