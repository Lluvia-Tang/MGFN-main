'''
Description: 
version: 
'''
import copy
from logging import root
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .attention import StructuredAttention as StructuredAtt

Device = "cuda:0"
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class LatGatSemBertClassifier(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn_model = GCNAbsaModel(bert, opt=opt)
        self.classifier = nn.Linear(opt.bert_dim * 3, opt.polarities_dim)
        
        self.v_linear = nn.Linear(in_features=opt.bert_dim,
                                  out_features=1,
                                  bias=False)
        

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, attention_mask, src_mask, aspect_mask, lex, _, _ = inputs
        
        outputs, sem_outputs, sequence_output, adj_latent, adj_ag, loss_root, pooled_output, G_l = self.gcn_model(inputs) # 
        
        final_outputs = outputs 
        # final_outputs = torch.cat((final_outputs, sem_outputs, pooled_output), dim=-1) #  
        final_outputs = torch.cat((final_outputs, pooled_output), dim=-1)  #
        
        logits = self.classifier(final_outputs)
        
        # lex
        e = self.v_linear(G_l).squeeze(dim=2)  # (B, S)
        latent_weights = torch.nn.functional.softmax(e, dim=1) * 50  # (B, S) 
        
        mask = copy.deepcopy(src_mask)
        mask[:,0] = 0
        latent_weights = torch.mul(latent_weights , mask)

        lex = lex.to(torch.float32)
        lexicon_loss = F.mse_loss(latent_weights, lex)  
        return (logits,loss_root), lexicon_loss


class GCNAbsaModel(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        # gcn layer
        self.gcn = GCNBert(bert, opt, opt.num_layers)
        self.v_linear = nn.Linear(in_features=opt.bert_dim,
                                  out_features=1,
                                  bias=False)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, attention_mask, src_mask, aspect_mask, lex, ori_tag, head = inputs           # unpack inputs
        h, sem_h, sequence_output, adj_latent, adj_ag, loss_root, bert_output = self.gcn(inputs) 
        
        asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)
        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, self.opt.bert_dim) 
        mask = copy.deepcopy(src_mask)
        mask[:,0] = 1
        mask = mask.unsqueeze(-1).repeat(1, 1, self.opt.bert_dim) 
        h_l = h * mask
        h_sem = sem_h * mask     
        sequence_output = sequence_output * mask        
       
        beta_mat = torch.matmul(h_l, h_sem.transpose(1, 2)) / 1000 #[B,S,dim] [B,dim,S] => [B,S,S]
        beta = beta_mat.sum(1, keepdim=True)
        attn = F.softmax(beta, dim=2) #[B,1,S]   
        outputs = torch.matmul(attn, sem_h).squeeze(1) #[B,dim]
        sem_h = (sem_h * aspect_mask).sum(dim=1) / (asp_wn + 1e-9)
        return outputs, sem_h, sequence_output, adj_latent, adj_ag, loss_root, bert_output, h
        
class GCNBert(nn.Module):
    def __init__(self, bert, opt, num_layers):
        super(GCNBert, self).__init__()
        self.bert = bert
        self.opt = opt
        self.layers = num_layers
        self.bert_dim = opt.bert_dim
        self.bert_drop = nn.Dropout(opt.bert_dropout)
        self.pooled_drop = nn.Dropout(opt.bert_dropout)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)
        self.layernorm = LayerNorm(opt.bert_dim)
        self.mem_dim = self.bert_dim // 2
        
        #dep_tag adj embed
        self.dep_embed = nn.Embedding(opt.dep_size, 300)
        self.fc1 = nn.Linear(300, 256) #hidden_dim=64
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)
        
        self.linear_keys = nn.Linear(opt.bert_dim, opt.bert_dim)
        self.linear_query = nn.Linear(opt.bert_dim, opt.bert_dim)

        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.bert_dim 
            self.W.append(nn.Linear(input_dim, input_dim))

        self.attention_heads = opt.attention_heads
        self.head_dim = self.mem_dim // self.layers
        self.attn = MultiHeadAttention(self.attention_heads, self.bert_dim)

        self.str_att = StructuredAtt(opt)
        
        self.fc3 = nn.Linear(opt.bert_dim, opt.bert_dim)
        
    
    def forward(self,inputs):
        text_bert_indices, bert_segments_ids, attention_mask, src_mask, aspect_mask, lex, ori_tag, head = inputs      
        src_mask1 = src_mask.unsqueeze(-2)
        
        sequence_output, pooled_output = self.bert(text_bert_indices, attention_mask=attention_mask, token_type_ids=bert_segments_ids)
        sequence_output = self.layernorm(sequence_output)
        gcn_inputs = self.bert_drop(sequence_output)
        pooled_output = self.pooled_drop(pooled_output)
        
        fmask = src_mask
        fmask[:,0] = 1 #[B,L] src_mask

        # embedding dep adj
        dep_feature = self.dep_embed(ori_tag) # ori-tree
        dep_feature = self.fc1(dep_feature)  # [B,S,64]
        dep_feature = self.relu(dep_feature)
        dep_feature = self.fc2(dep_feature)  # (B, S, 1)
        dep_feature = dep_feature.squeeze(2)
        dep_feature = F.softmax(mask_logits(dep_feature, fmask), dim=1) #(B,S)
        dep_adj = ids2ori_adj(dep_feature, self.opt.max_length, head)
        dep_adj = np.array(dep_adj)
        dep_adj = torch.from_numpy(dep_adj) #[B,S,S]
        dep_adj = dep_adj.to(self.opt.device)   

        attn_tensor = self.attn(gcn_inputs, gcn_inputs, src_mask1)
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
        adj_ag = None

        # * Average Multi-head Attention matrixes
        for i in range(self.attention_heads):
            if adj_ag is None:
                adj_ag = attn_adj_list[i]
            else:
                adj_ag = adj_ag + attn_adj_list[i]
        adj_ag = adj_ag / self.attention_heads    

        for j in range(adj_ag.size(0)):
            adj_ag[j] -= torch.diag(torch.diag(adj_ag[j]))
            adj_ag[j] += torch.eye(adj_ag[j].size(0)).to(Device)
        adj_ag = src_mask1.transpose(1, 2) * adj_ag    

        # get latent graph
        extended_attention_mask = src_mask[:, None, None, :]
        root_mask = extended_attention_mask
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        structured_output, adj_latent, loss_root = self.str_att(
            gcn_inputs, dep_adj, extended_attention_mask, aspect_mask, root_mask)
        
        
        for j in range(adj_latent.size(0)):
            adj_latent[j] -= torch.diag(torch.diag(adj_latent[j]))
            adj_latent[j] += torch.eye(adj_latent[j].size(0)).to(Device)
 
        adj_latent = src_mask1.transpose(1, 2) * adj_latent 
        

        H_l = gcn_inputs
        
        # *********begin multul******
        for l in range(self.layers):
            si = nn.Sigmoid()
            g_l = si(H_l)
            
            # **********combine*********
            AH_sem = adj_ag.bmm(H_l)
            I_sem = self.W[l](AH_sem)
            AH_lat = adj_latent.bmm(H_l)
            I_lat = self.W[l](AH_lat)
            g = si(I_lat)
            lam_g = self.opt.lam * g # [16, 100, 768]
            I_com = torch.mul((1-lam_g),I_sem) + torch.mul(lam_g,I_lat) # 1- ?
            relu = nn.ReLU()
            H_out = relu(self.fc3(I_com))
            
            H_l = torch.mul(g_l, H_out) + torch.mul((1 - g_l),H_l)
    
        return H_l, I_sem, gcn_inputs, adj_latent, adj_ag, loss_root, pooled_output
        

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)

def ids2ori_adj(ori_tag, sent_len, head):
    adj = []
    # print(sent_len)
    for b in range(ori_tag.size()[0]):
        ret = np.ones((sent_len, sent_len), dtype='float32')
        fro_list = head[b]
        for i in range(len(fro_list) - 1):
            to = i + 1
            fro = fro_list[i]
            ret[fro][to] = ori_tag[b][i]
            ret[to][fro] =ori_tag[b][i]
        adj.append(ret)

    return adj

def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / (math.sqrt(d_k) + 1e-9)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    # p_attn = entmax15(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn

class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]

        attn = attention(query, key, mask=mask, dropout=self.dropout)
        return attn
    
    

