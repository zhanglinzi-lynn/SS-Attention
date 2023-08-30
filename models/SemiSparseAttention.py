import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random


# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/SubLayers.py

class ScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn, v

def similarity(x, y):
    return 1 / (1 + np.sqrt(np.sum(np.square(x - y))))

def SparseSemiWeight(data, t=50, phi=20, random_state=None):
    if random_state:
        random.seed(random_state)
        np.random.seed(random_state)

    n = data.shape[0]
    m = data.shape[1]
    x = np.reshape(np.arange(n), [n, 1])
    y = np.reshape(np.arange(m), [1, m])
    z = np.zeros([n, m])
    q = z + x
    k = z + y
    c1 = q >= k
    c2 = np.equal(np.fmod(q - k, 2), 0)
    c3 = np.logical_and(c1, c2)


    data_score = np.zeros((n, 1))
    sample = set()
    for i in range(n):
        score = 0
        for j in range(t):
            sample.clear()
            while len(sample) < phi and len(sample) < n:
                sample.add(np.random.randint(0, n - 1))
            nn_sim = 0
            for each in sample:
                nn_sim = max(nn_sim, similarity(data[i, :], data[each, :]))
            score += nn_sim
        if score:
            data_score[i] = (t / score)

    data_score = c3 * np.array(data_score)

    return data_score



class SparseMultiHeadAttention(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.2))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv


        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn, origin_v = self.attention(q, k, v, mask=mask)

        for i in range(d_v):
            origin_v[:, :, i] = torch.tensor(SparseSemiWeight(origin_v[:, :, i].detach().numpy()))



        output = torch.bmm(attn, origin_v).view(n_head, sz_b, len_q, d_v)


        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)  # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output
