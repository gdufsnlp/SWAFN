import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.w = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v = nn.Linear(self.hidden_dim, 1)

    def forward(self,h, l):

        # h = torch.cat([l, v, a], dim=2)
        w = F.tanh(self.w(h))
        weight = self.v(w)
        # print(weight.size())
        weight = weight.squeeze(dim=-1)
        # print(weight.size())
        # weight = self.Mask(weight, seq_len, mode='add')
        # print(weight.size())
        weight = F.softmax(weight)
        weight1 = weight.detach().cpu().data.numpy()
        weight = weight.unsqueeze(dim=-1)

        out = torch.mul(l, weight.repeat(1, 1, l.size(2)))
        # print(out.size())
        out = torch.sum(out, dim=1)
        # print(out.size())
        # out = self.Mask(out, seq_len, mode='mul')
        return out, weight1

class Wordnet_Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Wordnet_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.w = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, hiddens):
        w = F.relu(self.w(hiddens))
        w = self.dropout(w)
        weight = self.v(w)
        # print(weight.size())
        weight = weight.squeeze(dim=-1)
        # print(weight.size())
        # weight = self.Mask(weight, seq_len, mode='add')
        # print(weight.size())
        weight = F.softmax(weight, dim=2)
        weight1 = weight.detach().cpu().data.numpy()
        weight = weight.unsqueeze(dim=-1)
        # print(weight.repeat(1, 1, 1, hiddens.size(3)).size())
        out = torch.mul(hiddens, weight.repeat(1, 1, 1, hiddens.size(3)))
        # print(out.size())
        out = torch.sum(out, dim=2)
        # print(out.size())
        # out = self.Mask(out, seq_len, mode='mul')
        return out, weight1
