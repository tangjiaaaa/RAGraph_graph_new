import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import AvgReadout
from mp.cell_mp import CochainConv  # CWN 里核心的 CochainConv
import numpy as np

# 多层 CochainConv
class CochainEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, activation=F.relu):
        super(CochainEncoder, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        for _ in range(num_layers):
            self.layers.append(CochainConv(in_dim, hidden_dim))
            in_dim = hidden_dim  # 下层用上层输出作为输入

    def forward(self, cochain_params):
        x = cochain_params.x
        for conv in self.layers:
            x = conv(cochain_params, x)
            x = self.activation(x)
        return x

# 完整 PrePrompt Complex 版
class PrePromptComplex(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, activation=F.relu):
        super(PrePromptComplex, self).__init__()
        self.encoder_0 = CochainEncoder(in_dim, hidden_dim, num_layers, activation)  # 0-dim
        self.encoder_1 = CochainEncoder(in_dim, hidden_dim, num_layers, activation)  # 1-dim
        self.encoder_2 = CochainEncoder(in_dim, hidden_dim, num_layers, activation)  # 2-dim (胞腔)

        self.read = AvgReadout()

    def forward(self, complex_batch, negative_sample, temperature=1.5):
        # 取每一阶 cochain
        params0 = complex_batch.get_cochain_params(0)
        params1 = complex_batch.get_cochain_params(1)
        params2 = complex_batch.get_cochain_params(2)

        h0 = self.encoder_0(params0)
        h1 = self.encoder_1(params1)
        h2 = self.encoder_2(params2)

        # 可以选择做哪个阶对比
        logits = self.compareloss(h2, negative_sample, temperature)
        return logits

    def embed(self, complex_batch):
        params2 = complex_batch.get_cochain_params(2)
        h2 = self.encoder_2(params2)
        readout = self.read(h2, msk=None)
        return h2.detach(), readout.detach()

    def compareloss(self, features, tuples, temperature):
        h_tuples = mygather(features, tuples)
        temp = torch.arange(0, len(tuples)).reshape(-1, 1)
        temp = torch.broadcast_to(temp, (temp.size(0), tuples.size(1))).to(features.device)
        h_i = mygather(features, temp)

        sim = F.cosine_similarity(h_i, h_tuples, dim=2)
        exp = torch.exp(sim) / temperature
        exp = exp.permute(1, 0)
        numerator = exp[0].reshape(-1, 1)
        denominator = exp[1:].permute(1, 0).sum(dim=1, keepdim=True)
        res = -1 * torch.log(numerator / denominator)
        return res.mean()

def mygather(feature, index):
    input_size = index.size(0)
    index = index.flatten().reshape(len(index), 1)
    index = torch.broadcast_to(index, (len(index), feature.size(1)))
    res = torch.gather(feature, dim=0, index=index)
    return res.reshape(input_size, -1, feature.size(1))
