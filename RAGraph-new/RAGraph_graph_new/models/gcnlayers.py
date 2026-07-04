import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DGI, GraphCL, Lp
from layers import GCN, AvgReadout
import tqdm
import numpy as np


class GcnLayers(torch.nn.Module):
    def __init__(self, n_in, n_h,num_layers_num,dropout):
        super(GcnLayers, self).__init__()

        self.act=torch.nn.ReLU()
        self.num_layers_num=num_layers_num
        self.g_net, self.bns = self.create_net(n_in,n_h,self.num_layers_num)
        self.out_dim = n_h
        self.dropout=torch.nn.Dropout(p=dropout)

    def create_net(self,input_dim, hidden_dim,num_layers):

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_layers):

            if i:
                nn = GCN(hidden_dim, hidden_dim)
            else:
                nn = GCN(input_dim, hidden_dim)
            conv = nn
            bn = torch.nn.BatchNorm1d(hidden_dim)

            self.convs.append(conv)
            self.bns.append(bn)

        return self.convs, self.bns


    def forward(self, seq, adj,sparse,LP=False):
        graph_output = torch.squeeze(seq,dim=0)
        graph_len = adj
        # print("seq",seq.shape)
        # print("adj",adj.shape)
        xs = []
        for i in range(self.num_layers_num):
            # print("i",i)
            input=(graph_output,adj)
            graph_output = self.convs[i](input)
            # print("graphout1",graph_output)
            # print("graphout1",graph_output.shape)
            if LP:
                # print("graphout1",graph_output.shape)
                graph_output = self.bns[i](graph_output)
                # print("graphout2",graph_output.shape)
                graph_output = self.dropout(graph_output)
            # print("graphout2",graph_output)
            # print("graphout2",graph_output.shape)
            xs.append(graph_output)
            # print("Xs",xs)
        # xpool= []
        # for x in xs:
        #     graph_embedding = split_and_batchify_graph_feats(x, graph_len)[0]
        #     graph_embedding = torch.sum(graph_embedding, dim=1)
        #     xpool.append(graph_embedding)
        # x = torch.cat(xpool, -1).unsqueeze(dim=0)
        return graph_output
