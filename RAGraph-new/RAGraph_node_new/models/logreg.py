#LogReg 模块，位于 ragraph_node/models/logreg.py。
# 它实现了一个非常简单的线性分类器（Logistic Regression），
# 用于在图表示学习之后完成分类任务，如节点分类或图分类，
# 是 RAGRAPH 框架中用于下游任务评估的关键组件之一。
import torch
import torch.nn as nn
import torch.nn.functional as F

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

