# author: sunshine
# datetime:2021/8/5 下午3:17
import torch
import torch.nn as nn
from transformers import BertModel


class SMPNet(nn.Module):
    def __init__(self, args, num_class):
        super(SMPNet, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_path)
        self.fc1 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_class)
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(768 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_class)
        )

    def forward(self, x1=None, x2=None):
        out = ()
        if x1 is not None:
            x1 = self.bert(*x1, output_hidden_states=True)
            x1_hidden = x1.hidden_states
            concat1 = torch.cat([x1_hidden[-1][:, 0], x1_hidden[-2][:, 0], x1_hidden[-3][:, 0], x1.pooler_output],
                                dim=-1)
            out1 = self.fc1(concat1)
            out = out + (out1,)
        if x2 is not None:
            x2 = self.bert(*x2, output_hidden_states=True)
            x2_hidden = x2.hidden_states
            concat2 = torch.cat([x2_hidden[-1][:, 0], x2_hidden[-2][:, 0], x2_hidden[-3][:, 0], x2.pooler_output],
                                dim=-1)
            out2 = self.fc2(concat2)
            out = out + (out2,)
        return out
