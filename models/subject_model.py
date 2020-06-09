from .Basic_model import BasicModel
from torch import nn
from torch.nn import functional as F


class sub_model(BasicModel):

    def __init__(self, config):

        super(sub_model, self).__init__()

        self.subject_head_layer = nn.Linear(config.hidden_size, 1)
        self.subject_tail_layer = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_model_weights)

    def forward(self, bert_output):
        pred_sub_heads = F.sigmoid(self.subject_head_layer(bert_output))  # [batch_size, seq_len, 1]
        pred_sub_tails = F.sigmoid(self.subject_tail_layer(bert_output))
        return pred_sub_heads, pred_sub_tails