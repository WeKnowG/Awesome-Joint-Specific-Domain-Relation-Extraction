import torch
from torch import nn
from torch.nn import functional as F

from .Basic_model import BasicModel


class obj_model(BasicModel):

    def __init__(self, config):

        super(obj_model, self).__init__()

        self.object_head_layer = nn.Linear(config.hidden_size, config.rel_nums)
        self.object_tail_layer = nn.Linear(config.hidden_size, config.rel_nums)
        self.apply(self.init_model_weights)

    def gather_info(self, input_tensor, positions):
        batch_size, seq_len, hidden_size = input_tensor.size()
        flat_offsets = torch.linspace(0, batch_size-1, steps=batch_size).long().view(-1, 1)*seq_len
        flat_positions = positions.long() + flat_offsets.cuda()
        flat_positions = flat_positions.view(-1)
        flat_seq_tensor = input_tensor.view(batch_size*seq_len, hidden_size)
        output_tensor = torch.index_select(flat_seq_tensor, 0, flat_positions).view(batch_size, -1, hidden_size)
        return output_tensor

    def forward(self,
                bert_output,
                positions):
        V_subject = torch.mean(self.gather_info(bert_output, positions), 1, keepdim=True)  # [batch_size, 1, hidden_size]
        tokens_feature = bert_output + V_subject  # [batch_size, seq_len, hidden_size]

        # obj_logits
        pred_obj_heads = F.sigmoid(self.object_head_layer(tokens_feature))   # [batch_size, seq, rel_nums]
        pred_obj_tails = F.sigmoid(self.object_tail_layer(tokens_feature))   # [batch_size, seq, rel_nums]

        return pred_obj_heads, pred_obj_tails
