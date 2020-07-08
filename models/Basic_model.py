import torch.nn as nn


class BasicModel(nn.Module):

    def __init__(self):

        super(BasicModel, self).__init__()

    def init_model_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()