from pytorch_pretrained_bert.modeling import PreTrainedBertModel, BertModel


class bert(PreTrainedBertModel):
    def __init__(self, config):

        super(bert, self).__init__(config=config)
        self.model = BertModel(config=config)

    def forward(self, input_ids, token_type_ids, attention_mask):

        sequence_output, _ = self.model(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        return sequence_output