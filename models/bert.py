from transformers.modeling_bert import BertPreTrainedModel, BertModel


class Bert_model(BertPreTrainedModel):

    def __init__(self, config):

        super(Bert_model, self).__init__(config)
        self.model = BertModel(config=config)

    def forward(self, input_ids, attention_mask, token_type_ids):

        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        return outputs[0]