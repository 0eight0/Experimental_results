import torch.nn as nn
from .layers.crf import CRF
from transformers import BertModel, BertPreTrainedModel
from .layers.entity_enumerator import Estor

class BertEstorCrf(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEstorCrf, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.estor = Estor(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.contrastive_alpha = config.contrastive_alpha
        self.init_weights()

    def forward(self, input_ids,tag_to_spans, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        # output,contrastive_loss = self.estor(sequence_output,tag_to_spans)
        output,contrastive_loss = self.estor(sequence_output,attention_mask,tag_to_spans)

        logits = self.classifier(output)
        outputs = (logits,)
        if labels is not None:
            loss = -self.crf(emissions=logits, tags=labels, mask=attention_mask)
            loss += contrastive_loss*self.contrastive_alpha
            outputs = (loss,)+outputs
        return outputs  # (loss), scores



