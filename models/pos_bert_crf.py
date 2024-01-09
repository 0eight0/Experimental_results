import torch.nn as nn
from .layers.crf import CRF
from transformers import BertModel, BertPreTrainedModel
from .layers.peace_of_shit import POS,POSV2


class PosBertCrf(BertPreTrainedModel):
    def __init__(self, config):
        super(PosBertCrf, self).__init__(config)
        self.bert = BertModel(config)
        self.embedding_layer = self.bert.get_input_embeddings()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.pos = POS(config)
        self.init_weights()

    def forward(self, input_ids,tag_to_spans, token_type_ids=None, attention_mask=None, labels=None):
        word_embedding = self.embedding_layer(input_ids)
        joint_embedding = self.pos(word_embedding,tag_to_spans)
        outputs = self.bert(inputs_embeds = joint_embedding, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1 * loss,) + outputs
        return outputs  # (loss), scores

class PosV2BertCrf(BertPreTrainedModel):
    def __init__(self, config):
        super(PosV2BertCrf, self).__init__(config)
        self.bert = BertModel(config)
        self.embedding_layer = self.bert.get_input_embeddings()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.pos = POSV2(config)
        self.init_weights()

    def forward(self, input_ids,tag_to_spans, token_type_ids=None, attention_mask=None, labels=None):
        word_embedding = self.embedding_layer(input_ids)
        joint_embedding = self.pos(word_embedding,tag_to_spans)
        outputs = self.bert(inputs_embeds = joint_embedding, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1 * loss,) + outputs
        return outputs  # (loss), scores



