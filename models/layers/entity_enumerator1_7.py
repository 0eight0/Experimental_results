import torch
import torch.nn as nn
# from encoder import Encoder
from .position_embeddings import FixedAbsolutePositionEmbedding

class Estor(nn.Module):
    def __init__(self, config):
        super(Estor, self).__init__()
        self.config = config
        self.num_tags, hidden_size = config.num_tags, config.hidden_size
        self.tagging_rate = config.tagging_rate
        self.if_add_a_self_attention = config.if_add_a_self_attention
        self.enumerate_mode = config.enumerate_mode
        self.if_merge_by_add = config.if_merge_by_add
        self.gate_scaling_rate = config.gate_scaling_rate
        self.gate_dropout_rate = config.gate_dropout_rate
        self.if_contrastive_learn = config.if_contrastive_learn
        self.tag_hidden_size = hidden_size if self.if_merge_by_add else int(hidden_size*self.tagging_rate)

        self.tag_embedding_layer = nn.Embedding(self.num_tags, self.tag_hidden_size)
        self.positional_embedding = FixedAbsolutePositionEmbedding(512, config.hidden_size,
                                                                   position_embedding_type='rope')

        self.hidden_size_to_tag_hidden_size = nn.Linear(hidden_size,self.tag_hidden_size)
        self.self_attention = nn.MultiheadAttention(self.tag_hidden_size, num_heads=config.num_attention_heads,
                                                    dropout=config.attention_probs_dropout_prob, batch_first=False)
        self.attention = nn.MultiheadAttention(self.tag_hidden_size, num_heads=config.num_attention_heads,
                                               dropout=config.attention_probs_dropout_prob, batch_first=False)
        self.att_norm = nn.LayerNorm(self.tag_hidden_size, eps=config.layer_norm_eps)
        self.gate_linear = nn.Linear(self.tag_hidden_size,1)
        self.gate_dropout = nn.Dropout(p=self.gate_dropout_rate)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.reshape_linear = nn.Linear(self.tag_hidden_size+hidden_size,hidden_size)

        # self.encoder = Encoder(d_model=hidden_size,nhead=config.num_attention_heads,dim_feedforward=config.intermediate_size,dropout=config.attention_probs_dropout_prob)
        self.encoder = nn.TransformerEncoderLayer(d_model=hidden_size,nhead=config.num_attention_heads,dim_feedforward=config.intermediate_size,dropout=config.attention_probs_dropout_prob,batch_first=True)

        # self.feed_forward = nn.Sequential(
        #     nn.Linear(hidden_size, config.intermediate_size),
        #     nn.ReLU(),
        #     nn.Linear(config.intermediate_size, hidden_size))
        # self.ff_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def _init_tags(self,device):
        self.tag_distri = torch.zeros(self.num_tags).to(device)

    def gate_scaling(self, X):
        return self.gate_scaling_rate * X + (1 - self.gate_scaling_rate) / 2

    def dual_tower_function(self,X):
        abs_X = torch.abs(X)
        epsilon = 1e-10
        abs_X += epsilon
        return -torch.log(abs_X)

    def get_contrastive_loss(self):
        num_elements = torch.count_nonzero(self.tag_distri)
        if num_elements <= 1:return 0
        tags = self.tag_embedding_layer.weight
        tags = torch.nn.functional.normalize(tags, p=2, dim=1)
        similarity_matrix = torch.matmul(tags, tags.T)
        tag_distri = torch.unsqueeze(self.tag_distri, -1)
        scale_matrix = torch.matmul(tag_distri, tag_distri.T)
        scale_matrix = torch.triu(scale_matrix, diagonal=1)
        scale_matrix /= scale_matrix.sum()
        # print(similarity_matrix,'\n',scale_matrix)
        loss_matrix = self.dual_tower_function(similarity_matrix)
        upper_triangle_strict = loss_matrix*scale_matrix
        num_elements_triangle_strict = num_elements * (num_elements - 1) / 2
        loss = upper_triangle_strict.sum() / num_elements_triangle_strict

        self._init_tags(tags.device)
        return loss

    def get_result(self,X,tag_id):
        if not self.if_merge_by_add:
            X = self.hidden_size_to_tag_hidden_size(X)
        tag_embedding = self.tag_embedding_layer(torch.tensor([tag_id]).to(X.device))
        if self.if_add_a_self_attention:# old
            new_X, _ = self.self_attention(X, X, X)
            X = new_X+X
        # if self.if_add_a_self_attention:# new
        #     qkv = torch.cat([tag_embedding, X], 0)
        #     new_X, _ = self.self_attention(qkv, qkv, qkv)
        #     new_X = new_X[1:]
        #     X = new_X + X
        if self.enumerate_mode == 'attention':
            result, _ = self.attention(query=X, key=tag_embedding, value=tag_embedding)
        elif self.enumerate_mode=='raw':
            result = torch.zeros_like(X)
            result += tag_embedding
        elif self.enumerate_mode=='gate':
            X = self.gate_linear(X)
            gate = self.gate_scaling(self.sigmoid(X))
            result = gate*tag_embedding
            result = result*result.size(0)
        if self.if_contrastive_learn:
            self.tag_distri[tag_id]+=1
        return result

    def forward(self, word_embedding,attention_mask, tag_to_spans):  # tag_to_span是一个加了batch的dict
        batch_size, seq_len, hidden_size = word_embedding.shape
        if self.if_contrastive_learn:self._init_tags(word_embedding.device)

        raw_word_embedding = self.dropout(word_embedding)
        word_embedding = self.positional_embedding(word_embedding)
        taged_word_embedding = torch.zeros((batch_size,seq_len,self.tag_hidden_size)).to(word_embedding.device)
        for idx_batch in range(batch_size):
            for tag, spans in tag_to_spans[idx_batch].items():
                for span in spans:
                    start, end = span
                    X = word_embedding[idx_batch, start:end, :]
                    result=self.get_result(X,tag)
                    taged_word_embedding[idx_batch, start:end, :] += result
        if self.if_merge_by_add:
            output_embedding = raw_word_embedding + taged_word_embedding * self.tagging_rate
        else:
            output_embedding = torch.cat([raw_word_embedding, taged_word_embedding], dim=-1)
            output_embedding = self.reshape_linear(output_embedding)
        output_embedding = self.att_norm(output_embedding)

        mask = attention_mask == 0
        output = self.encoder(output_embedding,src_key_padding_mask=mask)

        output = output+output_embedding

        if self.if_contrastive_learn and self.training:
            return output,self.get_contrastive_loss()
        else:
            return output,0


if __name__ == '__main__':
    from dataclasses import dataclass

    word_embedding = torch.randn(2, 32, 768)
    tag_to_span = [{2: [[0, 4], [2, 5]], 5: [[3, 8]], 3: [[12, 32], [17, 25]]},
                   {2: [[0, 4], [2, 5]], 5: [[3, 8]], 3: [[12, 32], [17, 25]]}]
    # tag_to_span = [{5: [[3, 8],[0,4]]},{}]
    from transformers import BertConfig

    config = BertConfig.from_pretrained(
        '/home/yixian.yx/project/yixian-try/banma_ner_torch/prev_trained_model/bert-base-chinese', num_labels=5, )
    config.num_tags = 6
    config.tagging_rate = 1 / 16
    config.if_add_a_self_attention = True
    config.enumerate_mode = 'gate'
    config.if_merge_by_add = True
    config.gate_scaling_rate = 0.6
    config.gate_dropout_rate = 0.5
    config.if_contrastive_learn = 1

    peace_of_shit = Estor(config)
    peace_of_shit(word_embedding, tag_to_span)

