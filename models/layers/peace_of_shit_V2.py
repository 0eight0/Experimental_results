import torch
import torch.nn as nn
from .position_embeddings import FixedAbsolutePositionEmbedding

class Estor_attention(nn.Module):
    def __init__(self, config):
        super(Estor_attention, self).__init__()
        self.config = config
        self.num_tags, hidden_size = config.num_tags, config.hidden_size
        self.tagging_rate = config.tagging_rate

        self.tag_embedding_layer = nn.Embedding(self.num_tags, hidden_size)

        self.positional_embedding = FixedAbsolutePositionEmbedding(512, config.hidden_size,
                                                                   position_embedding_type='rope')

        self.self_attention = nn.MultiheadAttention(hidden_size, num_heads=config.num_attention_heads,
                                                    dropout=config.attention_probs_dropout_prob, batch_first=False)
        self.self_att_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=config.num_attention_heads,
                                               dropout=config.attention_probs_dropout_prob, batch_first=False)
        self.att_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, config.intermediate_size),
            nn.ReLU(),
            nn.Linear(config.intermediate_size, hidden_size)
        )
        self.ff_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.output_linear = nn.Linear(hidden_size, config.num_labels)



    def forward(self, word_embedding, tag_to_spans):  # tag_to_span是一个加了batch的dict
        batch_size, seq_len, hidden_size = word_embedding.shape
        # print(torch.LongTensor([i for i in range(self.num_tags)]).to('cuda').device)
        tag_embedding = self.tag_embedding_layer(
            torch.LongTensor([i for i in range(self.num_tags)]).to(word_embedding.device))  # (num_tagging,hidden_size)

        raw_word_embedding = self.dropout(word_embedding)
        word_embedding = self.positional_embedding(word_embedding)
        taged_word_embedding = torch.zeros(word_embedding.shape).to(word_embedding.device)  # num_tag,batch_size,seq_len,hidden_size
        for idx_batch in range(batch_size):
            for tag, spans in tag_to_spans[idx_batch].items():
                kv = tag_embedding[tag:tag + 1, :]  # (1,hidden_size)
                for span in spans:
                    start, end = span
                    q = word_embedding[idx_batch, start:end, :]  # (span_size,hidden_size)
                    q = torch.cat([kv,q],0)
                    #自注意力运算
                    q, _ = self.self_attention(q, q, q)
                    # q += word_embedding[idx_batch, start:end, :]

                    attention_output = q[1:]
                    # attention_output += kv  # 加上tag_embedding的残差，利用广播机制
                    taged_word_embedding[idx_batch, start:end, :] += attention_output

        output_embedding = raw_word_embedding + taged_word_embedding * self.tagging_rate
        output_embedding = self.att_norm(output_embedding)

        output = self.feed_forward(output_embedding)
        output = self.dropout(output)
        output += output_embedding
        output = self.ff_norm(output)

        return self.output_linear(output)

class Estor_raw(nn.Module):
    def __init__(self, config):
        super(Estor_raw, self).__init__()
        self.config = config
        self.num_tags, hidden_size = config.num_tags, config.hidden_size
        self.tagging_rate = config.tagging_rate
        new_hidden_size = hidden_size+int(hidden_size * self.tagging_rate)

        self.tag_embedding_layer = nn.Embedding(self.num_tags, hidden_size)

        self.positional_embedding = FixedAbsolutePositionEmbedding(512, config.hidden_size,
                                                                   position_embedding_type='rope')

        self.self_attention = nn.MultiheadAttention(hidden_size, num_heads=config.num_attention_heads,
                                                    dropout=config.attention_probs_dropout_prob, batch_first=False)
        self.self_att_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=config.num_attention_heads,
                                               dropout=config.attention_probs_dropout_prob, batch_first=False)
        self.att_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, config.intermediate_size),
            nn.ReLU(),
            nn.Linear(config.intermediate_size, hidden_size)
        )
        self.ff_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.output_linear = nn.Linear(hidden_size, config.num_labels)


    def forward(self, word_embedding, tag_to_spans):  # tag_to_span是一个加了batch的dict
        batch_size, seq_len, hidden_size = word_embedding.shape
        # print(torch.LongTensor([i for i in range(self.num_tags)]).to('cuda').device)
        tag_embedding = self.tag_embedding_layer(
            torch.LongTensor([i for i in range(self.num_tags)]).to(word_embedding.device))  # (num_tagging,hidden_size)

        raw_word_embedding = self.dropout(word_embedding)
        word_embedding = self.positional_embedding(word_embedding)
        taged_word_embedding = torch.zeros(word_embedding.shape).to(word_embedding.device)  # num_tag,batch_size,seq_len,hidden_size
        for idx_batch in range(batch_size):
            for tag, spans in tag_to_spans[idx_batch].items():
                kv = tag_embedding[tag:tag + 1, :]  # (1,hidden_size)
                for span in spans:
                    start, end = span
                    q = word_embedding[idx_batch, start:end, :]  # (span_size,hidden_size)
                    #自注意力运算
                    # q, _ = self.self_attention(q, q, q)
                    # q += word_embedding[idx_batch, start:end, :]

                    attention_output = torch.zeros_like(q)
                    attention_output += kv  # 加上tag_embedding的残差，利用广播机制
                    taged_word_embedding[idx_batch, start:end, :] += attention_output

        output_embedding = raw_word_embedding + taged_word_embedding * self.tagging_rate
        output_embedding = self.att_norm(output_embedding)

        output = self.feed_forward(output_embedding)
        output = self.dropout(output)
        output += output_embedding
        output = self.ff_norm(output)

        return self.output_linear(output)

class Estor_concat(nn.Module):
    def __init__(self, config):
        super(Estor_concat, self).__init__()
        self.config = config
        self.num_tags, hidden_size = config.num_tags, config.hidden_size
        self.tagging_rate = config.tagging_rate
        new_hidden_size = hidden_size+int(hidden_size * self.tagging_rate)

        self.tag_embedding_layer = nn.Embedding(self.num_tags, hidden_size)

        self.positional_embedding = FixedAbsolutePositionEmbedding(512, config.hidden_size,
                                                                   position_embedding_type='rope')

        self.self_attention = nn.MultiheadAttention(hidden_size, num_heads=config.num_attention_heads,
                                                    dropout=config.attention_probs_dropout_prob, batch_first=False)
        self.self_att_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=config.num_attention_heads,
                                               dropout=config.attention_probs_dropout_prob, batch_first=False)
        self.att_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size * self.num_tags, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, int(hidden_size * self.tagging_rate))
        )
        self.ff_norm = nn.LayerNorm(new_hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.output_linear = nn.Linear(new_hidden_size,config.num_labels)


    def forward(self, word_embedding, tag_to_spans):  # tag_to_span是一个加了batch的dict
        batch_size, seq_len, hidden_size = word_embedding.shape
        # print(torch.LongTensor([i for i in range(self.num_tags)]).to('cuda').device)
        tag_embedding = self.tag_embedding_layer(
            torch.LongTensor([i for i in range(self.num_tags)]).to(word_embedding.device))  # (num_tagging,hidden_size)

        raw_word_embedding = self.dropout(word_embedding)
        word_embedding = self.positional_embedding(word_embedding)
        taged_word_embedding = torch.zeros(
            (self.num_tags,) + word_embedding.shape).to(word_embedding.device)  # num_tag,batch_size,seq_len,hidden_size
        for idx_batch in range(batch_size):
            for tag, spans in tag_to_spans[idx_batch].items():
                kv = tag_embedding[tag:tag + 1, :]  # (1,hidden_size)
                for span in spans:
                    start, end = span
                    q = word_embedding[idx_batch, start:end, :]  # (span_size,hidden_size)
                    #自注意力运算
                    # q, _ = self.self_attention(q, q, q)
                    # q += word_embedding[idx_batch, start:end, :]

                    attention_output, _ = self.attention(query=q, key=kv, value=kv)
                    # attention_output += kv  # 加上tag_embedding的残差，利用广播机制
                    taged_word_embedding[tag, idx_batch, start:end, :] += attention_output

        taged_word_embedding = torch.cat([tensor for tensor in taged_word_embedding], dim=-1)
        taged_word_embedding = self.feed_forward(taged_word_embedding)
        output_embedding = torch.cat([raw_word_embedding, taged_word_embedding], dim=-1)

        output = self.dropout(output_embedding)
        output = self.ff_norm(output)

        return self.output_linear(output)


class Estor(nn.Module):
    def __init__(self, config):
        super(Estor, self).__init__()
        self.config = config
        self.num_tags, hidden_size = config.num_tags, config.hidden_size
        self.tagging_rate = config.tagging_rate

        self.if_add_a_self_attention = config.if_add_a_self_attention
        self.enumerate_mode = config.enumerate_mode
        self.if_merge_by_add = config.if_merge_by_add

        self.tag_embedding_layer = nn.Embedding(self.num_tags, hidden_size)
        self.tag_embeddings = self.tag_embedding_layer(torch.LongTensor([i for i in range(self.num_tags)]))
        self.positional_embedding = FixedAbsolutePositionEmbedding(512, config.hidden_size,
                                                                   position_embedding_type='rope')

        self.self_attention = nn.MultiheadAttention(hidden_size, num_heads=config.num_attention_heads,
                                                    dropout=config.attention_probs_dropout_prob, batch_first=False)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=config.num_attention_heads,
                                               dropout=config.attention_probs_dropout_prob, batch_first=False)

        self.reshape_linear = nn.Linear(hidden_size,hidden_size*self.tagging_rate)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, config.intermediate_size),
            nn.ReLU(),
            nn.Linear(config.intermediate_size, hidden_size))
        self.ff_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.output_linear = nn.Linear(hidden_size, config.num_labels)

    def get_result(self,X,tag_id):
        tag_embedding = self.tag_embeddings[tag_id]
        result = 0 #有问题就返回0
        if self.if_add_a_self_attention:
            new_X, _ = self.self_attention(X, X, X)
            X += new_X
        if self.enumerate_mode=='self_attention':
            qkv= torch.cat([tag_embedding,X],0)
            result,_=self.attention(qkv,qkv,qkv)
            result = result[1:]
        elif self.enumerate_mode == 'attention':
            result, _ = self.attention(query=X, key=tag_embedding, value=tag_embedding)
        elif self.enumerate_mode=='raw':
            result = torch.zeros_like(X)
            result += tag_embedding
        return result

    def forward(self, word_embedding, tag_to_spans):  # tag_to_span是一个加了batch的dict
        batch_size, seq_len, hidden_size = word_embedding.shape

        raw_word_embedding = self.dropout(word_embedding)
        word_embedding = self.positional_embedding(word_embedding)
        taged_word_embedding = torch.zeros_like(word_embedding)
        for idx_batch in range(batch_size):
            for tag, spans in tag_to_spans[idx_batch].items():
                for span in spans:
                    start, end = span
                    result=self.get_result(word_embedding[idx_batch, start:end, :],tag)
                    taged_word_embedding[idx_batch, start:end, :] += result
        if self.if_merge_by_add:
            output_embedding = raw_word_embedding + taged_word_embedding * self.tagging_rate
        else:
            taged_word_embedding = self.reshape_linear(taged_word_embedding)
            output_embedding = torch.cat([raw_word_embedding, taged_word_embedding], dim=-1)
        output_embedding = self.att_norm(output_embedding)

        output = self.feed_forward(output_embedding)
        output = self.dropout(output)
        output += output_embedding
        output = self.ff_norm(output)

        return self.output_linear(output)


if __name__ == '__main__':
    from dataclasses import dataclass

    word_embedding = torch.randn(2, 32, 768)
    tag_to_span = [{2: [[0, 4], [2, 5]], 5: [[3, 8]], 3: [[12, 32], [17, 25]]},
                   {2: [[0, 4], [2, 5]], 5: [[3, 8]], 3: [[12, 32], [17, 25]]}]
    from transformers import BertConfig

    config = BertConfig.from_pretrained(
        '/home/yixian.yx/project/yixian-try/banma_ner_torch/prev_trained_model/bert-base-chinese', num_labels=5, )
    config.num_tags = 6
    config.tagging_rate = 1 / 16

    peace_of_shit = Estor_concat(config)
    peace_of_shit(word_embedding, tag_to_span)

    print(config.num_labels)
