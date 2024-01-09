import torch
import torch.nn as nn

class POS(nn.Module):
    def __init__(self, config):
        super(POS,self).__init__()
        self.config = config
        self.num_tags, hidden_size = config.num_tags, config.hidden_size
        self.tagging_rate = config.tagging_rate

        self.tag_embedding_layer = nn.Embedding(self.num_tags,hidden_size)

        self.attention = nn.MultiheadAttention(hidden_size,num_heads=config.num_attention_heads,dropout=config.attention_probs_dropout_prob,batch_first=False)
        self.att_norm = nn.LayerNorm(hidden_size,eps=config.layer_norm_eps)

        self.word_embedding_liner = nn.Linear(hidden_size,hidden_size-int(self.tagging_rate*hidden_size))
        self.tag_embedding_liner = nn.Linear(hidden_size,int(self.tagging_rate*hidden_size))

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, config.intermediate_size),
            nn.ReLU(),
            nn.Linear(config.intermediate_size, hidden_size)
        )
        self.ff_norm = nn.LayerNorm(hidden_size,eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, word_embedding, tag_to_spans):#tag_to_span是一个加了batch的dict
        batch_size,seq_len,hidden_size = word_embedding.shape
        # print(torch.LongTensor([i for i in range(self.num_tags)]).to('cuda').device)
        tag_embedding = self.tag_embedding_layer(torch.LongTensor([i for i in range(self.num_tags)]).to(word_embedding.device))#(num_tagging,hidden_size)

        taged_word_embedding = torch.zeros_like(word_embedding)
        for idx_batch in range(batch_size):
            for tag,spans in tag_to_spans[idx_batch].items():
                kv = tag_embedding[tag:tag+1,:]#(1,hidden_size)
                for span in spans:
                    start,end = span
                    q = word_embedding[idx_batch,start:end,:]#(span_size,hidden_size)
                    attention_output,_ = self.attention(query=q,key = kv,value=kv)
                    # print(attention_output.shape)
                    taged_word_embedding[idx_batch,start:end,:] += attention_output

        word_embedding = self.word_embedding_liner(word_embedding)
        word_embedding = self.dropout(word_embedding)
        taged_word_embedding = self.tag_embedding_liner(taged_word_embedding)
        output_embedding = torch.cat([word_embedding,taged_word_embedding],dim=-1)
        output_embedding = self.att_norm(output_embedding)


        output = self.feed_forward(output_embedding)
        output = self.dropout(output)
        output += output_embedding
        output = self.ff_norm(output)

        return output

class POSV2(nn.Module):
    def __init__(self, config):
        super(POSV2,self).__init__()
        self.config = config
        self.num_tags, hidden_size = config.num_tags, config.hidden_size
        self.tagging_rate = config.tagging_rate

        self.tag_embedding_layer = nn.Embedding(self.num_tags,hidden_size)

        self.attention = nn.MultiheadAttention(hidden_size,num_heads=config.num_attention_heads,dropout=config.attention_probs_dropout_prob,batch_first=False)
        self.att_norm = nn.LayerNorm(hidden_size,eps=config.layer_norm_eps)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, config.intermediate_size),
            nn.ReLU(),
            nn.Linear(config.intermediate_size, hidden_size)
        )
        self.ff_norm = nn.LayerNorm(hidden_size,eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, word_embedding, tag_to_spans):#tag_to_span是一个加了batch的dict
        batch_size,seq_len,hidden_size = word_embedding.shape
        # print(torch.LongTensor([i for i in range(self.num_tags)]).to('cuda').device)
        tag_embedding = self.tag_embedding_layer(torch.LongTensor([i for i in range(self.num_tags)]).to(word_embedding.device))#(num_tagging,hidden_size)

        taged_word_embedding = torch.zeros_like(word_embedding)
        for idx_batch in range(batch_size):
            for tag,spans in tag_to_spans[idx_batch].items():
                kv = tag_embedding[tag:tag+1,:]#(1,hidden_size)
                for span in spans:
                    start,end = span
                    q = word_embedding[idx_batch,start:end,:]#(span_size,hidden_size)
                    attention_output,_ = self.attention(query=q,key = kv,value=kv)
                    # print(attention_output.shape)
                    taged_word_embedding[idx_batch,start:end,:] += attention_output

        word_embedding = self.dropout(word_embedding)
        output_embedding = word_embedding+taged_word_embedding*self.tagging_rate
        output_embedding = self.att_norm(output_embedding)


        output = self.feed_forward(output_embedding)
        output = self.dropout(output)
        output += output_embedding
        output = self.ff_norm(output)

        return output

if __name__ == '__main__':
    from dataclasses import dataclass

    word_embedding = torch.randn(2, 32, 768)
    tag_to_span = [{2:[[0,4],[2,5]],5:[[3,8]],3:[[12,32],[17,25]]},
                   {2:[[0,4],[2,5]],5:[[3,8]],3:[[12,32],[17,25]]}]
    from transformers import BertConfig
    config = BertConfig.from_pretrained('/home/yixian.yx/project/yixian-try/banma_ner_torch/prev_trained_model/bert-base-chinese', num_labels=5, )
    config.num_tags = 6
    config.tagging_rate = 1/16

    peace_of_shit = POS(config)
    peace_of_shit(word_embedding,tag_to_span)

    print(config.num_labels)