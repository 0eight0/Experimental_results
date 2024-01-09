import collections
import json

import tqdm


class Tag2Spans():
    def __init__(self,tag_list,tokenizer):
        self.num_tags = len(tag_list)
        self.id2tag = {i: tag for i, tag in enumerate(tag_list)}
        self.tag2id = {tag: i for i, tag in enumerate(tag_list)}
        self.tokenizer = tokenizer
        with open('/home/yixian.yx/project/yixian-try/torch_ner/datasets/ecommerce/dics/merge.json', 'r') as j:
            self.dic = json.load(j)
        temp_list = []
        for k in self.dic.keys():
            for v in self.dic[k]:
                temp_list.append((k,v))
        self.id_dict = [(tokenizer.convert_tokens_to_ids(list(seq)),self.tag2id[tag]) for seq,tag in temp_list]
        # print(self.id_dict)

    def _find_sublist_spans(self,lst, sublist):
        spans = []
        sublist_len = len(sublist)
        if sublist_len == 0:return []
        for i in range(len(lst) - sublist_len + 1):
            if lst[i:i + sublist_len] == sublist:
                start = i
                end = i + sublist_len
                spans.append((start, end))
        return spans
    def get_tag_to_spans(self,input_ids):
        tag_to_spans = collections.defaultdict(list)
        for tag_ids,tag in self.id_dict:
            spans = self._find_sublist_spans(input_ids, tag_ids)
            if spans:
                print(tag,self.tag2id,input_ids,tag_ids)
                tag_to_spans[tag]+=spans
        return tag_to_spans
    def get_tag_to_spans_batch(self,tokens_batch,mask_batch):
        tag_to_spans_batch = []
        for tokens,mask in tqdm.tqdm(zip(tokens_batch,mask_batch)):
            print(tokens,mask)
            tokens =tokens[mask.bool()]
            print(tokens)
            tag_to_spans = self.get_tag_to_spans(tokens)
            tag_to_spans_batch.append(tag_to_spans)
        return tag_to_spans_batch

if __name__ == '__main__':
    import transformers
    tokenizer = transformers.BertTokenizer.from_pretrained('/home/yixian.yx/project/yixian-try/banma_ner_torch/prev_trained_model/bert-base-chinese',
                                                do_lower_case=False, )
    ids = tokenizer.convert_tokens_to_ids(list('你是一只小猪猪，卓凯飞机哇哦附件二哦ife213'))

    print(ids)
    tag2spans = Tag2Spans(['product','brand'],tokenizer)
    print(tag2spans.get_tag_to_spans(ids))



