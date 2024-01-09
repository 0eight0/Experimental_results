import collections
import json
import os

import tqdm

class Tag2Spans():
    def __init__(self,data_dir,if_with_cls=True):
        # data_dir ='/home/yixian.yx/project/yixian-try/torch_ner/datasets/ecommerce/dics/format'
        fns = os.listdir(data_dir)
        self.id2tag = [fn for fn in fns]
        self.tag2id = {tag:i for i,tag in enumerate(self.id2tag)}
        self.num_tags = len(self.id2tag)
        self.words2tags=collections.defaultdict(list)
        self.max_word_len = 0
        self.move = 1 if if_with_cls else 0
        for fn in tqdm.tqdm(fns):
            tag = fn
            with open(os.path.join(data_dir,fn),'r') as f:
                for line in f:
                    word = line.strip()
                    self.words2tags[word].append(tag)
                    self.max_word_len = max(len(word),self.max_word_len)
        self.words_sets = [set() for i in range(self.max_word_len+1)]
        for fn in tqdm.tqdm(fns):
            with open(os.path.join(data_dir,fn),'r') as f:
                for line in f:
                    word = line.strip()
                    self.words_sets[len(word)].add(word)
        print(self.id2tag)


    def _find_sublist_spans(self,sentence,word_len):
        spans = []
        if word_len == 0:return []
        for i in range(len(sentence) - word_len + 1):
            if sentence[i:i + word_len] in self.words_sets[word_len]:
                start = i
                end = i + word_len
                spans.append((start, end,sentence[i:i + word_len]))
        return spans
    def get_tag_to_spans(self,input_sentence,mask = None):
        tag_to_spans = collections.defaultdict(list)
        for i in range(1,self.max_word_len+1):
            spans = self._find_sublist_spans(input_sentence, i)
            if spans:
                for start,end,word in spans:
                    for tag in self.words2tags[word]:
                        tag_to_spans[self.tag2id[tag]].append([start+self.move,end+self.move])
        return tag_to_spans
    def get_tag_to_spans_batch(self,tokens_batch,mask_batch):
        tag_to_spans_batch = []
        for tokens,mask in tqdm.tqdm(zip(tokens_batch,mask_batch)):
            tag_to_spans = self.get_tag_to_spans(tokens,mask)
            tag_to_spans_batch.append(tag_to_spans)
        return tag_to_spans_batch

if __name__ == '__main__':
    import transformers
    tokenizer = transformers.BertTokenizer.from_pretrained('/home/yixian.yx/project/yixian-try/banma_ner_torch/prev_trained_model/bert-base-chinese',
                                                do_lower_case=False, )
    ids = tokenizer.convert_tokens_to_ids(['[CLS]', 't', 'o', 'p', '2', '2', '4', 'p', ' ', 'd', 'i', 'p', '-', '8', '[SEP]'])

    print(ids)
    tag2spans = Tag2Spans(['product','brand'],tokenizer)
    print(list('t o p 2 2 4 p   d i p - 8'.split(' ')))
    print(tag2spans.get_tag_to_spans('top224p dip-8'))



