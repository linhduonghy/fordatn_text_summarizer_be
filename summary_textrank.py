from typing import List
from model import Model
from textrank import TextRank

class TextRankSummarizer(object):
    def __init__(self, unprocessed_text: List[str], processed_text: List[str], n_summ: int, model: Model) -> str:
        super().__init__()
        self.unprocessed_text = unprocessed_text
        self.processed_text = processed_text
        self.n_summ = n_summ
        self.model = model

    def summarize(self) -> List[str]:
        # get sentence embedding
        sents_emb = self.model.get_sentences_embedding(self.processed_text)

        # model textrank
        textrank = TextRank(sents_emb)
        sent_ranking = textrank.pagerank()

        # get n_numm first ranking
        summ_index = [k for k, v in sent_ranking[:self.n_summ]]
        summ_index = sorted(summ_index)

        print('summary index: ', summ_index)
        summ_text = [self.unprocessed_text[i] for i in summ_index]
        return summ_text
