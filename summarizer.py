from typing import List
from underthesea import sent_tokenize
from preprocess import TextPreprocess
from summary_textrank import TextRankSummarizer
from model import get_model
import math

class Summarizer(object):

    def __init__(self, text: str, n_summ: int, type: str) -> None:
        super().__init__()
        self.preprocess = TextPreprocess()
        self.model = get_model()
        self.unprocessed_text = sent_tokenize(text)
        self.processed_text = self.preprocess.process(self.unprocessed_text)

        if not n_summ:
          self.n_summ = math.ceil(len(self.processed_text) ** .5)
        else:
          self.n_summ = n_summ

        print(self.n_summ)
        print(self.processed_text)

    def summarize(self) -> List[str]:
        return self.__textrank_summarizer()
  
    def __textrank_summarizer(self) -> List[str]: 
        summarizer = TextRankSummarizer(self.unprocessed_text, self.processed_text, self.n_summ, self.model)
        return summarizer.summarize()
    
    def __sentence_features_summarizer(self) -> List[str]: 
        pass

    def __clustering_summarizer(self) -> List[str]:
        pass