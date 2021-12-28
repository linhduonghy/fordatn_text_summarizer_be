from typing import List
from underthesea import word_tokenize, sent_tokenize
import utils

stopword_path = "/content/text_sum_v2/vietnamese-stopwords.txt"

REDUNDANCY_CHARACTER = ',|\"\'”“‘’'

class TextPreprocess():
    def __init__(self, stopword: List[str] = None) -> None:
        if stopword is None:
            stopwords = utils.read(stopword_path)
            self.stopword = stopwords.split("\n")
        else:
            self.stopword = stopword

    def __remove_punctuation(self, text: str) -> str:
        return text.translate({ord(c): ' ' for c in REDUNDANCY_CHARACTER})

    def __remove_stopword(self, tokenized_text: List[str]) -> str:
        words = [word for word in tokenized_text if word not in self.stopword and word != '']
        return ' '.join(words)

    def process(self, unprocessed_text) -> List[str]:
        processed_text = []
        for sentence in unprocessed_text:
            s = self.__remove_punctuation(sentence)
            s = self.__remove_stopword(word_tokenize(s))    
            processed_text.append(s)  
        return processed_text
    
file_path = "D:\PTIT\DATN\_final\\source\\text_sum\\data\\test.txt"

if __name__=="__main__":
    preprocess = TextPreprocess()
    with open(file_path, 'r', encoding="utf8") as f:
        text = f.read()
        unprocessed_text = sent_tokenize(text)
        print(unprocessed_text)
        processed_text = preprocess.process(unprocessed_text)
        print(processed_text)