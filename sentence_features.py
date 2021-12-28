# -*- coding: utf-8 -*-

from typing import List
from vncorenlp import VnCoreNLP
from utils import cos_sim
from math import log2, ceil
from collections import Counter


class SentenceFeature():
    '''
        Sentence features
        *** The score of every feature will be normalized between 0 and 1
    '''

    def __init__(self, text, title, unprocessed_text, processed_text, title_emb, sents_emb, annotator: VnCoreNLP):
        '''
            text: origin document
            title: title of document
            unprocessed_text: list unprocessed sentence of document 
            processed_text: list processed sentence of document 
            sents_emb: list sentence embedding
            annotator: vietnamese nlp annotator
        '''
        self.text = text
        self.title = title
        self.unprocessed_text = unprocessed_text
        self.processed_text = processed_text
        # self.title_emb = title_emb        
        self.sents_emb = sents_emb
        self.annotator = annotator

        self.max_len = self.__get_max_sent_length()        
        self.dict = self.__build_dict()
        self.tf_isf = self.__build_tf_isf()
        self.keywords = self.__build_keywords()
        
    # all features    
    def get_sentence_features(self, sent_id):
        print("=============================")
        print(self.processed_text[sent_id])
        sim_title = 0
        if self.title is not None:
            sim_title = self.get_sim_to_title(sent_id)          # 0.2
        length = self.get_sent_length(sent_id)                  # 0.1
        # position = self.get_sent_position(sent_id)              
        number_of_nouns = self.get_noun(sent_id)                # 0.1
        number_of_numerals = self.get_numeral(sent_id)          # 0.1
        number_of_name_entities = self.get_ner(sent_id)         # 0.2
        keyword_score = self.get_keyword_score(sent_id)         # 0.3

        features = [sim_title, length, number_of_nouns, number_of_numerals, number_of_name_entities, keyword_score]
        w = [0.2, 0.1, 0.1, 0.1, 0.2, 0.3]
        features_score = sum([w[i] * features[i] for i in range(len(features))])
        print("Final: ", features_score)
        return features_score


    def get_list_sim_to_title(self) -> List[str]:
        sim_title = []
        for e in self.sents_emb:
            sim_title.append(cos_sim(self.title_emb, e))

        return sim_title

    def get_sim_to_title(self, sent_id: int) -> float:
        print("sim_title: ", cos_sim(self.title_emb, self.sents_emb[sent_id]))
        return cos_sim(self.title_emb, self.sents_emb[sent_id])

    def get_sent_position(self, sent_id: int) -> float:
        '''
            sentence postion relative with paragraph | document
        '''
        return 1 - sent_id / len(self.processed_text)

    def __get_max_sent_length(self):
        '''
            get max length of sentences
        '''
        max_len = 0
        for sent in self.processed_text:
            tokenized_words = self.annotator.tokenize(sent)
            tokenized_words = tokenized_words[self.__get_max_index_tokenized(
                tokenized_words)]
            max_len = max(max_len, len(tokenized_words))
        return max_len

    def get_sent_length(self, sent_id: int):
        '''
            number of words in sentence

        '''
        tokenized_words = self.annotator.tokenize(self.processed_text[sent_id])
        tokenized_words = tokenized_words[self.__get_max_index_tokenized(
            tokenized_words)]
        print("length: ", len(tokenized_words), len(tokenized_words) / self.max_len)
        return len(tokenized_words) / self.max_len

    def __get_max_index_tokenized(self, tokenized_word: list):
        '''
            check if tokenize vncorenlp different with underthesea
            return sentence index with max number of word
        '''
        if len(tokenized_word) > 1:
            # get sentence with max length
            max_len = max([len(x) for x in tokenized_word])
            for i in range(len(tokenized_word)):
                if len(tokenized_word[i]) == max_len:
                    return i
        return 0

    def get_noun(self, sent_id: int):
        '''
            number of nouns in sentence 
        '''
        pos_tags = self.annotator.pos_tag(self.processed_text[sent_id])
        pos_tags = pos_tags[self.__get_max_index_tokenized(pos_tags)]
        list_noun = [word for word, pt in pos_tags if 'N' in pt]  # filter noun
        print("nouns: ", list_noun, len(list_noun) / len(pos_tags))
        return len(list_noun) / len(pos_tags)

    def get_numeral(self, sent_id: int):
        '''
            number of numerals in sentence
        '''
        pos_tags = self.annotator.pos_tag(self.processed_text[sent_id])
        pos_tags = pos_tags[self.__get_max_index_tokenized(pos_tags)]
        # filter numeral, quantity
        list_numeral = [word for word, pt in pos_tags if pt == 'M']
        print("numeral: ", list_numeral, len(list_numeral) / len(pos_tags))
        return len(list_numeral) / len(pos_tags)

    def get_ner(self, sent_id: int):
        '''
            number of named entities in sentence
            named entities: person, location, organization
        '''
        ners = self.annotator.ner(self.processed_text[sent_id])
        ners = ners[self.__get_max_index_tokenized(ners)]
        # filter word : person, location, organization
        list_filter_ner = [word for word, ner in ners if ner ==
                           'B-PER' or ner == "B-LOC" or ner == "B-ORG"]
        print("ner: ", list_filter_ner, len(list_filter_ner) / len(ners))
        return len(list_filter_ner) / len(ners)

    def get_keyword_score(self, sent_id):
        '''
            number of keywords in sentence
            list keywords is predefined
        '''
        tokenized_word = self.annotator.tokenize(self.processed_text[sent_id])
        tokenized_word = tokenized_word[self.__get_max_index_tokenized(
            tokenized_word)]
        num_kw = [word for word in tokenized_word if word in self.keywords]
        print("keywords: ", num_kw, len(num_kw) / len(tokenized_word))
        return len(num_kw) / len(tokenized_word)

    def __build_dict(self):
        '''
            all unique term in document
        '''
        dict = set()
        for sent in self.processed_text:
            tokenized_word = self.annotator.tokenize(sent)
            tokenized_word = tokenized_word[self.__get_max_index_tokenized(
                tokenized_word)]
            dict.update(tokenized_word)
        return dict

    def __build_isf(self):
        '''
            inverted sentence frequency
        '''
        isf = dict.fromkeys(self.dict, 0)
        N = len(self.processed_text)  # number of sentences in document
        tokenized_words = []
        for sent in self.processed_text:
            tokenized_word = self.annotator.tokenize(sent)
            tokenized_word = tokenized_word[self.__get_max_index_tokenized(
                tokenized_word)]
            tokenized_words.append(tokenized_word)
        for term in self.dict:
            # number of sentences contain word
            n = len(
                [1 for tokenized_word in tokenized_words if term in tokenized_word])
            isf[term] = log2(float(N) / n)
        return isf

    def __build_tf(self):
        '''
            term frequency of term i in sentence j
        '''
        tf = []
        for sent in self.processed_text:
            tokenized_word = self.annotator.tokenize(sent)
            tokenized_word = tokenized_word[self.__get_max_index_tokenized(
                tokenized_word)]
            counter = Counter(tokenized_word)
            tf.append({term: c / len(tokenized_word)
                      for term, c in counter.items()})
        return tf

    def __build_tf_isf(self):
        '''
            every sentence, build term frequency - inverted sentence frequency of each term
            return: dict
                key: term in document
                value: list weight of term with every sentence
        '''
        isf = self.__build_isf()
        tf = self.__build_tf()
        tf_isf = dict.fromkeys(self.dict)

        for term in self.dict:
            tf_isf[term] = []
            for i in range(len(self.processed_text)):
                if term in tf[i]:
                    # tf_isf(t,s) = tf(t,s)*isf(t)
                    tf_isf[term].append(tf[i][term] * isf[term])
                else:
                    tf_isf[term].append(0)
        return tf_isf

    def __build_keywords(self, keywords_ratio=0.1):
        '''
            get top k terms with highest tf-isf as keywords of document
            in here: k = 10% of dictionary size
        '''

        tf_isf = self.__build_tf_isf()
        
        # get average tf_isf term weight
        # sum(tf_isf[term]) / len(tf_isf[term])
        tf_isf_average = {term: sum(term_ts_isf) / len(term_ts_isf)
                       for term, term_ts_isf in tf_isf.items()}

        # sort tf_isf descending by term weight
        tf_isf_sorted = [(term, weight) for term, weight in sorted(
            tf_isf_average.items(), key=lambda item: item[1], reverse=True)]

        # top k keywords
        k = ceil(keywords_ratio * len(self.dict))
        # select top k terms in tf_isf as keywords
        topk_keywords = [k for k, v in tf_isf_sorted[:k]]

        print("Top keywords: ", topk_keywords)
        return topk_keywords