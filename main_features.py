
from underthesea import word_tokenize, sent_tokenize
import numpy as np
from vncorenlp import VnCoreNLP
from preprocess import TextPreprocess
import utils
from sentence_features import SentenceFeature
import os

file_path = "D:\PTIT\DATN\source_l20\\text_sum\\data\\law.txt"
emb_path = "D:\PTIT\DATN\source_l20\\text_sum\sents_emb\\law_emb.npy"
vncorenlp_path = "D:\PTIT\DATN\libs\VnCoreNLP\VnCoreNLP-1.1.1.jar"
out_path = "D:\PTIT\DATN\source_l20\\text_sum\\output"

if __name__ == "__main__":

    # read text
    text = utils.read(file_path)
    # title, text = utils.split_text(text)

    text_preprocess = TextPreprocess(tokenizer=word_tokenize)
    unprocessed_text = sent_tokenize(text)
    processed_text = text_preprocess.process_text(sents=unprocessed_text)
    
    # get sent emb after processed text
    sents_emb = np.load(emb_path, allow_pickle=True)

    # vncorenlp
    vncorenlp = VnCoreNLP(address="http://127.0.0.1", port=9000)

    sentence_features = SentenceFeature(text=text, title=None, unprocessed_text=unprocessed_text,
                                    processed_text=processed_text, title_emb=None, sents_emb=sents_emb,
                                    annotator=vncorenlp)
    
    sent_features = []
    for i in range(len(processed_text)):
        sent_features.append((i, sentence_features.get_sentence_features(i)))
    sent_features = sorted(sent_features, key=lambda x:x[1], reverse=True)

    print("===============SENTENCE Ranking===============")
    print([i for i, sc in sent_features])

    try:
        os.remove(os.path.join(out_path, "s_features.txt"))
    except OSError:
        pass
    
    n_summ = np.ceil(len(sent_features) ** .5)
    print(n_summ)
    # for i, score in sent_features[:n_summ]:
    #     utils.write_append(str(i) + " | " + unprocessed_text[i] + "\n\n", os.path.join(out_path, "s_features.txt"))

