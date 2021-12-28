from typing import List
from sentence_transformers import SentenceTransformer

class Model(object):
    def __init__(self, model_path) -> None:
        super(Model, self).__init__()
        self.sbert = SentenceTransformer(model_path)

    def get_sentences_embedding(self, text: List[str]):
        return self.sbert.encode(text)

model = Model('/content/vn_sbert_deploy/phobert_base_mean_tokens_NLI_STS')

def get_model():
  return model