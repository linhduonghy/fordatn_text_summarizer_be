from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

class TextRank():

  def __init__(self, sents_emb, threshold=0.5, lambda_=0.5):
    self.sents_emb = sents_emb
    self.sim_mat = None
    self.threshold = threshold
    self.lambda_ = lambda_

  def build_graph(self):
    # calculate similarity matrix
    self.sim_mat = cosine_similarity(self.sents_emb)

    nodes = [i for i in range(self.sim_mat.shape[0])]

    graph = nx.Graph()
    graph.add_nodes_from(nodes)

    for i in range(self.sim_mat.shape[0]):
      for j in range(i + 1, self.sim_mat.shape[0]):
        if self.sim_mat[i][j] > self.threshold:
          graph.add_edge(i, j, weight=self.sim_mat[i][j])       
    return graph

  def pagerank(self):

    graph = self.build_graph()
    pr_scores = nx.pagerank(graph, max_iter=1000, weight='weight')
    
    temp = [[k, v] for k, v in pr_scores.items()]
    return sorted(temp, key=lambda x:x[1], reverse=True)

  def get_summary_index(self, n_summ, t=0.5):
    rank = self.pagerank()
    s_index = [r[0] for r in rank]
    summ = [s_index[0]]
    s_index.remove(s_index[0])
    while len(summ) < n_summ:
      flag = False
      for k in s_index:
        if self.check(k, summ, t):          
          summ.append(k)
          s_index.remove(k)
          flag = True
          break
      if not flag:
        for k in s_index:
          summ.append(k)
          if len(summ) == n_summ:
            break
        break
    return sorted(summ)


  def check(self, k: int, summ, t) -> bool:
    for i in summ:
      if self.sim_mat[i][k] < t:
        return True
    return False

  def pagerank_score(self):
    graph = self.build_graph()
    pr_scores = nx.pagerank(graph, max_iter=1000, weight='weight')
    return pr_scores

  def re_rank_mmr(self, pr_scores):
    
    s = []
    r = [[k, v] for k, v in pr_scores.items()]
    r = sorted(r, key=lambda x: x[1], reverse=True)    
    r = [i[0] for i in r]

    while len(r) > 0:
      max_mmr, index = float("-inf"), 0
      for i in r:
        # sim1, sim2
        sim1 = pr_scores[i]
        sim2 = float("-inf")        
        for j in s:
          sim2 = max(sim2, self.sim_mat[i][j])

        # mmr
        mmr = self.lambda_*sim1 - (1-self.lambda_)*sim2

        # update mmr
        if mmr > max_mmr:
          max_mmr = mmr
          index = i
      
      # add index to s, remove from r
      s.append(index)
      r.remove(index)
    
    return s

  def get_sentences_ranking(self):
    pr_scores = self.pagerank_score()    
    return self.re_rank_mmr(pr_scores)