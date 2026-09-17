"""TF-IDF indexing and vote-weighted-centroid recommendation.

Replaces arxiv-sanity-lite's tf-idf + linear SVM classifier with a simpler,
dependency-free scheme: build a tf-idf vector per paper, take the mean tf-idf
vector of upvoted papers as the user's interest centroid, and rank unseen
papers by cosine similarity to it.
"""

import math
import re
from collections import Counter

from .store import Paper, Store

TOKEN_RE = r"[a-z0-9]{2,}"
STOPWORDS = {
    "the", "of", "and", "to", "in", "we", "for", "on", "a", "is", "are", "with",
    "that", "this", "by", "as", "an", "be", "or", "from", "it", "our", "at",
    "which", "can", "these", "such", "into", "than", "its", "have", "has",
}


def tokenize(text):
    return [t for t in re.findall(TOKEN_RE, text.lower())
            if t not in STOPWORDS]


class TfIdfIndex:
    def __init__(self, papers):
        self.papers = list(papers)
        self.counts = [Counter(tokenize(p.text)) for p in self.papers]
        self.vocab = sorted({t for c in self.counts for t in c})
        self.df = Counter()
        for c in self.counts:
            for t in c:
                self.df[t] += 1
        self._idf = {t: math.log((1 + len(self.papers)) / (1 + self.df[t])) + 1
                     for t in self.vocab}

    def vector(self, i):
        c = self.counts[i]
        return {t: (1 + math.log(n)) * self._idf[t] for t, n in c.items()}

    @staticmethod
    def cosine(a, b):
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        num = sum(a[t] * b[t] for t in common)
        den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
        return num / den if den else 0.0

    def centroid(self, indices):
        indices = list(indices)
        if not indices:
            return {}
        acc = Counter()
        for i in indices:
            for t, v in self.vector(i).items():
                acc[t] += v
        return {t: v / len(indices) for t, v in acc.items()}


class Recommender:
    def __init__(self, store: Store):
        self.store = store
        self._sig = None

    def _index(self):
        # rebuild only when the corpus or votes changed
        n_papers = self.store.conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        n_votes = len(self.store.upvoted_ids())
        sig = (n_papers, n_votes)
        if sig == self._sig:
            return
        self._sig = sig
        self.papers = list(self.store.all_papers())
        self.index = TfIdfIndex(self.papers)
        self.id_to_i = {p.arxiv_id: i for i, p in enumerate(self.papers)}

    def similar(self, arxiv_id, k=5):
        self._index()
        i = self.id_to_i.get(arxiv_id)
        if i is None:
            return []
        v = self.index.vector(i)
        scored = [(self.index.cosine(v, self.index.vector(j)), self.papers[j])
                  for j in range(len(self.papers)) if j != i]
        scored.sort(key=lambda x: -x[0])
        return [{"score": round(s, 4), "paper": p.__dict__} for s, p in scored[:k]]

    def recommend(self, k=5, exclude_voted=True):
        """Rank unseen papers by similarity to the upvote centroid."""
        self._index()
        upvoted = self.store.upvoted_ids()
        if not upvoted:
            return []
        centroid = self.index.centroid(
            self.id_to_i[a] for a in upvoted if a in self.id_to_i)
        scored = []
        for i, p in enumerate(self.papers):
            if exclude_voted and p.arxiv_id in upvoted:
                continue
            scored.append((self.index.cosine(centroid, self.index.vector(i)), p))
        scored.sort(key=lambda x: -x[0])
        return [{"score": round(s, 4), "paper": p.__dict__} for s, p in scored[:k]]

    def digest(self, k=5):
        """Daily digest: the recommendations, or latest papers if no votes yet."""
        recs = self.recommend(k=k)
        if recs:
            return {"mode": "recommended", "papers": recs}
        self._index()
        return {"mode": "latest", "papers": [p.__dict__ for p in self.papers[:k]]}
