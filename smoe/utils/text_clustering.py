import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


class TextClustering:
    def __init__(self, num_experts: int = 16) -> None:
        self.kmeans = KMeans(n_clusters=num_experts)
        self.emb = SentenceTransformer("all-mpnet-base-v2")

    def encode_emb(self, sentences: list[str]) -> np.ndarray:
        arr: np.ndarray = self.emb.encode(sentences=sentences)
        return arr

    def fit_emb(self, emb: np.ndarray):
        self.kmeans.fit(emb)

    def fit(self, sentences: list[str]):
        emb_arr = self.encode_emb(sentences)
        self.kmeans.fit(emb_arr)

    def predict_emb(self, emb: np.ndarray) -> list[int]:
        return self.kmeans.predict(emb)

    def predict(self, sentences: list[str]) -> list[int]:
        emb_arr = self.encode_emb(sentences)
        return self.predict_emb(emb_arr)
