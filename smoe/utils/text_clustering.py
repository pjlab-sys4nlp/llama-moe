from pathlib import Path

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from smoe.utils.vars import CLUSTERING_MODEL_NAME


class TextClustering:
    def __init__(
        self, num_clusters: int = 16, encoder: str = "all-mpnet-base-v2"
    ) -> None:
        self.kmeans = KMeans(n_clusters=num_clusters)
        self.emb = SentenceTransformer(encoder)

    @property
    def num_clusters(self) -> int:
        return self.kmeans.n_clusters

    def encode_emb(self, sentences: list[str]) -> np.ndarray:
        arr: np.ndarray = self.emb.encode(sentences=sentences, show_progress_bar=False)
        return arr

    def fit_emb(self, emb: np.ndarray):
        self.kmeans.fit(emb)

    def fit(self, sentences: list[str]):
        emb_arr = self.encode_emb(sentences)
        self.kmeans.fit(emb_arr)

    def predict_emb(self, emb: np.ndarray) -> list[int]:
        return self.kmeans.predict(emb).tolist()

    def predict(self, sentences: list[str]) -> list[int]:
        emb_arr = self.encode_emb(sentences)
        return self.predict_emb(emb_arr)

    def save_pretrained(self, folder: str):
        model_path = Path(folder) / CLUSTERING_MODEL_NAME
        model_path.parent.mkdir(exist_ok=True, parents=True)
        joblib.dump(self.kmeans, model_path)

    @classmethod
    def from_pretrained(cls, folder: str):
        model_path = Path(folder) / CLUSTERING_MODEL_NAME
        kmeans = joblib.load(model_path)
        model = cls()
        model.kmeans = kmeans
        return model
