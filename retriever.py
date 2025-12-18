import json
import torch
import faiss
import numpy as np
import torch.nn.functional as F
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel

class CLSEmbedding:
    def __init__(self, model_dir: str="BAAI/bge-m3"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True
        )

        self.model = AutoModel.from_pretrained(
            model_dir,
            local_files_only=True
        ).to(self.device)

        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: List[str], return_tensor=False) -> np.ndarray:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        # CLS embedding
        cls_embed = outputs.last_hidden_state[:, 0]
        cls_embed = F.normalize(cls_embed, dim=1)

        if return_tensor:
            return cls_embed
        else:
            return cls_embed.cpu().numpy()


class SemanticRetriever:
    def __init__(self, embedding_model: CLSEmbedding):
        self.embedder = embedding_model
        self.index = None
        self.corpus_texts = []

    def build_index(self, corpus: List[str]):
        """
        corpus: list of documents / sentences
        """
        self.corpus_texts = corpus
        embeddings = self.embedder.encode(corpus)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 5):
        query_emb = self.embedder.encode([query])
        scores, indices = self.index.search(query_emb, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            results.append({
                "text": self.corpus_texts[idx],
                "similarity": float(score)
            })
        return results


    def compute_embedding_shift_metrics(self,
        context1: str,
        context2: str,
        knn_k: int = 5
    ) -> Dict[str, float]:

        emb_1 = self.embedder.encode([context1])
        emb_2 = self.embedder.encode([context2])
        assert emb_1.shape == emb_2.shape

        metrics = {}

        metrics["cosine_shift_mean"] = self.cosine_shift(emb_1, emb_2).mean()
        metrics["l2_shift_mean"] = self.l2_shift(emb_1, emb_2).mean()
        metrics["mmd_rbf"] = self.mmd_rbf(emb_1, emb_2)
        metrics[f"knn_overlap@{knn_k}"] = self.knn_overlap(emb_1, emb_2, k=knn_k)

        return metrics


    def cosine_shift(self, E_big: np.ndarray, E_small: np.ndarray) -> np.ndarray:
        """
        return: shape [N]
        """
        cos = np.sum(E_big * E_small, axis=1)
        return 1.0 - cos


    def l2_shift(self, E_big: np.ndarray, E_small: np.ndarray) -> np.ndarray:
        """
        return: shape [N]
        """
        return np.linalg.norm(E_big - E_small, axis=1)

    def rbf_kernel(self, X: np.ndarray, Y: np.ndarray, gamma: float = None):
        """
        X: [N, D], Y: [M, D]
        """
        XX = np.sum(X * X, axis=1, keepdims=True)
        YY = np.sum(Y * Y, axis=1, keepdims=True)
        distances = XX - 2 * np.dot(X, Y.T) + YY.T

        if gamma is None:
            gamma = 1.0 / X.shape[1]

        return np.exp(-gamma * distances)

    def mmd_rbf(self, E_big: np.ndarray, E_small: np.ndarray, gamma: float = None) -> float:
        """
        Maximum Mean Discrepancy between two embedding distributions
        """
        K_xx = self.rbf_kernel(E_big, E_big, gamma)
        K_yy = self.rbf_kernel(E_small, E_small, gamma)
        K_xy = self.rbf_kernel(E_big, E_small, gamma)

        mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
        return float(mmd)

    def knn_overlap(self,
        E_big: np.ndarray,
        E_small: np.ndarray,
        k: int = 5
    ) -> float:
        """
        Compute average kNN overlap ratio
        """

        dim = E_big.shape[1]

        index_big = faiss.IndexFlatIP(dim)
        index_small = faiss.IndexFlatIP(dim)

        index_big.add(E_big)
        index_small.add(E_small)

        _, nn_big = index_big.search(E_big, k + 1)
        _, nn_small = index_small.search(E_small, k + 1)

        overlaps = []
        for i in range(E_big.shape[0]):
            # remove self
            set_big = set(nn_big[i][1:])
            set_small = set(nn_small[i][1:])
            overlaps.append(len(set_big & set_small) / k)

        return float(np.mean(overlaps))
