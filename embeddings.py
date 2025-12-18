import os
import json
from typing import List, Dict
from langchain.schema.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel
from FlagEmbedding import BGEM3FlagModel
import torch
import torch.nn.functional as F


class Embedding(Embeddings):
    """
    Call embeddingng model, encode text to vector.
    """
    def __init__(self, model_name, **kwargs):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.local_dir = ""
        if model_name == "BAAI/bge-m3":
            self.model = BGEM3FlagModel(self.local_dir, local_files_only=True, **kwargs)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_dir, local_files_only=True)
            self.model_hf = AutoModel.from_pretrained(self.local_dir, local_files_only=True)
    
    def embed_documents(self, texts, max_token=500):
        """
        Encode documents to dense vectors and return.
        param: 
         - texts: documents to be encoded, format: [text1, text2, ...].
         - max_token: Maximum number of tokens to be encoded, max limit is 8192, set default 500.
        return:
         - dense matrix, dim=[len(documents), 1024].
        """
        return self.model.encode(texts, return_dense=True, max_length=max_token)['dense_vecs'].tolist()

    def embed_query(self, text, max_token=500):
        """
        Encode one sentence to dense vector and return.
        param: 
         - texts: sentence to be encoded, string format.
         - max_token: Maximum number of tokens to be encoded, max limit is 8192, set default 500.
        return:
         - dense vector number, dim=[1, 1024].
        """
        return self.model.encode([text], return_dense=True, max_length=max_token)['dense_vecs'][0].tolist()

    def dense_encode(self,
                    content1: List[str],
                    content2: List[str], 
                    max_token: int =500) -> List[float]:
        """
        Encode texts to dense vectors and return their similarity score(dense).
        param: 
         - context1: texts to be encoded, format: [text1, text2, ...].
         - context2: texts to be encoded, format: [text1, text2, ...].
         - max_token: Maximum number of tokens to be encoded, max limit is 8192, set default 500.
        return:
         - dense score matrix of content1 and content2, shape=[len(content1), len(content2)].
        """
        embeddings_1 = self.model.encode(content1, max_length=max_token)['dense_vecs'] 
        embeddings_2 = self.model.encode(content2, max_length=max_token)['dense_vecs'] 
        score_dense = embeddings_1 @ embeddings_2.T
        return score_dense

    def sparse_encode(self,
                    content1: List[str],
                    content2: List[str]) -> List[float]:
        """
        Encode texts to sparse vectors based on token and return their token similarity scores, high score for important token(keywords) and 0 for others.
        param: 
         - context1: texts to be encoded, format: [text1, text2, ...].
         - context2: texts to be encoded, format: [text1, text2, ...].
        return:
         - sparse score matrix of content1 and content2, shape=[len(content1), len(content2)].
        """
        embeddings_1 = self.model.encode(content1, return_sparse=True)
        embeddings_2 = self.model.encode(content2, return_sparse=True)
        # print(self.model.convert_id_to_token(embeddings_1['lexical_weights']))

        # compute the scores via token mathcing
        score_sparse20 = self.model.compute_lexical_matching_score(embeddings_1['lexical_weights'][0], embeddings_2['lexical_weights'][0])
        score_sparse21 = self.model.compute_lexical_matching_score(embeddings_1['lexical_weights'][0], embeddings_2['lexical_weights'][1])

        print(score_sparse20)
        print(score_sparse21)
        return [score_sparse21, score_sparse20]
    
    def encode_file(self,
                    json_path: str,
                    text_key: str = "title") -> List[float]:
        """
        Encode text from a JSON file to vectors.
        param: 
         - json_path: Path to the JSON file, format: [{"key1": "value1", "key2": "value2", ...}, ...].
         - text_key: The key in the JSON objects that contains the text to be encoded.
        return:
         - Normalised embedded vectors for the file, format:[len(file)=texts_number, 1024=vector_length].
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        texts = [item[text_key] for item in data]

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            out = self.model_hf(**inputs)

        mask = inputs["attention_mask"]
        last_hidden = out.last_hidden_state
        embed = (last_hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        embed_norm = F.normalize(embed, p=2, dim=1)    # p=2 for L2 normalization, dim=1 for row-wise normalization 
        return embed_norm

    def encode_topic(self,
                    topics: List[str]) -> List[float]:
        """
        Encode topics to vectors.
        param: 
         - topics: the list of topics.
        return:
         - Normalised embedded vectors for the topics, format:[len(topics)=topics_number, 1024=vector_length].
        """
        inputs = self.tokenizer(topics, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            out = self.model_hf(**inputs)

        mask = inputs["attention_mask"]
        last_hidden = out.last_hidden_state
        embed = (last_hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        embed_norm = F.normalize(embed, p=2, dim=1)    # p=2 for L2 normalization, dim=1 for row-wise normalization 
        return embed_norm
    

############## test ##############
if __name__ == "__main__":
    embedding_model = Embedding("BAAI/bge-m3/no_fp16", use_fp16=False)
    print(embedding_model.encode_topic(["What is the capital of France?"]))







