"""
IVF-PQ + HNSW: 2-stage retrieval system
==================================

Design:
 - Stage 1: IVF-PQ(compression storage, fast filtering)
 - Stage 2: HNSW(high precision graph retrieval)
"""

import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional
import pickle
import os
import json
import time
from dataclasses import dataclass
from retriever import CLSEmbedding, SemanticRetriever
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IndexConfig:
    """Index configuration parameters"""
    # IVF-PQ parameters
    nlist: int = 4096              # number of clusters(suggested: sqrt(N) to 4*sqrt(N))
    m_pq: int = 64                 # number of subvectors in PQ(suggested: dimension/8, must be divisible by dimension, larger m means the quantization is more accurate)
    nbits_pq: int = 8              # number of bits per subvector(suggested: 8)
    nprobe: int = 32               # number of clusters to probe/search(suggested: 32-256)
    
    # HNSW parameters
    M_hnsw: int = 32               # number of maximum connections per node in HNSW(suggested: 16-64)
    efConstruction: int = 200      # number of candidates in the construction phase(suggested: 200-1000)
    efSearch: int = 128            # number of candidates in the search phase(suggested: 128-512)
    
    # Hybrid strategy parameters
    use_two_stage: bool = True     # whether to use two-stage retrieval
    stage1_k: int = 200            # number of candidates in the first stage(suggested: 200-1000)
    final_k: int = 10              # number of candidates to return(suggested: 10-100)
    
    # Training parameters
    train_size: int = 1000000      # number of training samples(suggested: 1000000-10000000)


class HybridVectorIndex:
    """
    IVF-PQ + HNSW 
    
    Workflow:
    1. Training stage:
       - Use samples to train IVF cluster centers
       - Train PQ encoder
       
    2. Indexing stage:
       - Store original vectors in IVF-PQ (compression storage)
       - Store selected vectors in HNSW (precise retrieval)
       
    3. Retrieval stage:
       - Stage 1: IVF-PQ rough screening, quickly get top-K candidates
       - Stage 2: HNSW precise ranking, ensure high recall rate
    """
    
    def __init__(self, dimension: int, config: Optional[IndexConfig] = None):
        """
        Args:
            dimension: Vector dimension
            config: Index configuration, default is None
        """
        self.dimension = dimension
        self.config = config or IndexConfig()
        self.corpus_texts = []
        
        # Validate PQ parameters
        if dimension % self.config.m_pq != 0:
            raise ValueError(
                f"Vector dimension ({dimension}) must be divisible by m_pq ({self.config.m_pq})"
            )
        
        self.ivf_pq_index = None
        self.hnsw_index = None
        self.is_trained = False
        
        logger.info(f"Initialize hybrid index: dimension={dimension}")
        logger.info(f"IVF-PQ configuration: nlist={self.config.nlist}, "
                   f"m={self.config.m_pq}, nbits={self.config.nbits_pq}")
        logger.info(f"HNSW configuration: M={self.config.M_hnsw}, "
                   f"efConstruction={self.config.efConstruction}")
    
    def _create_ivf_pq_index(self) -> faiss.IndexIVFPQ:
        logger.info("Create IVF-PQ index...")
        
        # Use HNSW as a coarse quantizer (faster than Flat)
        quantizer = faiss.IndexHNSWFlat(
            self.dimension,
            self.config.M_hnsw
        )
        quantizer.hnsw.efConstruction = self.config.efConstruction
        
        print("debug", self.config.nlist)
        # Create IVF-PQ index
        index = faiss.IndexIVFPQ(
            quantizer,
            self.dimension,
            self.config.nlist,      # Cluster number
            self.config.m_pq,       # Subvector number
            self.config.nbits_pq    # Bits per subvector
        )
        
        # Set search parameters
        index.nprobe = self.config.nprobe
        
        return index
    
    def _create_hnsw_index(self) -> faiss.IndexHNSWFlat:
        """
        Create HNSW index for precise ranking
        
        Features:
        - High precision graph structure
        - Fast approximate nearest neighbor search
        - No training required
        """
        logger.info("Create HNSW index for precise ranking...")
        
        index = faiss.IndexHNSWFlat(
            self.dimension,
            self.config.M_hnsw
        )
        
        # Set search parameters
        index.hnsw.efConstruction = self.config.efConstruction
        index.hnsw.efSearch = self.config.efSearch
        
        return index
    
    def train(self, training_data: np.ndarray):
        """
        Train index (only IVF-PQ needs training)
        
        Args:
            training_data: Training data, shape=(n_samples, dimension)
        
        Recommended training sample size:
        - Minimum: 30 * nlist
        - Recommended: 100-256 * nlist
        - For nlist=4096: 400K - 1M samples
        """
        logger.info(f"Train IVF-PQ index, training data: {training_data.shape}")
        
        # Check training data size
        min_samples = 30 * self.config.nlist
        if len(training_data) < min_samples:
            logger.warning(
                f"Training data size ({len(training_data)}) is less than the recommended value ({min_samples})"
            )
        
        # Normalize (if using inner product similarity)
        faiss.normalize_L2(training_data)
        
        # Create and train IVF-PQ index
        self.ivf_pq_index = self._create_ivf_pq_index()
        
        start_time = time.time()
        self.ivf_pq_index.train(training_data)
        train_time = time.time() - start_time
        
        logger.info(f"✓ IVF-PQ training completed, time: {train_time:.2f} seconds")
        
        self.is_trained = True
    
    def build_index(self, corpus: List[str], embeddings: np.ndarray):
        """
        Build two-stage index
        
        Args:
            corpus: Document list
            embeddings: Corresponding vectors, shape=(n_docs, dimension)
        
        Workflow:
        1. Normalize vectors
        2. Add to IVF-PQ (all vectors, compression storage)
        3. Add to HNSW (all vectors, for precise ranking)
        """
        logger.info(f"Build index: {len(corpus)} documents")
        
        self.corpus_texts = corpus
        
        # Normalize vectors
        faiss.normalize_L2(embeddings)
    
        if self.config.use_two_stage:
            # Train and create IVF-PQ index
            self.train(embeddings)
            if not self.is_trained:
                raise RuntimeError("Index not trained, please call train() method")
            # Add to IVF-PQ index
            logger.info("Add vectors to IVF-PQ...")
            start_time = time.time()
            self.ivf_pq_index.add(embeddings)
            ivfpq_time = time.time() - start_time
        
        # Create HNSW index (no training required)
        self.hnsw_index = self._create_hnsw_index()
        # Add to HNSW index
        logger.info("Add vectors to HNSW...")
        start_time = time.time()
        self.hnsw_index.add(embeddings)
        hnsw_time = time.time() - start_time
        
        logger.info(f"✓ Index built")
        logger.info(f"  - IVF-PQ added: {ivfpq_time:.2f} seconds" if self.config.use_two_stage else "No IVF-PQ")
        logger.info(f"  - HNSW added: {hnsw_time:.2f} seconds")
        
        # Print memory usage statistics
        self._print_memory_stats()
    
    def _print_memory_stats(self):
        """Print memory usage statistics"""
        n_vectors = self.ivf_pq_index.ntotal if self.config.use_two_stage else self.hnsw_index.ntotal
        
        # IVF-PQ memory estimation
        # Per vector = m_pq * nbits_pq / 8 bytes
        bytes_per_vector_pq = self.config.m_pq * self.config.nbits_pq // 8
        ivfpq_size_mb = (n_vectors * bytes_per_vector_pq) / (1024 ** 2)
        
        # HNSW memory estimation
        # Per vector = (dimension * 4 + M * 2 * 4) bytes
        bytes_per_vector_hnsw = (self.dimension * 4 + 
                                 self.config.M_hnsw * 2 * 4)
        hnsw_size_mb = (n_vectors * bytes_per_vector_hnsw) / (1024 ** 2)
        
        # Original vector size
        raw_size_mb = (n_vectors * self.dimension * 4) / (1024 ** 2)
        
        logger.info(f"\nMemory usage statistics ({n_vectors:,} vectors):")
        logger.info(f"  - Original vectors: {raw_size_mb:.1f} MB")
        logger.info(f"  - IVF-PQ: {ivfpq_size_mb:.1f} MB "
                   f"(compression ratio: {raw_size_mb/ivfpq_size_mb:.1f}x)")
        logger.info(f"  - HNSW: {hnsw_size_mb:.1f} MB")
        logger.info(f"  - Total: {ivfpq_size_mb + hnsw_size_mb:.1f} MB")
    
    def retrieve(self, query: str, embedder, top_k: int = 10) -> List[Dict]:
        """
        Two-stage retrieval
        
        Args:
            query: Query text
            embedder: Encoder
            top_k: Number of return results
        
        Returns:
            Retrieval results list
        
        Workflow:
        1. Encode query
        2. Stage 1 (IVF-PQ): Rough screening, get top-K*20 candidates
        3. Stage 2 (HNSW): Precise ranking, return top-K
        """
        if not self.is_trained and self.config.use_two_stage:
            raise RuntimeError("Index not built, please call build_index() method")
        
        # Encode query
        query_emb = embedder.encode([query])
        faiss.normalize_L2(query_emb)
        
        if self.config.use_two_stage:
            return self._two_stage_retrieve(query_emb, top_k)
        else:
            return self._single_stage_retrieve(query_emb, top_k)
    
    def _two_stage_retrieve(self, query_emb: np.ndarray, 
                           top_k: int) -> List[Dict]:
        """
        Two-stage retrieval strategy
        
        Stage 1 (IVF-PQ): Rough screening, get top-K*20 candidates
        - Advantages: Fast, low memory
        - Disadvantages: Low precision (compression loss)
        
        Stage 2 (HNSW): Precise ranking, return top-K
        - Advantages: High precision
        - Disadvantages: High memory
        
        Combined advantages: Speed + precision
        """
        # Stage 1: IVF-PQ rough screening, get more candidates
        stage1_k = max(self.config.stage1_k, top_k * 10)
        
        start_time = time.time()
        scores_stage1, indices_stage1 = self.ivf_pq_index.search(
            query_emb, stage1_k
        )
        stage1_time = (time.time() - start_time) * 1000
        
        # Stage 2: HNSW precise ranking
        # Only calculate the exact distance for the candidates in stage1
        start_time = time.time()
        
        # Get the IDs of the candidate vectors
        candidate_ids = indices_stage1[0]
        valid_ids = candidate_ids[candidate_ids >= 0]  # Filter invalid IDs
        
        if len(valid_ids) == 0:
            logger.warning("Stage 1: No valid candidates found")
            return []

        id_selector = faiss.IDSelectorArray(len(valid_ids),
                                        faiss.swig_ptr(valid_ids.astype(np.int64)))
        search_params = faiss.SearchParametersHNSW()
        # search_params.efSearch = self.config.efSearch
        search_params.sel = id_selector

        scores_stage2, indices_stage2 = self.hnsw_index.search(
            query_emb,
            top_k,
            params=search_params
        )
        
        stage2_time = (time.time() - start_time) * 1000
        
        # Build results
        results = []
        for idx, score in zip(indices_stage2[0], scores_stage2[0]):
            if idx >= 0 and idx < len(self.corpus_texts):
                results.append({
                    "text": self.corpus_texts[idx],
                    "similarity": float(score),
                    "index": int(idx)
                })
        
        logger.debug(
            f"Retrieval time: Stage1={stage1_time:.2f}ms, "
            f"Stage2={stage2_time:.2f}ms, "
            f"Total={stage1_time+stage2_time:.2f}ms"
        )
        
        return results
    
    def _single_stage_retrieve(self, query_emb: np.ndarray, 
                              top_k: int) -> List[Dict]:
        """Single-stage retrieval (only using HNSW)"""
        start_time = time.time()
        scores, indices = self.hnsw_index.search(query_emb, top_k)
        search_time = (time.time() - start_time) * 1000
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0 and idx < len(self.corpus_texts):
                results.append({
                    "text": self.corpus_texts[idx],
                    "similarity": float(score),
                    "index": int(idx)
                })
        
        logger.debug(f"HNSW retrieval time: {search_time:.2f}ms")
        
        return results
    
    def optimize_search_params(self, 
                              queries: List[np.ndarray],
                              ground_truth: List[np.ndarray],
                              target_recall: float = 0.95):
        """
        Automatically optimize search parameters(nprobe) to achieve target recall
        
        Args:
            queries: Query vectors list
            ground_truth: True nearest neighbors (for calculating recall)
            target_recall: Target recall
        """
        logger.info(f"Start parameter optimization, target recall: {target_recall}")
        
        # Test different nprobe values
        nprobe_values = [8, 16, 32, 64, 128, 256]
        best_nprobe = self.config.nprobe
        best_recall = 0.0
        
        for nprobe in nprobe_values:
            self.ivf_pq_index.nprobe = nprobe
            recall = self._compute_recall(queries, ground_truth, k=10)
            logger.info(f"nprobe={nprobe}: recall={recall:.4f}")
            
            if recall >= target_recall and recall > best_recall:
                best_recall = recall
                best_nprobe = nprobe
            
            if recall >= target_recall:
                break
        
        self.config.nprobe = best_nprobe
        self.ivf_pq_index.nprobe = best_nprobe
        
        logger.info(f"✓ Optimization completed: nprobe={best_nprobe}, recall={best_recall:.4f}")
    
    def _compute_recall(self, queries: List[np.ndarray], 
                       ground_truth: List[np.ndarray], k: int) -> float:
        """Calculate recall"""
        total_recall = 0.0
        
        for query, gt in zip(queries, ground_truth):
            _, indices = self.ivf_pq_index.search(query.reshape(1, -1), k)
            retrieved = set(indices[0])
            true_neighbors = set(gt[:k])
            recall = len(retrieved & true_neighbors) / k
            total_recall += recall
        
        return total_recall / len(queries)
    
    def save(self, save_dir: str):
        """Save index to disk"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save index
        faiss.write_index(
            self.ivf_pq_index, 
            os.path.join(save_dir, "ivf_pq.index")
        )
        faiss.write_index(
            self.hnsw_index,
            os.path.join(save_dir, "hnsw.index")
        )
        
        # Save text and configuration
        with open(os.path.join(save_dir, "corpus.pkl"), "wb") as f:
            pickle.dump(self.corpus_texts, f)
        
        with open(os.path.join(save_dir, "config.pkl"), "wb") as f:
            pickle.dump(self.config, f)
        
        logger.info(f"✓ Index saved to: {save_dir}")
    
    def load(self, save_dir: str):
        """Load index from disk"""
        self.ivf_pq_index = faiss.read_index(
            os.path.join(save_dir, "ivf_pq.index")
        )
        self.hnsw_index = faiss.read_index(
            os.path.join(save_dir, "hnsw.index")
        )
        
        with open(os.path.join(save_dir, "corpus.pkl"), "rb") as f:
            self.corpus_texts = pickle.load(f)
        
        with open(os.path.join(save_dir, "config.pkl"), "rb") as f:
            self.config = pickle.load(f)
        
        self.is_trained = True
        
        logger.info(f"✓ Index loaded from {save_dir}")
        logger.info(f"  - Number of vectors: {self.ivf_pq_index.ntotal:,}")


# ============================================================================
# 使用示例
# ============================================================================

def example_usage():
    """演示完整的使用流程"""
    print("\n" + "="*80)
    print("IVF-PQ + HNSW 混合索引示例")
    print("="*80 + "\n")
    # 1. 准备数据
    print("1. 准备数据...")
    questions = json.load(open("gold/QAmedqa.json"))
    questions = [f"question: {q['question']}\noptions: {q['options']}" for id, q in questions.items()]

    corpus = json.load(open("gold/textbooks/corpus_1000.json"))
    corpus = [c["contents"] for c in corpus]
    corpus_embeddings = np.load("corpus_embeddings/ce1000.npy")

    dimension = corpus_embeddings.shape[1] # 1024
    n_train = 100      # 训练样本
    n_corpus = len(corpus)   # 语料库大小
    # 生成训练数据
    train_data = corpus_embeddings[:n_train]

    # 2. 配置索引
    print("\n2. 配置索引...")
    config = IndexConfig(
        nlist=5,           
        m_pq=16,              # 1024维度可以被整除
        nbits_pq=2,
        nprobe=3,
        M_hnsw=16,
        efConstruction=100,
        efSearch=64,
        use_two_stage=False,
        stage1_k=100
    )
    embedder = CLSEmbedding(model_dir="BAAI/bge-m3")

    # 3. 初始化索引
    print("\n3. 初始化索引...")
    index = HybridVectorIndex(dimension, config)

    # 4. 构建索引
    print("\n4. 构建索引...")
    index.build_index(corpus, corpus_embeddings)

    # 5. 检索
    print("\n5. 执行检索...")
    query = questions[596]
    results = index.retrieve(query, embedder, top_k=3)

    print(f"\n查询: {query}")
    print(f"找到 {len(results)} 个结果:\n")
    for i, result in enumerate(results[:5], 1):
        print(f"{i}. {result['text']}")
        print(f"   相似度: {result['similarity']:.4f}\n")

    # 6. 保存索引
    print("6. 保存索引...")
    index.save("./hybrid_index")

    print("\n" + "="*80)
    print("示例完成！")
    print("="*80 + "\n")


if __name__ == "__main__":
    # 运行示例
    example_usage()

