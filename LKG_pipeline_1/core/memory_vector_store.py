"""
内存版向量索引 + BM25 全文检索

替代 Neo4j 的 db.index.vector.queryNodes 和 db.index.fulltext.queryNodes，
使 LKG pipeline 摆脱 Neo4j 依赖、显著提速、可线程并行。

- 向量索引：numpy 矩阵 + 余弦相似度（每库 < 1000 节点完全够用）
- BM25 索引：rank_bm25.BM25Okapi（已是事实标准实现）
- Embedding 模型：复用 sentence-transformers，进程内单例缓存
"""
from __future__ import annotations

import logging
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from LKG_pipeline_1 import config

logger = logging.getLogger(__name__)


# ─── 全局单例：embedding 模型（避免每个 MemoryGraphManager 实例重复加载） ───
_MODEL_LOCK = threading.Lock()
_GLOBAL_MODEL: Optional[SentenceTransformer] = None


def _get_embedding_model() -> SentenceTransformer:
    global _GLOBAL_MODEL
    with _MODEL_LOCK:
        if _GLOBAL_MODEL is None:
            logger.info(f"[MemoryVectorStore] Loading embedding model: {config.EMBEDDING_MODEL}")
            _GLOBAL_MODEL = SentenceTransformer(config.EMBEDDING_MODEL)
        return _GLOBAL_MODEL


# ─── 简单 BM25 实现（避免引入 rank_bm25 依赖） ───
_TOKEN_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class _BM25:
    """BM25 Okapi 简化实现（k1=1.5, b=0.75）。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: List[List[str]] = []
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0.0

    def add(self, doc: str):
        toks = _tokenize(doc)
        self.docs.append(toks)
        freqs: Dict[str, int] = {}
        for t in toks:
            freqs[t] = freqs.get(t, 0) + 1
        self.doc_freqs.append(freqs)
        self.doc_lengths.append(len(toks))

    def build(self):
        if not self.docs:
            self.avgdl = 0.0
            return
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths)
        df: Dict[str, int] = {}
        for freqs in self.doc_freqs:
            for term in freqs:
                df[term] = df.get(term, 0) + 1
        n = len(self.docs)
        # Robertson-Spärck-Jones idf with smoothing
        self.idf = {term: max(0.0, np.log((n - cnt + 0.5) / (cnt + 0.5) + 1.0))
                    for term, cnt in df.items()}

    def score(self, query: str) -> np.ndarray:
        if not self.docs:
            return np.zeros(0)
        q_terms = _tokenize(query)
        scores = np.zeros(len(self.docs), dtype=np.float32)
        for i, freqs in enumerate(self.doc_freqs):
            dl = self.doc_lengths[i]
            denom_norm = 1.0 - self.b + self.b * dl / self.avgdl if self.avgdl else 1.0
            s = 0.0
            for term in q_terms:
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                idf = self.idf.get(term, 0.0)
                s += idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * denom_norm)
            scores[i] = s
        return scores


class MemoryVectorStore:
    """内存向量 + BM25 索引，按 (label) 维护多个独立索引。

    每个 GraphManager 实例持有自己的 MemoryVectorStore（线程安全）。
    """

    def __init__(self):
        self._model = _get_embedding_model()
        # vectors[label] = (ids: List[str], matrix: np.ndarray, nodes: List[dict])
        self._vectors: Dict[str, Dict[str, Any]] = {}
        # bm25[label] = (ids: List[str], texts: List[str], nodes: List[dict], _BM25 instance)
        self._bm25: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._dirty_bm25: set = set()
        self._dirty_vec: set = set()

    # ─── encode 接口（与 Neo4j 版 VectorStore.encode 兼容） ───
    def encode(self, text):
        # 关闭 sentence-transformers 默认的 tqdm 进度条（避免刷屏）
        emb = self._model.encode(text, show_progress_bar=False)
        if isinstance(emb, np.ndarray):
            return emb.tolist()
        return emb

    # ─── 添加节点 ───
    def add_node(self, label: str, node_id: str, embedding: Optional[List[float]],
                 fulltext_doc: Optional[str], node_props: Dict[str, Any]):
        with self._lock:
            if embedding is not None:
                store = self._vectors.setdefault(label, {"ids": [], "vecs": [], "nodes": [], "matrix": None})
                store["ids"].append(node_id)
                store["vecs"].append(np.asarray(embedding, dtype=np.float32))
                store["nodes"].append(node_props)
                self._dirty_vec.add(label)
            if fulltext_doc is not None:
                bstore = self._bm25.setdefault(label, {"ids": [], "texts": [], "nodes": [], "bm25": None})
                bstore["ids"].append(node_id)
                bstore["texts"].append(fulltext_doc)
                bstore["nodes"].append(node_props)
                self._dirty_bm25.add(label)

    # ─── 重建索引（lazy，仅在搜索前触发） ───
    def _ensure_vector_index(self, label: str):
        if label not in self._vectors:
            return None
        if label in self._dirty_vec:
            store = self._vectors[label]
            if store["vecs"]:
                mat = np.stack(store["vecs"]).astype(np.float32)
                # L2 归一化以加速 cosine（即 dot）
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                store["matrix"] = mat / norms
            else:
                store["matrix"] = np.zeros((0, 1), dtype=np.float32)
            self._dirty_vec.discard(label)
        return self._vectors[label]

    def _ensure_bm25_index(self, label: str):
        if label not in self._bm25:
            return None
        if label in self._dirty_bm25:
            bstore = self._bm25[label]
            bm = _BM25()
            for t in bstore["texts"]:
                bm.add(t)
            bm.build()
            bstore["bm25"] = bm
            self._dirty_bm25.discard(label)
        return self._bm25[label]

    # ─── 检索 API ───
    def search_similar(self, label: str, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """返回 [{"node": dict, "score": float}, ...] —— 与 Neo4j VectorStore 输出兼容。"""
        store = self._ensure_vector_index(label)
        if not store or store["matrix"] is None or store["matrix"].shape[0] == 0:
            return []
        q = np.asarray(self.encode(query_text), dtype=np.float32)
        qn = np.linalg.norm(q)
        if qn == 0:
            return []
        q = q / qn
        scores = store["matrix"] @ q  # (N,)
        k = min(top_k, len(scores))
        if k == 0:
            return []
        # 取 top-k
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [{"node": store["nodes"][i], "score": float(scores[i])} for i in top_idx]

    def fulltext_search(self, label: str, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """BM25 全文检索，输出格式与 Neo4j VectorStore.fulltext_search 一致。"""
        bstore = self._ensure_bm25_index(label)
        if not bstore or bstore["bm25"] is None:
            return []
        scores = bstore["bm25"].score(query_text)
        if scores.size == 0:
            return []
        k = min(top_k, len(scores))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        # 过滤分数 <= 0 的
        return [{"node": bstore["nodes"][i], "score": float(scores[i])}
                for i in top_idx if scores[i] > 0]

    def clear(self):
        with self._lock:
            self._vectors.clear()
            self._bm25.clear()
            self._dirty_vec.clear()
            self._dirty_bm25.clear()
