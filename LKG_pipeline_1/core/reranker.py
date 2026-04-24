"""
Cross-encoder Reranker 模块

使用 BAAI/bge-reranker-v2-m3 对 bi-encoder 召回的候选进行精排。
全局单例加载（与 embedding 模型相同策略），避免重复加载 568M 参数模型。

用法：
    from LKG_pipeline_1.core.reranker import get_reranker
    reranker = get_reranker()
    results = reranker.rerank("Who founded Orion Pictures?",
                               ["Orion Pictures", "Safari School", "Mike Medavoy"],
                               top_k=3)
    # results = [("Mike Medavoy", 0.92), ("Orion Pictures", 0.88), ("Safari School", 0.05)]
"""
from __future__ import annotations

import logging
import threading
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_RERANKER_LOCK = threading.Lock()
_GLOBAL_RERANKER: Optional["Reranker"] = None

RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"


def get_reranker() -> "Reranker":
    """获取全局单例 Reranker 实例。"""
    global _GLOBAL_RERANKER
    with _RERANKER_LOCK:
        if _GLOBAL_RERANKER is None:
            _GLOBAL_RERANKER = Reranker()
        return _GLOBAL_RERANKER


class Reranker:
    """Cross-encoder reranker wrapping BAAI/bge-reranker-v2-m3.

    Uses sentence-transformers CrossEncoder API for simplicity.
    Falls back gracefully if model unavailable.
    """

    def __init__(self, model_name: str = RERANKER_MODEL_ID):
        self._model = None
        self._model_name = model_name
        self._available = False
        self._load()

    def _load(self):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            logger.warning("[Reranker] sentence-transformers not installed. Reranker disabled.")
            return

        # 尝试多个模型路径：HuggingFace cache → ModelScope cache → 在线下载
        import os, glob
        candidates = [self._model_name]
        hf_org, model_short = "BAAI", "bge-reranker-v2-m3"
        for cache_dir in [
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
            os.path.join("D:\\", "hugging face", "hub"),
        ]:
            pattern = os.path.join(cache_dir, f"models--{hf_org}--{model_short}", "snapshots", "*", "config.json")
            matches = glob.glob(pattern)
            if matches:
                candidates.insert(0, os.path.dirname(matches[0]))
        for ms_dir in [
            os.path.join("D:\\", "hugging face", "modelscope_cache"),
            os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub"),
        ]:
            ms_path = os.path.join(ms_dir, hf_org, model_short)
            if os.path.exists(os.path.join(ms_path, "config.json")):
                candidates.insert(0, ms_path)

        for path in candidates:
            try:
                logger.info(f"[Reranker] Trying {path}...")
                self._model = CrossEncoder(path, max_length=512)
                self._available = True
                logger.info(f"[Reranker] Loaded successfully from {path}")
                return
            except Exception as e:
                logger.debug(f"[Reranker] {path} failed: {e}")
                continue

        logger.warning(f"[Reranker] All paths failed. Reranker disabled. "
                       f"Download with: python download_bge_m3.py --modelscope "
                       f"or: huggingface-cli download BAAI/bge-reranker-v2-m3")

    @property
    def available(self) -> bool:
        return self._available

    def rerank(self, query: str, candidates: List[str],
               top_k: int = 8) -> List[Tuple[str, float]]:
        """对候选列表做 cross-encoder 精排。

        Args:
            query: 问题文本
            candidates: 候选文本列表（如节点名、三元组文本等）
            top_k: 返回 top-K 个

        Returns:
            [(candidate_text, score)] 按 score 降序排列
        """
        if not self._available or not candidates:
            return [(c, 0.0) for c in candidates[:top_k]]

        pairs = [[query, c] for c in candidates]
        try:
            scores = self._model.predict(pairs, show_progress_bar=False)
            scored = list(zip(candidates, [float(s) for s in scores]))
            scored.sort(key=lambda x: -x[1])
            return scored[:top_k]
        except Exception as e:
            logger.warning(f"[Reranker] predict failed: {e}")
            return [(c, 0.0) for c in candidates[:top_k]]

    def rerank_with_keys(self, query: str,
                         candidates: List[Tuple[str, str]],
                         top_k: int = 8) -> List[Tuple[str, str, float]]:
        """对 (key, display_text) 候选做精排，返回 (key, display_text, score)。

        适用于 KG 节点检索场景：key=node_key, display_text=node_name。
        """
        if not self._available or not candidates:
            return [(k, t, 0.0) for k, t in candidates[:top_k]]

        texts = [t for _, t in candidates]
        pairs = [[query, t] for t in texts]
        try:
            scores = self._model.predict(pairs, show_progress_bar=False)
            scored = [(candidates[i][0], candidates[i][1], float(scores[i]))
                      for i in range(len(candidates))]
            scored.sort(key=lambda x: -x[2])
            return scored[:top_k]
        except Exception as e:
            logger.warning(f"[Reranker] predict failed: {e}")
            return [(k, t, 0.0) for k, t in candidates[:top_k]]
