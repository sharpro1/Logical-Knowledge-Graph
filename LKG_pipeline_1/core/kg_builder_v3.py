"""
多跳 QA 知识图谱构建与检索 v3

核心改进（vs v2）：
  1. 从 ontology JSON 加载 schema，注入 LLM 抽取 prompt
  2. 存储为真正的图结构（节点 dict + 邻接表），不是事实列表
  3. 检索用"向量找种子 → 图 BFS 遍历 1-2 跳"，不是向量近似多跳

架构：
  KnowledgeGraph  — 图数据结构（节点 + 边 + 邻接表 + 向量索引）
  KGBuilder       — 从文本分段抽取并构图
  KGRetriever     — 向量种子 + BFS 子图检索
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ONTOLOGY_PATH = os.path.join(os.path.dirname(__file__), "..", "schemas", "multihop_qa_ontology.json")
CHUNK_SIZE = 4


# ═══════════════════════════════════════════════════════════
#  图数据结构
# ═══════════════════════════════════════════════════════════

class KnowledgeGraph:
    """真正的图：节点 + 有向边 + 邻接表 + 向量索引。"""

    def __init__(self):
        # 节点：name_lower → {name, type, evidence, paragraph, embedding}
        self.nodes: Dict[str, Dict[str, Any]] = {}
        # 边列表：[{source, target, predicate, evidence, paragraph}]
        self.edges: List[Dict[str, str]] = []
        # 邻接表（双向）：name_lower → [(edge_index, neighbor_name_lower)]
        self.adj: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        # 节点向量矩阵（lazy build）
        self._node_names: List[str] = []
        self._node_matrix: Optional[np.ndarray] = None

    def add_node(self, name: str, entity_type: str = "",
                 evidence: str = "", paragraph: str = ""):
        key = name.strip().lower()
        if not key:
            return
        if key in self.nodes:
            # 合并：追加 evidence
            existing = self.nodes[key]
            if evidence and evidence not in existing.get("evidence", ""):
                existing["evidence"] = (existing.get("evidence", "") + " | " + evidence)[:600]
            return
        self.nodes[key] = {
            "name": name.strip(),
            "type": entity_type,
            "evidence": evidence[:300],
            "paragraph": paragraph,
        }

    def add_edge(self, source: str, predicate: str, target: str,
                 evidence: str = "", paragraph: str = ""):
        s_key = source.strip().lower()
        t_key = target.strip().lower()
        if not s_key or not t_key or s_key == t_key:
            return
        # 确保端点节点存在
        if s_key not in self.nodes:
            self.add_node(source, evidence=evidence, paragraph=paragraph)
        if t_key not in self.nodes:
            self.add_node(target, evidence=evidence, paragraph=paragraph)
        idx = len(self.edges)
        self.edges.append({
            "source": s_key, "target": t_key,
            "predicate": predicate,
            "evidence": evidence[:300],
            "paragraph": paragraph,
        })
        self.adj[s_key].append((idx, t_key))
        self.adj[t_key].append((idx, s_key))

    def build_vectors(self, model):
        """用嵌入模型为所有节点生成向量，构建检索矩阵。"""
        self._node_names = list(self.nodes.keys())
        if not self._node_names:
            self._node_matrix = np.zeros((0, 1), dtype=np.float32)
            return
        texts = [self.nodes[k]["name"] for k in self._node_names]
        vecs = np.array([model.encode(t, show_progress_bar=False) for t in texts],
                        dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._node_matrix = vecs / norms

    def vector_search_nodes(self, query_emb: np.ndarray, top_k: int = 10,
                            threshold: float = 0.35) -> List[Tuple[str, float]]:
        """向量检索最相关的节点，返回 [(name_lower, score)]。"""
        if self._node_matrix is None or self._node_matrix.shape[0] == 0:
            return []
        scores = self._node_matrix @ query_emb
        top_idx = np.argsort(-scores)[:top_k]
        return [(self._node_names[i], float(scores[i]))
                for i in top_idx if scores[i] > threshold]

    def bfs_subgraph(self, seed_nodes: Set[str], max_hops: int = 2) -> Dict[str, Any]:
        """从种子节点出发做 BFS，收集 1-2 跳内的所有节点和边。

        Returns:
            {"nodes": [node_dict, ...], "edges": [edge_dict, ...]}
        """
        visited_nodes: Set[str] = set()
        visited_edges: Set[int] = set()
        frontier = set(seed_nodes)

        for hop in range(max_hops):
            next_frontier = set()
            for node_key in frontier:
                if node_key not in self.adj:
                    continue
                visited_nodes.add(node_key)
                for edge_idx, neighbor in self.adj[node_key]:
                    if edge_idx not in visited_edges:
                        visited_edges.add(edge_idx)
                    if neighbor not in visited_nodes:
                        next_frontier.add(neighbor)
            frontier = next_frontier
        visited_nodes.update(frontier)

        result_nodes = [self.nodes[k] for k in visited_nodes if k in self.nodes]
        result_edges = [self.edges[i] for i in sorted(visited_edges)]
        return {"nodes": result_nodes, "edges": result_edges}

    @property
    def stats(self) -> Dict[str, int]:
        return {"nodes": len(self.nodes), "edges": len(self.edges)}


# ═══════════════════════════════════════════════════════════
#  Ontology 驱动的抽取 prompt
# ═══════════════════════════════════════════════════════════

def _load_ontology() -> Dict[str, Any]:
    if os.path.exists(ONTOLOGY_PATH):
        with open(ONTOLOGY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _build_extraction_prompt(ontology: Dict[str, Any]) -> str:
    """从 ontology JSON 动态构建抽取 prompt。"""
    # 实体类型列表
    ent_types = ontology.get("entity_types", {})
    type_lines = []
    for t, info in ent_types.items():
        examples = ", ".join(info.get("examples", []))
        type_lines.append(f"    - {t}: {info.get('description', '')} (e.g. {examples})")
    type_block = "\n".join(type_lines)

    # 关系类型列表
    rel_types = ontology.get("relationship_types", {})
    rel_lines = []
    for category, rels in rel_types.items():
        rel_lines.append(f"    {category}: {', '.join(rels)}")
    rel_block = "\n".join(rel_lines)

    # 抽取规则
    rules = ontology.get("extraction_rules", [])
    rules_block = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(rules))

    return """Extract a knowledge graph from the paragraphs below.

ENTITY TYPES (use ONLY these):
{types}

RELATIONSHIP TYPES (use these or similar specific verbs):
{rels}

EXTRACTION RULES:
{rules}

OUTPUT FORMAT (strict JSON):
{{
  "entities": [
    {{"name": "<specific name>", "type": "<type from list above>", "evidence": "<exact source sentence>", "paragraph": "<paragraph title>"}}
  ],
  "relationships": [
    {{"source": "<entity name>", "predicate": "<specific verb>", "target": "<entity name>", "evidence": "<exact source sentence>", "paragraph": "<paragraph title>"}}
  ]
}}

CRITICAL:
- "name" must be a SPECIFIC proper noun, date, or number — NOT a generic word.
- "predicate" must be a SPECIFIC verb — NOT "is a" or "related to".
- "evidence" must be the EXACT sentence from the text.
- "paragraph" is the [Title] at the beginning of each paragraph.
- Extract facts from EVERY paragraph. Each should yield 2-5 facts.

PARAGRAPHS:
{{text}}""".format(types=type_block, rels=rel_block, rules=rules_block)


# ═══════════════════════════════════════════════════════════
#  KG Builder（分段抽取 + 构图）
# ═══════════════════════════════════════════════════════════

class KGBuilder:
    """从文本构建 KnowledgeGraph：分段抽取 + 实体合并。"""

    def __init__(self, provider: str = "deepseek", api_key: str = None,
                 model_name: str = None, base_url: str = None):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.model = model_name or "deepseek-chat"
        except ImportError:
            self.client = None
            self.model = None

        ontology = _load_ontology()
        self._prompt_template = _build_extraction_prompt(ontology)
        logger.info(f"KGBuilder: loaded ontology with "
                    f"{len(ontology.get('entity_types', {}))} entity types, "
                    f"{sum(len(v) for v in ontology.get('relationship_types', {}).values())} relation types")

    def build(self, context_text: str, chunk_size: int = CHUNK_SIZE) -> KnowledgeGraph:
        """从全文构建知识图谱。"""
        kg = KnowledgeGraph()
        if not self.client:
            return kg

        paragraphs = [p.strip() for p in context_text.split("\n\n") if p.strip()]
        for i in range(0, len(paragraphs), chunk_size):
            chunk = "\n\n".join(paragraphs[i:i + chunk_size])
            self._extract_chunk_into(chunk, kg)

        logger.info(f"KGBuilder: {len(paragraphs)} paragraphs → "
                    f"{kg.stats['nodes']} nodes, {kg.stats['edges']} edges")
        return kg

    @staticmethod
    def _repair_truncated_json(text: str) -> Optional[Dict]:
        """修复被 max_tokens 截断的 JSON。

        策略：找到最后一个完整的 JSON 对象/数组元素，截断后面的残片，然后闭合括号。
        """
        if not text or not isinstance(text, str):
            return None
        text = text.strip()
        if not text.startswith("{"):
            start = text.find("{")
            if start < 0:
                return None
            text = text[start:]

        # 多轮尝试修复
        for _ in range(8):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
            # 如果在字符串中间被截断，砍掉最后一个不完整的元素
            # 找最后一个 }, ] 或完整的 "..." 作为安全截断点
            last_safe = max(text.rfind("},"), text.rfind("}]"), text.rfind('"}'))
            if last_safe > 0:
                text = text[:last_safe + 1]
            # 闭合
            if text.count('"') % 2 == 1:
                text += '"'
            open_sq = text.count("[") - text.count("]")
            open_br = text.count("{") - text.count("}")
            text += "]" * max(0, open_sq) + "}" * max(0, open_br)
        return None

    def _extract_chunk_into(self, text: str, kg: KnowledgeGraph):
        prompt = self._prompt_template.replace("{text}", text)
        try:
            # 尝试带 response_format（OpenAI/DeepSeek 兼容端点支持）
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Extract knowledge graph as JSON. Output valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=16384,
                    response_format={"type": "json_object"}
                )
            except Exception:
                # Fallback: 不带 response_format（Llama/Claude 等模型）
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Extract knowledge graph as JSON. Output valid JSON only. Do NOT output anything except the JSON object."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=16384,
                )
            content = response.choices[0].message.content or ""
            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)
            if not content.startswith("{"):
                start = content.find("{")
                if start >= 0:
                    content = content[start:]
            data = json.loads(content)
        except json.JSONDecodeError as je:
            data = self._repair_truncated_json(content if 'content' in dir() else "")
            if data is None:
                logger.warning(f"KGBuilder chunk extraction failed (JSON parse): {je}")
                return
        except Exception as e:
            logger.warning(f"KGBuilder chunk extraction failed: {e}")
            return

        # 添加实体节点
        for ent in data.get("entities", []):
            if not isinstance(ent, dict) or not ent.get("name"):
                continue
            kg.add_node(
                name=str(ent["name"]),
                entity_type=str(ent.get("type", "")),
                evidence=str(ent.get("evidence", "")),
                paragraph=str(ent.get("paragraph", "")),
            )

        # 添加关系边
        for rel in data.get("relationships", []):
            if not isinstance(rel, dict):
                continue
            src = str(rel.get("source", "")).strip()
            tgt = str(rel.get("target", "")).strip()
            pred = str(rel.get("predicate", "")).strip()
            if not src or not tgt or not pred:
                continue
            pred_lower = pred.lower()
            if pred_lower in ("is a", "is an", "type of", "instance of", "kind of"):
                continue
            kg.add_edge(
                source=src, predicate=pred, target=tgt,
                evidence=str(rel.get("evidence", "")),
                paragraph=str(rel.get("paragraph", "")),
            )


# ═══════════════════════════════════════════════════════════
#  KG Retriever（向量种子 + 图 BFS）
# ═══════════════════════════════════════════════════════════

class KGRetriever:
    """从 KnowledgeGraph 中检索与问题相关的子图。

    流程：
      1. 用嵌入模型编码 question
      2. 向量搜索 KG 中最相关的种子节点（top-K）
      3. 从种子节点出发 BFS 1-2 跳，收集子图
      4. 格式化子图为 LLM 可读文本
    """

    def __init__(self, embedding_model=None, use_rerank: bool = False):
        if embedding_model is not None:
            self._model = embedding_model
        else:
            from .memory_vector_store import _get_embedding_model
            self._model = _get_embedding_model()

        self._reranker = None
        if use_rerank:
            try:
                from .reranker import get_reranker
                self._reranker = get_reranker()
                if not self._reranker.available:
                    self._reranker = None
                    logger.warning("[KGRetriever] Reranker not available, falling back to bi-encoder only.")
            except Exception as e:
                logger.warning(f"[KGRetriever] Failed to load reranker: {e}")

    def retrieve(self, kg: KnowledgeGraph, question: str,
                 seed_top_k: int = 8, seed_threshold: float = 0.35,
                 max_hops: int = 2) -> Dict[str, Any]:
        """检索子图。

        Returns:
            {"subgraph": {"nodes": [...], "edges": [...]},
             "seed_nodes": [...],
             "kg_text": "formatted for LLM",
             "rerank_used": bool,
             "stats": {"total_nodes": N, "total_edges": M, "sub_nodes": n, "sub_edges": m}}
        """
        # Step 1: 建向量索引
        kg.build_vectors(self._model)

        # Step 2: 向量检索候选节点
        if self._reranker:
            # 先编码 question，再召回更多候选（top-30），然后用 cross-encoder 精排到 top-K
            q_emb = np.array(self._model.encode(question, show_progress_bar=False), dtype=np.float32)
            qn = np.linalg.norm(q_emb)
            if qn > 0:
                q_emb = q_emb / qn
            recall_k = max(seed_top_k * 4, 30)
            candidates = kg.vector_search_nodes(q_emb, top_k=recall_k, threshold=0.15)

            if candidates:
                cand_pairs = [(key, kg.nodes[key]["name"]) for key, _ in candidates if key in kg.nodes]
                reranked = self._reranker.rerank_with_keys(question, cand_pairs, top_k=seed_top_k)
                seeds = [(key, score) for key, _, score in reranked if score > -10]
            else:
                seeds = []
        else:
            q_emb = np.array(self._model.encode(question, show_progress_bar=False), dtype=np.float32)
            qn = np.linalg.norm(q_emb)
            if qn > 0:
                q_emb = q_emb / qn
            seeds = kg.vector_search_nodes(q_emb, top_k=seed_top_k, threshold=seed_threshold)

        seed_keys = {s[0] for s in seeds}

        # Step 3: BFS 遍历 1-2 跳
        subgraph = kg.bfs_subgraph(seed_keys, max_hops=max_hops)

        # Step 4: 格式化
        kg_text = self._format(subgraph, seeds)

        return {
            "subgraph": subgraph,
            "seed_nodes": [(kg.nodes[k]["name"], score) for k, score in seeds if k in kg.nodes],
            "kg_text": kg_text,
            "rerank_used": self._reranker is not None,
            "stats": {
                "total_nodes": kg.stats["nodes"],
                "total_edges": kg.stats["edges"],
                "sub_nodes": len(subgraph["nodes"]),
                "sub_edges": len(subgraph["edges"]),
            },
        }

    @staticmethod
    def _format(subgraph: Dict[str, Any], seeds: List[Tuple[str, float]]) -> str:
        nodes = subgraph["nodes"]
        edges = subgraph["edges"]
        if not nodes and not edges:
            return ""

        lines = ["[Knowledge Subgraph]", ""]

        if edges:
            lines.append("Facts (entity → relationship → entity):")
            for e in edges:
                src_name = e.get("source", "")
                tgt_name = e.get("target", "")
                # 还原为原始大小写名
                lines.append(f"  • {src_name} —[{e['predicate']}]→ {tgt_name}")
                if e.get("evidence"):
                    lines.append(f"    Evidence: \"{e['evidence'][:200]}\"")
            lines.append("")

        # 列出有 evidence 但不在边上的孤立实体
        edge_entities = set()
        for e in edges:
            edge_entities.add(e["source"])
            edge_entities.add(e["target"])
        lonely = [n for n in nodes if n["name"].lower() not in edge_entities and n.get("evidence")]
        if lonely:
            lines.append("Additional entity evidence:")
            for n in lonely[:10]:
                lines.append(f"  • {n['name']} ({n.get('type','')}): \"{n['evidence'][:150]}\"")
            lines.append("")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
#  对外统一接口
# ═══════════════════════════════════════════════════════════

def build_and_retrieve(question: str, context_text: str,
                       provider: str = "deepseek", api_key: str = None,
                       model_name: str = None, base_url: str = None,
                       embedding_model=None, use_rerank: bool = False,
                       seed_top_k: int = 8, seed_threshold: float = 0.35,
                       max_hops: int = 2, chunk_size: int = CHUNK_SIZE
                       ) -> Dict[str, Any]:
    """一站式接口：构建 KG → 检索子图 → 返回格式化文本。"""
    builder = KGBuilder(provider=provider, api_key=api_key,
                        model_name=model_name, base_url=base_url)
    kg = builder.build(context_text, chunk_size=chunk_size)

    retriever = KGRetriever(embedding_model=embedding_model, use_rerank=use_rerank)
    result = retriever.retrieve(kg, question,
                                seed_top_k=seed_top_k,
                                seed_threshold=seed_threshold,
                                max_hops=max_hops)
    result["kg"] = kg
    return result
