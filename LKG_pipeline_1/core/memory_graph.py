"""
内存版 GraphManager，用于摆脱 Neo4j 依赖、支持真并行、降低单题延迟。

- 与 graph_ops.GraphManager 暴露完全相同的方法签名（鸭子类型，可即插即用）
- 每个实例完全独立 → 可在多线程中并行使用，无全局锁竞争
- 检索流程：向量召回（cosine） + BM25 召回 + 1-跳邻居遍历

替代映射：
    Neo4j 节点 (Entity/Rule/Constraint) → 内部 dict 列表
    Neo4j 关系 (a)-[r]->(b)            → adjacency dict（带反向索引）
    向量索引 / 全文索引                 → MemoryVectorStore（NumPy + BM25）
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from LKG_pipeline_1.models.base import Entity, Rule
from LKG_pipeline_1.models.constraints import Constraint

from .memory_vector_store import MemoryVectorStore

logger = logging.getLogger(__name__)


class MemoryGraphManager:
    """与 GraphManager 完全兼容的内存图管理器。"""

    # ── 跨节点公用的元关系名（从图遍历邻居中排除） ──
    META_RELATIONS = {"MENTIONS", "APPLIES_TO"}

    def __init__(self):
        self.vector_store = MemoryVectorStore()

        # 节点存储
        self.entities: Dict[str, Dict[str, Any]] = {}      # id -> props
        self.rules: Dict[str, Dict[str, Any]] = {}
        self.constraints: Dict[str, Dict[str, Any]] = {}

        # id -> "Entity" / "Rule" / "Constraint"  （向量库 label 映射）
        self.node_kind: Dict[str, str] = {}

        # 关系存储：邻接表 (out / in)，每条关系: {source, target, type, props}
        self.out_edges: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.in_edges: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # ════════════════════════════════════════════════════════
    #  与 GraphManager 完全对应的 API
    # ════════════════════════════════════════════════════════

    def initialize_schema(self):
        """no-op：内存版无需建索引"""
        pass

    def clear_database(self):
        self.entities.clear()
        self.rules.clear()
        self.constraints.clear()
        self.node_kind.clear()
        self.out_edges.clear()
        self.in_edges.clear()
        self.vector_store.clear()

    # ─── 添加实体 ───
    def add_entity(self, entity: Entity):
        if not entity.properties.get("embedding"):
            norm_name = entity.properties.get("normalized_name") or entity.name
            text = f"{norm_name} ({entity.entity_type})"
            try:
                entity.properties["embedding"] = self.vector_store.encode(text)
            except Exception as e:
                logger.warning(f"Failed to encode entity {entity.name}: {e}")

        # 节点属性（与 Neo4j 版字段保持一致）
        node_props = {
            "id": entity.id,
            "name": entity.name,
            "entity_type": entity.entity_type,
        }
        # 合并原始 properties（除 embedding 外）
        for k, v in (entity.properties or {}).items():
            if k != "embedding":
                node_props[k] = v

        self.entities[entity.id] = node_props
        self.node_kind[entity.id] = "Entity"

        emb = entity.properties.get("embedding")
        # BM25 全文检索字段：name + entity_type
        fulltext = f"{entity.name} {entity.entity_type}"
        self.vector_store.add_node("Entity", entity.id, emb, fulltext, node_props)

    # ─── 添加关系 ───
    def add_relationship(self, source_id: str, target_id: str,
                         relation_type: str, properties: Optional[dict] = None):
        properties = properties or {}
        safe_type = "".join(c for c in relation_type if c.isalnum() or c == "_").upper()
        if not safe_type:
            safe_type = "RELATED_TO"
        if source_id not in self.entities and source_id not in self.rules and source_id not in self.constraints:
            return  # 未注册节点
        if target_id not in self.entities and target_id not in self.rules and target_id not in self.constraints:
            return
        edge = {
            "source": source_id,
            "target": target_id,
            "type": safe_type,
            "props": dict(properties),
        }
        self.out_edges[source_id].append(edge)
        self.in_edges[target_id].append(edge)

    # ─── 添加规则 ───
    def add_rule(self, rule: Rule, related_entity_ids: Optional[List[str]] = None):
        text_to_embed = rule.description or rule.expression
        if not rule.embedding and text_to_embed:
            try:
                rule.embedding = self.vector_store.encode(text_to_embed)
            except Exception as e:
                logger.warning(f"Failed to encode rule: {e}")

        node_props = {
            "id": rule.id,
            "expression": rule.expression,
            "description": rule.description,
        }
        self.rules[rule.id] = node_props
        self.node_kind[rule.id] = "Rule"

        fulltext = f"{rule.expression} {rule.description}"
        self.vector_store.add_node("Rule", rule.id, rule.embedding, fulltext, node_props)

        if related_entity_ids:
            for eid in related_entity_ids:
                if eid:
                    # 与 Neo4j 版一致：(Rule)-[:MENTIONS]->(Entity)
                    self.add_relationship(rule.id, eid, "MENTIONS")

    # ─── 添加约束 ───
    def add_constraint_node(self, constraint: Constraint, related_entity_ids: List[str]):
        if not constraint.properties.get("embedding"):
            text = (constraint.raw_expression
                    or constraint.properties.get("expression")
                    or constraint.properties.get("description") or "")
            if text:
                try:
                    constraint.properties["embedding"] = self.vector_store.encode(text)
                except Exception as e:
                    logger.warning(f"Failed to encode constraint: {e}")

        node_props = {
            "id": constraint.id,
            "constraint_type": constraint.constraint_type,
            "raw_expression": constraint.raw_expression,
        }
        for k, v in (constraint.properties or {}).items():
            if k != "embedding":
                node_props[k] = v

        self.constraints[constraint.id] = node_props
        self.node_kind[constraint.id] = "Constraint"

        emb = constraint.properties.get("embedding")
        fulltext = f"{constraint.raw_expression or ''} {constraint.properties.get('raw_context', '')}"
        self.vector_store.add_node("Constraint", constraint.id, emb, fulltext, node_props)

        if related_entity_ids:
            for eid in related_entity_ids:
                if eid:
                    self.add_relationship(constraint.id, eid, "APPLIES_TO")

    # ════════════════════════════════════════════════════════
    #  hybrid_search（与 graph_ops.hybrid_search 行为对齐）
    # ════════════════════════════════════════════════════════

    def hybrid_search(self, query_text: str, top_k: int = 5,
                      entity_threshold: float = 0.4, text_threshold: float = 0.4,
                      use_graph_traversal: bool = True, pruning_callback=None,
                      verbose: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        if verbose:
            print(f"\n[MemHybridSearch] Query: '{query_text}' "
                  f"(thresholds: ent={entity_threshold}, text={text_threshold})")

        results = {"matched_entities": [], "similar_rules": [], "related_constraints": []}
        all_candidates: List[Dict[str, Any]] = []

        # ── 1. Rule 向量召回 ──
        for hit in self.vector_store.search_similar("Rule", query_text, top_k + 3):
            score = hit["score"]
            if score < text_threshold:
                continue
            node = hit["node"]
            all_candidates.append({
                "id": node["id"], "type": "Rule",
                "content": f"Expression: {node.get('expression')}, Description: {node.get('description')}",
                "score": score, "original_node": node,
            })

        # ── 2. Constraint 向量召回 ──
        for hit in self.vector_store.search_similar("Constraint", query_text, top_k + 3):
            score = hit["score"]
            if score < text_threshold:
                continue
            node = hit["node"]
            all_candidates.append({
                "id": node["id"], "type": "Constraint",
                "content": f"Type: {node.get('constraint_type')}, Expression: {node.get('raw_expression')}",
                "score": score, "original_node": node,
            })

        # ── 3a. Entity 向量召回 ──
        kept_entity_ids = set()  # 仅记录被实际加入候选的，不含被阈值过滤的
        for hit in self.vector_store.search_similar("Entity", query_text, top_k + 3):
            score = hit["score"]
            node = hit["node"]
            if score < entity_threshold:
                continue
            kept_entity_ids.add(node["id"])
            all_candidates.append({
                "id": node["id"], "type": "Entity",
                "content": f"Name: {node.get('name')}, Type: {node.get('entity_type')}",
                "score": score, "original_node": node,
            })

        # ── 3b. Entity BM25 召回（精确关键词；可补回向量未召回的实体） ──
        for hit in self.vector_store.fulltext_search("Entity", query_text, top_k):
            node = hit["node"]
            if node["id"] in kept_entity_ids:
                continue
            kept_entity_ids.add(node["id"])
            all_candidates.append({
                "id": node["id"], "type": "Entity",
                "content": f"Name: {node.get('name')}, Type: {node.get('entity_type')}",
                "score": 0.5, "original_node": node,
            })

        # ── 3.5 LLM 剪枝（可选） ──
        start_node_ids: Set[str] = set()
        if pruning_callback and all_candidates:
            try:
                kept = set(pruning_callback(query_text, all_candidates))
                kept_cands = [c for c in all_candidates if c["id"] in kept]
            except Exception as e:
                logger.warning(f"Pruning callback failed: {e}; keeping all.")
                kept_cands = all_candidates
        else:
            kept_cands = all_candidates

        for cand in kept_cands:
            start_node_ids.add(cand["id"])
            node = cand["original_node"]
            if cand["type"] == "Rule":
                results["similar_rules"].append({
                    "id": cand["id"], "expression": node.get("expression"),
                    "description": node.get("description"),
                    "score": cand["score"], "source": "vector",
                })
            elif cand["type"] == "Constraint":
                if not any(c["id"] == cand["id"] for c in results["related_constraints"]):
                    results["related_constraints"].append({
                        "id": cand["id"], "type": node.get("constraint_type"),
                        "expression": node.get("raw_expression"),
                        "score": cand["score"], "source": "vector",
                    })
            elif cand["type"] == "Entity":
                entry = {
                    "id": cand["id"], "name": node.get("name"),
                    "type": node.get("entity_type"),
                    "score": cand["score"], "source": "vector",
                }
                if node.get("source_context"):
                    entry["context"] = node["source_context"]
                results["matched_entities"].append(entry)

        # ── 4. 图遍历扩展（1-跳邻居） ──
        if use_graph_traversal and start_node_ids:
            seen_rule = {r["id"] for r in results["similar_rules"]}
            seen_const = {c["id"] for c in results["related_constraints"]}
            seen_entity = {e["id"] for e in results["matched_entities"]}

            for nid in list(start_node_ids):
                neighbors = self.out_edges.get(nid, []) + self.in_edges.get(nid, [])
                for edge in neighbors:
                    other_id = edge["target"] if edge["source"] == nid else edge["source"]
                    kind = self.node_kind.get(other_id)
                    if kind == "Rule" and other_id not in seen_rule:
                        seen_rule.add(other_id)
                        n = self.rules[other_id]
                        results["similar_rules"].append({
                            "id": other_id, "expression": n.get("expression"),
                            "description": n.get("description"), "source": "graph_traversal",
                        })
                    elif kind == "Constraint" and other_id not in seen_const:
                        seen_const.add(other_id)
                        n = self.constraints[other_id]
                        results["related_constraints"].append({
                            "id": other_id, "type": n.get("constraint_type"),
                            "expression": n.get("raw_expression"), "source": "graph_traversal",
                        })
                    elif kind == "Entity" and other_id not in seen_entity:
                        seen_entity.add(other_id)
                        n = self.entities[other_id]
                        entry = {
                            "id": other_id, "name": n.get("name"),
                            "type": n.get("entity_type"), "source": "graph_traversal",
                        }
                        if n.get("source_context"):
                            entry["context"] = n["source_context"]
                        results["matched_entities"].append(entry)

        return results

    # ════════════════════════════════════════════════════════
    #  其他与 GraphManager 兼容的接口
    # ════════════════════════════════════════════════════════

    def dump_full_graph(self) -> Dict[str, List[Dict[str, Any]]]:
        result = {"entities": [], "relationships": [], "rules": [], "constraints": []}
        for n in self.entities.values():
            entry = {
                "name": n.get("name", ""), "type": n.get("entity_type", "Entity"),
                "id": n.get("id", ""),
            }
            if n.get("source_context"):
                entry["context"] = n["source_context"]
            result["entities"].append(entry)
        for src, edges in self.out_edges.items():
            for edge in edges:
                if edge["type"] in self.META_RELATIONS:
                    continue
                src_name = self._lookup_name(edge["source"])
                tgt_name = self._lookup_name(edge["target"])
                props = edge.get("props") or {}
                entry = {
                    "source": src_name, "relation": edge["type"], "target": tgt_name,
                    "desc": props.get("desc", ""),
                }
                if props.get("source_context"):
                    entry["context"] = props["source_context"]
                result["relationships"].append(entry)
        for n in self.rules.values():
            result["rules"].append({
                "expression": n.get("expression", ""), "description": n.get("description", ""),
            })
        for n in self.constraints.values():
            result["constraints"].append({
                "type": n.get("constraint_type", ""),
                "expression": n.get("raw_expression", ""),
                "raw_text": n.get("raw_context", "") or n.get("raw_text", ""),
                "description": n.get("description", ""),
            })
        return result

    def get_relationships_between(self, entity_ids: List[str]) -> List[Dict[str, Any]]:
        if not entity_ids:
            return []
        id_set = set(entity_ids)
        out = []
        for src in entity_ids:
            for edge in self.out_edges.get(src, []):
                if edge["type"] in self.META_RELATIONS:
                    continue
                if edge["target"] not in id_set:
                    continue
                props = edge.get("props") or {}
                entry = {
                    "source": self._lookup_name(edge["source"]),
                    "relation": edge["type"],
                    "target": self._lookup_name(edge["target"]),
                    "desc": props.get("desc", ""),
                }
                if props.get("source_context"):
                    entry["context"] = props["source_context"]
                out.append(entry)
        return out

    # ─── 内部辅助 ───
    def _lookup_name(self, node_id: str) -> str:
        if node_id in self.entities:
            return self.entities[node_id].get("name", node_id)
        if node_id in self.rules:
            return (self.rules[node_id].get("expression", "") or node_id)[:30]
        if node_id in self.constraints:
            return (self.constraints[node_id].get("raw_expression", "") or node_id)[:30]
        return node_id
