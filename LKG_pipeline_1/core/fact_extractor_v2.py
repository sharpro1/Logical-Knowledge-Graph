"""
多跳事实 QA 知识图谱抽取 v2

v1 的问题：
  1. 抽了 127 条事实，大部分来自干扰段
  2. 没有检索步骤，简单关键词排序找不到桥接事实
  3. 所有事实平等对待，关键信息被淹没

v2 改进：
  1. 抽取时每段标注来源段落标题（paragraph_title）
  2. 用 BGE-M3 向量模型做两轮检索：
     Round 1: question → 最相关的事实（直接匹配）
     Round 2: Round 1 的 object → 更多事实（追踪多跳链路）
  3. 只返回检索到的子图（~15-25 条），不是全部 127 条

架构：
  extract_and_retrieve(question, context)
    → Step 1: 分段抽取全量事实（每 4 段一批）
    → Step 2: 向量编码全部事实
    → Step 3: 两轮检索（question → hop1 facts → hop2 facts）
    → 返回紧凑子图
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

CHUNK_SIZE = 4

ONTOLOGY_PATH = os.path.join(os.path.dirname(__file__), "..", "schemas", "multihop_qa_ontology.json")

EXTRACT_PROMPT = """You are a knowledge graph extraction expert. Extract structured facts from the paragraphs.

OUTPUT FORMAT (strict JSON):
{{
  "facts": [
    {{
      "subject": "<specific entity name>",
      "subject_type": "<Person|Organization|Location|CreativeWork|Event|DateTime|Quantity>",
      "predicate": "<specific verb phrase>",
      "object": "<specific entity name or value>",
      "object_type": "<Person|Organization|Location|CreativeWork|Event|DateTime|Quantity>",
      "evidence": "<EXACT sentence from the paragraph>",
      "paragraph": "<title of the paragraph this fact comes from>"
    }}
  ]
}}

RULES:
1. "subject" and "object" MUST be specific named entities, NOT generic types.
   GOOD: "Steve Hillage", "Orion Pictures", "Sydney Harbour"
   BAD: "a person", "the company", "it"

2. "predicate" MUST be a specific relationship verb:
   GOOD: "directed", "born_in", "capital_of", "spouse_of", "released_in", "flows_into"
   BAD: "is a", "related to", "associated with"

3. "evidence" MUST be the EXACT sentence from the input. Do NOT paraphrase.

4. "paragraph" is the [Title] at the start of each paragraph.

5. Extract ALL factual relationships from EVERY paragraph. Each paragraph should yield 2-5 facts.

6. Focus on facts that NAME specific entities, dates, locations, or quantities.

PARAGRAPHS:
{text}
"""


class FactExtractorV2:
    """两阶段事实抽取器：分段抽取 + 向量检索。"""

    def __init__(self, provider: str = "deepseek", api_key: str = None,
                 model_name: str = None, base_url: str = None,
                 embedding_model=None):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.model = model_name or "deepseek-chat"
        except ImportError:
            logger.error("OpenAI SDK not found.")
            self.client = None
            self.model = None

        # 嵌入模型（用于检索阶段）
        if embedding_model is not None:
            self._emb_model = embedding_model
        else:
            from .memory_vector_store import _get_embedding_model
            self._emb_model = _get_embedding_model()

    def extract_and_retrieve(self, question: str, context_text: str,
                             max_hops: int = 2, top_k_per_hop: int = 10,
                             chunk_size: int = CHUNK_SIZE) -> Dict[str, Any]:
        """完整流程：抽取 → 向量检索 → 返回紧凑子图。

        Returns:
            {"facts": [...], "all_facts_count": N, "retrieved_count": M,
             "entity_list": [...], "kg_text": "formatted text for LLM"}
        """
        # ── Phase 1: 分段抽取全量事实 ──
        all_facts = self._extract_all(context_text, chunk_size)
        if not all_facts:
            return {"facts": [], "all_facts_count": 0, "retrieved_count": 0,
                    "entity_list": [], "kg_text": ""}

        # ── Phase 2: 向量编码 ──
        fact_texts = [f"{f['subject']} {f['predicate']} {f['object']}" for f in all_facts]
        fact_embs = np.array(
            [self._emb_model.encode(t, show_progress_bar=False) for t in fact_texts],
            dtype=np.float32
        )
        norms = np.linalg.norm(fact_embs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        fact_embs = fact_embs / norms

        # ── Phase 3: 多轮检索 ──
        retrieved_idx = set()

        # Round 1: question → top-K 最相关事实（直接匹配第 1 跳）
        q_emb = np.array(self._emb_model.encode(question, show_progress_bar=False), dtype=np.float32)
        qn = np.linalg.norm(q_emb)
        if qn > 0:
            q_emb = q_emb / qn
        scores_r1 = fact_embs @ q_emb
        top_r1 = np.argsort(-scores_r1)[:top_k_per_hop]
        for idx in top_r1:
            if scores_r1[idx] > 0.35:
                retrieved_idx.add(int(idx))

        # Round 2+: 用已检索事实的每个 subject/object 分别查询（不平均，逐个追踪）
        for hop in range(1, max_hops):
            hop_entities = set()
            for idx in list(retrieved_idx):
                hop_entities.add(all_facts[idx]["object"])
                hop_entities.add(all_facts[idx]["subject"])
            if not hop_entities:
                break
            new_hits = set()
            for entity_name in hop_entities:
                e_emb = np.array(self._emb_model.encode(entity_name, show_progress_bar=False),
                                 dtype=np.float32)
                en = np.linalg.norm(e_emb)
                if en > 0:
                    e_emb = e_emb / en
                scores_hop = fact_embs @ e_emb
                # 每个实体取 top-3 最相关（而不是全部 top-K）
                top_hop = np.argsort(-scores_hop)[:3]
                for idx in top_hop:
                    if scores_hop[idx] > 0.45 and int(idx) not in retrieved_idx:
                        new_hits.add(int(idx))
            retrieved_idx.update(new_hits)

        # ── Phase 4: 组装子图 ──
        retrieved_facts = [all_facts[i] for i in sorted(retrieved_idx)]
        entity_set = set()
        for f in retrieved_facts:
            entity_set.add(f["subject"])
            entity_set.add(f["object"])

        kg_text = self._format_subgraph(retrieved_facts, question, list(entity_set))

        return {
            "facts": retrieved_facts,
            "all_facts_count": len(all_facts),
            "retrieved_count": len(retrieved_facts),
            "entity_list": sorted(entity_set),
            "kg_text": kg_text,
        }

    def _extract_all(self, context_text: str, chunk_size: int) -> List[Dict[str, str]]:
        """分段抽取全量事实。"""
        if not self.client:
            return []
        paragraphs = [p.strip() for p in context_text.split("\n\n") if p.strip()]
        all_facts = []
        for i in range(0, len(paragraphs), chunk_size):
            chunk = paragraphs[i:i + chunk_size]
            facts = self._extract_chunk("\n\n".join(chunk))
            all_facts.extend(facts)
        logger.info(f"FactExtractorV2: {len(paragraphs)} paragraphs → {len(all_facts)} facts")
        return all_facts

    def _extract_chunk(self, text: str) -> List[Dict[str, str]]:
        prompt = EXTRACT_PROMPT.format(text=text)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Extract facts as JSON. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=4096,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            facts = data.get("facts", [])
            # 验证 + 过滤
            valid = []
            for f in facts:
                if not isinstance(f, dict):
                    continue
                if not f.get("subject") or not f.get("object"):
                    continue
                pred = (f.get("predicate") or "").lower()
                if pred in ("is a", "is an", "type of", "instance of", "kind of"):
                    continue
                valid.append({
                    "subject": str(f["subject"]).strip(),
                    "subject_type": str(f.get("subject_type", "")).strip(),
                    "predicate": str(f.get("predicate", "related_to")).strip(),
                    "object": str(f["object"]).strip(),
                    "object_type": str(f.get("object_type", "")).strip(),
                    "evidence": str(f.get("evidence", "")).strip(),
                    "paragraph": str(f.get("paragraph", "")).strip(),
                })
            return valid
        except Exception as e:
            logger.warning(f"FactExtractorV2 chunk failed: {e}")
            return []

    @staticmethod
    def _format_subgraph(facts: List[Dict], question: str, entities: List[str]) -> str:
        if not facts:
            return ""
        lines = ["[Retrieved Knowledge Subgraph]", ""]
        lines.append(f"Entities found: {', '.join(entities[:30])}")
        lines.append(f"Facts relevant to the question ({len(facts)} retrieved):")
        lines.append("")
        for f in facts:
            src_tag = f" [from: {f['paragraph']}]" if f.get("paragraph") else ""
            lines.append(f"  • {f['subject']} —[{f['predicate']}]→ {f['object']}{src_tag}")
            if f.get("evidence"):
                lines.append(f"    Evidence: \"{f['evidence'][:200]}\"")
        return "\n".join(lines)
