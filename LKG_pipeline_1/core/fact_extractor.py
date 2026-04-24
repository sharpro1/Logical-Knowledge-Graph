"""
多跳事实 QA 专用的知识图谱抽取器

与 LLMBuilder 不同点：
  - Schema 极简：只抽 (subject, predicate, object, evidence) 四元组
  - 分段抽取：每 3-4 段调一次 LLM，避免 JSON 截断
  - 跨段落实体合并：同名实体自动合并，建立多跳链路
  - 每个事实保留原文证据（evidence），供 LLM 直接引用

设计原则：
  "不如不抽就不抽，抽出来的必须有用"
  - 不抽 IS_A / SUBCLASS_OF 等本体关系（对 QA 无用）
  - 不抽概念节点（Concept）
  - 只抽实体间的事实性关系
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 每次给 LLM 的段落数（3-4 段 ≈ 600-1000 字，输出 JSON 通常 < 2000 tokens，不会截断）
CHUNK_SIZE = 4

EXTRACTION_PROMPT = """Extract factual knowledge from the paragraphs below as a JSON object.

SCHEMA (strict):
{{
  "facts": [
    {{
      "subject": "<entity name>",
      "predicate": "<relationship verb/phrase>",
      "object": "<entity name or value>",
      "evidence": "<the EXACT sentence from the paragraph that states this fact>"
    }}
  ]
}}

RULES:
1. Extract ONLY concrete facts stated in the text. Do NOT infer or generalize.
2. "subject" and "object" must be specific named entities, dates, numbers, or places — NOT generic types like "Person" or "Country".
3. "predicate" should be a specific verb phrase: "directed", "born in", "capital of", "married to", "founded in", "borders", "member of", etc. Do NOT use "is a" or "related to".
4. "evidence" must be the EXACT sentence (or clause) from the input — do NOT paraphrase.
5. Extract ALL facts, even from seemingly unimportant paragraphs. Each paragraph should yield at least 1-2 facts.
6. For biographical paragraphs, extract: birth/death dates, nationality, occupation, notable works, family relations.
7. For geographical paragraphs, extract: location, borders, capital, population, area.
8. Keep the JSON compact. Do NOT add explanations outside the JSON.

PARAGRAPHS:
{text}
"""


class FactExtractor:
    """多跳 QA 专用的分段式事实抽取器。

    Usage:
        extractor = FactExtractor(provider="deepseek", api_key="...", ...)
        facts = extractor.extract_facts(context_text)
        # facts = [{"subject": ..., "predicate": ..., "object": ..., "evidence": ...}, ...]
    """

    def __init__(self, provider: str = "deepseek", api_key: str = None,
                 model_name: str = None, base_url: str = None):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.model = model_name or "deepseek-chat"
        except ImportError:
            logger.error("OpenAI SDK not found.")
            self.client = None
            self.model = None

    def extract_facts(self, context_text: str, chunk_size: int = CHUNK_SIZE) -> List[Dict[str, str]]:
        """从完整上下文中分段抽取事实三元组。

        Returns:
            list of {"subject", "predicate", "object", "evidence"}
        """
        if not self.client:
            return []

        paragraphs = [p.strip() for p in context_text.split("\n\n") if p.strip()]
        all_facts: List[Dict[str, str]] = []

        # 分块抽取
        for i in range(0, len(paragraphs), chunk_size):
            chunk = paragraphs[i:i + chunk_size]
            chunk_text = "\n\n".join(chunk)
            facts = self._extract_chunk(chunk_text)
            all_facts.extend(facts)

        logger.info(f"FactExtractor: {len(paragraphs)} paragraphs → {len(all_facts)} facts "
                    f"({len(paragraphs) // chunk_size + 1} API calls)")
        return all_facts

    def _extract_chunk(self, text: str) -> List[Dict[str, str]]:
        """对单个 chunk 调 LLM 抽取事实。"""
        prompt = EXTRACTION_PROMPT.format(text=text)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a fact extraction expert. Output valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=4096,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            facts = data.get("facts", [])
            return [f for f in facts if isinstance(f, dict) and f.get("subject") and f.get("object")]
        except json.JSONDecodeError:
            # 尝试修复截断的 JSON
            try:
                content = content.strip()
                if not content.endswith("}"):
                    content = content + '"}]}'
                data = json.loads(content)
                return data.get("facts", [])
            except Exception:
                pass
            logger.warning(f"FactExtractor: JSON parse failed for chunk ({len(text)} chars)")
            return []
        except Exception as e:
            logger.warning(f"FactExtractor: API call failed: {e}")
            return []


def build_fact_graph(facts: List[Dict[str, str]]) -> Dict[str, Any]:
    """从事实列表构建图结构。

    Returns:
        {
            "entities": {name: {"mentions": [evidence1, ...], "predicates": set()}},
            "triples": [(subj, pred, obj, evidence), ...],
            "entity_list": [name1, name2, ...],
        }
    """
    entities: Dict[str, Dict[str, Any]] = {}
    triples = []

    def _norm(name: str) -> str:
        return name.strip().lower()

    def _register(name: str, evidence: str = ""):
        norm = _norm(name)
        if norm not in entities:
            entities[norm] = {"name": name, "mentions": [], "predicates": set()}
        if evidence:
            entities[norm]["mentions"].append(evidence[:200])

    for fact in facts:
        subj = fact.get("subject", "").strip()
        pred = fact.get("predicate", "").strip()
        obj = fact.get("object", "").strip()
        evidence = fact.get("evidence", "").strip()
        if not subj or not obj:
            continue
        # 过滤 IS_A / 类型关系（对 QA 无用）
        pred_lower = pred.lower()
        if pred_lower in ("is a", "is an", "type of", "instance of", "subclass of", "kind of"):
            continue
        _register(subj, evidence)
        _register(obj, evidence)
        triples.append((subj, pred, obj, evidence))

    return {
        "entities": entities,
        "triples": triples,
        "entity_list": sorted(set(e["name"] for e in entities.values())),
    }


def format_kg_for_llm(graph: Dict[str, Any], question: str,
                      max_triples: int = 30) -> str:
    """将事实图谱格式化为 LLM 可读的文本。

    格式设计原则：
    - 每个三元组自带证据原文 → LLM 可以直接引用
    - 按与问题的相关度排序（简单关键词匹配）
    - 实体出现次数多的排前面（更可能是桥接实体）
    """
    triples = graph["triples"]
    if not triples:
        return ""

    # 按与问题的关键词重叠度排序
    q_words = set(re.findall(r'\w+', question.lower()))
    def relevance(triple):
        s, p, o, ev = triple
        words = set(re.findall(r'\w+', f"{s} {p} {o}".lower()))
        return len(words & q_words)

    sorted_triples = sorted(triples, key=relevance, reverse=True)[:max_triples]

    lines = ["[Knowledge Graph — extracted facts with source evidence]", ""]
    for s, p, o, ev in sorted_triples:
        lines.append(f"  FACT: {s} → {p} → {o}")
        if ev:
            lines.append(f"  EVIDENCE: \"{ev}\"")
        lines.append("")

    # 列出所有实体（供 LLM 做 entity linking）
    lines.append(f"[All entities mentioned] {', '.join(graph['entity_list'][:50])}")
    return "\n".join(lines)
