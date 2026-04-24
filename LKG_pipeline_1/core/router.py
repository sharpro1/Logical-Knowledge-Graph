import logging
import json
import re
from enum import Enum
from typing import Dict, List, Any
import os
from openai import OpenAI

# 配置 Logger
logger = logging.getLogger(__name__)

class SolverType(Enum):
    Z3 = "Z3"           # 用于数值计算、排序约束、逻辑谜题 (CSP/SMT)
    PROVER9 = "PROVER9" # 用于纯一阶逻辑证明、反证法、蕴涵判断 (Full FOL ATP)
    PYKE = "PYKE"       # 用于基于知识的推理引擎、前向/后向链、专家系统 (Rule Engine)
    PYTHON = "PYTHON"   # 用于简单的图遍历、事实查询 (Graph Traversal)

class LogicRouter:
    def __init__(self, provider: str = "openai", api_key: str = None, base_url: str = None, model_name: str = None):
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        
        # 初始化 LLM 客户端
        if provider == "openai" or provider == "deepseek" or provider == "ollama":
            self.client = OpenAI(api_key=api_key or "", base_url=base_url)
            if not self.model_name:
                if provider == "openai":
                    self.model_name = "gpt-3.5-turbo"
                elif provider == "deepseek":
                    self.model_name = "deepseek-chat"
                else:  # ollama
                    self.model_name = "deepseek-r1:32b"
        elif provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model_name = model_name or "gemini-pro"
            self.client = genai.GenerativeModel(self.model_name)
        else:
            self.client = None
            logger.warning(f"Unsupported provider: {provider}, Router will default to PYTHON")

    def _call_llm(self, prompt: str) -> str:
        """统一的 LLM 调用接口"""
        if not self.client:
            return "PYKE"
            
        try:
            if self.provider in ["openai", "deepseek", "ollama"]:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a logic reasoning expert. You output ONLY the solver name."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )
                return response.choices[0].message.content.strip()
            
            elif self.provider == "gemini":
                response = self.client.generate_content(prompt)
                return response.text.strip()
                
        except Exception as e:
            logger.error(f"LLM call failed in Router: {e}")
            return "PYTHON" # Fallback

    def route(self, query: str, search_results: Dict[str, Any]) -> str:
        """
        根据 Query 和 检索结果 决定使用哪个求解器。
        """
        
        # --- 1. 分析检索结果 (Schema-Aware Analysis) ---
        rules = search_results.get("similar_rules", [])
        constraints = search_results.get("related_constraints", [])
        entities = search_results.get("matched_entities", [])
        
        # 提取约束类型
        constraint_types = [c.get('type', 'GenericConstraint') for c in constraints]
        
        rule_count = len(rules)
        entity_count = len(entities)
        
        logger.info(f"[Router] Context Analysis: {len(constraints)} Constraints ({constraint_types}), {rule_count} Rules, {entity_count} Entities.")

        # --- 2. 构造 Prompt ---
        has_ordering = any("Ordering" in t for t in constraint_types)
        has_alldiff = any("AllDifferent" in t for t in constraint_types)
        has_arithmetic = any("Arithmetic" in t for t in constraint_types)
        has_constraints = len(constraints) > 0
        
        # 从 query 中提取问题模式特征
        q_lower = query.lower()
        has_true_false_uncertain = "true, false, or uncertain" in q_lower or "true, false or uncertain" in q_lower
        has_true_or_false = ("true or false" in q_lower or "true? false?" in q_lower) and not has_true_false_uncertain
        has_multi_option = any(kw in q_lower for kw in ["which one", "which of the following", "could be", "must be", "cannot be", "except"])
        has_options_abcde = bool(re.search(r'\([A-E]\)', query))

        prompt = f"""Analyze the problem to determine the best Solver Engine.

### Solver Definitions:
1. **Z3** (SMT Solver): Best for:
   - Constraint Satisfaction Problems (CSP): scheduling, seating, ordering, assignments.
   - Arithmetic/numeric constraints (greater than, less than, equals).
   - AllDifferent / unique assignment constraints.
   - Multi-option questions with options (A)-(E) asking "which COULD/MUST be true".

2. **PROVER9** (First-Order Logic Theorem Prover): Best for:
   - Pure logical entailment: "Does premise set imply conclusion?"
   - Questions asking "is the following statement true, false, or uncertain?"
   - Real-world knowledge with universal/existential quantifiers and implications.
   - NOT suitable for: scheduling, ordering, or multi-option (A)-(E) problems.

3. **PYKE** (Rule Engine): Best for:
   - Simple chain reasoning: "Every X is a Y. Y is Z. Is X a Z?"
   - Questions with made-up/fantasy creature names (Wumpus, Dumpus, Jompus, etc.)
   - True/False queries about property inheritance through a chain of rules.
   - NOT suitable for: ordering, scheduling, numeric constraints, or multi-option problems.

### Problem Analysis:
User Query: "{query}"
Constraints Found: {json.dumps(constraint_types)}
Rules Found: {rule_count}
Entities Found: {entity_count}
Has Ordering Constraints: {has_ordering}
Has AllDifferent Constraints: {has_alldiff}
Has Arithmetic Constraints: {has_arithmetic}

### Key Pattern Signals:
- Multi-option question (A)-(E): {has_multi_option or has_options_abcde}
- "true, false, or uncertain" pattern: {has_true_false_uncertain}
- "true or false" pattern: {has_true_or_false}

### Decision Rules (apply in order):
1. If multi-option (A)-(E) or Ordering/AllDifferent/Arithmetic constraints → Z3
2. If "true, false, or uncertain" with real-world entities → PROVER9
3. If "true or false" with simple chain rules or fantasy entity names → PYKE
4. If only logical implications, no numeric constraints → PROVER9

Return ONLY one word: Z3, PROVER9, or PYKE.
"""
        
        # --- 3. 获取决策 ---
        decision = self._call_llm(prompt)
        
        # 清洗输出
        decision_clean = decision.upper().replace(".", "").strip()
        
        # 解析决策
        if "Z3" in decision_clean:
            return SolverType.Z3.value
        elif "PROVER" in decision_clean:
            return SolverType.PROVER9.value
        elif "PYKE" in decision_clean:
            return SolverType.PYKE.value
        else:
            return SolverType.PYKE.value
