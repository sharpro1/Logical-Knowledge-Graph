import json
import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import os
from LKG_pipeline_1.models.base import Entity, Rule, Concept
from LKG_pipeline_1.models.constraints import OrderingConstraint, AllDifferentConstraint

if TYPE_CHECKING:
    from .graph_ops import GraphManager  # 仅用于类型提示，避免运行时强依赖 neo4j

logger = logging.getLogger(__name__)

class LLMBackend:
    """LLM 后端抽象基类"""
    def generate_json(self, prompt: str) -> Dict[str, Any]:
        raise NotImplementedError

class OpenAICompatibleBackend(LLMBackend):
    """支持 OpenAI 和 DeepSeek 等兼容 API"""
    def __init__(self, api_key: str, model_name: str, base_url: Optional[str] = None):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.model_name = model_name
        except ImportError:
            logger.error("OpenAI SDK not found. Please install `openai`.")
            self.client = None

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        if not self.client:
            return {}

        try:
            # 尝试带 response_format（DeepSeek/OpenAI 支持）
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a Knowledge Graph extraction expert. Output strictly valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=8192,
                    response_format={"type": "json_object"}
                )
            except Exception:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a Knowledge Graph extraction expert. Output strictly valid JSON only. No other text."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=8192,
                )
            content = response.choices[0].message.content or ""
            content = content.strip()
            if content.startswith("```"):
                import re as _re
                content = _re.sub(r"^```(?:json)?\s*", "", content)
                content = _re.sub(r"\s*```$", "", content)
            if not content.startswith("{"):
                start = content.find("{")
                if start >= 0:
                    content = content[start:]
            return json.loads(content)
        except json.JSONDecodeError as je:
            repaired = self._repair_truncated_json(content if 'content' in dir() else "")
            if repaired:
                logger.warning(f"JSON was truncated, repaired successfully ({len(repaired)} keys)")
                return repaired
            logger.error(f"OpenAI/DeepSeek JSON parse failed (unrepairable): {je}")
            return {}
        except Exception as e:
            logger.error(f"OpenAI/DeepSeek generation failed: {e}")
            return {}

    @staticmethod
    def _repair_truncated_json(text: str) -> Optional[Dict]:
        """尝试修复被截断的 JSON（关闭未闭合的字符串/数组/对象）。"""
        if not text or not text.strip():
            return None
        text = text.strip()
        if not text.startswith("{"):
            start = text.find("{")
            if start == -1:
                return None
            text = text[start:]

        for attempt in range(5):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
            open_braces = text.count("{") - text.count("}")
            open_brackets = text.count("[") - text.count("]")
            in_string = text.count('"') % 2 == 1
            if in_string:
                text += '"'
            text += "]" * max(0, open_brackets)
            text += "}" * max(0, open_braces)
        return None

class GeminiBackend(LLMBackend):
    """支持 Google Gemini API"""
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
        except ImportError:
            logger.error("Google Generative AI SDK not found. Please install `google-generativeai`.")
            self.model = None

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        if not self.model:
            return {}
        
        try:
            full_prompt = prompt + "\n\nPlease output the result in strict JSON format."
            response = self.model.generate_content(
                full_prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return {}

class LLMBuilder:
    """基于多模型后端的知识图谱构建器"""
    
    def __init__(self, graph_manager: "GraphManager", provider: str = "openai", 
                 api_key: str = None, model_name: str = None, base_url: str = None):
        self.gm = graph_manager
        self.provider = provider.lower()
        api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            logger.warning(f"No API Key provided for {provider}.")
        
        if self.provider == "openai":
            self.backend = OpenAICompatibleBackend(
                api_key=api_key, 
                model_name=model_name or "gpt-4o",
                base_url=base_url
            )
        elif self.provider == "deepseek":
            self.backend = OpenAICompatibleBackend(
                api_key=api_key,
                model_name=model_name or "deepseek-chat",
                base_url=base_url or "https://api.deepseek.com"
            )
        elif self.provider == "gemini":
            self.backend = GeminiBackend(
                api_key=api_key,
                model_name=model_name or "gemini-1.5-flash"
            )
        elif self.provider == "ollama":
            # 使用OpenAICompatibleBackend连接本地Ollama
            self.backend = OpenAICompatibleBackend(
                api_key="",  # Ollama不需要API密钥
                model_name=model_name or "deepseek-r1:32b",
                base_url=base_url or "http://localhost:11434/api"
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def build_from_text(self, text: str, verbose: bool = False):
        """通用构建入口"""
        prompt = self._construct_prompt(text)
        data = self.backend.generate_json(prompt)
        
        if verbose:
            print("\n" + "="*20 + " LLM Raw Output " + "="*20)
            raw_json_str = json.dumps(data, indent=2, ensure_ascii=False)
            print(raw_json_str)
            print("="*56 + "\n")
            
            # Log raw output to logger as well for persistence
            # logger.info(f"LLM Extraction Result:\n{raw_json_str}")
            
        if data:
            self._ingest_data(data)
        else:
            logger.warning("No data extracted from LLM.")

    def _construct_prompt(self, text: str) -> str:
        return f"""
        Task: Extract a structured Logic Knowledge Graph from the text based on the strict Ontology Rules below.
        
        Strict Ontology Rules:
        1. **Nodes Classification**:
           - **Entity**: Specific instances/individuals. 
             * IMPORTANT: For the 'type' field, infer the specific semantic class (e.g., 'Person', 'Book', 'Location'), DO NOT just put 'Entity'.
             * **CRITICAL: Include a "context" field** — the exact sentence(s) from the input where this entity is mentioned. This preserves the source evidence.
           - **Concept**: Abstract classes/types (e.g., 'Student', 'Course', 'Exam').
             * The 'type' field for concepts should be 'Concept'.
        
        2. **Relationships Restrictions**:
           - Entity -> Concept: MUST use relation "IS_A".
           - Concept -> Concept: Abstract relationships (e.g., SUBCLASS_OF).
           - Entity -> Entity: **EXTRACT SPECIFIC RELATIONSHIPS** from text.
             * Use specific verbs (FATHER_OF, LEFT_OF, DIRECTED, BORN_IN, etc.), avoid generic "RELATED_TO".
             * **Include a "context" field** — the sentence from the input that states this relationship.
           - **FORBIDDEN**: Concept -> Entity.
        
        3. **Logic & Constraints (STRICT TYPES)**:
           - Extract 'Rules' (implications/logic formulas) and 'Constraints' (restrictions).
           - **Structure for Rules/Constraints**:
             * "expression": The formal logic or mathematical formula.
             * "raw_text": The exact natural language sentence that defines this rule/constraint.
             * "description": A short summary.
           - **Constraint Type MUST be one of**: "Ordering", "AllDifferent", "Arithmetic", "Generic".
        
        4. **Linking Strategy (CRITICAL)**:
           - For every Rule and Constraint, list **ALL** related Entity/Concept names in 'related_entities' or 'entities' field.
        
        Example Input:
        "Peter is the father of John. John is the father of Mike. If X is the father of Y and Y is the father of Z, then X is the grandfather of Z."
        
        Example Output JSON:
        {{
            "entities": [
                {{"name": "Peter", "type": "Person", "context": "Peter is the father of John."}},
                {{"name": "John", "type": "Person", "context": "Peter is the father of John. John is the father of Mike."}},
                {{"name": "Mike", "type": "Person", "context": "John is the father of Mike."}}
            ],
            "concepts": [
                {{"name": "Father", "type": "Concept"}},
                {{"name": "Grandfather", "type": "Concept"}}
            ],
            "relationships": [
                {{"source": "Peter", "target": "John", "relation": "FATHER_OF", "context": "Peter is the father of John."}},
                {{"source": "John", "target": "Mike", "relation": "FATHER_OF", "context": "John is the father of Mike."}},
                {{"source": "Peter", "target": "Father", "relation": "IS_A"}},
                {{"source": "John", "target": "Father", "relation": "IS_A"}},
                {{"source": "Father", "target": "Grandfather", "relation": "SUBCLASS_OF"}}
            ],
            "rules": [
                {{
                    "expression": "father(X, Y) AND father(Y, Z) -> grandfather(X, Z)",
                    "raw_text": "If X is the father of Y and Y is the father of Z, then X is the grandfather of Z.",
                    "description": "The father of a father is a grandfather.",
                    "related_entities": ["Father", "Grandfather"]
                }}
            ],
            "constraints": []
        }}

        IMPORTANT REMINDERS:
        - Every entity MUST have a "context" field with the source sentence(s) from the input.
        - Every entity-to-entity relationship SHOULD have a "context" field.
        - The "context" is the EXACT text from the input — do NOT paraphrase.
        
        Now process this text:
        {text}
        """

    def _ingest_data(self, data: Dict[str, Any]):
        """将提取的数据存入图谱。

        改进点：
          - 维护 id_map（精确名）+ norm_map（小写去标点的名 → id）双索引
          - 关系/规则中找不到的实体名，先模糊匹配，再退化为自动补建
          - 自动补建的实体类型默认 'Entity'，会随后被检索流程使用
        """
        id_map: Dict[str, str] = {}        # 精确名 -> id
        norm_map: Dict[str, str] = {}      # 归一化名 -> id

        def _norm(name: str) -> str:
            """归一化：小写、去除标点和多余空白。"""
            if not name:
                return ""
            import re as _re
            s = str(name).lower()
            s = _re.sub(r"[^\w\s\u4e00-\u9fff]", " ", s)
            s = _re.sub(r"\s+", " ", s).strip()
            return s

        def _register(name: str, ent_id: str):
            id_map[name] = ent_id
            n = _norm(name)
            if n:
                norm_map.setdefault(n, ent_id)

        def _resolve_or_create(name: str, default_type: str = "Entity") -> Optional[str]:
            """按精确名 → 归一化名 → 子串匹配 解析；都失败则自动补建一个实体。"""
            if not name or not isinstance(name, str):
                return None
            # 1. 精确匹配
            if name in id_map:
                return id_map[name]
            # 2. 归一化匹配
            n = _norm(name)
            if n in norm_map:
                return norm_map[n]
            # 3. 双向子串匹配（短名匹长名 / 长名匹短名）
            for known_n, kid in norm_map.items():
                if not known_n:
                    continue
                if (n in known_n and len(n) >= 3) or (known_n in n and len(known_n) >= 3):
                    return kid
            # 4. 自动补建
            try:
                new_e = Entity(name=name, entity_type=default_type)
                self.gm.add_entity(new_e)
                _register(name, new_e.id)
                logger.debug(f"Auto-created missing entity '{name}' as {default_type}")
                return new_e.id
            except Exception as e:
                logger.warning(f"Failed to auto-create entity '{name}': {e}")
                return None

        # 1. Ingest Concepts
        for item in data.get("concepts", []):
            if isinstance(item, str):
                entity = Entity(name=item, entity_type="Concept")
                self.gm.add_entity(entity)
                _register(item, entity.id)
                continue
            if not isinstance(item, dict):
                logger.warning(f"Skipping invalid concept: {item}")
                continue
            entity = Entity(name=item["name"], entity_type="Concept")
            self.gm.add_entity(entity)
            _register(item["name"], entity.id)

        # 2. Ingest Entities (now with source_context)
        for item in data.get("entities", []):
            if isinstance(item, str):
                entity = Entity(name=item, entity_type="Entity")
                self.gm.add_entity(entity)
                _register(item, entity.id)
                continue
            if not isinstance(item, dict):
                logger.warning(f"Skipping invalid entity: {item}")
                continue
            etype = item.get("type", "Entity")
            entity = Entity(name=item["name"], entity_type=etype)
            ctx = item.get("context", "")
            if ctx:
                entity.properties["source_context"] = str(ctx)[:500]
            self.gm.add_entity(entity)
            _register(item["name"], entity.id)

        # 3. Ingest Relationships  (使用模糊匹配 + 自动补建)
        rel_kept = rel_fuzzy = rel_created = 0
        for rel in data.get("relationships", []):
            if rel is None or isinstance(rel, str):
                if rel is not None:
                    logger.debug(f"Skipping string format relationship: {rel}")
                continue
            if not isinstance(rel, dict):
                logger.debug(f"Skipping invalid relationship: {rel}")
                continue

            src_name = rel.get("source")
            tgt_name = rel.get("target")
            if not src_name or not tgt_name:
                continue

            src_was_new = src_name not in id_map
            tgt_was_new = tgt_name not in id_map
            src_id = _resolve_or_create(src_name)
            tgt_id = _resolve_or_create(tgt_name)

            if src_id and tgt_id:
                rel_props = {"desc": rel.get("desc", "")}
                rel_ctx = rel.get("context", "")
                if rel_ctx:
                    rel_props["source_context"] = str(rel_ctx)[:500]
                self.gm.add_relationship(
                    src_id, tgt_id,
                    rel.get("relation", "RELATED_TO"),
                    properties=rel_props,
                )
                if src_was_new or tgt_was_new:
                    rel_fuzzy += 1
                else:
                    rel_kept += 1
                if src_was_new and src_name in id_map:
                    rel_created += 1
                if tgt_was_new and tgt_name in id_map:
                    rel_created += 1
            else:
                logger.warning(f"Skipping relationship {src_name}->{tgt_name}: nodes not found.")

        if rel_fuzzy or rel_created:
            logger.info(
                f"Relationships: kept={rel_kept}, recovered_via_fuzzy_or_create={rel_fuzzy}, "
                f"auto_created_entities={rel_created}"
            )
            
        # 4. Rules
        for item in data.get("rules", []):
            if item is None:
                continue
            if isinstance(item, str):
                expression = item
                description = item
                related_names = []
            elif not isinstance(item, dict):
                logger.warning(f"Skipping invalid rule: {item}")
                continue
            else:
                expression = item.get("expression", "")
                description = item.get("description", "")
                related_names = item.get("related_entities", [])
                
            rule = Rule(
                expression=expression,
                description=description
            )
            related_ids = [eid for eid in
                           (_resolve_or_create(name) for name in (related_names or []))
                           if eid]
            self.gm.add_rule(rule, related_entity_ids=related_ids)
            
        # 5. Constraints
        for item in data.get("constraints", []):
            if item is None:
                continue
            if isinstance(item, str):
                ctype = "Generic"
                raw_expr = item
                raw_text = item
                related_names = []
            elif not isinstance(item, dict):
                logger.warning(f"Skipping invalid constraint: {item}")
                continue
            else:
                ctype = item.get("type", "Generic")
                # 尝试多种可能的键名获取表达式
                raw_expr = item.get("expression") or item.get("expr") or item.get("formula") or ""
                raw_text = item.get("raw_text", "") # Extract raw natural language text
                related_names = item.get("entities", [])
                
            related_ids = [eid for eid in
                           (_resolve_or_create(name) for name in (related_names or []))
                           if eid]

            if ctype == "Ordering":
                const = OrderingConstraint()
                const.labels.append("OrderingConstraint")
                const.constraint_type = "OrderingConstraint"
            elif ctype == "AllDifferent":
                const = AllDifferentConstraint()
                const.labels.append("AllDifferentConstraint")
                const.constraint_type = "AllDifferentConstraint"
            elif ctype == "Arithmetic": # 增加 Arithmetic 支持
                from LKG_pipeline_1.models.constraints import ArithmeticConstraint
                const = ArithmeticConstraint()
                const.labels.append("ArithmeticConstraint")
                const.constraint_type = "ArithmeticConstraint"
            else:
                # 即使是未知类型，也应该保留 LLM 返回的原始类型名，而不是 Generic
                from LKG_pipeline_1.models.constraints import Constraint
                const = Constraint()
                const.constraint_type = ctype # 保留原始类型
            
            const.raw_expression = raw_expr
            const.raw_context = raw_text
            # 确保属性字典里也有，以防万一
            const.properties['raw_expression'] = raw_expr
            const.properties['raw_context'] = raw_text
            
            self.gm.add_constraint_node(const, related_ids)
            
        logger.info(f"Ingested {len(data.get('entities', []))} entities, {len(data.get('concepts', []))} concepts.")
