import re
import logging
from typing import List, Dict, Set
from .graph_ops import GraphManager
from LKG_pipeline_1.models.base import Entity, Rule
from LKG_pipeline_1.models.constraints import OrderingConstraint, AllDifferentConstraint, ArithmeticConstraint

logger = logging.getLogger(__name__)

class KGBuilder:
    """自动化知识图谱构建器：从 JSON 记录中提取逻辑并构建图谱"""
    
    def __init__(self, graph_manager: GraphManager):
        self.gm = graph_manager
        # 内存中的实体缓存: name -> entity_id
        self.entity_cache: Dict[str, str] = {}

    def clear_cache(self):
        self.entity_cache.clear()

    def get_or_create_entity(self, name: str, entity_type: str = "Thing") -> str:
        """获取或创建实体，返回 ID"""
        from LKG_pipeline_1.core.text_processor import processor
        
        name = name.strip()
        # Use normalized name for caching to prevent duplication (e.g., "Apple" vs "apple")
        normalized_key = processor.lemmatize(name)
        
        if normalized_key in self.entity_cache:
            return self.entity_cache[normalized_key]
        
        # 创建新实体
        entity = Entity(name=name, entity_type=entity_type)
        self.gm.add_entity(entity)
        self.entity_cache[normalized_key] = entity.id
        return entity.id

    def extract_entities_from_text(self, text: str) -> List[str]:
        """从文本中提取已知实体 ID"""
        from LKG_pipeline_1.core.text_processor import processor
        
        found_ids = []
        # Iterate over normalized keys in cache
        for norm_name, eid in self.entity_cache.items():
            # Check if the normalized name appears in the text
            # This is a heuristic; ideally we'd lemmatize the text first, but that's expensive.
            # Instead, we check if the cache key matches.
            # A better approach for the future: Aho-Corasick or similar.
            
            # Simple check: does the normalized entity name appear?
            if re.search(r'\b' + re.escape(norm_name) + r'\b', text, re.IGNORECASE):
                found_ids.append(eid)
                
            # Fallback: Check if we can find words in text that lemmatize to this key
            # (Skipped for performance in this demo loop, assuming norm_name is sufficient)
            
        return list(set(found_ids))

    def build_from_record(self, record: dict):
        """处理单条数据记录，根据 ID 前缀分发处理逻辑"""
        record_id = record.get('id', '')
        
        # 清除上一条记录的缓存？取决于是否跨记录共享实体。
        # 通常每个逻辑题是独立的，所以建议清除，或者是独立的命名空间。
        # 这里为了简单，我们假设每条记录独立，清空缓存。
        self.clear_cache()
        
        try:
            if 'FOLIO' in record_id:
                self._build_folio(record)
            elif 'logical_deduction' in record_id.lower():
                self._build_logical_deduction(record)
            elif 'ar_lsat' in record_id.lower():
                self._build_ar_lsat(record)
            else:
                logger.warning(f"Unknown dataset type for ID: {record_id}")
        except Exception as e:
            logger.error(f"Error building record {record_id}: {e}")

    def _build_folio(self, record: dict):
        """处理 FOLIO (First-Order Logic) 格式"""
        context = record.get('context', '')
        
        # 1. 提取 Predicates 并注册为 Concept
        pred_section = re.search(r'Predicates:(.*?)(?=Premises:|Question:|$)', context, re.DOTALL)
        if pred_section:
            lines = pred_section.group(1).strip().split('\n')
            for line in lines:
                # 格式: "- Dependent(x): x is a person dependent on caffeine."
                match = re.match(r'-?\s*(\w+)\((.*?)\):\s*(.*)', line.strip())
                if match:
                    pred_name = match.group(1)
                    # 注册为 Concept
                    self.get_or_create_entity(pred_name, "Concept")

        # 2. 提取 Premises (规则)
        premises_section = re.search(r'Premises:(.*?)(?=Question:|$)', context, re.DOTALL)
        if premises_section:
            lines = premises_section.group(1).strip().split('\n')
            for line in lines:
                line = line.strip()
                if not line: continue
                
                parts = line.split(':', 1)
                if len(parts) == 2:
                    raw_expr = re.sub(r'^\d+\.\s*', '', parts[0]).strip()
                    desc = parts[1].strip()
                else:
                    raw_expr = line
                    desc = line
                
                related_ids = []
                
                # 策略 A: 解析原子公式 Predicate(Arg)
                atom_pattern = re.compile(r'(\w+)\s*\(([^)]+)\)')
                atoms = atom_pattern.findall(raw_expr)
                
                for pred, args in atoms:
                    # Predicate 通常是 Concept
                    if pred[0].isupper():
                        self.get_or_create_entity(pred, "Concept")
                    
                    # 处理参数
                    arg_list = [a.strip() for a in args.split(',')]
                    for arg in arg_list:
                        # 忽略单个小写字母变量 (x, y, z)
                        if len(arg) == 1 and arg.islower():
                            continue
                        
                        # 其他视为实体 (即使是小写，如 caffeine)
                        # 注意：如果 arg 已经在 Predicates 定义中出现过（即它是 Concept），则保持为 Concept
                        # 这里简单处理：如果不在缓存中，创建为 Entity
                        if arg not in self.entity_cache:
                            eid = self.get_or_create_entity(arg, "Entity")
                        else:
                            eid = self.entity_cache[arg]
                        related_ids.append(eid)

                # 策略 B: 补充提取 (针对不在 Predicate 结构中的词)
                potential_entities = re.findall(r'\b[A-Z][a-zA-Z0-9_]*\b', raw_expr)
                keywords = {'∀x', '∃x', 'V', '¬', '→', '∧', '∨'}
                
                for word in potential_entities:
                    if word not in keywords and len(word) > 1:
                        # 如果已经在策略 A 中处理过，entity_cache 里会有
                        if word not in self.entity_cache:
                            eid = self.get_or_create_entity(word, "Concept")
                            related_ids.append(eid)
                        else:
                            if self.entity_cache[word] not in related_ids:
                                related_ids.append(self.entity_cache[word])
                
                # 创建 Rule
                rule = Rule(expression=raw_expr, description=desc)
                self.gm.add_rule(rule, related_entity_ids=list(set(related_ids)))

    def _build_logical_deduction(self, record: dict):
        """处理 Logical Deduction (CSP) 格式"""
        context = record.get('context', '')
        
        # 1. 解析变量 (Variables) -> 实体
        # "Variables:\ngreen_book [IN] [1, 2, 3, 4, 5]"
        var_section = re.search(r'Variables:(.*?)(?=Constraints:|$)', context, re.DOTALL)
        if var_section:
            lines = var_section.group(1).strip().split('\n')
            for line in lines:
                match = re.match(r'(\w+)\s*\[IN\]', line.strip())
                if match:
                    entity_name = match.group(1)
                    # 规范化：将下划线替换为空格，以匹配自然语言查询
                    entity_name = entity_name.replace('_', ' ')
                    self.get_or_create_entity(entity_name, "Object")

        # 2. 解析约束 (Constraints)
        # "blue_book == yellow_book + 1 ::: The blue book is to the right of the yellow book."
        const_section = re.search(r'Constraints:(.*?)(?=Query:|$)', context, re.DOTALL)
        if const_section:
            lines = const_section.group(1).strip().split('\n')
            for line in lines:
                line = line.strip()
                if not line: continue
                
                # 分离表达式和描述
                parts = line.split(':::', 1)
                expr = parts[0].strip()
                desc = parts[1].strip() if len(parts) > 1 else ""
                
                # 识别相关实体
                # 规范化表达式中的下划线，以便匹配已清洗的实体名
                expr_cleaned = expr.replace('_', ' ')
                related_ids = self.extract_entities_from_text(expr_cleaned)
                
                # 分类约束类型
                if "AllDifferent" in expr:
                    const = AllDifferentConstraint()
                    const.raw_expression = expr
                    const.properties['description'] = desc
                    self.gm.add_constraint_node(const, related_ids)
                elif ">" in expr or "<" in expr:
                    const = OrderingConstraint(relation="order")
                    const.raw_expression = expr
                    const.properties['description'] = desc
                    self.gm.add_constraint_node(const, related_ids)
                else:
                    # 默认为普通规则或算术约束
                    # 这里也可以用 ArithmeticConstraint
                    const = ArithmeticConstraint()
                    const.raw_expression = expr
                    const.properties['description'] = desc
                    self.gm.add_constraint_node(const, related_ids)

    def _build_ar_lsat(self, record: dict):
        """处理 AR-LSAT (类似 CSP 但更复杂)"""
        # 结构类似 Logical Deduction，可以复用部分逻辑
        self._build_logical_deduction(record)

