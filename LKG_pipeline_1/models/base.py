from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import uuid
import hashlib
from LKG_pipeline_1.core.text_processor import processor

@dataclass
class LKGNode:
    """基础图节点"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    labels: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "id": self.id,
            "labels": self.labels,
            "properties": self.properties
        }

@dataclass
class Entity(LKGNode):
    """实体节点"""
    name: str = ""
    entity_type: str = "Thing"
    
    def __post_init__(self):
        # 使用 TextProcessor 生成规范化的 ID String，结合 name 和 type
        # 这实现了论文中提到的 "Entities are normalized via lemmatization"
        if self.name:
            # unique_str = processor.get_canonical_id_str(self.name, self.entity_type)
            # 为了保持现有行为并逐步迁移，我们可以在 properties 里存储 normalized_name
            # 但为了 ID 的一致性，ID 生成应该基于规范化后的名称
            
            # 这里的 hash 策略：hash(normalized_type + ":" + normalized_name)
            # 例如: hash("person:run") 用于 "Running" (type: Person)
            canonical_str = processor.get_canonical_id_str(self.name, self.entity_type)
            self.id = hashlib.md5(canonical_str.encode('utf-8')).hexdigest()
            
        if "Entity" not in self.labels:
            self.labels.append("Entity")
        if self.entity_type and self.entity_type not in self.labels:
            self.labels.append(self.entity_type)
        self.properties["name"] = self.name
        self.properties["entity_type"] = self.entity_type
        # Store normalized name in properties for visibility in Neo4j
        self.properties["normalized_name"] = processor.lemmatize(self.name)

@dataclass
class Rule(LKGNode):
    """逻辑规则节点"""
    expression: str = ""
    description: str = ""  # 用于向量化
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        # 如果表达式存在，使用表达式的 Hash 作为 ID，实现去重
        if self.expression:
            self.id = hashlib.md5(self.expression.encode('utf-8')).hexdigest()
            
        if "Rule" not in self.labels:
            self.labels.append("Rule")
        self.properties["expression"] = self.expression
        self.properties["description"] = self.description
        # Embedding 不直接存入 properties (因为是向量)，由 specialized 方法处理

@dataclass
class Concept(LKGNode):
    """概念/类节点 (Ontology Layer)"""
    name: str = ""
    
    def __post_init__(self):
        if "Concept" not in self.labels:
            self.labels.append("Concept")
        self.properties["name"] = self.name

