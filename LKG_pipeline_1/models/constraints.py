from dataclasses import dataclass, field
from typing import List, Any
from .base import LKGNode

@dataclass
class Constraint(LKGNode):
    """基础约束节点"""
    constraint_type: str = "GenericConstraint"
    raw_expression: str = ""
    raw_context: str = ""  # New field for natural language source text
    
    def __post_init__(self):
        if "Constraint" not in self.labels:
            self.labels.append("Constraint")
        self.properties["constraint_type"] = self.constraint_type
        self.properties["raw_expression"] = self.raw_expression
        self.properties["raw_context"] = self.raw_context

@dataclass
class OrderingConstraint(Constraint):
    """顺序约束 (e.g., A left_of B)"""
    relation: str = "left_of"  # left_of, right_of, newer_than, etc.
    
    def __post_init__(self):
        super().__post_init__()
        self.labels.append("OrderingConstraint")
        self.properties["relation"] = self.relation
        # 更新类型属性
        self.constraint_type = "OrderingConstraint"
        self.properties["constraint_type"] = "OrderingConstraint"

@dataclass
class AllDifferentConstraint(Constraint):
    """全不同约束"""
    def __post_init__(self):
        super().__post_init__()
        self.labels.append("AllDifferentConstraint")
        self.properties["constraint_type"] = "AllDifferent"

@dataclass
class ArithmeticConstraint(Constraint):
    """算术约束 (e.g., A = B + 1)"""
    operation: str = "add"
    value: float = 0.0
    
    def __post_init__(self):
        super().__post_init__()
        self.labels.append("ArithmeticConstraint")
        self.properties["operation"] = self.operation
        self.properties["value"] = self.value
        self.constraint_type = "ArithmeticConstraint"
        self.properties["constraint_type"] = "ArithmeticConstraint"

