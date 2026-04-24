import logging
import sys
import os

# 将父目录添加到路径，确保可以作为包导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LKG_pipeline_1.core.graph_ops import GraphManager
from LKG_pipeline_1.models.base import Entity, Rule, Concept
from LKG_pipeline_1.models.constraints import OrderingConstraint, AllDifferentConstraint, ArithmeticConstraint

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_folio_scenario(gm: GraphManager):
    """场景 1: FOLIO 数据集 - 一阶逻辑推理"""
    logger.info("\n=== Scenario 1: FOLIO (First-Order Logic) ===")
    
    # 1. 定义实体
    rina = Entity(name="Rina", entity_type="Person")
    coffee = Entity(name="Coffee", entity_type="Object")
    caffeine = Entity(name="Caffeine", entity_type="Substance")
    
    gm.add_entity(rina)
    gm.add_entity(coffee)
    gm.add_entity(caffeine)
    
    # 2. 添加逻辑规则
    # Rule 1: All people who regularly drink coffee are dependent on caffeine.
    # ∀x (Drinks(x) → Dependent(x))
    rule1 = Rule(
        expression="∀x (Drinks(x) → Dependent(x))",
        description="All people who regularly drink coffee are dependent on caffeine"
    )
    # 这条规则隐含地关联了 Coffee 和 Caffeine
    gm.add_rule(rule1, related_entity_ids=[coffee.id, caffeine.id])
    
    # Rule 2: No one who jokes about being addicted to caffeine is unaware that caffeine is a drug.
    # ∀x (Jokes(x) → ¬Unaware(x))
    rule2 = Rule(
        expression="∀x (Jokes(x) → ¬Unaware(x))",
        description="No one who jokes about being addicted to caffeine is unaware that caffeine is a drug"
    )
    gm.add_rule(rule2, related_entity_ids=[caffeine.id])
    
    logger.info("Added FOLIO entities and rules.")

def run_logical_deduction_scenario(gm: GraphManager):
    """场景 2: LogicalDeduction 数据集 - 约束满足问题"""
    logger.info("\n=== Scenario 2: LogicalDeduction (Constraint Satisfaction) ===")
    
    # 1. 定义实体 (书籍)
    books = [
        Entity(name="Blue Book", entity_type="Book"),
        Entity(name="Yellow Book", entity_type="Book"),
        Entity(name="Green Book", entity_type="Book"),
        Entity(name="Red Book", entity_type="Book")
    ]
    
    for book in books:
        gm.add_entity(book)
        
    blue_book = books[0]
    yellow_book = books[1]
    
    # 2. 添加约束
    # Constraint 1: The blue book is to the right of the yellow book.
    const_right = OrderingConstraint(relation="right_of")
    const_right.raw_expression = "Position(BlueBook) > Position(YellowBook)"
    gm.add_constraint_node(const_right, [blue_book.id, yellow_book.id])
    
    # Constraint 2: All books have different positions.
    const_diff = AllDifferentConstraint()
    const_diff.raw_expression = "AllDifferent([b.position for b in books])"
    gm.add_constraint_node(const_diff, [b.id for b in books])
    
    logger.info("Added LogicalDeduction entities and constraints.")

def run_arlsat_scenario(gm: GraphManager):
    """场景 3: AR-LSAT 数据集 - 复杂布局约束"""
    logger.info("\n=== Scenario 3: AR-LSAT (Complex Layout) ===")
    
    # 1. 定义实体 (参议员/代表)
    poirier = Entity(name="Poirier", entity_type="Representative")
    neri = Entity(name="Neri", entity_type="Representative")
    osata = Entity(name="Osata", entity_type="Representative")
    manley = Entity(name="Manley", entity_type="Representative")
    
    gm.add_entity(poirier)
    gm.add_entity(neri)
    gm.add_entity(osata)
    gm.add_entity(manley)
    
    # 2. 添加约束
    # Constraint: Poirier sits immediately next to Neri.
    # 这是一种特殊的 Ordering，距离为 1
    const_next = OrderingConstraint(relation="immediately_next_to")
    const_next.properties["distance"] = 1
    const_next.raw_expression = "|Position(Poirier) - Position(Neri)| == 1"
    gm.add_constraint_node(const_next, [poirier.id, neri.id])
    
    # Complex Rule/Constraint: If Osata sits immediately next to Poirier, Osata does not sit immediately next to Manley.
    # 这是一个条件约束，我们可以将其作为一条复杂的 Rule 存入，或者用复合约束节点（当前架构暂未实现复合节点，用 Rule 代替）
    rule_complex = Rule(
        expression="(Next(Osata, Poirier)) -> ¬(Next(Osata, Manley))",
        description="If Osata sits immediately next to Poirier, Osata does not sit immediately next to Manley"
    )
    gm.add_rule(rule_complex, related_entity_ids=[osata.id, poirier.id, manley.id])
    
    logger.info("Added AR-LSAT entities and constraints.")

def demonstrate_search(gm: GraphManager):
    """演示混合检索能力"""
    logger.info("\n=== Demonstration: Hybrid Search ===")
    
    queries = [
        "dependent on caffeine",  # 测试语义检索 (FOLIO)
        "blue book right",        # 测试实体+约束检索 (LogicalDeduction)
        "Osata sits next to",     # 测试复杂规则检索 (AR-LSAT)
        "Rina"                    # 测试实体检索
    ]
    
    for q in queries:
        print(f"\n>>> Searching for: '{q}'")
        results = gm.hybrid_search(q, top_k=3)
        
        if results["matched_entities"]:
            print(f"  Matched Entities ({len(results['matched_entities'])}):")
            for e in results["matched_entities"]:
                print(f"    - {e['name']} ({e['type']})")
        
        if results["similar_rules"]:
            print(f"  Similar Rules ({len(results['similar_rules'])}):")
            for r in results["similar_rules"]:
                print(f"    - [{r['score']:.4f}] {r['description']}")
                print(f"      Expr: {r['expression']}")

def main():
    gm = GraphManager()
    
    try:
        # 1. 初始化与清理
        logger.info("Initializing Schema and Clearing Database...")
        gm.initialize_schema()
        gm.clear_database()
        
        # 2. 运行场景
        run_folio_scenario(gm)
        run_logical_deduction_scenario(gm)
        run_arlsat_scenario(gm)
        
        # 3. 演示检索
        demonstrate_search(gm)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

