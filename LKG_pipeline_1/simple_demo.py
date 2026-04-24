import logging
import sys
import os

# 将父目录添加到路径，确保可以作为包导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LKG_pipeline_1.core.graph_ops import GraphManager
from LKG_pipeline_1.models.base import Entity, Rule
from LKG_pipeline_1.models.constraints import OrderingConstraint, AllDifferentConstraint

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting LKG Pipeline Demo...")
    
    gm = GraphManager()
    
    try:
        # 1. 初始化 Schema
        logger.info("Initializing Schema...")
        gm.initialize_schema()
        
        # 2. 清理旧数据 (可选)
        gm.clear_database()
        
        # 3. 模拟逻辑题场景: "Logic Deduction"
        # 题目: "The blue book is to the right of the yellow book. The blue book is 4."
        
        logger.info("Adding Entities...")
        blue_book = Entity(name="Blue Book", entity_type="Book")
        yellow_book = Entity(name="Yellow Book", entity_type="Book")
        
        gm.add_entity(blue_book)
        gm.add_entity(yellow_book)
        
        # 4. 添加规则 (支持向量化)
        logger.info("Adding Rules...")
        rule1 = Rule(
            expression="Position(BlueBook) = Position(YellowBook) + 1",
            description="The blue book is to the right of the yellow book"
        )
        gm.add_rule(rule1, related_entity_ids=[blue_book.id, yellow_book.id])
        
        # 5. 添加结构化约束 (CSP 增强)
        logger.info("Adding Constraints...")
        # 约束: Blue Book right_of Yellow Book
        const_order = OrderingConstraint(relation="right_of")
        const_order.raw_expression = "blue_book > yellow_book"
        # 注意：这里的逻辑连接应该是有方向的，为了简单演示，我们在 add_constraint_node 里只是简单地关联了实体
        gm.add_constraint_node(const_order, [blue_book.id, yellow_book.id])
        
        # 约束: AllDifferent
        const_diff = AllDifferentConstraint()
        gm.add_constraint_node(const_diff, [blue_book.id, yellow_book.id])
        
        # 6. 执行混合检索
        logger.info("Performing Hybrid Search...")
        query = "blue book right"
        results = gm.hybrid_search(query)
        
        print("\n--- Search Results ---")
        print(f"Query: '{query}'")
        print("Matched Entities:", results["matched_entities"])
        print("Similar Rules:")
        for rule in results["similar_rules"]:
            print(f"  - [{rule['score']:.4f}] {rule['description']}")
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"An error occurred: {e}")
        print("\n[Note] Please ensure Neo4j is running and configured in config.py")

if __name__ == "__main__":
    main()

