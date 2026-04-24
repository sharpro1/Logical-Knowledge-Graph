import json
import logging
import sys
import os

# 路径设置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LKG_pipeline_1.core.graph_ops import GraphManager
from LKG_pipeline_1.core.builder import KGBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_json_records(file_path, limit=5):
    """读取 JSON 文件中的前 N 条记录"""
    records = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                records = data[:limit]
            else:
                logger.warning(f"JSON root is not a list in {file_path}")
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e}")
    return records

def main():
    gm = GraphManager()
    builder = KGBuilder(gm)
    
    try:
        # 1. 初始化
        logger.info("Initializing Schema and Clearing Database...")
        gm.initialize_schema()
        gm.clear_database()
        
        # 2. 定义要处理的数据集文件
        # 使用项目中的真实路径
        datasets = [
            ("FOLIO", "results/None_FOLIO_train_gpt-3.5-turbo.json"),
            ("LogicalDeduction", "results/None_LogicalDeduction_dev_gpt-3.5-turbo.json"),
            # ("AR-LSAT", "results/None_AR-LSAT_train_gpt-3.5-turbo.json") # 可选
        ]
        
        for name, path in datasets:
            logger.info(f"\n=== Processing Dataset: {name} ===")
            full_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
            
            records = load_json_records(full_path, limit=3) # 每个数据集处理前3条
            if not records:
                logger.warning(f"No records found for {name}")
                continue
                
            for i, record in enumerate(records):
                logger.info(f"Building Record {i+1}: {record.get('id')}")
                builder.build_from_record(record)
                
        # 3. 验证检索
        logger.info("\n=== Verifying Auto-Built Knowledge Graph ===")
        test_queries = ["caffeine", "blue book", "dependent"]
        
        for q in test_queries:
            print(f"\n>>> Searching for: '{q}'")
            results = gm.hybrid_search(q, top_k=3)
            
            print(f"  Matched Entities: {len(results['matched_entities'])}")
            for e in results["matched_entities"]:
                print(f"    - {e['name']} ({e['type']})")
                
            print(f"  Similar Rules: {len(results['similar_rules'])}")
            for r in results["similar_rules"]:
                source = r.get('source', 'vector')
                print(f"    - [{source}] {r['description'][:80]}...")
                
            if results.get("related_constraints"):
                print(f"  Related Constraints: {len(results['related_constraints'])}")
                for c in results["related_constraints"]:
                    print(f"    - [{c['type']}] {c['expression'][:80]}...")

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

