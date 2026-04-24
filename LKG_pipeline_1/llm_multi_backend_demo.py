import logging
import sys
import os
import argparse
import json

# 路径设置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LKG_pipeline_1.core.graph_ops import GraphManager
from LKG_pipeline_1.core.llm_builder import LLMBuilder

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_json_example(file_path, limit=1):
    """读取 JSON 文件中的 Context 字段"""
    examples = []
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    for record in data[:limit]:
                        # 优先使用 original_context，如果没有则使用 context
                        text = record.get('original_context') or record.get('context')
                        if text:
                            examples.append({
                                "source": os.path.basename(file_path),
                                "id": record.get('id'),
                                "text": text
                            })
        else:
            logger.warning(f"File not found: {file_path}")
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
    return examples

def main():
    parser = argparse.ArgumentParser(description="Multi-Backend LLM Knowledge Graph Builder Demo")
    parser.add_argument("--provider", type=str, choices=["openai", "deepseek", "gemini"], default="openai", help="LLM Provider")
    parser.add_argument("--api_key", type=str, help="API Key (optional if env var set)")
    parser.add_argument("--base_url", type=str, help="Base URL for OpenAI/DeepSeek (optional)")
    parser.add_argument("--model", type=str, help="Model name (optional)")
    
    args = parser.parse_args()
    
    # 1. 检查 API Key
    api_key = args.api_key
    if not api_key:
        if args.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif args.provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
        elif args.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            
    if not api_key:
        print(f"\n[Error] No API Key found for provider '{args.provider}'.")
        print(f"Please set the corresponding environment variable or use --api_key.")
        return

    gm = GraphManager()
    
    logger.info(f"Initializing LLMBuilder with provider: {args.provider}")
    llm_builder = LLMBuilder(
        gm, 
        provider=args.provider, 
        api_key=api_key,
        base_url=args.base_url,
        model_name=args.model
    )
    
    try:
        # 2. 初始化环境
        logger.info("Initializing Schema and Clearing Database...")
        gm.initialize_schema()
        gm.clear_database()
        
        # 3. 准备测试用例
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        test_cases = [
            # Case 1: Hardcoded Example (验证结构化约束)
            {
                "source": "Manual Example",
                "id": "manual_01",
                "text": """
                Context:
                Alice and Bob are students. Math101 is a course. Every course has a exam. Every exam has a grade.
                Rules:
                If a student studies hard, they will pass the exam. 
                Constraints:
                Alice's grade is higher than Bob's grade.
                """
            }
        ]
        
        # Case 2: FOLIO (验证逻辑规则提取)
        test_cases.extend(load_json_example(os.path.join(root_dir, "results/None_FOLIO_dev_gpt-3.5-turbo.json")))
        
        # Case 3: AR-LSAT (验证约束提取)
        # 注意：文件较大，只读第一条
        test_cases.extend(load_json_example(os.path.join(root_dir, "results/None_AR-LSAT_train_gpt-3.5-turbo.json")))

        # 4. 执行构建与验证
        for case in test_cases:
            print(f"\n{'='*50}")
            logger.info(f"Processing Case: {case['source']} (ID: {case['id']})")
            print(f"Text Preview: {case['text'][:100]}...")
            
            # 调用 LLM 构建
            llm_builder.build_from_text(case['text'], verbose=True)
            
            print(f"\n--- Verification for {case['id']} ---")
            
            # 简单的验证搜索 (尝试搜索文本中的关键词)
            # 提取前几个单词作为搜索词
            words = case['text'].split()[:3]
            query = " ".join(words)
            if "Alice" in case['text']: query = "Alice"
            elif "Klosnik" in case['text']: query = "Klosnik"
            
            print(f">>> Searching for '{query}'")
            res = gm.hybrid_search(query, top_k=3, entity_threshold=0.8, text_threshold=0.6)
            
            entities = [e['name'] for e in res['matched_entities'] if e['type'] != 'Concept']
            concepts = [e['name'] for e in res['matched_entities'] if e['type'] == 'Concept']
            
            print(f"  Matched Entities: {entities}")
            print(f"  Related Concepts (via Graph): {concepts}")
            # 这里的 related concepts 其实包含在 matched_entities 里（如果它们被识别为 Concept 且名字匹配）
            # 或者通过 hybrid_search 的扩展逻辑被包含在 similar_rules/constraints 关联的实体中
            
            if res['similar_rules']:
                print(f"  Rules found: {len(res['similar_rules'])}")
                print(f"  - First rule: {res['similar_rules'][0]['description'][:80]}...")
                
            if res.get('related_constraints'):
                print(f"  Constraints found: {len(res['related_constraints'])}")
                print(f"  - First const: {res['related_constraints'][0]['expression'][:80]}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
