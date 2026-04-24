import argparse
import logging
import os
import sys
import json
import re
import datetime

# Configure logging
# Remove default config to allow custom handlers
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Console Handler (Always active)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

# Add parent directory to path for package imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LKG_pipeline_1.core.graph_ops import GraphManager
from LKG_pipeline_1.config import get_graph_manager
from LKG_pipeline_1.core.builder import KGBuilder
from LKG_pipeline_1.core.llm_builder import LLMBuilder
from LKG_pipeline_1.core.router import LogicRouter, SolverType

# --- Logger Helper ---
class DualLogger:
    """
    Context manager to redirect stdout (print) and logging to a file
    while keeping them visible on console.
    """
    def __init__(self, filename):
        self.filename = filename
        self.file = None
        self.original_stdout = sys.stdout
        self.file_handler = None

    def __enter__(self):
        # 1. Open file for raw print capturing
        self.file = open(self.filename, 'w', encoding='utf-8')
        
        # 2. Redirect sys.stdout to write to both console and file
        class Tee:
            def __init__(self, original, file_handle):
                self.original = original
                self.file_handle = file_handle
            def write(self, message):
                self.original.write(message)
                self.file_handle.write(message)
                self.file_handle.flush()
            def flush(self):
                self.original.flush()
                self.file_handle.flush()
        
        sys.stdout = Tee(self.original_stdout, self.file)
        
        # 3. Add FileHandler to Logger — set to WARNING to exclude Neo4j INFO notifications
        self.file_handler = logging.FileHandler(self.filename, mode='a', encoding='utf-8')
        self.file_handler.setLevel(logging.WARNING)
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(self.file_handler)
        
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore stdout
        sys.stdout = self.original_stdout
        if self.file:
            self.file.close()
            
        # Remove the file handler to prevent duplicate logs in subsequent runs
        if self.file_handler:
            logging.getLogger().removeHandler(self.file_handler)
            self.file_handler.close()

def load_json_example(file_path, limit=1):
    """Reads the Context field from JSON file records"""
    examples = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            for record in data[:limit]:
                context_text = record.get('context') or record.get('original_context')
                if context_text:
                    # For AR-LSAT, context might be structured with ### sections
                    # We want the full natural language context for LLM
                    if 'ar_lsat' in record.get('id', '').lower():
                        # Extract only the initial natural language description
                        match = re.search(r'^(.*?)(?=\n### Domain|\n### Variables|\n### Constraints|$)', context_text, re.DOTALL)
                        if match:
                            context_text = match.group(1).strip()
                    
                    examples.append({
                        "source": record.get('source', os.path.basename(file_path)),
                        "id": record.get('id'),
                        "question": record.get('question', ''),
                        "original_context": context_text
                    })
    except Exception as e:
        logger.error(f"Error loading JSON example from {file_path}: {e}")
    return examples

def llm_pruner_factory(router):
    """Factory to create a pruning function using the router's client"""
    def prune_nodes(query, candidates):
        if not candidates:
            return []
            
        # Format candidates for LLM
        candidates_text = ""
        for i, c in enumerate(candidates):
            candidates_text += f"ID: {c['id']}\nType: {c['type']}\nContent: {c['content']}\n\n"
            
        prompt = f"""You are a strict relevance filter for a Logical Knowledge Graph.
Your task is to identify which of the provided Graph Nodes are relevant to answering the specific Query.
Ignore nodes that belong to completely different contexts (e.g., ignore 'Apple' constraints if the query is about 'Family relationships').

Query: "{query}"

Candidates:
{candidates_text}

Return strictly a JSON list of the IDs of the relevant nodes. 
Example: ["ID1", "ID3"]
Do not explain. Return ONLY the JSON.
"""
        try:
            # Use the router's provider/client logic
            if router.provider == "openai":
                response = router.client.chat.completions.create(
                    model=router.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                content = response.choices[0].message.content
            elif router.provider == "deepseek":
                 response = router.client.chat.completions.create(
                    model=router.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    stream=False
                )
                 content = response.choices[0].message.content
            elif router.provider == "gemini":
                response = router.client.generate_content(prompt)
                content = response.text
            else:
                logger.warning("Unknown provider for pruning, returning all.")
                return [c['id'] for c in candidates]
                
            # Parse JSON
            # Clean markdown code blocks if present
            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(json)?|```$", "", content, flags=re.MULTILINE).strip()

            import json
            try:
                keep_ids = json.loads(content)
                return keep_ids
            except json.JSONDecodeError:
                 # Try to find JSON array in text if pure parse fails
                match = re.search(r'\[.*?\]', content, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    keep_ids = json.loads(json_str)
                    return keep_ids
                else:
                    logger.warning(f"Could not parse JSON from pruning response: {content[:100]}...")
                    return [c['id'] for c in candidates] # Fallback
                
        except Exception as e:
            logger.error(f"Pruning error: {e}")
            return [c['id'] for c in candidates] # Fallback
            
    return prune_nodes

def _build_subgraph_text(results: dict, query: str, solver_choice: str) -> str:
    """构建干净的逻辑子图文本，作为 LLM 代码生成的输入"""
    lines = []
    lines.append(f"[Query]: {query}")
    lines.append(f"[Router Decision]: {solver_choice}")
    lines.append("")

    entities = results.get("matched_entities", [])
    rules = results.get("similar_rules", [])
    constraints = results.get("related_constraints", [])

    if entities:
        lines.append("[Retrieved Entities]")
        for e in entities:
            lines.append(f"  - {e.get('name', '')} (Type: {e.get('type', '')})")
        lines.append("")

    if rules:
        lines.append("[Retrieved Rules]")
        for r in rules:
            expr = r.get("expression", "")
            desc = r.get("description", "")
            raw = r.get("raw_text", "")
            lines.append(f"  - Expression: {expr}")
            if desc:
                lines.append(f"    Description: {desc}")
            if raw:
                lines.append(f"    Raw Text: {raw}")
        lines.append("")

    if constraints:
        lines.append("[Retrieved Constraints]")
        for c in constraints:
            ctype = c.get("type", "")
            expr = c.get("expression", "")
            raw = c.get("raw_text", "")
            desc = c.get("description", "")
            lines.append(f"  - Type: {ctype}, Expression: {expr}")
            if desc:
                lines.append(f"    Description: {desc}")
            if raw:
                lines.append(f"    Raw Text: {raw}")
        lines.append("")

    return "\n".join(lines)


def verify_graph_with_router(gm: GraphManager, router: LogicRouter, query: str, enable_pruning: bool = False, subgraph_log_path: str = None):
    print(f"\n>>> [Inference Pipeline] Step 1: Hybrid Search for '{query}' (Pruning: {enable_pruning})")
    
    pruner = llm_pruner_factory(router) if enable_pruning else None
    
    entity_th = 0.3 if enable_pruning else 0.6
    text_th = 0.3 if enable_pruning else 0.55

    results = gm.hybrid_search(
        query, 
        top_k=10, 
        entity_threshold=entity_th, 
        text_threshold=text_th,
        pruning_callback=pruner
    )
    
    entities = [e for e in results["matched_entities"] if e.get('type') != 'Concept']
    rules = results['similar_rules']
    constraints = results['related_constraints']
    
    print(f"    Found: {len(entities)} Entities, {len(rules)} Rules, {len(constraints)} Constraints")
    
    solver_choice = "PYTHON"
    if router:
        print(f"\n>>> [Inference Pipeline] Step 2: Solver Routing")
        solver_choice = router.route(query, results)
        print(f"    [Router Decision]: {solver_choice}") 
        
        print(f"\n>>> [Inference Pipeline] Step 3: Execution Plan (Simulation)")
        if solver_choice == "Z3":
            print("    -> Action: Translate Arithmetic/Ordering constraints to Z3 Python API.")
            print("    -> Action: Check Satisfiability or Optimize.")
        elif solver_choice == "PROVER9":
            print("    -> Action: Translate Rules to First-Order Logic (FOL) format.")
            print("    -> Action: Run Prover9 binary for entailment check.")
        elif solver_choice == "PYKE":
            print("    -> Action: Translate Rules to Pyke Knowledge Base (.kfb) format.")
            print("    -> Action: Perform Forward/Backward Chaining Inference.")
        else:
            print("    -> Action: Return retrieved facts directly or run graph traversal (NetworkX).")
    
    # 写入干净的逻辑子图日志（单独文件，只给 LLM 代码生成使用）
    if subgraph_log_path:
        subgraph_text = _build_subgraph_text(results, query, solver_choice)
        os.makedirs(os.path.dirname(subgraph_log_path), exist_ok=True)
        with open(subgraph_log_path, 'w', encoding='utf-8') as f:
            f.write(subgraph_text)

def run_llm_based_pipeline(gm: GraphManager, provider: str, api_key: str, model_name: str, base_url: str, data_limit: int, enable_pruning: bool = False, input_file: str = None):
    # Ensure logs directory exists
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger.info(f"--- Running LLM-Based KG Construction Pipeline with {provider} (Pruning: {enable_pruning}) ---")
    
    # Initialize Builder
    llm_builder = LLMBuilder(
        gm, 
        provider=provider, 
        api_key=api_key,
        base_url=base_url,
        model_name=model_name
    )
    
    # Initialize Router
    logger.info("Initializing Logic Router...")
    router = LogicRouter(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model_name=model_name
    )
    
    # Fix: root_dir was going up one level too many. 
    # Current file is LKG_pipeline_1/unified_demo.py
    # 1. dirname -> LKG_pipeline_1
    # 2. dirname -> SymbCoT (Project Root)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    test_cases = []
    
    # Load test cases from input file if provided
    if input_file:
        logger.info(f"Loading test cases from input file: {input_file}")
        input_data = load_json_example(input_file, limit=data_limit)
        if input_data:
            test_cases.extend(input_data)
        else:
            logger.error("No valid test cases found in input file.")
            return
    else:
        # Use hardcoded example only if no input file is provided
        logger.info("No input file provided, using hardcoded example.")
        # Case 1: Hardcoded example (Math/Logic -> Z3)
        test_cases.append({
            "source": "Hardcoded_Math",
            "id": "Hardcoded_Math_001",
            "original_context": """
            Alice has 5 apples. Bob has 2 more apples than Alice. If someone has more than 10 apples, they are rich. Alice's apples + Bob's apples = Total.
            """,
            "question": "How many apples does Bob have?"
        })


    # 子图日志目录（干净版，仅供 LLM 代码生成使用）
    subgraph_log_dir = os.path.join(os.path.dirname(log_dir), "subgraph_logs")
    os.makedirs(subgraph_log_dir, exist_ok=True)

    for i, case in enumerate(test_cases):
        safe_source = re.sub(r'[\\/*?:"<>|]', '_', case['source'])
        safe_id = re.sub(r'[\\/*?:"<>|]', '_', case.get('id', f'unknown_{i+1}'))
        if safe_id.startswith(safe_source + '_'):
            safe_id = safe_id[len(safe_source) + 1:]
        log_filename = os.path.join(log_dir, f"{safe_source}_{safe_id}.log")
        subgraph_log_filename = os.path.join(subgraph_log_dir, f"{safe_source}_{safe_id}.log")
        
        with DualLogger(log_filename):
            logger.info(f"\n{'='*50}")
            logger.info(f"=== Processing Test Case {i+1} from {case['source']} ===")
            logger.info(f"=== ID: {case.get('id')} ===")
            logger.info(f"=== Log File: {log_filename} ===")
            logger.info(f"{'='*50}")
            
            logger.info(f"Input Context: {case['original_context'][:100]}...")
            logger.info(f"Question: {case['question']}")
            
            llm_builder.build_from_text(case['original_context'], verbose=True)
            
            query = case.get("question", "Who is involved?")
            verify_graph_with_router(gm, router, query, enable_pruning, subgraph_log_path=subgraph_log_filename)

def main():
    parser = argparse.ArgumentParser(description="Unified Knowledge Graph Builder Demo with Routing")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode of operation")

    # LLM-based mode parser only for this demo
    llm_parser = subparsers.add_parser("llm", help="Use LLM-based extraction and routing")
    llm_parser.add_argument("--provider", type=str, choices=["openai", "deepseek", "gemini", "ollama"], default="openai", help="LLM Provider")
    llm_parser.add_argument("--api_key", type=str, help="API Key")
    llm_parser.add_argument("--base_url", type=str, help="Base URL")
    llm_parser.add_argument("--model", type=str, help="Model name")
    llm_parser.add_argument("--limit", type=int, default=1, help="Number of records")
    llm_parser.add_argument("--enable_pruning", action="store_true", help="Enable LLM-based candidate pruning for higher precision")
    llm_parser.add_argument("--input_file", type=str, help="Input JSON file path containing the sample to process")

    args = parser.parse_args()
    
    # API Key handling
    api_key = args.api_key
    if not api_key:
        if args.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif args.provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
        elif args.provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        elif args.provider == "ollama":
            # Ollama doesn't require API key
            api_key = ""
            
    if not api_key and args.provider != "ollama":
        print(f"\n[Error] No API Key found for provider '{args.provider}'.")
        return

    gm = get_graph_manager()
    
    try:
        gm.initialize_schema()
        # Add initial clear here to start fresh, but keep data during loop
        gm.clear_database()
        logger.info("Database cleared for new run.")
    except Exception as e:
        logger.warning(f"Schema initialization warning (may already exist): {e}")

    if args.mode == "llm":
        run_llm_based_pipeline(gm, args.provider, api_key, args.model, args.base_url, args.limit, args.enable_pruning, args.input_file)

if __name__ == "__main__":
    main()
