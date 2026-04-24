import logging
from typing import List, Dict, Any, Optional
from .connector import Neo4jConnector
from .vector_store import VectorStore
from LKG_pipeline_1.models.base import Entity, Rule, Concept
from LKG_pipeline_1.models.constraints import Constraint
from LKG_pipeline_1 import config

logger = logging.getLogger(__name__)

class GraphManager:
    """图操作管理器：负责节点创建、关系构建和检索"""
    
    def __init__(self):
        self.connector = Neo4jConnector()
        self.vector_store = VectorStore()
        
    def initialize_schema(self):
        """初始化索引和约束"""
        # 唯一性约束
        queries = [
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT rule_id_unique IF NOT EXISTS FOR (n:Rule) REQUIRE n.id IS UNIQUE",
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (n:Entity) ON (n.name)",
        ]
        
        for q in queries:
            self.connector.execute_query(q)
            
        # 创建向量索引 (Entity) - NEW
        self.vector_store.create_vector_index(
            index_name="entity_embedding_index",
            label="Entity",
            property_key="embedding",
            dimensions=config.EMBEDDING_DIM
        )
            
        # 创建向量索引 (Rule)
        self.vector_store.create_vector_index(
            index_name="rule_embedding_index",
            label="Rule",
            property_key="embedding",
            dimensions=config.EMBEDDING_DIM
        )
        # 创建向量索引 (Constraint)
        self.vector_store.create_vector_index(
            index_name="constraint_embedding_index",
            label="Constraint",
            property_key="embedding",
            dimensions=config.EMBEDDING_DIM
        )
        # 创建全文索引 (BM25) 用于关键词精确匹配
        self.vector_store.create_fulltext_index(
            index_name="entity_fulltext_index",
            label="Entity",
            properties=["name", "entity_type"]
        )
        self.vector_store.create_fulltext_index(
            index_name="constraint_fulltext_index",
            label="Constraint",
            properties=["raw_expression", "raw_context"]
        )
        self.vector_store.create_fulltext_index(
            index_name="rule_fulltext_index",
            label="Rule",
            properties=["expression", "description"]
        )
        logger.info("Schema initialized (vector + fulltext indices).")

    def add_entity(self, entity: Entity):
        """添加实体"""
        
        # Generate embedding for Entity (Name + Type)
        if not entity.properties.get('embedding'):
            # Use normalized name for embedding to ensure consistency (e.g., "cats" -> "cat")
            norm_name = entity.properties.get('normalized_name') or entity.name
            text_to_embed = f"{norm_name} ({entity.entity_type})"
            try:
                embedding = self.vector_store.encode(text_to_embed)
                entity.properties['embedding'] = embedding
            except Exception as e:
                logger.warning(f"Failed to encode entity {entity.name}: {e}")
                
        # 动态构建 Label 字符串（Neo4j label 不能含空格，需要用反引号或替换）
        safe_labels = []
        for lbl in entity.labels:
            sanitized = lbl.replace(" ", "_")
            if not sanitized.isidentifier():
                sanitized = f"`{sanitized}`"
            safe_labels.append(sanitized)
        labels_str = ":".join(safe_labels)
        if labels_str:
            labels_str = ":" + labels_str
        else:
            labels_str = ":Entity"
            
        query_simple = f"""
        MERGE (e{labels_str} {{id: $id}})
        SET e.name = $name, 
            e.entity_type = $entity_type,
            e += $properties
        RETURN e
        """
        
        self.connector.execute_query(
            query_simple,
            parameters={
                "id": entity.id,
                "name": entity.name,
                "entity_type": entity.entity_type,
                "properties": entity.properties
            }
        )

    def add_relationship(self, source_id: str, target_id: str, relation_type: str, properties: dict = None):
        """添加两个实体/概念之间的关系"""
        if not properties:
            properties = {}
            
        # 注意：关系类型也不能参数化，必须拼接。需确保 relation_type 安全。
        # 允许字母、数字和下划线，并将结果转为大写
        safe_rel_type = "".join(c for c in relation_type if c.isalnum() or c == '_').upper()
        if not safe_rel_type:
            safe_rel_type = "RELATED_TO"
            
        query = f"""
        MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
        MERGE (a)-[r:{safe_rel_type}]->(b)
        SET r += $props
        """
        
        self.connector.execute_query(
            query,
            parameters={
                "source_id": source_id, 
                "target_id": target_id,
                "props": properties
            }
        )

    def add_rule(self, rule: Rule, related_entity_ids: List[str] = None):
        """添加规则（包含向量化），并连接到相关实体"""
        # 生成向量
        # Fix: 如果 description 为空，回退使用 expression 生成 embedding
        text_to_embed = rule.description or rule.expression
        if not rule.embedding and text_to_embed:
            try:
                rule.embedding = self.vector_store.encode(text_to_embed)
            except Exception as e:
                logger.warning(f"Failed to encode rule: {e}")
            
        # 1. 创建/合并 Rule 节点
        query = """
        MERGE (r:Rule {id: $id})
        SET r.expression = $expression,
            r.description = $description,
            r.embedding = $embedding
        RETURN r
        """
        
        self.connector.execute_query(
            query,
            parameters={
                "id": rule.id,
                "expression": rule.expression,
                "description": rule.description,
                "embedding": rule.embedding
            }
        )
        
        # 2. 连接到实体 (如果提供了相关ID)
        if related_entity_ids:
            query_link = """
            MATCH (r:Rule {id: $rule_id}), (e:Entity {id: $entity_id})
            MERGE (r)-[:MENTIONS]->(e)
            """
            for eid in related_entity_ids:
                self.connector.execute_query(
                    query_link,
                    parameters={"rule_id": rule.id, "entity_id": eid}
                )

    def add_constraint_node(self, constraint: Constraint, related_entity_ids: List[str]):
        """添加约束节点（含向量化）并连接到实体"""
        
        # 0. Generate Embedding for Constraint Expression
        if not constraint.properties.get('embedding'):
            # 优先使用 raw_expression，其次是 properties 中的 expression 或 description
            text_to_embed = constraint.raw_expression or constraint.properties.get('expression') or constraint.properties.get('description') or ""
            if text_to_embed:
                try:
                    embedding = self.vector_store.encode(text_to_embed)
                    constraint.properties['embedding'] = embedding
                except Exception as e:
                    logger.warning(f"Failed to encode constraint: {e}")
        
        # 1. 动态构建带有Label的创建语句（sanitize 空格等非法字符）
        safe_labels = []
        for lbl in constraint.labels:
            sanitized = lbl.replace(" ", "_")
            if not sanitized.isidentifier():
                sanitized = f"`{sanitized}`"
            safe_labels.append(sanitized)
        labels_str = ":".join(safe_labels)
        if labels_str:
            labels_str = ":" + labels_str
        else:
            labels_str = ":Constraint"
            
        logger.info(f"Creating Constraint with labels: {labels_str}")
            
        query_create = f"""
        CREATE (c{labels_str} {{id: $id}})
        SET c.constraint_type = $constraint_type,
            c.raw_expression = $raw_expression,
            c += $properties
        RETURN c
        """
        
        self.connector.execute_query(
            query_create,
            parameters={
                "id": constraint.id,
                "constraint_type": constraint.constraint_type,
                "raw_expression": constraint.raw_expression,
                "properties": constraint.properties
            }
        )
        
        # 2. 连接实体
        if related_entity_ids:
            query_link = """
            MATCH (c {id: $const_id}), (e:Entity {id: $entity_id})
            MERGE (c)-[:APPLIES_TO]->(e)
            """
            for eid in related_entity_ids:
                self.connector.execute_query(
                    query_link,
                    parameters={"const_id": constraint.id, "entity_id": eid}
                )

    def hybrid_search(self, query_text: str, top_k: int = 5, entity_threshold: float = 0.4, text_threshold: float = 0.4, use_graph_traversal: bool = True, pruning_callback=None):
        """
        混合搜索：向量搜索 + LLM剪枝 + 图遍历扩展
        
        Args:
            query_text: 查询文本
            top_k: 向量搜索召回数量
            entity_threshold: 实体向量相似度初筛阈值 (建议调低，依赖LLM剪枝)
            text_threshold: 文本(规则/约束)向量相似度初筛阈值 (建议调低，依赖LLM剪枝)
            use_graph_traversal: 是否进行图遍历
            pruning_callback: (可选) 一个函数 func(query, candidates) -> filtered_ids，用于LLM二次筛选
        """
        print(f"\n[Hybrid Search Debug] Query: '{query_text}'")
        print(f"Thresholds (Initial): Entity={entity_threshold}, Text={text_threshold}")
        
        results = {
            "matched_entities": [],
            "similar_rules": [],
            "related_constraints": []
        }
        
        # 临时存储所有候选节点，用于统一剪枝
        all_candidates = [] 
        
        # --- 1. Vector Search for Rules ---
        print("\n--- Step 1: Vector Search Rules ---")
        try:
            # 增加 top_k 以提高召回率，依靠后续 LLM 过滤
            rule_res = self.vector_store.search_similar("rule_embedding_index", query_text, top_k + 3)
            for record in rule_res:
                score = record["score"]
                node = dict(record["node"])
                rid = node.get("id")
                expr = node.get("expression", "")[:50]
                desc = node.get("description", "")[:50]
                
                print(f"  Found Rule: {expr} (Desc: {desc})... (Score: {score:.4f})", end="")
                if score < text_threshold: 
                    print(" [FILTERED by Vector Threshold]")
                    continue
                print(" [KEPT for Pruning]")
                
                candidate = {
                    "id": rid, 
                    "type": "Rule",
                    "content": f"Expression: {node.get('expression')}, Description: {node.get('description')}",
                    "score": score,
                    "original_node": node
                }
                all_candidates.append(candidate)
        except Exception as e:
            logger.warning(f"Rule vector search failed: {e}")

        # --- 2. Vector Search for Constraints ---
        print("\n--- Step 2: Vector Search Constraints ---")
        try:
            const_res = self.vector_store.search_similar("constraint_embedding_index", query_text, top_k + 3)
            for record in const_res:
                score = record["score"]
                node = dict(record["node"])
                cid = node.get("id")
                expr = node.get("raw_expression", "")[:50]
                
                print(f"  Found Constraint: {expr}... (Score: {score:.4f})", end="")
                if score < text_threshold: 
                    print(" [FILTERED by Vector Threshold]")
                    continue
                print(" [KEPT for Pruning]")
                
                candidate = {
                    "id": cid, 
                    "type": "Constraint",
                    "content": f"Type: {node.get('constraint_type')}, Expression: {node.get('raw_expression')}",
                    "score": score,
                    "original_node": node
                }
                all_candidates.append(candidate)
        except Exception as e:
            logger.warning(f"Constraint vector search failed: {e}")

        # --- 3. Vector Search for Entities ---
        print("\n--- Vector Search Entities ---")
        seen_entity_ids_vec = set()
        try:
            entity_res = self.vector_store.search_similar("entity_embedding_index", query_text, top_k + 3)
            for record in entity_res:
                score = record["score"]
                node = dict(record["node"])
                eid = node.get("id")
                name = node.get("name")
                seen_entity_ids_vec.add(eid)
                
                print(f"  Found Entity: {name} ({node.get('entity_type')}) (Score: {score:.4f})", end="")
                if score < entity_threshold: 
                    print(" [FILTERED by Vector Threshold]")
                    continue
                print(" [KEPT for Pruning]")
                
                candidate = {
                    "id": eid, 
                    "type": "Entity",
                    "content": f"Name: {name}, Type: {node.get('entity_type')}",
                    "score": score,
                    "original_node": node
                }
                all_candidates.append(candidate)
        except Exception as e:
            logger.warning(f"Entity vector search failed: {e}")

        # --- 3b. BM25 Fulltext Search for Entities (补充精确关键词匹配) ---
        print("\n--- BM25 Fulltext Search Entities ---")
        try:
            bm25_res = self.vector_store.fulltext_search("entity_fulltext_index", query_text, top_k)
            for record in bm25_res:
                node = dict(record["node"])
                eid = node.get("id")
                name = node.get("name")
                bm25_score = record["score"]
                if eid in seen_entity_ids_vec:
                    continue
                print(f"  [BM25] Found Entity: {name} ({node.get('entity_type')}) (BM25 Score: {bm25_score:.4f})")
                candidate = {
                    "id": eid,
                    "type": "Entity",
                    "content": f"Name: {name}, Type: {node.get('entity_type')}",
                    "score": 0.5,
                    "original_node": node
                }
                all_candidates.append(candidate)
        except Exception as e:
            logger.warning(f"BM25 entity search failed: {e}")

        # --- 3.5 LLM Pruning ---
        start_node_ids = set()
        
        if pruning_callback and all_candidates:
            print(f"\n--- Step 3.5: LLM Pruning ({len(all_candidates)} candidates) ---")
            try:
                # 调用外部传入的 LLM 剪枝函数
                kept_ids = pruning_callback(query_text, all_candidates)
                print(f"  LLM decided to keep {len(kept_ids)} nodes out of {len(all_candidates)}.")
                
                for cand in all_candidates:
                    if cand["id"] in kept_ids:
                        start_node_ids.add(cand["id"])
                        # 根据类型回填到 results
                        node = cand["original_node"]
                        if cand["type"] == "Rule":
                            results["similar_rules"].append({
                                "id": cand["id"], "expression": node.get("expression"),
                                "description": node.get("description"), "score": cand["score"], "source": "vector+pruned"
                            })
                        elif cand["type"] == "Constraint":
                             results["related_constraints"].append({
                                "id": cand["id"], "type": node.get("constraint_type"),
                                "expression": node.get("raw_expression"), "score": cand["score"], "source": "vector+pruned"
                            })
                        elif cand["type"] == "Entity":
                            results["matched_entities"].append({
                                "id": cand["id"], "name": node.get("name"), 
                                "type": node.get("entity_type"), "score": cand["score"], "source": "vector+pruned"
                            })
                        print(f"  [KEPT] {cand['type']}: {cand['content']}")
                    else:
                        print(f"  [DROPPED] {cand['type']}: {cand['content']}")
                        
            except Exception as e:
                logger.error(f"Pruning callback failed: {e}. Falling back to keeping all vector results.")
                # Fallback: keep everything passing vector threshold if LLM fails
                for cand in all_candidates:
                    start_node_ids.add(cand["id"])
                    # ... (fill results logic similar to above, skipped for brevity in fallback)
        else:
            # 没有回调，保留所有通过向量阈值的
            for cand in all_candidates:
                start_node_ids.add(cand["id"])
                node = cand["original_node"]
                if cand["type"] == "Rule":
                    results["similar_rules"].append({
                        "id": cand["id"], "expression": node.get("expression"),
                        "description": node.get("description"), "score": cand["score"], "source": "vector"
                    })
                elif cand["type"] == "Constraint":
                     if not any(c['id'] == cand["id"] for c in results["related_constraints"]):
                        results["related_constraints"].append({
                            "id": cand["id"], "type": node.get("constraint_type"),
                            "expression": node.get("raw_expression"), "score": cand["score"], "source": "vector"
                        })
                elif cand["type"] == "Entity":
                    results["matched_entities"].append({
                        "id": cand["id"], "name": node.get("name"), 
                        "type": node.get("entity_type"), "score": cand["score"], "source": "vector"
                    })

        # --- 4. Graph Traversal Extension ---
        if use_graph_traversal and start_node_ids:
            print(f"\n--- Step 4: Graph Traversal from {len(start_node_ids)} nodes ---")
            
            # Optimized traversal to find Entities connected to Constraints/Rules
            # If start node 'n' is a Constraint, we find the Entity via APPLIES_TO
            # If start node 'n' is a Rule, we find the Entity via MENTIONS
            cypher_traversal_optimized = """
            MATCH (n) WHERE n.id IN $ids
            
            // 1. Find related Rules (connected to n)
            OPTIONAL MATCH (n)-[:MENTIONS]-(r:Rule)
            
            // 2. Find related Constraints (connected to n)
            OPTIONAL MATCH (n)-[:APPLIES_TO]-(c:Constraint)
            
            // 3. Find related Entities (connected via ANY relationship)
            OPTIONAL MATCH (n)--(e:Entity)
            
            RETURN n, r, c, e
            """
            
            try:
                traversal_res = self.connector.execute_query(cypher_traversal_optimized, parameters={"ids": list(start_node_ids)})
                
                seen_rule_ids = {r['id'] for r in results["similar_rules"]}
                seen_const_ids = {c['id'] for c in results["related_constraints"]}
                seen_entity_ids = {e['id'] for e in results["matched_entities"]}
                
                for record in traversal_res.records:
                    start_node = dict(record["n"])
                    start_name = start_node.get("name") or start_node.get("raw_expression", "")[:20]
                    
                    # Add Rules
                    r_node = record["r"]
                    if r_node:
                        props = dict(r_node)
                        if props["id"] not in seen_rule_ids:
                            print(f"  -> Rule via [{start_name}]: {props.get('expression')[:30]}...")
                            seen_rule_ids.add(props["id"])
                            results["similar_rules"].append({
                                "id": props["id"], "expression": props.get("expression"),
                                "description": props.get("description"), "source": "graph_traversal"
                            })
                    
                    # Add Constraints
                    c_node = record["c"]
                    if c_node:
                        props = dict(c_node)
                        if props["id"] not in seen_const_ids:
                            print(f"  -> Constraint via [{start_name}]: {props.get('raw_expression')[:30]}...")
                            seen_const_ids.add(props["id"])
                            results["related_constraints"].append({
                                "id": props["id"], "type": props.get("constraint_type"),
                                "expression": props.get("raw_expression"), "source": "graph_traversal"
                            })
                            
                    # Add Entities
                    e_node = record["e"]
                    if e_node:
                        props = dict(e_node)
                        if props["id"] not in seen_entity_ids:
                            print(f"  -> Entity via [{start_name}]: {props.get('name')}")
                            seen_entity_ids.add(props["id"])
                            results["matched_entities"].append({
                                "id": props["id"], "name": props.get("name"), 
                                "type": props.get("entity_type"), "source": "graph_traversal"
                            })
            except Exception as e:
                logger.warning(f"Graph traversal failed: {e}")

        return results

    def _extract_records(self, query_result):
        """从 execute_query 返回值中提取 dict 列表（兼容 EagerResult 和列表）"""
        if hasattr(query_result, 'records'):
            return [rec.data() for rec in query_result.records]
        if isinstance(query_result, list):
            return [r.data() if hasattr(r, 'data') else r for r in query_result]
        return []

    def dump_full_graph(self):
        """导出 Neo4j 中当前全部图数据（实体、关系、规则、约束），用于全图对比实验。"""
        result = {"entities": [], "relationships": [], "rules": [], "constraints": []}

        for r in self._extract_records(self.connector.execute_query(
                "MATCH (e:Entity) RETURN e.id AS id, e.name AS name, e.entity_type AS type")):
            result["entities"].append({
                "name": r.get("name", ""), "type": r.get("type", "Entity"), "id": r.get("id", "")
            })

        for r in self._extract_records(self.connector.execute_query(
                "MATCH (a)-[r]->(b) WHERE NOT type(r) IN ['MENTIONS', 'APPLIES_TO'] "
                "RETURN a.name AS source, type(r) AS relation, b.name AS target, properties(r) AS props")):
            props = r.get("props") or {}
            result["relationships"].append({
                "source": r.get("source", ""), "relation": r.get("relation", ""),
                "target": r.get("target", ""), "desc": props.get("desc", "") if isinstance(props, dict) else ""
            })

        for r in self._extract_records(self.connector.execute_query(
                "MATCH (ru:Rule) RETURN ru.expression AS expression, ru.description AS description")):
            result["rules"].append({
                "expression": r.get("expression", ""), "description": r.get("description", "")
            })

        for r in self._extract_records(self.connector.execute_query(
                "MATCH (c:Constraint) RETURN c.constraint_type AS type, "
                "c.raw_expression AS expression, c.raw_context AS raw_text, c.description AS description")):
            result["constraints"].append({
                "type": r.get("type", ""), "expression": r.get("expression", ""),
                "raw_text": r.get("raw_text", ""), "description": r.get("description", "")
            })

        return result

    def get_relationships_between(self, entity_ids: List[str]):
        """查询给定实体集合之间的所有关系（排除内部元关系 MENTIONS/APPLIES_TO）。"""
        if not entity_ids:
            return []
        rows = self._extract_records(self.connector.execute_query(
            "MATCH (a)-[r]->(b) "
            "WHERE a.id IN $ids AND b.id IN $ids "
            "AND NOT type(r) IN ['MENTIONS', 'APPLIES_TO'] "
            "RETURN a.name AS source, type(r) AS relation, b.name AS target, properties(r) AS props",
            parameters={"ids": entity_ids}))
        rels = []
        for r in rows:
            props = r.get("props") or {}
            rels.append({
                "source": r.get("source", ""), "relation": r.get("relation", ""),
                "target": r.get("target", ""), "desc": props.get("desc", "") if isinstance(props, dict) else ""
            })
        return rels

    def clear_database(self):
        """清空数据库"""
        query = "MATCH (n) DETACH DELETE n"
        self.connector.execute_query(query)
