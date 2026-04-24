import logging
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer
from .connector import Neo4jConnector
from LKG_pipeline_1 import config

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.connector = Neo4jConnector()
        # 延迟加载模型以节省资源，除非显式调用
        self._model = None
    
    @property
    def model(self):
        if self._model is None:
            logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
            self._model = SentenceTransformer(config.EMBEDDING_MODEL)
        return self._model

    def encode(self, text: Union[str, List[str]]) -> List[float]:
        """生成文本向量"""
        embeddings = self.model.encode(text)
        if isinstance(embeddings, np.ndarray):
            return embeddings.tolist()
        return embeddings

    def create_vector_index(self, index_name: str, label: str, property_key: str, dimensions: int):
        """在 Neo4j 中创建向量索引"""
        query = f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (n:{label})
        ON (n.{property_key})
        OPTIONS {{indexConfig: {{
         `vector.dimensions`: {dimensions},
         `vector.similarity_function`: 'cosine'
        }}}}
        """
        try:
            self.connector.execute_query(query)
            logger.info(f"Vector index '{index_name}' created or already exists.")
        except Exception as e:
            logger.error(f"Failed to create vector index: {e}")
            raise

    def search_similar(self, index_name: str, query_text: str, top_k: int = 5):
        """执行向量相似度搜索"""
        query_vector = self.encode(query_text)
        
        cypher = f"""
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
        YIELD node, score
        RETURN node, score
        """
        
        result = self.connector.execute_query(
            cypher, 
            parameters={
                "index_name": index_name,
                "top_k": top_k,
                "query_vector": query_vector
            }
        )
        return result.records

    def create_fulltext_index(self, index_name: str, label: str, properties: list):
        """创建 Neo4j 全文索引（BM25）"""
        props_str = ", ".join([f"n.{p}" for p in properties])
        query = f"""
        CREATE FULLTEXT INDEX {index_name} IF NOT EXISTS
        FOR (n:{label})
        ON EACH [{props_str}]
        """
        try:
            self.connector.execute_query(query)
            logger.info(f"Fulltext index '{index_name}' created or already exists.")
        except Exception as e:
            if "already exists" in str(e).lower() or "equivalent index" in str(e).lower():
                logger.info(f"Fulltext index '{index_name}' already exists.")
            else:
                logger.warning(f"Failed to create fulltext index '{index_name}': {e}")

    def fulltext_search(self, index_name: str, query_text: str, top_k: int = 5):
        """执行 BM25 全文搜索"""
        safe_query = query_text.replace("'", "\\'").replace('"', '\\"')
        cypher = """
        CALL db.index.fulltext.queryNodes($index_name, $query_text)
        YIELD node, score
        RETURN node, score
        LIMIT $top_k
        """
        try:
            result = self.connector.execute_query(
                cypher,
                parameters={"index_name": index_name, "query_text": query_text, "top_k": top_k}
            )
            return result.records
        except Exception as e:
            logger.warning(f"Fulltext search failed on '{index_name}': {e}")
            return []

