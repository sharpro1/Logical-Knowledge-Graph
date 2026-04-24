from neo4j import GraphDatabase
import logging
from typing import Optional
import sys
import os

# 确保能导入 LKG_pipeline_1 包
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from LKG_pipeline_1 import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jConnector:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Neo4jConnector, cls).__new__(cls)
            cls._instance._driver = None
        return cls._instance
    
    def connect(self):
        """建立连接"""
        if self._driver is None:
            try:
                self._driver = GraphDatabase.driver(
                    config.NEO4J_URI, 
                    auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
                )
                self.verify_connection()
                logger.info("Successfully connected to Neo4j")
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                raise

    def verify_connection(self):
        """验证连接是否有效"""
        if self._driver:
            self._driver.verify_connectivity()
    
    def close(self):
        """关闭连接"""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("Neo4j connection closed")
            
    def get_driver(self):
        """获取驱动实例"""
        if self._driver is None:
            self.connect()
        return self._driver
    
    def execute_query(self, query: str, parameters: dict = None, db: str = None):
        """执行 Cypher 查询"""
        driver = self.get_driver()
        try:
            result = driver.execute_query(
                query, 
                parameters_=parameters, 
                database_=db
            )
            return result
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

