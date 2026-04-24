import os
import glob

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "fhz5228887")

# Embedding Configuration
# 支持两种模型，通过环境变量 EMBEDDING_BACKEND 切换:
#   "minilm" (默认) — paraphrase-multilingual-MiniLM-L12-v2, 384维, 轻量快速
#   "bge-m3"         — BAAI/bge-m3, 1024维, 论文中使用的模型, 精度更高
_EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "minilm").lower()

_MODEL_CONFIGS = {
    "minilm": {
        "model_name": "paraphrase-multilingual-MiniLM-L12-v2",
        "hf_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "hf_org": "sentence-transformers",
        "dim": 384,
    },
    "bge-m3": {
        "model_name": "bge-m3",
        "hf_id": "BAAI/bge-m3",
        "hf_org": "BAAI",
        "dim": 1024,
    },
    # 折中选项：BGE-base，比 MiniLM 准、比 BGE-M3 快
    "bge-base": {
        "model_name": "bge-base-en-v1.5",
        "hf_id": "BAAI/bge-base-en-v1.5",
        "hf_org": "BAAI",
        "dim": 768,
    },
    "bge-large": {
        "model_name": "bge-large-en-v1.5",
        "hf_id": "BAAI/bge-large-en-v1.5",
        "hf_org": "BAAI",
        "dim": 1024,
    },
}

_active_config = _MODEL_CONFIGS.get(_EMBEDDING_BACKEND, _MODEL_CONFIGS["minilm"])
_MODEL_NAME = _active_config["model_name"]
_HF_MODEL_ID = _active_config["hf_id"]

def _find_embedding_model():
    """按优先级查找 Embedding 模型路径：
    1. 项目 models_cache/ 子目录
    2. HuggingFace Hub 缓存（snapshots/）
    3. ModelScope 缓存（BAAI/bge-m3 等）
    4. 回退到 HF model ID（在线下载）
    """
    local_path = os.path.join(os.path.dirname(__file__), "models_cache", _MODEL_NAME)
    if os.path.exists(local_path):
        return local_path

    hf_cache = os.getenv("HF_HOME") or os.getenv("HUGGINGFACE_HUB_CACHE")
    if not hf_cache:
        hf_cache_candidates = [
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
            os.path.join("D:\\", "hugging face", "hub"),
        ]
    else:
        hf_cache_candidates = [os.path.join(hf_cache, "hub") if "hub" not in hf_cache else hf_cache]

    hf_org = _active_config["hf_org"]
    for cache_dir in hf_cache_candidates:
        pattern = os.path.join(cache_dir, f"models--{hf_org}--{_MODEL_NAME}", "snapshots", "*", "config.json")
        matches = glob.glob(pattern)
        if matches:
            return os.path.dirname(matches[0])

    # ModelScope 缓存路径（国内下载常用）
    ms_cache_candidates = [
        os.path.join("D:\\", "hugging face", "modelscope_cache"),
        os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub"),
    ]
    for cache_dir in ms_cache_candidates:
        ms_path = os.path.join(cache_dir, hf_org, _MODEL_NAME)
        if os.path.exists(os.path.join(ms_path, "config.json")):
            return ms_path

    return _HF_MODEL_ID

EMBEDDING_MODEL = _find_embedding_model()
EMBEDDING_DIM = _active_config["dim"]

if EMBEDDING_MODEL == _HF_MODEL_ID:
    print(f"[Config] Embedding: {_EMBEDDING_BACKEND} ({_HF_MODEL_ID}, {EMBEDDING_DIM}D) — will download from HuggingFace")
else:
    print(f"[Config] Embedding: {_EMBEDDING_BACKEND} ({EMBEDDING_DIM}D) — local: {EMBEDDING_MODEL}")


# ═══════════════════════════════════════════════════════════
#  Graph Backend 选择
# ═══════════════════════════════════════════════════════════
# LKG_BACKEND=memory  ->  纯 Python 内存版 (无需 Neo4j, 真并行, 推荐科研)
# LKG_BACKEND=neo4j   ->  Neo4j 后端 (默认, 兼容历史代码)
# 注意：每次调用 get_graph_manager() 时重新读取环境变量，
# 因此调用方可以在 import 后通过 os.environ 动态切换。


def get_lkg_backend() -> str:
    return os.getenv("LKG_BACKEND", "neo4j").lower()


def get_graph_manager():
    """根据 LKG_BACKEND 返回 GraphManager 实例（鸭子类型）。"""
    backend = get_lkg_backend()
    if backend == "memory":
        from .core.memory_graph import MemoryGraphManager
        return MemoryGraphManager()
    from .core.graph_ops import GraphManager
    return GraphManager()