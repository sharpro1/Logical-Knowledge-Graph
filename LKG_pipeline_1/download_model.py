import os
import sys

# 设置国内镜像环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

def download_model():
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    local_dir = os.path.join(os.path.dirname(__file__), "models_cache", "paraphrase-multilingual-MiniLM-L12-v2")
    
    print(f"Downloading {model_name} to {local_dir} using hf-mirror.com...")
    
    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # Windows 下建议关闭 symlinks
            resume_download=True
        )
        print("Download completed successfully!")
        print(f"Model path: {local_dir}")
        return local_dir
    except Exception as e:
        print(f"Download failed: {e}")
        return None

if __name__ == "__main__":
    download_model()

