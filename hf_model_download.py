import os
from huggingface_hub import snapshot_download

# 模型名称
model_name = "sentence-transformers/all-MiniLM-L6-v2"
# 指定本地保存路径
local_model_path = f"./models/{model_name.replace('/', '_')}"

# 使用镜像站下载
snapshot_download(
    repo_id=model_name,
    local_dir=local_model_path,
    local_dir_use_symlinks=False,
    endpoint="https://hf-mirror.com"  # 明确指定镜像
)
print(f"✅ 模型已下载到：{local_model_path}")