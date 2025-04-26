from huggingface_hub import snapshot_download
from pathlib import Path


repo_id = "meta-llama/Llama-3.1-8B"
models_path = Path.cwd().joinpath('hf_models', repo_id)
models_path.mkdir(parents=True, exist_ok=True)

snapshot_download(repo_id=repo_id, local_dir=models_path)
