from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="rasoul-nikbakht/TSpec-LLM",
    repo_type="dataset",
    local_dir="./TSpec-LLM",
    local_dir_use_symlinks=False  # IMPORTANT for Windows
)