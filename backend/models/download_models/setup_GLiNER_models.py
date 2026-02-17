from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="urchade/gliner_medium-v2.1",
    local_dir="backend/models/gliner",
    local_dir_use_symlinks=False)#type: ignore

snapshot_download(
    repo_id="microsoft/deberta-v3-base",
    local_dir="backend/models/transformers/deberta-v3-base",
    local_dir_use_symlinks=False) #type: ignore
