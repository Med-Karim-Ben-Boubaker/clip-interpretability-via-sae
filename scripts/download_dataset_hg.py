from huggingface_hub import snapshot_download
import dotenv
import os

dotenv.load_dotenv()

# Download the whole dataset repo snapshot to a folder
local_dir = snapshot_download(
    repo_id="ILSVRC/imagenet-1k",
    repo_type="dataset",
    local_dir="data/imagenet_1k",
    token=os.getenv("HF_TOKEN")
)
