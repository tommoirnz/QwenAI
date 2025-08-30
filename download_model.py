from huggingface_hub import snapshot_download

# Change the path if you prefer a different folder
LOCAL_DIR = r"C:\models\Qwen2_5_VL_7B"

p = snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False,   # Windows-friendly
    resume_download=True,
    allow_patterns=[
        "*.safetensors", "*.json", "*tokenizer*", "*processor*", "*preprocessor*"
    ],
)
print("Downloaded to:", p)
