# check_versions.py
import sys, shutil, importlib

def line(title, value): print(f"{title:<28} {value}")

print("=== Python & pip ===")
line("python", sys.version.split()[0])
try:
    import pip
    line("pip", pip.__version__)
except Exception as e:
    line("pip", f"ERROR: {e}")

print("\n=== PyTorch & CUDA ===")
try:
    import torch
    line("torch", torch.__version__)
    line("torch build CUDA", getattr(torch.version, "cuda", "n/a"))
    cuda_ok = torch.cuda.is_available()
    line("CUDA available", cuda_ok)
    if cuda_ok:
        line("GPU", torch.cuda.get_device_name(0))
    else:
        print("  NOTE: CUDA=False → install a CUDA wheel matching your driver.")
except Exception as e:
    line("torch", f"IMPORT ERROR: {e}")

print("\n=== Transformers & Qwen ===")
try:
    import transformers
    line("transformers", transformers.__version__)
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    line("Qwen2.5-VL import", "OK")
except Exception as e:
    line("transformers/Qwen2.5-VL", f"ERROR: {e}")

print("\n=== Vision utils & Pillow ===")
try:
    import qwen_vl_utils
    from importlib.metadata import version
    line("qwen-vl-utils", version("qwen-vl-utils"))
except Exception as e:
    line("qwen-vl-utils", f"ERROR: {e}")
try:
    import PIL
    line("pillow", PIL.__version__)
except Exception as e:
    line("pillow", f"ERROR: {e}")

print("\n=== UI / Audio stack ===")
def chk(modname):
    try:
        m = importlib.import_module(modname)
        v = getattr(m, "__version__", "n/a")
        line(modname, v)
    except Exception as e:
        line(modname, f"ERROR: {e}")

for m in ["gradio", "accelerate", "whisper", "sounddevice", "soundfile", "pyttsx3"]:
    chk(m)

print("\n=== FFmpeg ===")
ff = shutil.which("ffmpeg")
line("ffmpeg on PATH", ff or "NOT FOUND")

print("\nDone.")
