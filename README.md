
Qwen2.5-VL-7B multimodal assistant (Gradio 4.x)
-----------------------------------------------
* Model: Qwen/Qwen2.5-VL-7B-Instruct (4-bit NF4 on 16 GB VRAM; set 8-bit/fp16 if needed)
* Features: text + image chat, Whisper ASR, pyttsx3 TTS, HTTPS Gradio UI.
* Math: LaTeX -> MathML on the server (no MathJax needed; inline + block work in iframes).

Setup (PyCharm Terminal):
    pip install -U transformers accelerate pillow qwen-vl-utils[decord]==0.0.8 markdown latex2mathml gradio sounddevice soundfile pyttsx3


Has speech recognition using Whisper and speech synthesis using Sapi5 voices from your PC.

Download the model first by running download_model.py.  I save it to c:\models on my PC

Check you have the right version of Transformers and library by running check_versions. For example mine gives out:

Note you will also need ffmpeg

https://www.nextdiffusion.ai/tutorials/how-to-install-ffmpeg-on-windows-for-stable-diffusion-a-comprehensive-guide




=== Python & pip ===
python                       3.10.11
pip                          25.1.1

=== PyTorch & CUDA ===
torch                        2.9.0.dev20250827+cu128
torch build CUDA             12.8
CUDA available               True
GPU                          NVIDIA GeForce RTX 5070 Ti

=== Transformers & Qwen ===
transformers                 4.55.4
Qwen2.5-VL import            OK

=== Vision utils & Pillow ===
qwen-vl-utils                0.0.8
pillow                       11.3.0

=== UI / Audio stack ===
gradio                       5.44.0
accelerate                   1.10.1
whisper                      20250625
sounddevice                  0.5.2
soundfile                    0.13.1
pyttsx3                      n/a

=== FFmpeg ===
ffmpeg on PATH               C:\ffmpeg\bin\ffmpeg.EXE

Done.

Process finished with exit code 0
