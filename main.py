# WebQwen2_5_VL.py  – 2025-08-28 (Qwen2.5-VL + Gradio 4; server-side MathML)
# -*- coding: utf-8 -*-
"""
Qwen2.5-VL-7B multimodal assistant (Gradio 4.x)
-----------------------------------------------
* Model: Qwen/Qwen2.5-VL-7B-Instruct (4-bit NF4 on 16 GB VRAM; set 8-bit/fp16 if needed)
* Features: text + image chat, Whisper ASR, pyttsx3 TTS, HTTPS Gradio UI.
* Math: LaTeX -> MathML on the server (no MathJax needed; inline + block work in iframes).

Setup (PyCharm Terminal):
    pip install -U transformers accelerate pillow qwen-vl-utils[decord]==0.0.8 markdown latex2mathml gradio sounddevice soundfile pyttsx3
"""

import sys, asyncio, os, datetime, tempfile, re
from typing import List, Dict, Any, Optional

import markdown as md
import torch
import sounddevice as sd
import soundfile as sf
import PIL.Image as Image
import gradio as gr

# ---- Disable Gradio Brotli compression to avoid Content-Length mismatches on some files ----
try:
    import gradio.brotli_middleware as _bm


    class _NoBrotli(_bm.BrotliMiddleware):
        async def __call__(self, scope, receive, send):
            return await self.app(scope, receive, send)


    _bm.BrotliMiddleware = _NoBrotli
except Exception:
    pass
# -------------------------------------------------------------------------------------------
import whisper, pyttsx3
import numpy as np

# LaTeX -> MathML converter
from latex2mathml.converter import convert as tex2mml

# --- pyttsx3 SAPI5 language parsing fix (handles "809;409") ---
import pyttsx3.drivers.sapi5 as sapi5


def _safe_toVoice(self, attr):
    """Patched version of pyttsx3.drivers.sapi5.SAPI5Driver._toVoice"""
    try:
        lang_attr = attr.GetAttribute('Language') or ''
    except Exception:
        lang_attr = ''
    if ';' in lang_attr:
        lang_attr = lang_attr.split(';')[0].strip()
    try:
        language_code = int(lang_attr, 16) if lang_attr else 0
    except Exception:
        language_code = 0
    return sapi5.Voice(
        attr.Id,
        attr.GetDescription(),
        languages=[language_code],
        gender=attr.GetAttribute('Gender') or None,
        age=attr.GetAttribute('Age') or None,
    )


sapi5.SAPI5Driver._toVoice = _safe_toVoice
# --- end patch ---

# ───────────────────────  Environment  ───────────────────────
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["USE_TORCH_COMPILE"] = "0"
os.environ.setdefault("GRADIO_DISABLE_COMPRESSION", "1")  # extra guard against content-length mismatches

# ───────────────────────  Load Qwen2.5-VL  ───────────────────
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

MODEL_ID = r"C:\models\Qwen2_5_VL_7B"  # local folder path
MODEL_CACHE = r"C:\models"

processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE)

qcfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

qwen = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=qcfg,
    device_map={"": 0},  # keep EVERYTHING on GPU 0
    low_cpu_mem_usage=True,
    cache_dir=MODEL_CACHE,
    # attn_implementation="flash_attention_2",  # enable if FA2 is available
)
qwen.eval()

# ───────────────────────  Whisper & TTS  ─────────────────────
# whisper_mod = whisper.load_model("small", device="cpu")
whisper_mod = whisper.load_model("small", device="cuda")  # faster


def record_audio(duration: int, fs: int = 16000) -> str:
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    path = tempfile.mktemp(suffix=".wav")
    sf.write(path, audio, fs)
    return path


def transcribe(path: str) -> str:
    try:
        return whisper_mod.transcribe(path, task="translate")["text"].strip()
    except Exception as e:
        return f"Whisper error: {e}"
    finally:
        if os.path.exists(path):
            os.remove(path)


def voices() -> List[str]:
    engine = pyttsx3.init("sapi5")
    return [f"{i}: {v.name}" for i, v in enumerate(engine.getProperty("voices"))]


def speak(txt: str, idx: str, rate: int = 200):
    """Return (sample_rate, waveform) for gr.Audio(type="numpy").
    Avoids FileResponse + Content-Length races on Windows temp files.
    """
    engine = pyttsx3.init("sapi5")
    vs = engine.getProperty("voices")
    engine.setProperty("voice", vs[int(idx)].id)
    engine.setProperty("rate", rate)
    out = tempfile.mktemp(suffix=".wav")
    engine.save_to_file(txt, out)
    engine.runAndWait()
    try:
        engine.stop()
    except Exception:
        pass
    # Ensure the file is fully written/unlocked before reading
    import time, os
    last_size = -1
    for _ in range(50):  # up to ~2.5s
        try:
            size = os.path.getsize(out)
            if size == last_size and size > 0:
                break
            last_size = size
        except Exception:
            pass
        time.sleep(0.05)
    # Read into numpy and return
    import numpy as np
    import numpy as np
    try:
        data, sr = sf.read(out, dtype="float32")
        # collapse (N,1) → (N,) if needed
        if data.ndim == 2 and data.shape[1] == 1:
            data = data[:, 0]
        # sanitize then convert to int16
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        data = np.clip(data, -1.0, 1.0)
        data_i16 = (data * 32767.0).astype(np.int16)
    except Exception:
        sr, data_i16 = 16000, np.zeros(16000, dtype=np.int16)
    finally:
        try:
            os.remove(out)
        except Exception:
            pass
    return (sr, data_i16)


def make_speakable(text: str, speak_math: bool = False) -> str:
    t = text
    # remove code fences
    while '```' in t:
        start = t.find('```');
        end = t.find('```', start + 3)
        if end == -1: break
        t = t[:start] + ' [code] ' + t[end + 3:]
    # inline code
    t = t.replace('`', ' ')

    # $$ blocks
    def _proc_blocks(s: str, delim: str) -> str:
        out = '';
        i = 0
        L = len(delim)
        while True:
            start = s.find(delim, i)
            if start == -1:
                out += s[i:];
                break
            out += s[i:start]
            end = s.find(delim, start + L)
            if end == -1:
                out += s[start:];
                break
            inner = s[start + L:end]
            if speak_math:
                inner_clean = inner.replace('\\', '').replace('{', '').replace('}', '')
                inner_clean = inner_clean.replace('^2', ' squared').replace('^3', ' cubed')
                out += ' equation: ' + inner_clean + ' '
            else:
                out += ' [equation] '
            i = end + L
        return out

    t = _proc_blocks(t, '$$')

    # $ inline
    def _proc_inline(s: str) -> str:
        out = [];
        buf = [];
        in_math = False;
        i = 0
        while i < len(s):
            ch = s[i]
            if ch == '$':
                if in_math:
                    inner = ''.join(buf);
                    buf = []
                    if speak_math:
                        inner_clean = inner.replace('\\', '').replace('{', '').replace('}', '')
                        inner_clean = inner_clean.replace('^2', ' squared').replace('^3', ' cubed')
                        out.append(' equation: ' + inner_clean + ' ')
                    else:
                        out.append(' [equation] ')
                    in_math = False
                else:
                    in_math = True
                i += 1
                continue
            if in_math:
                buf.append(ch)
            else:
                out.append(ch)
            i += 1
        if buf:
            out.append(' ' + ''.join(buf))
        return ''.join(out)

    t = _proc_inline(t)
    # simplify markdown markers
    t = t.replace('*', '').replace('_', '')
    # symbols SAPI tends to spell out
    t = t.replace('\\', '').replace('#', '').replace('&', ' and ')
    # collapse whitespace
    t = ' '.join(t.split())
    return t


# ───────────────────────  Math-safe Markdown rendering (LaTeX -> MathML) ─────────────────────
# 1) Normalize: \(..\) → $..$; \[..\] → $$..$$; mathy [..] → $..$; keep $$..$$ on own lines; \thinspace → \,
# 2) Extract ALL math into placeholders (skip code fences/inline code).
# 3) Convert each TeX chunk to MathML (inline/block).
# 4) Run Markdown on non-math.
# 5) Restore MathML (raw) so browser renders it (no MathJax needed).

_CODE_SPLIT_RE = re.compile(r'(```.*?```|`[^`]*`)', re.DOTALL)  # skip code
_PARENS = re.compile(r'\\\(\s*(.*?)\s*\\\)', re.DOTALL)  # \( ... \)
_BRACKETS_BLOCK = re.compile(r'\\\[\s*(.*?)\s*\\\]', re.DOTALL)  # \[ ... \]
_BRACKETS_INLINE = re.compile(r'(?<!\!)\[\s*([^\[\]]+?)\s*\](?!\()', re.DOTALL)  # [ ... ] not link/image
_BLOCK_DOLLAR_PAT = re.compile(r'\$\$(.+?)\$\$', re.DOTALL)  # $$...$$
_INLINE_DOLLAR_PAT = re.compile(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', re.DOTALL)  # $...$

# Optional: trim boundary spacing commands like \, \; \quad at the edges of TeX
_BOUNDARY_SPACE = re.compile(r'^(?:\\,|\\;|\\:|\\!|\\quad|\\qquad)\s*|\s*(?:\\,|\\;|\\:|\\!|\\quad|\\qquad)$')


def _looks_mathy(text: str) -> bool:
    return bool(re.search(r'(\\[A-Za-z]+|[_^=+\-*/<>])', text))


def _normalize_delimiters(s: str) -> str:
    s = s.replace('\r\n', '\n')
    # Unescape accidental \$ / \$$
    s = re.sub(r'\\\s*\$\$', '$$', s)
    s = re.sub(r'\\\s*\$', '$', s)
    # \( ... \) → $ ... $
    s = _PARENS.sub(lambda m: f"${m.group(1).strip()}$", s)
    # \[ ... \] → $$ ... $$
    s = _BRACKETS_BLOCK.sub(lambda m: f"$${m.group(1).strip()}$$", s)

    # [ ... ] → $ ... $ (only if it looks mathy and not a link/image)
    def _br_to_inline(m):
        inner = m.group(1).strip()
        return f"${inner}$" if _looks_mathy(inner) else m.group(0)

    s = _BRACKETS_INLINE.sub(_br_to_inline, s)
    # Normalize thinspace macro
    s = s.replace(r'\thinspace', r'\,')
    return s


def _strip_tex_edges(tex: str) -> str:
    # remove one layer of spacing commands at edges
    tex = _BOUNDARY_SPACE.sub('', tex)
    # Strip stray alignment ampersands at edges (common when models emit align tokens)
    if tex.rstrip().endswith(' &'):
        tex = tex.rstrip()[:-2].rstrip()
    if tex.lstrip().startswith('& '):
        tex = tex.lstrip()[2:].lstrip()
    tex = tex.replace('&=', ' = ').replace('=&', ' = ')
    return tex


def _tex_to_mathml(tex: str, display: bool) -> str:
    # latex2mathml doesn't need $...$ wrappers; pass pure TeX
    try:
        mml = tex2mml(tex)
    except Exception:
        # If conversion fails, fall back to showing raw TeX
        return f"<code>{tex}</code>"
    # Ensure display attribute for block math
    if display:
        # Insert display="block" into the <math ...> tag
        mml = re.sub(r'<math\b', r'<math display="block"', mml, count=1)
    return mml


def _extract_and_convert_math(s: str) -> str:
    # Convert math to MathML using placeholders to avoid Markdown mangling
    tokens: list[str] = []

    def tok(html_snippet: str) -> str:
        i = len(tokens)
        tokens.append(html_snippet)
        return f"§§MATH{i}§§"

    def process_segment(seg: str) -> str:
        # Blocks first
        def rep_block(m):
            tex = _strip_tex_edges(m.group(1).strip())
            return tok(_tex_to_mathml(tex, display=True))

        seg = _BLOCK_DOLLAR_PAT.sub(rep_block, seg)

        # Inline next
        def rep_inline(m):
            tex = _strip_tex_edges(m.group(1).strip())
            return tok(_tex_to_mathml(tex, display=False))

        seg = _INLINE_DOLLAR_PAT.sub(rep_inline, seg)
        return seg

    parts = _CODE_SPLIT_RE.split(_normalize_delimiters(s))
    for i in range(0, len(parts), 2):  # only non-code parts
        parts[i] = process_segment(parts[i])
    combined = ''.join(parts)

    # Now run Markdown over the text that includes placeholders
    html_body = md.markdown(combined, extensions=["extra", "sane_lists"])

    # Restore MathML (raw)
    for i, snippet in enumerate(tokens):
        html_body = html_body.replace(f"§§MATH{i}§§", snippet)

    # Cleanup: remove any stray \( \) or \[ \] that might sit adjacent to MathML
    html_body = re.sub(r'\\\(\s*(?=<math\b)', '', html_body)
    html_body = re.sub(r'(?<=</math>)\s*\\\)', '', html_body)
    html_body = re.sub(r'\\\[\s*(?=<math\b)', '', html_body)
    html_body = re.sub(r'(?<=</math>)\s*\\\]', '', html_body)

    # Light CSS to make MathML look nice
    style = """
    <style>
      math[display="block"] { display:block; margin: 0.5em 0 0.75em 0; }
      math { font-size: 1.05em; }
    </style>
    """
    return style + f"<div style='white-space:pre-wrap'>{html_body}</div>"


def render_math_markdown(s: str) -> str:
    return _extract_and_convert_math(s)


# ───────────────────────  Chat helpers  ─────────────────────
SYSTEM_PROMPT = (
    "You are a helpful, concise AI assistant. "
    "Use LaTeX for math: inline with $...$ and blocks with $$...$$ (on their own lines). "
    "Do not escape dollar signs."
)

SANITY_TEXT = """**Inline tests (both should render):**

Dollar: $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$
Bracket-to-math: Gauss's Law: [ \\,\\nabla \\cdot \\mathbf{E} = \\frac{\\rho}{\\varepsilon_0} \\, ]

**Block test:**
$$
\\nabla \\cdot \\mathbf{E} = \\frac{\\rho}{\\varepsilon_0},\\qquad
\\nabla \\times \\mathbf{B} - \\mu_0\\varepsilon_0\\,\\frac{\\partial \\mathbf{E}}{\\partial t} = \\mu_0 \\mathbf{J}
$$

**Link should stay a link (not math):** [OpenAI](https://openai.com)
"""

hist: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]


def reset_history() -> tuple[str, str, Optional[str], Optional[Image.Image]]:
    hist.clear()
    hist.append({"role": "system", "content": SYSTEM_PROMPT})
    return render_math_markdown(SANITY_TEXT), "", None, None


def shortcut(msg: str) -> Optional[str]:
    text = (msg or "").lower()
    if "date" in text and any(k in text for k in ("today", "current", "what")):
        return datetime.date.today().strftime("Today's date is %B %d, %Y.")
    return None


def chat(message: str, image: Optional[Image.Image] = None) -> str:
    if not message or not message.strip():
        return ""
    if reply := shortcut(message):
        hist.append({"role": "assistant", "content": reply})
        return reply

    # Qwen: each user message can be a list of {"type": "image"/"text", ...}
    entry = ([{"type": "image", "image": image}, {"type": "text", "text": message}]
             if image is not None else [{"type": "text", "text": message}])

    messages = hist + [{"role": "user", "content": entry}]

    # 1) Render chat to template text
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 2) Prepare vision inputs (images/videos)
    image_inputs, video_inputs = process_vision_info(messages)

    # 3) Tokenize & move to device
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)

    # 4) Generate
    with torch.no_grad():
        out_ids = qwen.generate(**inputs, max_new_tokens=1024)

    # 5) Decode only new tokens (strip the prompt part)
    gen_only = out_ids[0, inputs.input_ids.shape[1]:]
    reply = processor.decode(gen_only, skip_special_tokens=True).strip()

    # Clean soft prefixes
    reply = re.sub(r'^(user|assistant):\s*', '', reply, flags=re.IGNORECASE).strip()

    # Update history
    hist.extend([{"role": "user", "content": message},
                 {"role": "assistant", "content": reply}])
    return reply  # raw; formatting happens in pipeline


# ───────────────────────  UI  ───────────────────────────────
CSS = """
textarea, input, .gradio-container {font-size:24px !important;}
button.btn-small {min-width:120px !important;}
#webcam-preview video, #webcam-preview img { transform: scaleX(-1); }   /* mirror fix */
.app-title { text-align:center; margin: 0.25rem 0 1rem; }
"""


def sanitize_pil(im: Image.Image | None) -> Image.Image | None:
    if im is None:
        return None
    try:
        # normalize color mode and strip odd encodings that can confuse servers/browsers
        im = im.convert("RGB")
    except Exception:
        pass
    return im


def build_ui():
    with gr.Blocks(css=CSS) as demo:
        gr.Markdown("<h1 class='app-title'>Qwen2.5-VL-7B Assistant</h1>")

        # ── Inputs ──
        with gr.Row():
            txt = gr.Textbox(lines=3, label="Your message", scale=8)
            send = gr.Button("Send", variant="primary", elem_classes="btn-small", scale=1)
            rec = gr.Button("🎤 Record", elem_classes="btn-small", scale=1)

        with gr.Row():
            dur_slider = gr.Slider(5, 20, value=5, step=1,
                                   label="Recording duration (seconds)", scale=10)

        with gr.Row():
            with gr.Tab("Upload"):
                img_upload = gr.Image(sources=["upload"], type="numpy", label="Upload")
            with gr.Tab("Webcam"):
                img_cam = gr.Image(sources=["webcam"], type="numpy", label="Webcam", elem_id="webcam-preview")
            img_state = gr.State()

            def sanitize_np(arr: np.ndarray | None) -> np.ndarray | None:
                if arr is None:
                    return None
                a = np.asarray(arr)
                if a.ndim == 2:
                    a = np.stack([a, a, a], axis=-1)
                if a.shape[-1] == 4:
                    a = a[..., :3]
                if a.dtype != np.uint8:
                    if np.issubdtype(a.dtype, np.floating):
                        a = np.clip(a, 0.0, 1.0)
                        a = (a * 255.0 + 0.5).astype(np.uint8)
                    else:
                        a = np.clip(a, 0, 255).astype(np.uint8)
                return a

            # (removed change hook to avoid reload loop and content-length issues)
            # (removed change hook to avoid reload loop and content-length issues)

        with gr.Row():
            auto = gr.Checkbox(True, label="Auto-TTS")
            speak_math = gr.Checkbox(False, label="Speak math (light)")
            vlist = voices()
            voice_dd = gr.Dropdown(choices=vlist, value=vlist[0] if vlist else None, label="Voice")
            reset = gr.Button("🔄 New topic")
            sanity = gr.Button("🧪 Sanity check", elem_classes="btn-small")

        # ── Outputs ──
        md_out = gr.HTML(value=render_math_markdown(SANITY_TEXT), elem_id="md-out")
        aud_out = gr.Audio(type="numpy", autoplay=True, label="TTS")

        # ── Pipeline glue ──
        def pipeline(msg, arr_upload, arr_cam, do_tts, voice, speak_math_flag):
            # choose source image lazily, with sanitize and webcam mirror
            pil = None
            if arr_upload is not None:
                a = sanitize_np(arr_upload)
                pil = Image.fromarray(a).convert("RGB")
            elif arr_cam is not None:
                a = sanitize_np(arr_cam)
                a = np.fliplr(a)
                pil = Image.fromarray(a).convert("RGB")

            # optional safety downscale to keep payloads small and stable
            if pil is not None and max(pil.size) > 1600:
                pil.thumbnail((1600, 1600))

            if not (msg and msg.strip()) and pil is None:
                return render_math_markdown(""), None, None
            reply = chat(msg, pil)  # raw text
            html_out = render_math_markdown(reply)  # Markdown + MathML
            if do_tts and voice:
                tts_text = make_speakable(reply, speak_math_flag)
                wav = speak(tts_text, int(voice.split(':')[0]))
            else:
                wav = None
            return html_out, wav, None

        # ── Events ──
        send.click(pipeline, [txt, img_upload, img_cam, auto, voice_dd, speak_math], [md_out, aud_out, img_state])
        txt.submit(pipeline, [txt, img_upload, img_cam, auto, voice_dd, speak_math], [md_out, aud_out, img_state])
        rec.click(lambda d: transcribe(record_audio(int(d))), dur_slider, txt)
        reset.click(reset_history, None, [md_out, txt, aud_out, img_state])
        sanity.click(lambda: (render_math_markdown(SANITY_TEXT), None, None),
                     None, [md_out, aud_out, img_state])

    return demo


# ───────────────────────  Launch  ───────────────────────────
demo = build_ui()

if __name__ == "__main__":
    os.environ["GRADIO_SSL_NO_VERIFY"] = "1"
    demo.launch(
        server_name="0.0.0.0", server_port=8443,
        ssl_certfile="C:/Users/tomsp/Downloads/cert.pem",
        ssl_keyfile="C:/Users/tomsp/Downloads/key.pem",
        ssl_verify=False,
        share=False,  # stay same-origin; MathML doesn't need JS anyway
    )
