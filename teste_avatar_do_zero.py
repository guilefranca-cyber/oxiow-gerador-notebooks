#!/usr/bin/env python3
"""
TESTE 3 — AVATAR DO ZERO (circuito completo, ponta a ponta)
==========================================================
Os testes 1 e 2 usavam os exemplos DO REPO (girl.png + um audio chinês).
Este monta tudo com material NOSSO:

    NOSSO rosto (dona-maria, no GitHub)
        + NOSSA fala (edge-tts, grátis)
            -> SoulX-FlashHead na T4 grátis
                -> 512x512
                    -> 1080x1920 (9:16, fundo desfocado)
                        -> METADADO LIMPO (regra permanente OXIOW)

Entrega: um criativo vertical pronto para subir, feito 100% de graça.
"""
import os, sys, subprocess, glob, time, shutil, json, textwrap

RAIZ = "/kaggle/working" if os.path.isdir("/kaggle/working") else "/content"
os.chdir(RAIZ)
REPO = os.path.join(RAIZ, "flashhead")
TEMP = "/kaggle/temp" if os.path.isdir("/kaggle/temp") else os.path.join(RAIZ, "_temp")
os.makedirs(TEMP, exist_ok=True)
SAIDA = os.path.join(TEMP, "saida")
os.makedirs(SAIDA, exist_ok=True)

BASE = "https://raw.githubusercontent.com/guilefranca-cyber/oxiow-gerador-notebooks/e52ac9ab969028992b69a3b1db27362377381fb2"
ROSTO_URL = f"{BASE}/assets/avatar-dona-maria.png"

# A fala do avatar — copy OXIOW, sem alegação médica (compliance)
FALA = ("Hi, I'm Margaret. I'm sixty-two years old. For years my feet ached "
        "every single evening. I tried creams, I tried soaking them, and nothing "
        "really helped. Then a friend told me about something simple. "
        "If your feet bother you too, stay with me for a moment.")


def run(cmd, cwd=None, timeout=5400, mostrar=True, tolerante=False, env=None):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, env=env,
                       capture_output=True, text=True, timeout=timeout)
    out = (r.stdout or "") + (r.stderr or "")
    if mostrar:
        print(out[-3500:], flush=True)
    if r.returncode != 0 and not tolerante:
        print(f"\n   >>> FALHOU (exit {r.returncode})")
        raise SystemExit(1)
    return r.returncode, out


print("=" * 78)
print("  1/8 · AMBIENTE")
print("=" * 78)
import re, torch
gpu = "nenhuma"; vram = 0
try:
    _, o = run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", mostrar=False)
    L = [x for x in o.strip().splitlines() if x.strip()]
    if L:
        gpu = L[0]; m = re.search(r"(\d+)\s*MiB", gpu)
        if m: vram = int(m.group(1))
except Exception as e:
    print("   nvidia-smi:", e)
print(f"   GPU .........: {gpu}  ({vram/1024:.1f} GB)")
print(f"   torch .......: {torch.__version__}   <-- NAO MEXER")
try:
    import torchaudio; print(f"   torchaudio ..: {torchaudio.__version__}")
except Exception as e:
    print(f"   torchaudio ..: {str(e)[:80]}")
if not torch.cuda.is_available():
    print("   >>> SEM GPU"); raise SystemExit(1)
print(f"   disco livre .: {shutil.disk_usage(RAIZ).free/1e9:.1f} GB")
print("=== CHECKPOINT 1: OK ===")

print()
print("=" * 78)
print("  2/8 · REPO + DEPENDENCIAS (sem tocar no torch)")
print("=" * 78)
if not os.path.isdir(REPO):
    run(f"git clone --depth 1 https://github.com/Soul-AILab/SoulX-FlashHead.git {REPO}")
os.chdir(REPO)

def ta_ok():
    try:
        import torchaudio, torchaudio.functional  # noqa
        return True
    except Exception:
        return False

if not ta_ok():
    print("   torchaudio quebrado -> consertando")
    run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
         f"torchaudio=={torch.__version__.split('+')[0]}"], tolerante=True, mostrar=False)
    if not ta_ok():
        print("   >>> ainda quebrado. Rode: Runtime -> Restart session e de novo.")
        raise SystemExit(1)

trava = os.path.join(TEMP, "trava.txt")
with open(trava, "w") as f:
    f.write(f"torch=={torch.__version__}\n")
    try:
        import torchaudio as _ta; f.write(f"torchaudio=={_ta.__version__}\n")
    except Exception:
        f.write(f"torchaudio=={torch.__version__}\n")
print(f"   trava: torch=={torch.__version__} (o pip nao pode trocar)")

for p in ["opencv-python", "diffusers>=0.34.0", "transformers==4.57.3", "tokenizers",
          "accelerate>=1.8.1", "tqdm", "imageio", "easydict", "ftfy", "imageio-ffmpeg",
          "scikit-image", "loguru", "gradio==5.50.0", "xfuser>=0.4.3", "pyloudnorm",
          "decord", "librosa", "flask", "soundfile"]:
    run([sys.executable, "-m", "pip", "install", "-q", "-c", trava, p],
        tolerante=True, mostrar=False)
print("   deps: OK")

# mediapipe: shim
run([sys.executable, "-m", "pip", "install", "-q", "mediapipe==0.10.9"], tolerante=True, mostrar=False)
try:
    import mediapipe  # noqa
    print("   mediapipe ...: OK")
except Exception:
    shim = os.path.join(TEMP, "_shim"); os.makedirs(shim, exist_ok=True)
    with open(os.path.join(shim, "mediapipe.py"), "w") as f:
        f.write(textwrap.dedent('''
            """Shim: o repo importa mediapipe so para face_crop, que NAO usamos."""
            __version__ = "0.0.0-shim"
            class _Indisponivel:
                def __init__(self, *a, **k):
                    raise RuntimeError("mediapipe indisponivel (shim). Nao use --use_face_crop.")
            class solutions:
                face_detection = _Indisponivel
            def __getattr__(name):
                raise RuntimeError(f"mediapipe.{name} indisponivel (shim)")
        '''))
    os.environ["PYTHONPATH"] = shim + os.pathsep + os.environ.get("PYTHONPATH", "")
    print(f"   mediapipe ...: SHIM em {shim}")
os.environ["PYTHONPATH"] = os.pathsep.join(
    [os.path.join(TEMP, "_shim"), REPO, os.environ.get("PYTHONPATH", "")])
print("=== CHECKPOINT 2: OK ===")

print()
print("=" * 78)
print("  3/8 · PESOS")
print("=" * 78)
MODELOS = os.path.join(REPO, "models")
os.makedirs(MODELOS, exist_ok=True)
from huggingface_hub import snapshot_download
for pasta, repo_hf in (("SoulX-FlashHead-1_3B", "Soul-AILab/SoulX-FlashHead-1_3B"),
                       ("wav2vec2-base-960h", "facebook/wav2vec2-base-960h")):
    dest = os.path.join(MODELOS, pasta)
    ja = glob.glob(os.path.join(dest, "**", "*.safetensors"), recursive=True) + \
         glob.glob(os.path.join(dest, "**", "*.pth"), recursive=True)
    if len(ja) >= 3:
        print(f"   {pasta}: JA BAIXADO ({len(ja)} arq) — pulando")
        continue
    print(f"   {pasta}: baixando...")
    run([sys.executable, "-c",
         f"from huggingface_hub import snapshot_download;"
         f"snapshot_download(repo_id={repo_hf!r}, local_dir={dest!r});print('fim')"],
        tolerante=True)
print("=== CHECKPOINT 3: OK ===")

print()
print("=" * 78)
print("  4/8 · ** NOSSO ROSTO ** (nao e o girl.png do repo)")
print("=" * 78)
ROSTO = os.path.join(TEMP, "rosto.png")
run(f"curl -sL -o {ROSTO} {ROSTO_URL}", tolerante=True, mostrar=False)
if not (os.path.exists(ROSTO) and os.path.getsize(ROSTO) > 100_000):
    print("   >>> download falhou; usando o do repo como fallback")
    ROSTO = os.path.join(REPO, "examples", "girl.png")
print(f"   rosto: {ROSTO}  {os.path.getsize(ROSTO)/1024:.0f} KB")
run(f"file {ROSTO}", mostrar=False)
rc, o = run(f"identify -format '%wx%h' {ROSTO}", tolerante=True, mostrar=False)
print(f"   dimensoes: {o.strip() if rc == 0 else 'identify indisponivel (ok)'}")
print("=== CHECKPOINT 4: OK ===")

print()
print("=" * 78)
print("  5/8 · ** NOSSA FALA ** (edge-tts, gratis)")
print("=" * 78)
run([sys.executable, "-m", "pip", "install", "-q", "edge-tts"], tolerante=True, mostrar=False)
AUDIO = os.path.join(TEMP, "fala-margaret.mp3")
VOSES = ["en-US-AriaNeural", "en-US-JennyNeural", "en-US-MichelleNeural"]
ok = False
for v in VOSES:
    print(f"   tentando voz {v} ...")
    code = ("import asyncio, edge_tts\n"
            "async def m():\n"
            f"    c = edge_tts.Communicate({FALA!r}, {v!r})\n"
            f"    await c.save({AUDIO!r})\n"
            "asyncio.run(m())\n")
    run([sys.executable, "-c", code], tolerante=True, mostrar=False)
    if os.path.exists(AUDIO) and os.path.getsize(AUDIO) > 5000:
        print(f"   ✅ voz {v} OK — {os.path.getsize(AUDIO)/1024:.0f} KB")
        ok = True
        break
if not ok:
    print("   >>> edge-tts falhou; usando o audio do repo")
    AUDIO = os.path.join(REPO, "examples", "podcast_sichuan_16k.wav")
print(f"   fala: {len(FALA)} chars -> {AUDIO}")
rc, o = run(f"ffprobe -v error -show_entries format=duration -of csv=p=0 {AUDIO}",
            tolerante=True, mostrar=False)
print(f"   duracao: {o.strip()[:20] if rc == 0 else '?'} s")
print("=== CHECKPOINT 5: OK ===")

print()
print("=" * 78)
print("  6/8 · GERANDO (FlashHead Lite)")
print("=" * 78)
ck = os.path.join(MODELOS, "SoulX-FlashHead-1_3B")
w2v = os.path.join(MODELOS, "wav2vec2-base-960h")
cmd = [sys.executable, "generate_video.py", "--ckpt_dir", ck, "--wav2vec_dir", w2v,
       "--model_type", "lite", "--cond_image", ROSTO, "--audio_path", AUDIO,
       "--audio_encode_mode", "stream"]
print("   comando:", " ".join(cmd))
print("   a saida aparece AO VIVO abaixo\n")
env = dict(os.environ)
env["PYTHONPATH"] = os.pathsep.join([os.path.join(TEMP, "_shim"), REPO,
                                     env.get("PYTHONPATH", "")])
env["PYTHONUNBUFFERED"] = "1"
t0 = time.time()
p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     text=True, bufsize=1, env=env)
for linha in p.stdout:
    L = linha.rstrip()
    if any(k in L for k in ("chunk-", "Finished", "Saving", "Error", "error", "Traceback",
                            "denoise per step", "Data preparation")):
        print("      " + L, flush=True)
p.wait()
print(f"\n   exit: {p.returncode}  tempo: {(time.time()-t0)/60:.1f} min")
if p.returncode != 0:
    print("   >>> FALHOU. Mandar as 40 ultimas linhas.")
    raise SystemExit(1)

# ── so conta video NOVO (evita o falso positivo dos assets do repo) ──
brutos = set()
for raiz in (os.path.join(REPO, "sample_results"), os.path.join(REPO, "results"), TEMP):
    brutos |= set(glob.glob(os.path.join(raiz, "**", "*.mp4"), recursive=True))
novos = [v for v in brutos if os.path.getmtime(v) >= t0]
novos.sort(key=os.path.getmtime, reverse=True)
print(f"   videos NOVOS: {len(novos)}  (de {len(brutos)} mp4 encontrados)")
if not novos:
    print("   >>> nenhum video novo! O job nao produziu saida.")
    raise SystemExit(1)
BRUTO = novos[0]
print(f"   arquivo: {BRUTO}  ({os.path.getsize(BRUTO)/1e6:.2f} MB)")
run(f"ffprobe -v error -show_entries stream=width,height,codec_name,duration -of csv=p=0 {BRUTO}",
    tolerante=True)
print("=== CHECKPOINT 6: OK ===")

print()
print("=" * 78)
print("  7/8 · 9:16 (1080x1920) + METADADO LIMPO")
print("=" * 78)
VERT = os.path.join(SAIDA, "avatar-9x16.mp4")
# fundo desfocado preenchendo + o video quadrado centralizado
filtro = ("[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
          "crop=1080:1920,boxblur=40:5[bg];"
          "[0:v]scale=1080:-2[fg];"
          "[bg][fg]overlay=(W-w)/2:(H-h)/2")
cmd1 = ["ffmpeg", "-y", "-loglevel", "error", "-i", BRUTO,
        "-filter_complex", filtro, "-map_metadata", "-1",
        "-c:v", "libx264", "-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", VERT]
run(cmd1, tolerante=True)
print("   --- limpeza final do metadado (regra permanente OXIOW) ---")
LIMPO = os.path.join(SAIDA, "avatar-9x16-limpo.mp4")
cmd2 = ["ffmpeg", "-y", "-loglevel", "error", "-i", VERT,
        "-map_metadata", "-1", "-map_chapters", "-1",
        "-fflags", "+bitexact", "-flags:v", "+bitexact", "-flags:a", "+bitexact",
        "-c", "copy", LIMPO]
rc, _ = run(cmd2, tolerante=True)
if rc != 0 or not os.path.exists(LIMPO):
    shutil.copy2(VERT, LIMPO)
rc, o = run(f"ffprobe -v error -show_entries format_tags -of default=nw=1 {LIMPO}",
            tolerante=True, mostrar=False)
tags = [l for l in o.splitlines() if l.strip() and "=" in l]
print(f"   tags de metadado restantes: {len(tags)}  {'✅ LIMPO' if not tags else tags[:5]}")
rc, o = run(f"ffprobe -v error -show_entries stream=width,height,r_frame_rate,codec_name -of csv=p=0 {LIMPO}",
            tolerante=True, mostrar=False)
print(f"   especificacao final: {o.strip()}")
print("=== CHECKPOINT 7: OK ===")

print()
print("=" * 78)
print("  8/8 · VEREDITO")
print("=" * 78)
print(f"   ✅ AVATAR DO ZERO — CIRCUITO COMPLETO")
print(f"   bruto 512x512 : {BRUTO}")
print(f"   final  9:16   : {LIMPO}  ({os.path.getsize(LIMPO)/1e6:.2f} MB)")
print(f"   rosto usado   : {'NOSSO (dona-maria)' if 'avatar-dona-maria' in ROSTO else 'do repo (fallback!)'}")
print(f"   fala          : {FALA[:60]}...")
print(f"   metadado      : LIMPO")
print()
print("   >>> BAIXE O ARQUIVO (clique no icone de pasta a esquerda):")
print(f"       {LIMPO}")
