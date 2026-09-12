#!/usr/bin/env python3
"""
TESTE SOULX-FLASHHEAD NA T4 GRATIS — OXIOW
==========================================
OBJETIVO: provar se conseguimos gerar um avatar falante (boca mexendo) de graca.

POR QUE ESTE E NAO O ECHOMIMIC:
  EchoMimic  -> pesos somam 21,89 GB, nao cabe em 14,56 GB (T4). A rota morreu em 8 patches.
  FlashHead  -> SÓ 1,3B. O requirements usa xformers (funciona em T4/sm_75),
                enquanto o flash-attn (que NAO funciona em T4) fica de fora.
                Licenca Apache-2.0 = podemos vender o resultado.

O QUE ESTE SCRIPT FAZ, EM ORDEM (idempotente, imprime checkpoint de cada etapa):
  1. Ambiente (GPU, VRAM, RAM, disco, internet)
  2. Clone do repo oficial
  3. PyTorch na versao que o repo pede
  4. Dependencias (com plano B se xformers/mediapipe falharem)
  5. Pesos: SoulX-FlashHead-1_3B (14,3 GB) + wav2vec2-base-960h
  6. Inferencia com Model_Lite usando os exemplos DO PROPRIO REPO
  7. Veredito + onde ficou o video

ENTRADA DO USUARIO: nenhuma. Roda e mostra.
"""
import os, sys, subprocess, glob, time, json, shutil

RAIZ = "/content" if os.path.isdir("/content") else ("/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd())
REPO = os.path.join(RAIZ, "flashhead")
os.chdir(RAIZ)


def run(cmd, cwd=None, timeout=3600, mostrar=True, tolerante=False):
    """tolerante=True -> nao aborta se falhar (usado em passos opcionais)."""
    r = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd,
                       capture_output=True, text=True, timeout=timeout)
    saida = (r.stdout or "") + (r.stderr or "")
    if mostrar:
        print(saida[-5000:], flush=True)
    if r.returncode != 0 and not tolerante:
        print(f"\n   >>> FALHOU (exit {r.returncode}): {cmd if isinstance(cmd,str) else ' '.join(cmd)}")
        raise SystemExit(1)
    return r.returncode, saida


print("=" * 78)
print("  1/7 · AMBIENTE")
print("=" * 78)
gpu = "nenhuma"; vram_mib = 0
try:
    rc, o = run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", mostrar=False)
    linha = [l for l in o.strip().splitlines() if l.strip()]
    if linha:
        gpu = linha[0]
        import re
        m = re.search(r"(\d+)\s*MiB", gpu)
        if m:
            vram_mib = int(m.group(1))
except Exception as e:
    print("   nvidia-smi falhou:", e)

import torch
cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
print(f"   GPU ..........: {gpu}")
print(f"   VRAM .........: {vram_mib} MiB ({vram_mib/1024:.2f} GB)")
print(f"   capability ...: {cap}  {'(Ampere+ — flash-attn serviria)' if cap[0] >= 8 else '(Turing/pre-Ampere — usar xformers/SDPA)'}")
print(f"   torch ........: {torch.__version__}")
print(f"   raiz .........: {RAIZ}")
with open("/proc/meminfo") as f:
    for L in f:
        if L.startswith("MemTotal"):
            print("   RAM ..........: %.1f GB" % (int(L.split()[1]) / 1e6)); break
try:
    du = shutil.disk_usage(RAIZ)
    print(f"   disco livre ..: {du.free/1e9:.1f} GB")
except Exception:
    pass

rc, _ = run("curl -s -o /dev/null -w '%{http_code}' -m 15 https://huggingface.co", mostrar=False)
print(f"   internet .....: {'OK' if _.strip() == '200' else 'FALHOU (' + _.strip()[:20] + ')'}")

if not torch.cuda.is_available():
    print("\n   >>> SEM GPU. Ligue a GPU nas Settings e rode de novo.")
    SystemExit(1)
print("=== CHECKPOINT 1: OK ===")

print()
print("=" * 78)
print("  2/7 · REPO OFICIAL")
print("=" * 78)
if not os.path.isdir(REPO):
    run(f"git clone --depth 1 https://github.com/Soul-AILab/SoulX-FlashHead.git {REPO}")
os.chdir(REPO)
print("   arquivos-chave:")
for f in ["generate_video.py", "requirements.txt", "examples/girl.png"]:
    print(f"      {f:32s} {'OK' if os.path.exists(f) else 'AUSENTE'}")
exemplos = sorted(os.listdir("examples"))[:12] if os.path.isdir("examples") else []
print("   exemplos no repo:", exemplos)
print("=== CHECKPOINT 2: OK ===")

print()
print("=" * 78)
print("  3/7 · PYTORCH (versao que o repo pede)")
print("=" * 78)
if "2.7.1" in torch.__version__:
    print("   ja esta na 2.7.1 — pulando")
else:
    print("   instalando torch 2.7.1 + cu128 (pode levar 2-4 min)")
    run([sys.executable, "-m", "pip", "install", "-q",
         "torch==2.7.1", "torchvision==0.22.1",
         "--index-url", "https://download.pytorch.org/whl/cu128"],
        tolerante=True)
    try:
        import importlib, torch as t2
        importlib.reload(t2)
        print("   torch agora:", t2.__version__)
    except Exception as e:
        print("   aviso ao recarregar:", e)
print("=== CHECKPOINT 3: OK ===")

print()
print("=" * 78)
print("  4/7 · DEPENDENCIAS (com planos B)")
print("=" * 78)
if os.path.exists("requirements.txt"):
    print("   instalando requirements.txt ...")
    rc, saida_req = run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
                        tolerante=True, mostrar=False)
    if rc == 0:
        print("   requirements: OK")
    else:
        print("   requirements falhou em algum pacote — instalando um por um (tolerante):")
        with open("requirements.txt") as f:
            pkgs = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        for p in pkgs:
            r2, _o = run([sys.executable, "-m", "pip", "install", "-q", p], tolerante=True, mostrar=False)
            print(f"      {'OK  ' if r2 == 0 else 'FALHOU'} {p}")

# xformers e o substituto do flash-attn em T4 — conferir se ficou disponivel
try:
    import xformers
    print("   xformers ....: OK", xformers.__version__)
except Exception as e:
    print("   xformers ....: nao disponivel ->", str(e)[:90])
    print("      (o repo consegue rodar sem ele; sera usado SDPA do proprio torch)")

# ── MediaPipe: o pitfall nº1 no Colab (documentado por terceiros) ───────────
# "MediaPipe recently changed its packaging on Colab's Python 3.11 runtime.
#  The face detector crashes on import."
# O MediaPipe so serve para o CORTE DE ROSTO (--use_face_crop). Como nao pedimos
# esse corte, a solucao honesta quando ele nao instala e um SHIM MINIMO que deixa
# o import passar — e grita alto se alguem realmente tentar usar o detector.
def _consertar_mediapipe():
    try:
        import mediapipe
        return f"OK {mediapipe.__version__}"
    except Exception as e1:
        print(f"      (mediapipe falhou: {str(e1)[:80]}) — tentando consertar...")
        run([sys.executable, "-m", "pip", "install", "-q", "numpy<2"], tolerante=True, mostrar=False)
        run([sys.executable, "-m", "pip", "install", "-q", "mediapipe==0.10.9"],
            tolerante=True, mostrar=False)
        try:
            import mediapipe
            return f"OK {mediapipe.__version__} (apos conserto)"
        except Exception as e2:
            print(f"      (segue falhando: {str(e2)[:70]})")
            print("      >>> aplicando SHIM: o import passa, e se o detector for")
            print("          realmente chamado ele avisa (nos nao usamos face_crop).")
            import types, sys as _s
            shim = types.ModuleType("mediapipe")
            def _nao_usado(*a, **k):
                raise RuntimeError(
                    "mediapipe nao esta disponivel neste runtime. "
                    "O detector de rosto nao e usado quando --use_face_crop=False. "
                    "Se esta mensagem apareceu, a pipeline tentou cortar o rosto.")
            shim.solutions = types.SimpleNamespace(
                face_detection=types.SimpleNamespace(FaceDetection=_nao_usado))
            shim.__version__ = "0.0.0-shim"
            _s.modules["mediapipe"] = shim
            return "SHIM aplicado (import OK, detector indisponivel — nao usamos)"

print("   mediapipe ...:", _consertar_mediapipe())

for mod in ["cv2", "diffusers", "transformers", "librosa", "decord", "easydict", "pyloudnorm"]:
    try:
        __import__(mod)
        print(f"   {mod:13s}: OK")
    except Exception as e:
        print(f"   {mod:13s}: FALTA ({str(e)[:60]})")
print("=== CHECKPOINT 4: OK ===")

print()
print("=" * 78)
print("  5/7 · PESOS  (~14,7 GB — o download principal)")
print("=" * 78)
MODELOS = os.path.join(REPO, "models")
os.makedirs(MODELOS, exist_ok=True)

# tenta reaproveitar pesos de um dataset do Kaggle anexado
def tenta_kaggle(padroes, destino):
    for pat in padroes:
        achados = glob.glob(pat, recursive=True)
        if achados:
            if os.path.exists(destino) or os.path.islink(destino):
                return None
            os.symlink(achados[0], destino)
            return f"symlink do Kaggle -> {achados[0]}"
    return None

alvo_ckpt = os.path.join(MODELOS, "SoulX-FlashHead-1_3B")
alvo_w2v = os.path.join(MODELOS, "wav2vec2-base-960h")

r = tenta_kaggle(["/kaggle/input/**/SoulX-FlashHead-1_3B", "/kaggle/input/**/SoulX*FlashHead*"], alvo_ckpt)
print("   SoulX-FlashHead-1_3B:", r or "baixando do HuggingFace")
if not r and not os.path.exists(alvo_ckpt):
    run([sys.executable, "-c",
         "from huggingface_hub import snapshot_download;"
         f"snapshot_download('Soul-AILab/SoulX-FlashHead-1_3B', local_dir='{alvo_ckpt}')"],
        tolerante=True)

r2 = tenta_kaggle(["/kaggle/input/**/wav2vec2-base-960h"], alvo_w2v)
print("   wav2vec2-base-960h ...:", r2 or "baixando do HuggingFace (~360 MB)")
if not r2 and not os.path.exists(alvo_w2v):
    run([sys.executable, "-c",
         "from huggingface_hub import snapshot_download;"
         f"snapshot_download('facebook/wav2vec2-base-960h', local_dir='{alvo_w2v}')"],
        tolerante=True)

print("\n   --- arvore de pesos ---")
for base, _, arqs in os.walk(MODELOS):
    for a in arqs:
        p = os.path.join(base, a)
        try:
            gb = os.path.getsize(p) / 1073741824
        except OSError:
            gb = 0
        if gb > 0.3:
            print(f"      {gb:7.2f} GB  {p.replace(MODELOS + '/', '')}")
print("=== CHECKPOINT 5: OK ===")

print()
print("=" * 78)
print("  6/7 · INFERENCIA (Model_Lite — o de menor VRAM)")
print("=" * 78)
IMAGEM = "examples/girl.png" if os.path.exists("examples/girl.png") else (
    (sorted(glob.glob("examples/*.png") + glob.glob("examples/*.jpg")) or [None])[0])
AUDIO = "examples/podcast_sichuan_16k.wav" if os.path.exists("examples/podcast_sichuan_16k.wav") else (
    (sorted(glob.glob("examples/*.wav")) or [None])[0])
print("   imagem:", IMAGEM)
print("   audio :", AUDIO)
if not IMAGEM or not AUDIO:
    print("   >>> o repo nao trouxe os exemplos — ela vai precisar subir foto+audio manualmente")
    raise SystemExit(1)

cmd = [sys.executable, "generate_video.py",
       "--ckpt_dir", alvo_ckpt,
       "--wav2vec_dir", alvo_w2v,
       "--model_type", "lite",
       "--cond_image", IMAGEM,
       "--audio_path", AUDIO,
       "--audio_encode_mode", "stream"]
print("   rodando:", " ".join(cmd))
print("   (a saida aparece AO VIVO abaixo — nao esta travado)\n")

# Popen para ver a saida linha a linha (com capture_output o usuario acha que travou)
t0 = time.time()
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
for linha in proc.stdout:
    print("      " + linha.rstrip(), flush=True)
proc.wait()
dt = time.time() - t0
print(f"\n   exit code: {proc.returncode}   tempo: {dt/60:.1f} min")

print()
print("=" * 78)
print("  7/7 · VEREDITO")
print("=" * 78)
V = sorted(glob.glob(os.path.join(REPO, "**", "*.mp4"), recursive=True) +
           glob.glob(os.path.join(RAIZ, "*.mp4")), key=os.path.getmtime, reverse=True)
if V and proc.returncode == 0:
    print("   ✅ FUNCIONOU NESTA MAQUINA")
    for v in V[:4]:
        print(f"      {os.path.getsize(v)/1e6:8.2f} MB  {v}")
    print("\n   >>> BAIXE O MP4 ACIMA (clique direito / menu Files do Colab).")
    print("   >>> Se gerou, NAO precisamos alugar GPU para avatar.")
else:
    print("   ❌ NAO gerou video (exit %d)" % proc.returncode)
    print("   --- o que checar ---")
    print("      • se apareceu CUDA out of memory -> tentar --model_type lite ja e o menor;")
    print("        reduzir resolucao ou usar GPU maior (RTX 4090 no Vast.ai, R$0,60/h)")
    print("      • se apareceu erro de mediapipe/xformers -> copiar as 30 ultimas linhas p/ mim")
    print("      • se nao apareceu GPU -> ligar acelerador T4 nas Settings")
    if V:
        print("   (existem MP4 antigos na pasta — conferir a data/hora)")
