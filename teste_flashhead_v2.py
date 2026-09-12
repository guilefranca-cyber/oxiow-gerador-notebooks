#!/usr/bin/env python3
"""
TESTE 1 (v2) — SoulX-FlashHead na GPU GRATIS DO KAGGLE/COLAB
===========================================================
CORRECAO DA v1: a v1 instalava "torch 2.7.1 (o que o repo pede)".
O requirements.txt OFICIAL NAO TEM TORCH. Baixar o torch quebrou o torchaudio
do Colab -> "undefined symbol: torch_library_impl". A v2 NAO TOCA NO TORCH.

Regras desta versao:
  * NUNCA instalar/baixar torch ou torchaudio
  * se o torchaudio estiver quebrado, CONSERTAR (reinstalar na versao do torch atual)
  * xformers incompativel = OK (o repo cai no SDPA do torch)
  * mediapipe sem wheel para 3.13 = OK (shim; nao usamos face_crop)
"""
import os, sys, subprocess, glob, time, shutil, re, textwrap

RAIZ = "/kaggle/working" if os.path.isdir("/kaggle/working") else "/content"
os.chdir(RAIZ)
REPO = os.path.join(RAIZ, "flashhead")


def run(cmd, cwd=None, timeout=5400, mostrar=True, tolerante=False):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd,
                       capture_output=True, text=True, timeout=timeout)
    out = (r.stdout or "") + (r.stderr or "")
    if mostrar:
        print(out[-4000:], flush=True)
    if r.returncode != 0 and not tolerante:
        print(f"\n   >>> FALHOU (exit {r.returncode})")
        raise SystemExit(1)
    return r.returncode, out


print("=" * 78)
print("  1/7 · AMBIENTE")
print("=" * 78)
gpu = "nenhuma"; vram = 0
try:
    rc, o = run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", mostrar=False)
    L = [x for x in o.strip().splitlines() if x.strip()]
    if L:
        gpu = L[0]
        m = re.search(r"(\d+)\s*MiB", gpu)
        if m: vram = int(m.group(1))
except Exception as e:
    print("   nvidia-smi:", e)

import torch
cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
print(f"   GPU .........: {gpu}")
print(f"   VRAM ........: {vram} MiB ({vram/1024:.2f} GB)")
print(f"   capability ..: {cap[0]}.{cap[1]}")
print(f"   torch .......: {torch.__version__}   <-- NAO VAMOS MEXER NESTE")
try:
    import torchaudio
    print(f"   torchaudio ..: {torchaudio.__version__}")
except Exception as e:
    print(f"   torchaudio ..: QUEBRADO ({str(e)[:90]})")

# ---- CHECAGEM QUE A v1 NAO FEZ: o torchaudio carrega? ----
SAUDAVEL = True
try:
    import torchaudio  # noqa
    import torchaudio.functional  # forca carregar a lib
except Exception as e:
    SAUDAVEL = False
    print(f"\n   >>> torchaudio NAO carrega: {str(e)[:150]}")
    print("   >>> consertando: reinstalando na versao EXATA do torch atual")

if not SAUDAVEL:
    ver = torch.__version__.split("+")[0]
    run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
         f"torchaudio=={ver}"], tolerante=True)
    try:
        import importlib, torchaudio
        importlib.reload(torchaudio)
        print("   >>> torchaudio consertado")
    except Exception as e:
        print(f"   >>> ainda quebrado: {str(e)[:120]}")
        print("   >>> ultimo recurso: REMOVER o torchaudio (o transformers cai no soundfile)")
        run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"],
            tolerante=True, mostrar=False)

with open("/proc/meminfo") as f:
    for L in f:
        if L.startswith("MemTotal"):
            print("   RAM .........: %.1f GB" % (int(L.split()[1])/1e6)); break
print(f"   disco livre .: {shutil.disk_usage(RAIZ).free/1e9:.1f} GB")
if not torch.cuda.is_available():
    print("\n   >>> SEM GPU. Ligue a aceleradora e rode de novo.")
    raise SystemExit(1)
print("=== CHECKPOINT 1: OK ===")

print()
print("=" * 78)
print("  2/7 · REPO OFICIAL")
print("=" * 78)
if not os.path.isdir(REPO):
    run(f"git clone --depth 1 https://github.com/Soul-AILab/SoulX-FlashHead.git {REPO}")
os.chdir(REPO)
for f in ("generate_video.py", "requirements.txt", "examples/girl.png"):
    print(f"   {f:26s}: {'OK' if os.path.exists(f) else 'AUSENTE'}")
ex = glob.glob("examples/*")
print(f"   exemplos no repo: {[os.path.basename(x) for x in ex]}")
print("=== CHECKPOINT 2: OK ===")

print()
print("=" * 78)
print("  3/7 · DEPENDENCIAS (SEM TOCAR NO TORCH)")
print("=" * 78)
# trava o torch/torchaudio na versao instalada -> pip NAO pode trocar
trava = os.path.join(RAIZ, "trava.txt")
with open(trava, "w") as f:
    f.write(f"torch=={torch.__version__}\n")
    f.write(f"torchaudio=={getattr(sys.modules.get('torchaudio'), '__version__', torch.__version__)}\n")
print(f"   trava escrita: torch=={torch.__version__} (+torchaudio)")
print("   (isso impede o pip de baixar outra versao por dependencia)")

rc, _ = run([sys.executable, "-m", "pip", "install", "-q", "-c", trava,
             "-r", "requirements.txt"], tolerante=True, mostrar=False)
print("   requirements:", "OK" if rc == 0 else "algum pacote falhou (seguindo)")

faltam = ["opencv-python>=4.12.0.88", "diffusers>=0.34.0", "transformers==4.57.3",
          "tokenizers>=0.20.3", "accelerate>=1.8.1", "tqdm", "imageio", "easydict",
          "ftfy", "imageio-ffmpeg", "scikit-image", "loguru", "gradio==5.50.0",
          "xfuser>=0.4.3", "pyloudnorm", "decord", "librosa", "flask"]
for p in faltam:
    rc, _ = run([sys.executable, "-m", "pip", "install", "-q", "-c", trava, p],
                tolerante=True, mostrar=False)
    print(f"   {'OK  ' if rc == 0 else 'FALHOU'} {p}")

# xformers: opcional (o repo cai no SDPA)
rc, _ = run([sys.executable, "-m", "pip", "install", "-q", "-c", trava, "xformers==0.0.31"],
            tolerante=True, mostrar=False)
print(f"   {'OK  ' if rc == 0 else 'NAO INSTALOU (ok, usa SDPA)'} xformers==0.0.31")

# mediapipe: sem wheel p/ 3.13 -> SHIM
print(f"   python deste ambiente: {sys.version.split()[0]}")
rc, _ = run([sys.executable, "-m", "pip", "install", "-q", "mediapipe==0.10.9"],
            tolerante=True, mostrar=False)
try:
    import mediapipe  # noqa
    print("   mediapipe ...: OK")
except Exception:
    print("   mediapipe ...: SHIM (nao usamos face_crop; import passa, detector avisa)")
    shim_dir = os.path.join(RAIZ, "_shim")
    os.makedirs(shim_dir, exist_ok=True)
    with open(os.path.join(shim_dir, "mediapipe.py"), "w") as f:
        f.write(textwrap.dedent('''
            """Shim: o repo importa mediapipe so para face_crop, que NAO usamos."""
            __version__ = "0.0.0-shim"
            class _Indisponivel:
                def __init__(self, *a, **k):
                    raise RuntimeError(
                        "mediapipe indisponivel neste ambiente. Este teste NAO usa "
                        "face_crop (--use_face_crop). Rode sem essa flag.")
            class solutions:
                face_detection = _Indisponivel
            def __getattr__(name):
                raise RuntimeError(f"mediapipe.{name} indisponivel (shim ativo)")
        '''))
    if shim_dir not in sys.path:
        sys.path.insert(0, shim_dir)
    os.environ["PYTHONPATH"] = shim_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
    print(f"   shim em: {shim_dir}")

print("\n   --- conferindo o que ficou ---")
for mod in ["torch", "torchaudio", "cv2", "diffusers", "transformers", "librosa", "decord",
            "easydict", "pyloudnorm", "mediapipe"]:
    try:
        __import__(mod)
        print(f"      {mod:14s} OK")
    except Exception as e:
        print(f"      {mod:14s} falhou ({str(e)[:60]})")
print("=== CHECKPOINT 3: OK ===")

print()
print("=" * 78)
print("  4/7 · PESOS (~14,7 GB — o download principal)")
print("=" * 78)
MODELOS = os.path.join(REPO, "models")
os.makedirs(MODELOS, exist_ok=True)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

alvo = {"SoulX-FlashHead-1_3B": "Soul-AILab/SoulX-FlashHead-1_3B",
        "wav2vec2-base-960h": "facebook/wav2vec2-base-960h"}
for pasta, repo_hf in alvo.items():
    destino = os.path.join(MODELOS, pasta)
    ja = glob.glob(os.path.join(destino, "**", "*.safetensors"), recursive=True) + \
         glob.glob(os.path.join(destino, "**", "*.pth"), recursive=True)
    if len(ja) >= 3:
        print(f"   {pasta}: JA BAIXADO ({len(ja)} arquivos) — pulando")
        continue
    print(f"   {pasta}: baixando de {repo_hf}")
    code = ("from huggingface_hub import snapshot_download\n"
            f"snapshot_download(repo_id={repo_hf!r}, local_dir={destino!r})\n"
            "print('fim')\n")
    run([sys.executable, "-c", code], tolerante=True)

print("\n   --- arvore de pesos ---")
for f in sorted(glob.glob(os.path.join(MODELOS, "**", "*"), recursive=True)):
    if os.path.isfile(f) and os.path.getsize(f) > 50_000_000:
        print(f"      {os.path.getsize(f)/1e9:6.2f} GB  {os.path.relpath(f, MODELOS)}")
print("=== CHECKPOINT 4: OK ===")

print()
print("=" * 78)
print("  5/7 · PATCH DE COMPATIBILIDADE (se necessario)")
print("=" * 78)
# o repo pode importar torchaudio explicitamente; garantimos que home/workspace
# apareca no sys.path para o shim, e que o SDPA seja usado se xformers falhar
patch = os.path.join(REPO, "sitecustomize.py")
with open(patch, "w") as f:
    f.write(textwrap.dedent(f'''
        # garante o shim do mediapipe e evita que o xformers quebrado derrube o run
        import os, sys
        _shim = {os.path.join(RAIZ, "_shim")!r}
        if os.path.isdir(_shim) and _shim not in sys.path:
            sys.path.insert(0, _shim)
        os.environ.setdefault("XFORMERS_DISABLED", "1")
        os.environ.setdefault("ATTENTION_BACKEND", "sdpa")
    '''))
os.environ["PYTHONPATH"] = os.pathsep.join(
    [os.path.join(RAIZ, "_shim"), REPO, os.environ.get("PYTHONPATH", "")])
print(f"   sitecustomize escrito em {patch}")

# mediapipe precisa importar mesmo com o shim
rc, o = run([sys.executable, "-c", "import mediapipe; print('mediapipe import OK')"],
            tolerante=True, mostrar=False)
print("   teste do import:", ("OK" if "OK" in o else o.strip()[:120]))

rc, o = run([sys.executable, "-c", "import torchaudio; print('torchaudio OK')"],
            tolerante=True, mostrar=False)
print("   torchaudio ......:", ("OK" if "OK" in o else o.strip()[-160:]))
print("=== CHECKPOINT 5: OK ===")

print()
print("=" * 78)
print("  6/7 · INFERENCIA (Model_Lite — o de menor VRAM)")
print("=" * 78)
img = "examples/girl.png"
aud = "examples/podcast_sichuan_16k.wav"
ck = os.path.join(MODELOS, "SoulX-FlashHead-1_3B")
w2v = os.path.join(MODELOS, "wav2vec2-base-960h")
print(f"   imagem: {img}\n   audio : {aud}")
cmd = [sys.executable, "generate_video.py", "--ckpt_dir", ck, "--wav2vec_dir", w2v,
       "--model_type", "lite", "--cond_image", img, "--audio_path", aud,
       "--audio_encode_mode", "stream"]
print("   rodando:", " ".join(cmd))
print("   (a saida aparece AO VIVO abaixo — nao esta travado)\n")

env = dict(os.environ)
env["PYTHONPATH"] = os.pathsep.join([os.path.join(RAIZ, "_shim"), REPO,
                                     env.get("PYTHONPATH", "")])
env["PYTHONUNBUFFERED"] = "1"
t0 = time.time()
p = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     text=True, bufsize=1, env=env)
for linha in p.stdout:
    print("      " + linha.rstrip(), flush=True)
p.wait()
print(f"\n   exit: {p.returncode}   tempo: {(time.time()-t0)/60:.1f} min")
print("=== CHECKPOINT 6: OK ===")

print()
print("=" * 78)
print("  7/7 · VEREDITO")
print("=" * 78)
vids = sorted(set(glob.glob(os.path.join(REPO, "**", "*.mp4"), recursive=True) +
                   glob.glob(os.path.join(RAIZ, "**", "*.mp4"), recursive=True)),
              key=os.path.getmtime, reverse=True)
if vids:
    for v in vids[:6]:
        print(f"   {os.path.getsize(v)/1e6:8.2f} MB  {v}")
    print(f"\n   tamanho do mais novo: {os.path.getsize(vids[0])/1e6:.2f} MB")
if vids and p.returncode == 0:
    print("   ✅ GEROU VIDEO — o FlashHead roda nesta GPU")
elif vids:
    print("   🟡 gerou video, mas o processo saiu com erro — o mp4 pode valer")
else:
    print("   ❌ nao gerou video")
    print("\n   --- o que me mandar de volta ---")
    print("      as 40 ultimas linhas e o CHECKPOINT onde parou")
    print("      • undefined symbol torch/torchaudio -> a trava falhou; me avise")
    print("      • CUDA out of memory -> ja e o Lite; reduzir resolucao")
    print("      • mediapipe -> rodar SEM --use_face_crop (ja e o default aqui)")
