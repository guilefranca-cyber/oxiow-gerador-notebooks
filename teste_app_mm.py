#!/usr/bin/env python3
"""
TESTE T4 — CAMINHO OFICIAL DO APP_MM (o dos 12 GB)
====================================================
O QUE ESTE SCRIPT PROVA:
  O README do EchoMimicV3 diz "12G VRAM is All YOU NEED — use o GradioUI app_mm.py".
  Este script instala e sobe ESSE app, sem patch nenhum nosso.

POR QUE O NOSSO run_flash.sh FALHOU (descoberto lendo o codigo oficial):
  - run_flash.sh  -> pipeline.to(device) empurra 21,89 GB de pesos. Numa T4 (14,56 GB) estoura. SEMPRE.
  - app_mm.py     -> usa `mmgp`: le a VRAM real, calcula o budget (90%) e reparte os modelos.
                     Gera o video EM BLOCOS (partial_video_length). EESTE e o caminho de 12 GB.

O QUE O USUARIO FAZ NO FIM:
  O Gradio imprime um link publico (--share). Voce abre no navegador, escolhe a foto e o audio
  e gera. E o teste definitivo: se gerar numa T4 gratis, nao precisamos alugar GPU.
"""
import os, sys, subprocess, shutil, time, json, glob

RAIZ = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
REPO = os.path.join(RAIZ, "em3_app")
os.chdir(RAIZ)

def run(cmd, cwd=None, timeout=3600, mostrar=True):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd,
                       capture_output=True, text=True, timeout=timeout)
    out = (r.stdout or "") + (r.stderr or "")
    if mostrar:
        print(out[-4000:], flush=True)
    return r.returncode, out

print("=" * 78)
print("  1/5 · AMBIENTE")
print("=" * 78)
gpu = "desconhecida"; vram = 0
try:
    rc, o = run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", mostrar=False)
    gpu = o.strip().splitlines()[0] if o.strip() else "nenhuma"
    import re
    m = re.search(r"(\d+)\s*MiB", gpu)
    if m: vram = int(m.group(1))
except Exception as e:
    print("   nvidia-smi falhou:", e)

import torch
print(f"   GPU .......: {gpu}")
print(f"   VRAM ......: {vram} MiB ({vram/1024:.2f} GB)")
print(f"   torch .....: {torch.__version__}")
print(f"   capability : {torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'sem cuda'}")
cap = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
print(f"   dtype .....: {'bfloat16' if cap >= 8 else 'float16'}  (o app usa isto sozinho)")
print(f"   raiz ......: {RAIZ}")
print(f"   RAM .......:")
with open("/proc/meminfo") as f:
    for L in f:
        if L.startswith("MemTotal"):
            print("               %.1f GB" % (int(L.split()[1])/1e6)); break
if not torch.cuda.is_available():
    print("   >>> SEM GPU. Ligue a GPU nas Settings e rode de novo.")
    sys.exit(1)
print("=== CHECKPOINT 1: OK ===")

print()
print("=" * 78)
print("  2/5 · REPO OFICIAL")
print("=" * 78)
if not os.path.isdir(REPO):
    rc, _ = run(f"git clone --depth 1 https://github.com/antgroup/echomimic_v3.git {REPO}")
    if rc != 0:
        print("   >>> clone FALHOU"); sys.exit(1)
os.chdir(REPO)
print("   repo em:", REPO)
print("   app_mm.py existe?", os.path.isfile("app_mm.py"))
print("=== CHECKPOINT 2: OK ===")

print()
print("=" * 78)
print("  3/5 · DEPENDENCIAS  (o requirements oficial + mmgp)")
print("=" * 78)
deficientes = []
if os.path.isfile("requirements.txt"):
    with open("requirements.txt") as f:
        linhas = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    print("   pacotes no requirements:", len(linhas))
    for p in linhas:
        nome = p.split("[")[0].split("==")[0].split(">=")[0].split("<")[0].strip()
        if not nome: continue
        mod = nome.replace("-", "_")
        try:
            __import__(mod)
        except Exception:
            deficientes.append(p)
    print("   faltando:", len(deficientes))
    print("   ", deficientes)
    if deficientes:
        run([sys.executable, "-m", "pip", "install", "-q"] + deficientes)

# mmgp = O ingrediente dos 12 GB
try:
    import mmgp
    print("   mmgp ......: OK", getattr(mmgp, "__version__", ""))
except Exception:
    print("   mmgp ......: instalando")
    run([sys.executable, "-m", "pip", "install", "-q", "mmgp"])
    try:
        import mmgp; print("   mmgp ......: OK agora")
    except Exception as e:
        print("   mmgp FALHOU:", e)
print("=== CHECKPOINT 3: OK ===")

print()
print("=" * 78)
print("  4/5 · PESOS")
print("=" * 78)
CONJUNTO = {
    "Wan2.1-Fun-V1.1-1.3B-InP": "alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP",
    "wav2vec2-base-960h":       "facebook/wav2vec2-base-960h",
}
MODELOS = os.path.join(REPO, "models")
os.makedirs(MODELOS, exist_ok=True)

# 1) o dataset do Kaggle, se estiver anexado (poupa o download de 19,81 GB)
origem_kaggle = None
for cand in glob.glob("/kaggle/input/*/echomimic-weights/Wan2.1-Fun-V1.1-1.3B-InP"):
    origem_kaggle = cand; break
for cand in glob.glob("/kaggle/input/**/Wan2.1-Fun-V1.1-1.3B-InP", recursive=True):
    origem_kaggle = cand; break

for nome, hf in CONJUNTO.items():
    destino = os.path.join(MODELOS, nome)
    if os.path.exists(destino) or os.path.islink(destino):
        print(f"   [ok] {nome} ja montado"); continue
    if nome.startswith("Wan2.1") and origem_kaggle:
        os.symlink(origem_kaggle, destino)
        n = len(os.listdir(destino))
        print(f"   [symlink] {nome} <- dataset do Kaggle ({n} arquivos, zero download)")
    else:
        print(f"   [download] {nome} (~{ '19,81 GB' if nome.startswith('Wan2.1') else '360 MB' })")
        run([sys.executable, "-c",
             f"from huggingface_hub import snapshot_download;"
             f"snapshot_download('{hf}', local_dir='{destino}')"])

# 2) o transformer do PREVIEW (app_mm usa a linhagem preview, nao a flash)
dt = os.path.join(MODELOS, "transformer", "diffusion_pytorch_model.safetensors")
if not os.path.isfile(dt):
    print("   [download] transformer do preview (~3,7 GB)")
    os.makedirs(os.path.dirname(dt), exist_ok=True)
    run([sys.executable, "-c",
         "from huggingface_hub import hf_hub_download;"
         "p=hf_hub_download('BadToBest/EchoMimicV3',"
         "'transformer/diffusion_pytorch_model.safetensors');"
         f"import shutil; shutil.copy(p, '{dt}')"])

print("\n   --- arvore de pesos montada ---")
for base, _, arqs in os.walk(MODELOS):
    for a in arqs:
        p = os.path.join(base, a)
        try:
            gb = os.path.getsize(p) / 1073741824
        except OSError:
            gb = 0
        if gb > 0.2:
            print(f"   {gb:7.2f} GB  {p.replace(MODELOS+'/', '')}")
print("=== CHECKPOINT 4: OK ===")

print()
print("=" * 78)
print("  5/5 · SUBINDO O APP OFICIAL (o dos 12 GB, com mmgp)")
print("=" * 78)
print("""
   O app vai imprimir duas linhas importantes:
     "自动调整最大显存占用为 XXXXXMB"  <- o budget que o mmgp calculou da SUA placa
     "Running on public URL: https://....gradio.live"  <- ABRA ESTE LINK

   No navegador: escolha a foto, o audio, e clique em gerar.
   Se gerar aqui numa T4 GRATIS, nao precisamos alugar GPU nenhuma.
""")
cmd = [sys.executable, "app_mm.py", "--share", "--server_name", "0.0.0.0"]
print("   rodando:", " ".join(cmd))
print("   (Ctrl+C para parar. Nao feche o navegador do Kaggle.)")
r = subprocess.run(cmd, cwd=REPO)
print("\n=== app encerrado (exit %d) ===" % r.returncode)
