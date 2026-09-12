#!/usr/bin/env python3
"""
TESTE 2 — LONGCAT AVATAR 1.5 via WanGP na GPU GRATIS DO KAGGLE
==============================================================
O MELHOR modelo aberto de avatar (MIT): do relatorio oficial, "competitive or
superior performance compared to leading closed-source systems (HeyGen,
OmniHuman, Kling Avatar 2.0)". Ganha 65,9% contra Kling Avatar 2.0.

POR QUE ELE NUNCA RODOU ANTES (3 paredes oficiais):
  1. "requires 2x A100-80GB"                       -> 160 GB, impossivel de graca
  2. exige flash_attn==2.7.4.post1                 -> NAO suporta T4 (sm_75)
  3. ComfyUI da comunidade travou (video preto)

O QUE MUDOU (descoberto lendo o codigo do WanGP em disco):
  * O WanGP implementa --attention sdpa  ("sdpa: Default, always available")
    -> substitui o flash-attn pelo SDPA do PyTorch. A parede 2 cai.
  * O WanGP usa mmgp com profile_type.LowRAM_LowVRAM (--profile 4)
    -> offload automatico, le a VRAM real. A parede 1 vira irrelevante.
  * Ele le GGUF nativamente (requirements: gguf==0.17.1).
  * Ja tem a arquitetura: models/longcat/modules/avatar/longcat_video_dit_avatar.py
    e o template defaults/longcat_avatar_v1_5.json (8 passos, 93 frames).
  * Os pesos que ja baixamos sao o repack QUANTIZADO do proprio autor do WanGP.

IDEMPOTENTE. 7 CHECKPOINTS. SAIDA AO VIVO.
"""
import os, sys, subprocess, glob, time, json, shutil, textwrap

RAIZ = "/kaggle/working" if os.path.isdir("/kaggle/working") else "/content"
os.chdir(RAIZ)
REPO = os.path.join(RAIZ, "WanGP")
# /kaggle/temp tem ~1 TB; /kaggle/working tem ~21 GB (nao cabe os ~22 GB de pesos)
TEMP = "/kaggle/temp" if os.path.isdir("/kaggle/temp") else os.path.join(RAIZ, "_temp")
os.makedirs(TEMP, exist_ok=True)


def run(cmd, cwd=None, timeout=3600, mostrar=True, tolerante=False):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd,
                       capture_output=True, text=True, timeout=timeout)
    out = (r.stdout or "") + (r.stderr or "")
    if mostrar:
        print(out[-5000:], flush=True)
    if r.returncode != 0 and not tolerante:
        print(f"\n   >>> FALHOU (exit {r.returncode})")
        raise SystemExit(1)
    return r.returncode, out


print("=" * 78)
print("  1/7 · AMBIENTE")
print("=" * 78)
gpu = "nenhuma"; vram = 0
import re
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
print(f"   GPU ..........: {gpu}")
print(f"   VRAM .........: {vram} MiB ({vram/1024:.2f} GB)")
print(f"   capability ...: {cap[0]}.{cap[1]}")
print(f"   torch ........: {torch.__version__}")
with open("/proc/meminfo") as f:
    for L in f:
        if L.startswith("MemTotal"):
            print("   RAM ..........: %.1f GB" % (int(L.split()[1])/1e6)); break
print(f"   working ......: {shutil.disk_usage(RAIZ).free/1e9:.1f} GB livres")
print(f"   temp (pesos) .: {TEMP} — {shutil.disk_usage(TEMP).free/1e9:.0f} GB livres")
if not torch.cuda.is_available():
    print("\n   >>> SEM GPU. Ligue a GPU e rode de novo.")
    raise SystemExit(1)
if vram and vram < 14000:
    print(f"   aviso: {vram/1024:.1f} GB de VRAM. O perfil 4 + sdpa deve aguentar, mas se der OOM")
    print("          reduzimos video_length de 93 para 49 frames.")
print("=== CHECKPOINT 1: OK ===")

print()
print("=" * 78)
print("  2/7 · REPO DO WANGP")
print("=" * 78)
if not os.path.isdir(REPO):
    run(f"git clone --depth 1 https://github.com/deepbeepmeep/Wan2GP.git {REPO}")
os.chdir(REPO)
print("   wgp.py .....:", "OK" if os.path.exists("wgp.py") else "AUSENTE")
print("   shared/api.py:", "OK" if os.path.exists("shared/api.py") else "AUSENTE")
print("   handler avatar:", "OK" if os.path.exists("models/longcat/modules/avatar/longcat_video_dit_avatar.py") else "AUSENTE")
print("   template .....:", "OK" if os.path.exists("defaults/longcat_avatar_v1_5.json") else "AUSENTE")
if not os.path.exists("shared/api.py"):
    print("   >>> repo sem a API headless — abortando")
    raise SystemExit(1)
print("=== CHECKPOINT 2: OK ===")

print()
print("=" * 78)
print("  3/7 · DEPENDENCIAS")
print("=" * 78)
try:
    import mmgp
    print("   mmgp .......: OK", getattr(mmgp, "__version__", ""))
except Exception:
    print("   mmgp .......: instalando")
    run([sys.executable, "-m", "pip", "install", "-q", "mmgp"], tolerante=True)
for mod in ["torchcodec", "gguf", "transformers", "diffusers"]:
    try:
        __import__(mod); print(f"   {mod:12s}: OK")
    except Exception:
        print(f"   {mod:12s}: instalando")
        run([sys.executable, "-m", "pip", "install", "-q", mod], tolerante=True)
print("   --- requirements do repo (tolerante) ---")
rc, _ = run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
            tolerante=True, mostrar=False)
print("   requirements:", "OK" if rc == 0 else "falhou em algum pacote (seguindo)")
print("=== CHECKPOINT 3: OK ===")

print()
print("=" * 78)
print("  4/7 · ENTRADAS (audio + rosto)")
print("=" * 78)
# AUDIO — edge-tts, gratis, sem chave
AUDIO = os.path.join(TEMP, "fala.mp3")
if not os.path.exists(AUDIO):
    print("   gerando o audio com edge-tts (gratis)...")
    run([sys.executable, "-m", "pip", "install", "-q", "edge-tts"], tolerante=True, mostrar=False)
    txt = ("Hello, this is a test of the LongCat avatar. "
           "If you can see my lips moving with these words, the pipeline works.")
    code = ("import asyncio, edge_tts\n"
            "async def m():\n"
            f"    c = edge_tts.Communicate({txt!r}, 'en-US-AriaNeural')\n"
            f"    await c.save({AUDIO!r})\n"
            "asyncio.run(m())\n")
    run([sys.executable, "-c", code], tolerante=True, mostrar=False)
print("   audio:", AUDIO if os.path.exists(AUDIO) else "FALHOU")

# ROSTO — tenta na ordem: dataset do Kaggle -> SoulX (verificado 200) -> nosso dona-maria
IMAGEM = None
candidatos = (
    glob.glob("/kaggle/input/**/girl.png", recursive=True) +
    glob.glob("/kaggle/input/**/dona*maria*.png", recursive=True) +
    glob.glob("/kaggle/input/**/*.jpg", recursive=True)[:3]
)
if candidatos:
    IMAGEM = candidatos[0]
    print("   rosto (do Kaggle):", IMAGEM)
if not IMAGEM:
    destino = os.path.join(TEMP, "rosto.png")
    if not os.path.exists(destino):
        for url in ["https://raw.githubusercontent.com/Soul-AILab/SoulX-FlashHead/main/examples/girl.png"]:
            rc, _ = run(f"curl -sL -o {destino} {url}", tolerante=True, mostrar=False)
            if os.path.exists(destino) and os.path.getsize(destino) > 10000:
                break
    if os.path.exists(destino) and os.path.getsize(destino) > 10000:
        IMAGEM = destino
        print("   rosto (baixado):", IMAGEM, f"{os.path.getsize(destino)/1024:.0f} KB")
if not IMAGEM:
    print("   >>> NAO consegui uma imagem de rosto.")
    print("       Solucao: anexe o dataset publico 'oxiowhub/dona-maria-white-bg-png' e rode de novo.")
    raise SystemExit(1)
print("=== CHECKPOINT 4: OK ===")

print()
print("=" * 78)
print("  5/7 · SUBINDO O WANGP (headless, --attention sdpa --profile 4)")
print("=" * 78)
SCRIPT = os.path.join(TEMP, "rodar_longcat.py")
with open(SCRIPT, "w") as f:
    f.write(textwrap.dedent(f'''
        import sys, os, json
        from pathlib import Path
        sys.path.insert(0, {REPO!r})
        os.chdir({REPO!r})

        from shared.api import init

        print("[api] init(...) com --attention sdpa --profile 4 (LowRAM_LowVRAM)", flush=True)
        sess = init(root=Path({REPO!r}), cli_args=["--attention", "sdpa", "--profile", "4"])
        print("[api] sessao pronta", flush=True)

        MODELO = "longcat_avatar_v1_5"
        cfg = sess.get_default_settings(MODELO)
        print("[api] defaults do proprio WanGP:", json.dumps(cfg, indent=2, default=str)[:2500], flush=True)

        cfg["model_type"] = MODELO
        cfg["prompt"] = "A person speaking naturally with expressive eye contact and accurate lip sync."
        cfg["negative_prompt"] = "Close-up, overexposed, static, blurred details, distorted mouth, extra teeth."
        cfg["num_inference_steps"] = 8
        cfg["sample_solver"] = "distill"
        cfg["video_length"] = 93

        # os nomes das chaves de imagem/audio vem dos defaults; preencher os provaveis
        # (o handler exige image_refs como LISTA e audio_guide como caminho)
        for k in ("image_refs", "ref_images", "reference_images"):
            if k in cfg: cfg[k] = [{IMAGEM!r}]
        cfg.setdefault("image_refs", [{IMAGEM!r}])
        for k in ("audio_guide", "audio_guide_path", "audio"):
            if k in cfg: cfg[k] = {AUDIO!r}
        cfg.setdefault("audio_guide", {AUDIO!r})
        if "image_prompt_type" in cfg: cfg["image_prompt_type"] = "S"
        else: cfg["image_prompt_type"] = "S"
        cfg["save_path"] = {TEMP!r}

        print("[api] settings finais:", json.dumps(cfg, indent=2, default=str)[:2000], flush=True)
        print("[api] submetendo o job (o download dos ~22 GB acontece AQUI na 1a vez)", flush=True)

        job = sess.submit_task(cfg)
        for ev in job.events.iter(timeout=0.5):
            k = getattr(ev, "kind", "")
            if k == "progress":
                d = ev.data
                print(f"[prog] {{getattr(d,'phase','')}} {{getattr(d,'progress',0)}} "
                      f"passo {{getattr(d,'current_step',0)}}/{{getattr(d,'total_steps',0)}} "
                      f"restam {{getattr(d,'remaining_time',None)}}", flush=True)
            elif k == "stream":
                print("[log]", str(getattr(ev.data, "text", ev.data))[:300], flush=True)

        r = job.result()
        print("\\n[RESULTADO] sucesso:", r.success, flush=True)
        if r.success:
            print("[RESULTADO] arquivos:", r.generated_files, flush=True)
        else:
            for e in r.errors:
                print("[ERRO]", getattr(e, "message", e), flush=True)
    '''))
print("   script da API escrito:", SCRIPT)
print("   rodando (a saida aparece AO VIVO):\n")
t0 = time.time()
p = subprocess.Popen([sys.executable, SCRIPT], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
for linha in p.stdout:
    print("      " + linha.rstrip(), flush=True)
p.wait()
print(f"\n   exit: {p.returncode}   tempo: {(time.time()-t0)/60:.1f} min")
print("=== CHECKPOINT 5: OK ===")

print()
print("=" * 78)
print("  6/7 · PROCURANDO O VIDEO")
print("=" * 78)
vids = sorted(set(
    glob.glob(os.path.join(TEMP, "**", "*.mp4"), recursive=True) +
    glob.glob(os.path.join(REPO, "outputs", "**", "*.mp4"), recursive=True) +
    glob.glob(os.path.join(RAIZ, "**", "*.mp4"), recursive=True)
), key=lambda v: os.path.getmtime(v), reverse=True)
if vids:
    for v in vids[:5]:
        print(f"   {os.path.getsize(v)/1e6:8.2f} MB  {v}")
else:
    print("   nenhum mp4 encontrado")

print()
print("=" * 78)
print("  7/7 · VEREDITO")
print("=" * 78)
if vids and p.returncode == 0:
    print("   ✅ O LONGCAT RODOU NESTA MAQUINA")
    print(f"   pegue o arquivo: {vids[0]}")
elif vids:
    print("   🟡 gerou video, mas o processo saiu com erro — o mp4 pode valer")
else:
    print("   ❌ nao gerou video")
    print("\n   --- o que me mandar de volta ---")
    print("      as 40 ultimas linhas da saida, e o numero do CHECKPOINT onde parou")
    print("      • CUDA out of memory -> baixar video_length de 93 para 49")
    print("      • erro no download    -> conferir espaco em", TEMP)
    print("      • erro de chave de settings -> me manda o bloco '[api] defaults do proprio WanGP'")
    print("        (ele imprime as chaves reais, e eu ajusto o script na hora)")
