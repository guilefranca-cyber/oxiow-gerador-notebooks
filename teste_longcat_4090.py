#!/usr/bin/env python3
"""
TESTE 4 — LONGCAT AVATAR 1.5 numa RTX 4090 (Vast.ai) com o NOSSO critério de qualidade
=====================================================================================
Os testes 1-3 rodaram o SoulX-FlashHead (rápido, grátis). Este roda o LONGAT, que é o
único dos dois TREINADO para o que a OXIOW precisa:

  Do relatório técnico do LongCat:
   - "preserve natural facial stillness ... suppressing unintended mouth movement"
   - "modeling gaze shifts, head motion, posture changes"
   - "three specialized data pipelines for multi-person, SILENT, and emotion-specific data"
   - v1.0 herdado: "Disentangled Unconditional Guidance: natural micro-movements
     (blinking, breathing, posture) during silent segments — silence does not freeze"

  O default do modelo é EXPRESSIVO. Este script INVERTE isso por prompt:
   positivo: calmo, sutil, boca discreta, pisca suave, respiração, luz ambiente
   negativo: expressão teatral, boca aberta, gesto exagerado, luz de estúdio, pele plástica

Alvo: RTX 4090 24 GB (24 GB de VRAM cabem o quanto_int8 + offload leve).
Entrega: 9:16 com metadado limpo, pronto para comparar com o do FlashHead.
"""
import os, sys, subprocess, glob, time, shutil, json, textwrap

RAIZ = "/workspace" if os.path.isdir("/workspace") else os.getcwd()
os.chdir(RAIZ)
REPO = os.path.join(RAIZ, "WanGP")
TEMP = os.path.join(RAIZ, "_temp"); os.makedirs(TEMP, exist_ok=True)
SAIDA = os.path.join(RAIZ, "saida_avatar"); os.makedirs(SAIDA, exist_ok=True)

BASE = "https://raw.githubusercontent.com/guilefranca-cyber/oxiow-gerador-notebooks/e52ac9ab969028992b69a3b1db27362377381fb2"
ROSTO_URL = f"{BASE}/assets/avatar-dona-maria.png"

FALA = ("Hi, I'm Margaret. I'm sixty-two years old. For years my feet ached "
        "every single evening. I tried creams, I tried soaking them, and nothing "
        "really helped. Then a friend told me about something simple. "
        "If your feet bother you too, stay with me for a moment.")

# ── O PROMPT É O CORAÇÃO DESTE TESTE (critério OXIOW: humano normal, sutil) ──
PROMPT = ("A calm ordinary woman sitting still and speaking quietly to the camera. "
          "Natural subtle lip movement, gentle slow blinking, steady breathing, "
          "minimal head motion, relaxed shoulders. Recorded on a handheld phone in "
          "ambient indoor light. Natural skin texture with visible pores and fine lines, "
          "no makeup gloss, no studio lighting. Documentary, unpolished, real person.")
NEGATIVO = ("theatrical expression, exaggerated gestures, wide open mouth, dramatic "
            "movement, shouting, overacting, studio lighting, beauty filter, plastic "
            "smooth skin, airbrushed, glossy, cartoon, anime, 3d render, cgi, "
            "distorted mouth, extra teeth, blurred details, low quality")
print("   prompt positivo:", PROMPT[:100], "...")
print("   prompt negativo:", NEGATIVO[:100], "...")


def run(cmd, cwd=None, timeout=5400, mostrar=True, tolerante=False, env=None):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, env=env,
                       capture_output=True, text=True, timeout=timeout)
    out = (r.stdout or "") + (r.stderr or "")
    if mostrar:
        print(out[-3000:], flush=True)
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
print(f"   torch .......: {torch.__version__}   <-- NAO MEXER (licao do teste 1)")
cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
print(f"   capability ..: {cap[0]}.{cap[1]}")
try:
    import torchaudio; print(f"   torchaudio ..: {torchaudio.__version__}")
except Exception as e:
    print(f"   torchaudio ..: {str(e)[:70]}")
if not torch.cuda.is_available():
    print("   >>> SEM GPU"); raise SystemExit(1)
print(f"   disco livre .: {shutil.disk_usage(RAIZ).free/1e9:.1f} GB")
# perfil conforme a VRAM (menos offload = mais rapido)
if vram >= 40000:   PERFIL = "2"
elif vram >= 22000: PERFIL = "3"
elif vram >= 15000: PERFIL = "4"
else:               PERFIL = "4"
print(f"   perfil WanGP escolhido: {PERFIL}  (24GB -> 3, 15GB -> 4)")
print("=== CHECKPOINT 1: OK ===")

print()
print("=" * 78)
print("  2/8 · WANGP (o motor que le quantizado + faz offload)")
print("=" * 78)
if not os.path.isdir(REPO):
    run(f"git clone --depth 1 https://github.com/deepbeepmeep/Wan2GP.git {REPO}")
os.chdir(REPO)
for f in ("wgp.py", "shared/api.py", "defaults/longcat_avatar_v1_5.json"):
    print(f"   {f:34s}: {'OK' if os.path.exists(f) else 'AUSENTE'}")
if not os.path.exists("shared/api.py"):
    print("   >>> repo sem a API headless"); raise SystemExit(1)
print("=== CHECKPOINT 2: OK ===")

print()
print("=" * 78)
print("  3/8 · DEPENDENCIAS (com a trava do torch, licão aprendida)")
print("=" * 78)
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
        print("   >>> não consertou; seguir sem torchaudio pode falhar")
trava = os.path.join(TEMP, "trava.txt")
with open(trava, "w") as f:
    f.write(f"torch=={torch.__version__}\n")
    try:
        import torchaudio as _t; f.write(f"torchaudio=={_t.__version__}\n")
    except Exception:
        f.write(f"torchaudio=={torch.__version__}\n")
print(f"   trava: torch=={torch.__version__}")
for p in ["mmgp", "torchcodec", "gguf"]:
    try:
        __import__(p); print(f"   {p:12s}: OK")
    except Exception:
        print(f"   {p:12s}: instalando")
        run([sys.executable, "-m", "pip", "install", "-q", "-c", trava, p],
            tolerante=True, mostrar=False)
for p in ["transformers", "diffusers"]:
    try:
        __import__(p); print(f"   {p:12s}: OK")
    except Exception:
        run([sys.executable, "-m", "pip", "install", "-q", "-c", trava, p],
            tolerante=True, mostrar=False)
rc, _ = run([sys.executable, "-m", "pip", "install", "-q", "-c", trava,
             "-r", "requirements.txt"], tolerante=True, mostrar=False)
print("   requirements:", "OK" if rc == 0 else "algum falhou (seguindo)")
if not ta_ok():
    run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
         f"torchaudio=={torch.__version__.split('+')[0]}"], tolerante=True, mostrar=False)
print("   torchaudio final:", "OK" if ta_ok() else "AINDA QUEBRADO")
print("=== CHECKPOINT 3: OK ===")

print()
print("=" * 78)
print("  4/8 · ** NOSSO ROSTO **")
print("=" * 78)
ROSTO = os.path.join(TEMP, "rosto.png")
run(f"curl -sL -o {ROSTO} {ROSTO_URL}", tolerante=True, mostrar=False)
if not (os.path.exists(ROSTO) and os.path.getsize(ROSTO) > 100_000):
    print("   >>> download falhou")
    raise SystemExit(1)
print(f"   rosto NOSSO: {ROSTO}  ({os.path.getsize(ROSTO)/1024:.0f} KB)")
run(f"identify -format '%wx%h' {ROSTO}", tolerante=True)
print("=== CHECKPOINT 4: OK ===")

print()
print("=" * 78)
print("  5/8 · ** NOSSA FALA **")
print("=" * 78)
run([sys.executable, "-m", "pip", "install", "-q", "edge-tts"], tolerante=True, mostrar=False)
AUDIO = os.path.join(TEMP, "fala-margaret.mp3")
for v in ["en-US-AriaNeural", "en-US-JennyNeural", "en-US-MichelleNeural"]:
    code = ("import asyncio, edge_tts\n"
            "async def m():\n"
            f"    c = edge_tts.Communicate({FALA!r}, {v!r})\n"
            f"    await c.save({AUDIO!r})\n"
            "asyncio.run(m())\n")
    run([sys.executable, "-c", code], tolerante=True, mostrar=False)
    if os.path.exists(AUDIO) and os.path.getsize(AUDIO) > 5000:
        print(f"   ✅ voz {v} — {os.path.getsize(AUDIO)/1024:.0f} KB")
        break
if not os.path.exists(AUDIO):
    print("   >>> edge-tts falhou"); raise SystemExit(1)
rc, o = run(f"ffprobe -v error -show_entries format=duration -of csv=p=0 {AUDIO}",
            tolerante=True, mostrar=False)
print(f"   duracao: {o.strip()[:20]} s")
print("=== CHECKPOINT 5: OK ===")

print()
print("=" * 78)
print("  6/8 · RODANDO O LONGCAT (o download de ~22 GB acontece AQUI na 1a vez)")
print("=" * 78)
SCRIPT = os.path.join(TEMP, "rodar_longcat.py")
with open(SCRIPT, "w") as f:
    f.write(textwrap.dedent(f'''
        import sys, os, json, time
        from pathlib import Path
        sys.path.insert(0, {REPO!r}); os.chdir({REPO!r})
        from shared.api import init
        print("[api] init com --attention sdpa --profile {PERFIL}", flush=True)
        sess = init(root=Path({REPO!r}),
                    cli_args=["--attention", "sdpa", "--profile", "{PERFIL}"])
        print("[api] sessao pronta", flush=True)
        M = "longcat_avatar_v1_5"
        cfg = sess.get_default_settings(M)
        print("[api] defaults do WanGP:", json.dumps(cfg, indent=2, default=str)[:2200], flush=True)
        cfg["model_type"] = M
        cfg["prompt"] = {PROMPT!r}
        cfg["negative_prompt"] = {NEGATIVO!r}
        cfg["num_inference_steps"] = 8
        cfg["sample_solver"] = "distill"
        cfg["video_length"] = 93
        for k in ("image_refs", "ref_images", "reference_images"):
            if k in cfg: cfg[k] = [{ROSTO!r}]
        cfg.setdefault("image_refs", [{ROSTO!r}])
        for k in ("audio_guide", "audio_guide_path", "audio"):
            if k in cfg: cfg[k] = {AUDIO!r}
        cfg.setdefault("audio_guide", {AUDIO!r})
        cfg["image_prompt_type"] = "S"
        cfg["save_path"] = {SAIDA!r}
        print("[api] settings finais:", json.dumps(cfg, indent=2, default=str)[:1800], flush=True)
        t0 = time.time()
        job = sess.submit_task(cfg)
        for ev in job.events.iter(timeout=0.5):
            k = getattr(ev, "kind", "")
            if k == "progress":
                d = ev.data
                print(f"[prog] {{getattr(d,'phase','')}} {{getattr(d,'progress',0)}} "
                      f"passo {{getattr(d,'current_step',0)}}/{{getattr(d,'total_steps',0)}} "
                      f"restam {{getattr(d,'remaining_time',None)}}", flush=True)
            elif k == "stream":
                print("[log]", str(getattr(ev.data, "text", ev.data))[:250], flush=True)
        r = job.result()
        print("\\n[RESULTADO] sucesso:", r.success, flush=True)
        print("[RESULTADO] arquivos:", getattr(r, "generated_files", None), flush=True)
        if not r.success:
            for e in getattr(r, "errors", []):
                print("[ERRO]", getattr(e, "message", e), flush=True)
        print(f"[TEMPO] {{(time.time()-t0)/60:.1f}} min", flush=True)
    '''))
env = dict(os.environ); env["PYTHONUNBUFFERED"] = "1"
t0 = time.time()
p = subprocess.Popen([sys.executable, SCRIPT], cwd=REPO, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
for linha in p.stdout:
    print("      " + linha.rstrip(), flush=True)
p.wait()
print(f"\n   exit: {p.returncode}   tempo: {(time.time()-t0)/60:.1f} min")
print("=== CHECKPOINT 6: OK ===")

print()
print("=" * 78)
print("  7/8 · PROCURANDO O VIDEO (só o criado AGORA)")
print("=" * 78)
achados = set()
for raiz in (SAIDA, TEMP, os.path.join(REPO, "outputs")):
    achados |= set(glob.glob(os.path.join(raiz, "**", "*.mp4"), recursive=True))
vids = sorted([v for v in achados if os.path.getmtime(v) >= t0],
              key=os.path.getmtime, reverse=True)
print(f"   novos: {len(vids)}  (de {len(achados)} mp4 no total)")
if not vids:
    print("   ❌ nenhum video NOVO gerado")
    print("      -> mandar o bloco '[api] defaults' e o bloco '[RESULTADO]'")
    raise SystemExit(1)
BRUTO = vids[0]
print(f"   ✅ {BRUTO}  ({os.path.getsize(BRUTO)/1e6:.2f} MB)")
run(f"ffprobe -v error -show_entries stream=width,height,codec_name,duration -of csv=p=0 {BRUTO}",
    tolerante=True)

print()
print("=" * 78)
print("  8/8 · 9:16 + METADADO LIMPO + VEREDITO")
print("=" * 78)
VERT = os.path.join(SAIDA, "longcat-9x16.mp4")
filtro = ("[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
          "crop=1080:1920,boxblur=40:5[bg];"
          "[0:v]scale=1080:-2[fg];"
          "[bg][fg]overlay=(W-w)/2:(H-h)/2")
run(["ffmpeg", "-y", "-loglevel", "error", "-i", BRUTO, "-filter_complex", filtro,
     "-map_metadata", "-1", "-c:v", "libx264", "-crf", "20", "-preset", "medium",
     "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", VERT], tolerante=True)
LIMPO = os.path.join(SAIDA, "longcat-9x16-limpo.mp4")
rc, _ = run(["ffmpeg", "-y", "-loglevel", "error", "-i", VERT, "-map_metadata", "-1",
             "-map_chapters", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact",
             "-flags:a", "+bitexact", "-c", "copy", LIMPO], tolerante=True)
if rc != 0 or not os.path.exists(LIMPO):
    shutil.copy2(VERT, LIMPO)
rc, o = run(f"ffprobe -v error -show_entries format_tags -of default=nw=1 {LIMPO}",
            tolerante=True, mostrar=False)
tags = [l for l in o.splitlines() if l.strip() and "=" in l]
print(f"   metadado: {'✅ LIMPO' if not tags else tags[:4]}")
print(f"   ✅ LONGCAT NA 4090 — CONCLUIDO")
print(f"   final: {LIMPO}  ({os.path.getsize(LIMPO)/1e6:.2f} MB)")
print(f"   rosto: NOSSO · prompt: SUTIL (critério OXIOW)")
print()
print("   >>> COMPARE com o do FlashHead (teste_avatar_do_zero, mesmo rosto e fala)")
print("       e me diga: boca mais sutil? pisca melhor? respiração? pele?")
