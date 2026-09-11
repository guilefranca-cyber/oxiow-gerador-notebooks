#!/usr/bin/env python3
"""
OXIOW — EchoMimicV3-Flash: patch + inferencia (celula 5c).

POR QUE ESTE ARQUIVO EXISTE
---------------------------
O repo EchoMimic tem um bug com diffusers >= 0.33: ele importa
`load_model_dict_into_meta` de `diffusers.models.modeling_utils`, mas a funcao
migrou para `diffusers.models.model_loading_utils`. O import falha, cai no
except, e o modo economico de memoria (low_cpu_mem_usage) e' desligado ->
tudo vai para a RAM -> o matador de memoria do Linux mata o processo (exit 137).

Este script injeta a funcao no lugar antigo ANTES de rodar a inferencia.
Precisa ser ARQUIVO (nao patch no kernel) porque a inferencia roda num
processo Python NOVO.

USO (uma linha, dentro do Colab):
    !cd /content && curl -sLO <url>/cell5c.py && python cell5c.py
"""
import os
import subprocess
import sys
import time
import glob
import json
import threading

# ─────────────────────────────────────────────────────────────────────────────
# 0. DETECTAR O AMBIENTE (Colab ou Kaggle) e localizar o repo
# ─────────────────────────────────────────────────────────────────────────────
CANDIDATOS = [
    '/content/em3/echomimic_v3',        # Colab (nosso)
    '/kaggle/temp/em3/echomimic_v3',    # Kaggle (nosso)
    '/kaggle/working/echomimic_v3',     # Kaggle (alternativo)
    'echomimic_v3',                     # cwd
]
REPO = next((c for c in CANDIDATOS if os.path.isdir(c)), None)
if REPO is None:
    achados = glob.glob('/content/**/echomimic_v3', recursive=True) + \
              glob.glob('/kaggle/**/echomimic_v3', recursive=True)
    REPO = achados[0] if achados else None

if REPO is None:
    print('>>> NAO ACHEI o repo echomimic_v3. Rode as celulas 1-4 primeiro.')
    sys.exit(1)

REPO = os.path.abspath(REPO)
FLASH = os.path.join(REPO, 'flash')
os.chdir(REPO)
os.makedirs('outputs', exist_ok=True)
print(f'REPO: {REPO}')
print(f'FLASH: {FLASH}')

# ─────────────────────────────────────────────────────────────────────────────
# 0b. MONTAR A ESTRUTURA ./flash (idempotente)
#     O repo espera: flash/Wan2.1-Fun-V1.1-1.3B-InP , flash/chinese-wav2vec2-base ,
#     flash/transformer/diffusion_pytorch_model.safetensors
# ─────────────────────────────────────────────────────────────────────────────
import shutil

BASE = os.environ.get('OXIOW_PESOS') or os.path.join(os.path.dirname(REPO), 'pesos')
# Se os pesos vierem de DATASET do Kaggle (/kaggle/input, somente leitura), usar de la:
# sobrevivem ao fim da sessao e nao custam download.
for _cand in glob.glob('/kaggle/input/*/echomimic-weights') + \
             glob.glob('/kaggle/input/*echomimic*/**/Wan2.1-Fun-V1.1-1.3B-InP',
                       recursive=True):
    _dir = _cand if os.path.basename(_cand) == 'echomimic-weights' else os.path.dirname(_cand)
    if os.path.isdir(os.path.join(_dir, 'Wan2.1-Fun-V1.1-1.3B-InP')):
        BASE = _dir
        print(f'>>> usando os pesos do DATASET: {BASE}')
        break
os.makedirs(FLASH, exist_ok=True)


def remover(p):
    """Apaga com o metodo certo: shutil.rmtree em symlink levanta OSError."""
    if os.path.islink(p):
        os.unlink(p)
    elif os.path.isdir(p):
        shutil.rmtree(p)
    elif os.path.exists(p):
        os.remove(p)


print(f'\n=== MONTANDO ./flash (origem: {BASE}) ===')
for nome in ['Wan2.1-Fun-V1.1-1.3B-InP', 'chinese-wav2vec2-base']:
    src = os.path.join(BASE, nome)
    dst = os.path.join(FLASH, nome)
    if not os.path.isdir(src):
        print(f'   !! falta a origem: {src}')
        continue
    if os.path.exists(dst) or os.path.islink(dst):
        remover(dst)
    try:
        os.symlink(src, dst)
        tam = sum(os.path.getsize(os.path.join(r, a))
                  for r, _d, fs in os.walk(src) for a in fs)
        print(f'   ligado {nome}  ({tam/1e9:.2f} GB)')
    except Exception as e:
        shutil.copytree(src, dst)
        print(f'   copiado {nome} (symlink falhou: {e})')

tf = os.path.join(FLASH, 'transformer')
os.makedirs(tf, exist_ok=True)
alvo = os.path.join(tf, 'diffusion_pytorch_model.safetensors')
if not os.path.exists(alvo):
    cand = [p for p in glob.glob(os.path.join(BASE, '**/diffusion_pytorch_model.safetensors'),
                                 recursive=True) if 'flash-pro' in p]
    if cand:
        # LINK, nao copy: copiar 3,73 GB travou no /kaggle/temp (o passo que
        # pendurou a execucao). O torch.load le por link igual.
        try:
            os.symlink(cand[0], alvo)
            print(f'   ligado transformer -> {os.path.basename(os.path.dirname(cand[0]))}')
        except Exception as e:
            print(f'   symlink falhou ({e}); copiando...')
            shutil.copy(cand[0], alvo)
    else:
        print(f'   !! nao achei os pesos do flash-pro em {BASE}')

# conferencia rapida
print('\n--- estrutura ./flash ---')
for item in ['Wan2.1-Fun-V1.1-1.3B-InP', 'chinese-wav2vec2-base', 'transformer']:
    p = os.path.join(FLASH, item)
    if os.path.isdir(p):
        tam = sum(os.path.getsize(os.path.join(r, a))
                  for r, _d, fs in os.walk(p) for a in fs)
    elif os.path.exists(p):
        tam = os.path.getsize(p)
    else:
        tam = 0
    ok = tam > 100_000_000
    print(f'   {"OK " if ok else "!! "} {tam/1e9:6.2f} GB  {item}')

# ─────────────────────────────────────────────────────────────────────────────
# 0c. AMBIENTE: quanto de RAM esta maquina tem (o que matou o Colab)
# ─────────────────────────────────────────────────────────────────────────────
try:
    with open('/proc/meminfo') as fh:
        info = {l.split(':')[0]: int(l.split()[1]) for l in fh if ':' in l}
    ram_gb = info['MemTotal'] / 1e6
    print(f'\n=== RAM DESTA MAQUINA: {ram_gb:.1f} GB ===')
    if ram_gb < 20:
        print('   >>> AVISO: menos de 20 GB. O modelo pede ~22 GB (T5 11,4 + CLIP 4,8 + resto).')
        print('   >>> O Colab gratis (12,7 GB) MORRE com exit 137. Use o Kaggle (33,7 GB).')
except Exception:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# 1. ONDE ESTA a funcao nesta versao do diffusers
# ─────────────────────────────────────────────────────────────────────────────
print('\n=== VERSAO E LOCALIZACAO ===')
r = subprocess.run('python -c "import diffusers,torch;print(diffusers.__version__, torch.__version__)"',
                   shell=True, capture_output=True, text=True)
print('   diffusers / torch:', (r.stdout or r.stderr).strip()[:120])

onde = None
for caminho in ['diffusers.models.model_loading_utils',
                'diffusers.models.modeling_utils']:
    r = subprocess.run(f'python -c "from {caminho} import load_model_dict_into_meta"',
                       shell=True, capture_output=True, text=True)
    ok = r.returncode == 0
    print(f'   {"OK   " if ok else "FALTA"} {caminho}.load_model_dict_into_meta')
    if ok and caminho.endswith('model_loading_utils'):
        onde = caminho

# ─────────────────────────────────────────────────────────────────────────────
# 2. ESCREVER O RUNNER COM O PATCH
#    (usa lista de linhas -> zero problema de aspas aninhadas)
# ─────────────────────────────────────────────────────────────────────────────
LINHAS_PATCH = [
    '"""Runner com patch: conserta low_cpu_mem_usage E o pico de RAM do torch.load."""',
    'import sys, runpy',
    'import diffusers',
    'import diffusers.models.modeling_utils as mu',
    'import torch',
    '',
    '# ── PATCH 1: religar low_cpu_mem_usage (o repo importa do modulo antigo) ──',
    '_f = None',
    'try:',
    '    from diffusers.models.model_loading_utils import load_model_dict_into_meta as _f',
    '    print("[patch1] load_model_dict_into_meta injetado de model_loading_utils")',
    'except Exception as e1:',
    '    try:',
    '        from diffusers.models.modeling_utils import load_model_dict_into_meta as _f',
    '        print("[patch1] ja existia no caminho antigo")',
    '    except Exception as e2:',
    '        print("[patch1] NAO ACHEI a funcao:", repr(e1), repr(e2))',
    'if _f is not None:',
    '    mu.load_model_dict_into_meta = _f',
    'try:',
    '    from diffusers.models.modeling_utils import load_model_dict_into_meta  # noqa',
    '    print("[patch1] VERIFICADO: o import do repo agora funciona")',
    'except Exception as e:',
    '    print("[patch1] AINDA FALHA:", repr(e))',
    '',
    '# ── PATCH 2: torch.load com mmap (o que o llama.cpp faz) ──',
    '# O T5 e um .pth de 11,36 GB. Sem mmap, torch.load le TUDO para a RAM e',
    '# o pico chega a ~22 GB -> o matador de memoria mata (exit 137).',
    '# Com mmap=True o arquivo e MAPEADO: a RAM so e usada pagina a pagina.',
    '_load_orig = torch.load',
    '_cont = {"n": 0}',
    'def _torch_load_mmap(*a, **kw):',
    '    if kw.get("map_location") is not None or "map_location" not in kw:',
    '        kw.setdefault("mmap", True)',
    '        try:',
    '            r = _load_orig(*a, **kw)',
    '            _cont["n"] += 1',
    '            print(f"[patch2] torch.load com mmap OK ({_cont[\'n\']}x)")',
    '            return r',
    '        except Exception as e:',
    '            print("[patch2] mmap falhou, caindo no modo normal:", repr(e)[:120])',
    '            kw.pop("mmap", None)',
    '    return _load_orig(*a, **kw)',
    'torch.load = _torch_load_mmap',
    'print("[patch2] torch.load com mmap=True instalado")',
    '',
    '# ── PATCH 3: aliviador de RAM rodando durante toda a carga ──',
    '# O coletor de lixo do Python segura tensores "mortos" do state_dict.',
    '# Forcar gc.collect() em paralelo devolve essa RAM enquanto o modelo carrega.',
    'import gc, threading, time',
    '_parar_gc = threading.Event()',
    'def _aliviar():',
    '    n = 0',
    '    while not _parar_gc.is_set():',
    '        gc.collect()',
    '        try:',
    '            import torch as _t',
    '            if _t.cuda.is_available():',
    '                _t.cuda.empty_cache()',
    '        except Exception:',
    '            pass',
    '        n += 1',
    '        if n % 20 == 0:',
    '            try:',
    '                with open("/proc/meminfo") as _f:',
    '                    _m = {l.split(":")[0]: int(l.split()[1]) for l in _f if ":" in l}',
    '                _usado = (_m["MemTotal"] - _m["MemAvailable"]) / 1e6',
    '                _tot = _m["MemTotal"] / 1e6',
    '                print(f"[ram] {_usado:.1f} / {_tot:.1f} GB usados", flush=True)',
    '            except Exception:',
    '                pass',
    '        time.sleep(0.5)',
    '_th = threading.Thread(target=_aliviar, daemon=True)',
    '_th.start()',
    'print("[patch3] aliviador de RAM ligado (gc.collect a cada 0,5 s)")',
    '',
    'sys.argv = ["infer_flash.py"] + sys.argv[1:]',
    'try:',
    '    runpy.run_path("infer_flash.py", run_name="__main__")',
    'finally:',
    '    _parar_gc.set()',
]
with open(os.path.join(REPO, 'run_patched.py'), 'w') as fh:
    fh.write('\n'.join(LINHAS_PATCH) + '\n')
print('\n[ok] run_patched.py escrito')

# ─────────────────────────────────────────────────────────────────────────────
# 3. ESCREVER O SCRIPT DE INFERENCIA (parametros leves p/ a T4)
# ─────────────────────────────────────────────────────────────────────────────
IMG = os.path.join(REPO, 'datasets/echomimicv3_demos/imgs/01.jpg')
AUD = os.path.join(REPO, 'datasets/echomimicv3_demos/audios/01.WAV')

ARG = [
    ('--image_path', f'"{IMG}"'),
    ('--audio_path', f'"{AUD}"'),
    ('--prompt', '"A person is speaking."'),
    ('--num_inference_steps', '8'),
    ('--config_path', '"config/config.yaml"'),
    ('--model_name', f'"{FLASH}/Wan2.1-Fun-V1.1-1.3B-InP"'),
    ('--ckpt_idx', '50000'),
    ('--transformer_path', f'"{FLASH}/transformer/diffusion_pytorch_model.safetensors"'),
    ('--save_path', '"outputs"'),
    ('--wav2vec_model_dir', f'"{FLASH}/chinese-wav2vec2-base"'),
    ('--sampler_name', '"Flow_Unipc"'),
    ('--video_length', '49'),
    ('--guidance_scale', '6.0'),
    ('--audio_guidance_scale', '2.0'),
    ('--audio_scale', '1.0'),
    ('--seed', '43'),
    ('--enable_teacache', ''),
    ('--teacache_threshold', '0.1'),
    ('--num_skip_start_steps', '5'),
    ('--riflex_k', '6'),
    ('--ulysses_degree', '1'),
    ('--ring_degree', '1'),
    ('--weight_dtype', '"float16"'),   # T4 = Turing: NAO tem bf16 em hardware
    ('--sample_size', '384 384'),      # leve, para o primeiro teste
    ('--fps', '25'),
    ('--add_prompt', '""'),
    ('--negative_prompt', '""'),
    ('--shift', '5.0'),
]
corpo = ' \\\n    '.join(f'{k} {v}'.rstrip() for k, v in ARG)
script = f'#!/bin/bash\ncd "{REPO}"\npython run_patched.py \\\n    {corpo}\n'
com = os.path.join(REPO, 'run_flash_patched.sh')
with open(com, 'w') as fh:
    fh.write(script)
os.chmod(com, 0o755)
print('[ok] run_flash_patched.sh escrito')
print('     (float16 porque a T4 nao tem bf16; 384x384 e video_length 49 = teste leve)')

# modo de conferencia: gera os arquivos e MOSTRA o comando, sem rodar a inferencia
if os.environ.get('OXIOW_DRY') == '1':
    print('\n' + '=' * 70)
    print('>>> MODO SECO (OXIOW_DRY=1) — mostrando os arquivos gerados <<<')
    print('=' * 70)
    print('\n--- run_flash_patched.sh ---')
    print(script)
    print('--- run_patched.py ---')
    print(open(os.path.join(REPO, 'run_patched.py')).read())
    print('>>> nada foi executado. para rodar de verdade, chame sem OXIOW_DRY <<<')
    sys.exit(0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. VIGIAR A RAM (o culpado do exit 137)
# ─────────────────────────────────────────────────────────────────────────────
parar = {'v': False}
pico = {'gb': 0.0}

def vigiar():
    while not parar['v']:
        try:
            with open('/proc/meminfo') as fh:
                info = {l.split(':')[0]: int(l.split()[1]) for l in fh if ':' in l}
            usado = (info['MemTotal'] - info['MemAvailable']) / 1e6
            pico['gb'] = max(pico['gb'], usado)
        except Exception:
            pass
        time.sleep(3)

# ─────────────────────────────────────────────────────────────────────────────
# 5. RODAR COM SAIDA AO VIVO
# ─────────────────────────────────────────────────────────────────────────────
th = threading.Thread(target=vigiar, daemon=True)
th.start()

print('\n' + '=' * 70)
print('>>> RODANDO A INFERENCIA (saida ao vivo abaixo) <<<')
print('=' * 70 + '\n')

t0 = time.time()
proc = subprocess.Popen(['bash', com], stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, bufsize=1)
linhas = []
try:
    for linha in proc.stdout:
        print(linha, end='')
        linhas.append(linha)
        if len(linhas) > 4000:
            linhas = linhas[-2000:]
except KeyboardInterrupt:
    proc.kill()
    print('\n>>> interrompido por voce <<<')
proc.wait()
parar['v'] = True

dur = round(time.time() - t0, 1)
rc = proc.returncode
saida = ''.join(linhas)

# ─────────────────────────────────────────────────────────────────────────────
# 6. VEREDITO
# ─────────────────────────────────────────────────────────────────────────────
vids = sorted(set(glob.glob(os.path.join(REPO, 'outputs/**/*.mp4'), recursive=True)
                 + glob.glob('**/*.mp4', recursive=True)))

print('\n' + '=' * 70)
print(f'  exit code ..........: {rc}')
print(f'  tempo ..............: {dur}s = {dur/60:.1f} min')
print(f'  pico de RAM usada ..: {pico["gb"]:.1f} GB')
print(f'  VIDEOS gerados .....: {len(vids)}')
for v in vids:
    print(f'      {os.path.getsize(v)/1e6:8.2f} MB  {v}')
print('=' * 70)

if rc == 137:
    print('\n>>> 137 = MATADOR DE MEMORIA (RAM), nao falta de VRAM.')
    print('>>> O Colab gratis tem ~12,7 GB de RAM; este modelo pede ~22 GB.')
    print('>>> SOLUCAO: rodar no KAGGLE, que tem ~33,7 GB de RAM e a MESMA T4.')
    print('>>>   1) abrir o notebook no kaggle.com')
    print('>>>   2) painel DIREITO -> Accelerator -> GPU T4 x2 -> Save')
    print('>>>   3) rodar as celulas 1-4 e depois este mesmo cell5c.py')
elif vids:
    print('\n>>> DEU CERTO! BAIXE O MP4 pelo painel de arquivos (a esquerda) <<<')
else:
    print('\n>>> NAO gerou video. O erro esta nas ultimas linhas acima.')

try:
    with open(os.path.join(os.path.dirname(REPO), 'resultado_echomimic.json'), 'w') as fh:
        json.dump({'sucesso': bool(vids), 'exit': rc, 'tempo_s': dur,
                   'pico_ram_gb': round(pico['gb'], 2), 'arquivos': vids,
                   'erro': None if vids else saida[-800:]}, fh, indent=1)
    print('[ok] resultado salvo em resultado_echomimic.json')
except Exception as e:
    print('[aviso] nao salvei o json:', e)
