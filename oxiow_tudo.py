#!/usr/bin/env python3
"""
OXIOW — EchoMimicV3-Flash: TUDO EM UM COMANDO.

Faz o trabalho das celulas 1-3 + o patch + a inferencia, sem depender de
rodar nada em ordem. Se a sessao reiniciar, roda esta linha e pronto.

USO (uma linha, no Kaggle ou Colab):
    !python /kaggle/temp/oxiow_tudo.py    (ou: curl -sLO <url> && python oxiow_tudo.py)

O que ele faz, em ordem:
  1. confere GPU (aborta em P100 — o PyTorch do Kaggle nao suporta sm_60)
  2. confere/usam os pesos: /kaggle/input (dataset) se existir, senao baixa do HF
  3. clona o repo EchoMimicV3
  4. instala as dependencias (lista COMPLETA do requirements oficial)
  5. monta ./flash (a estrutura que o run_flash.sh espera)
  6. injeta 3 patches: low_cpu_mem_usage, torch.load(mmap=True), aliviador de RAM
  7. roda a inferencia com saida AO VIVO
  8. diz se gerou video (le o campo 'sucesso', nao so o tempo)
"""
import os
import sys
import glob
import json
import shutil
import subprocess
import threading
import time

INICIO = time.time()


def sh(cmd, timeout=5400, quiet=False, mostrar=10):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    out = (p.stdout or '') + (p.stderr or '')
    if not quiet:
        linhas = [l for l in out.strip().splitlines() if l.strip()]
        if linhas:
            print('\n'.join(linhas[-mostrar:]), flush=True)
    return p.returncode, out


def bloco(titulo):
    print('\n' + '=' * 72)
    print(f'  {titulo}')
    print('=' * 72, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
bloco('1/8 · AMBIENTE')
# ─────────────────────────────────────────────────────────────────────────────
import torch

nome = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''
cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
with open('/proc/meminfo') as fh:
    _m = {l.split(':')[0]: int(l.split()[1]) for l in fh if ':' in l}
ram_gb = _m['MemTotal'] / 1e6

# o Kaggle usa /kaggle/temp (1,1 TB); o Colab usa /content
if os.path.isdir('/kaggle/temp'):
    RAIZ = '/kaggle/temp'
elif os.path.isdir('/content'):
    RAIZ = '/content'
else:
    RAIZ = os.path.expanduser('~')

print(f'GPU .......: {nome} | sm_{cap[0]}{cap[1]}')
print(f'RAM .......: {ram_gb:.1f} GB')
print(f'torch .....: {torch.__version__}')
print(f'raiz ......: {RAIZ}')

if 'P100' in nome or cap[0] < 7:
    print('\n' + '!' * 72)
    print(f'  GPU ERRADA: veio P100/sm_{cap[0]}. O PyTorch daqui NAO suporta (comeca em sm_70).')
    print('  ACAO: painel DIREITO -> Accelerator -> "GPU T4 x2" -> Save. Depois rode de novo.')
    print('  NAO faca push pela CLI: sobrescreve a escolha da interface.')
    print('!' * 72)
    sys.exit(1)

EM3 = os.path.join(RAIZ, 'em3')
REPO = os.path.join(EM3, 'echomimic_v3')
FLASH = os.path.join(REPO, 'flash')
os.makedirs(EM3, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
bloco('2/8 · PESOS (dataset do Kaggle, senao baixa do HuggingFace)')
# ─────────────────────────────────────────────────────────────────────────────
BASE = os.path.join(EM3, 'pesos')
achado = None
for c in glob.glob('/kaggle/input/*echomimic*/**/Wan2.1-Fun-V1.1-1.3B-InP', recursive=True):
    achado = os.path.dirname(c)
    break
if achado:
    BASE = achado
    print(f'usando o DATASET (instantaneo): {BASE}')
else:
    print('dataset nao anexado — baixando do HuggingFace (~28,5 GB, ~2-3 min)')
    sh('pip install -q huggingface_hub 2>&1 | tail -1', timeout=600, mostrar=2)
    from huggingface_hub import snapshot_download
    os.makedirs(BASE, exist_ok=True)
    TAREFAS = [
        ('Wan2.1-Fun-V1.1-1.3B-InP (T5 + CLIP + modelo)',
         'alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP', f'{BASE}/Wan2.1-Fun-V1.1-1.3B-InP'),
        ('chinese-wav2vec2-base', 'TencentGameMate/chinese-wav2vec2-base',
         f'{BASE}/chinese-wav2vec2-base'),
        ('EchoMimicV3 flash-pro', 'BadToBest/EchoMimicV3',
         f'{BASE}/echomimicv3-flash-pro'),
    ]
    for rot, repo, dest in TAREFAS:
        if os.path.isdir(dest) and sum(os.path.getsize(os.path.join(r, a))
                                       for r, _d, fs in os.walk(dest) for a in fs) > 1e8:
            print(f'   ja tem: {rot}')
            continue
        print(f'   baixando: {rot}', flush=True)
        try:
            # ignore_patterns (NAO allow_patterns — a licao do bug anterior)
            snapshot_download(repo_id=repo, local_dir=dest,
                              ignore_patterns=['*.md', '*.txt', '.gitattributes'],
                              max_workers=8)
        except Exception as e:
            print(f'   FALHOU: {type(e).__name__}: {str(e)[:200]}')

# ─────────────────────────────────────────────────────────────────────────────
bloco('3/8 · REPO')
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.isdir(REPO):
    sh(f'cd {EM3} && git clone --depth 1 https://github.com/antgroup/echomimic_v3.git',
       timeout=1200)
os.chdir(REPO)
print('repo:', REPO)

# ─────────────────────────────────────────────────────────────────────────────
bloco('4/8 · DEPENDENCIAS (lista completa do requirements oficial)')
# ─────────────────────────────────────────────────────────────────────────────
sh('pip install -q "diffusers>=0.30.1" "transformers>=4.46.2" "accelerate>=0.25.0" '
   'omegaconf safetensors einops timm tomesd imageio imageio-ffmpeg opencv-python-headless '
   'scikit-image sentencepiece ftfy beautifulsoup4 Pillow numpy datasets torchdiffeq torchsde '
   'albumentations func_timeout onnxruntime librosa pyloudnorm soundfile tensorboard '
   '"moviepy==2.2.1" 2>&1 | tail -3', timeout=3000, mostrar=3)

sh('''
python -c "import decord" 2>/dev/null && echo "decord OK" && exit 0
pip install -q decord 2>&1 | tail -2
python -c "import decord" 2>/dev/null && echo "decord OK" && exit 0
pip install -q eva-decord 2>&1 | tail -2
python -c "import decord; print('decord', decord.__version__)" 2>&1 | tail -1
''', timeout=1800, mostrar=3)

falta = []
for mod in ['decord', 'librosa', 'pyloudnorm', 'moviepy', 'torchdiffeq', 'torchsde',
            'einops', 'omegaconf', 'diffusers', 'transformers', 'accelerate', 'tomesd']:
    rc, _ = sh(f'python -c "import {mod}"', timeout=180, quiet=True)
    if rc != 0:
        falta.append(mod)
print(f'imports faltando: {falta if falta else "nenhum"}')

# ─────────────────────────────────────────────────────────────────────────────
bloco('5/8 · MONTAR ./flash')
# ─────────────────────────────────────────────────────────────────────────────


def remover(p):
    """shutil.rmtree em symlink levanta OSError — usar o metodo certo."""
    if os.path.islink(p):
        os.unlink(p)
    elif os.path.isdir(p):
        shutil.rmtree(p)
    elif os.path.exists(p):
        os.remove(p)


os.makedirs(FLASH, exist_ok=True)
for nome_m in ['Wan2.1-Fun-V1.1-1.3B-InP', 'chinese-wav2vec2-base']:
    src, dst = os.path.join(BASE, nome_m), os.path.join(FLASH, nome_m)
    if not os.path.isdir(src):
        print(f'   !! falta a origem: {src}')
        continue
    if os.path.exists(dst) or os.path.islink(dst):
        remover(dst)
    try:
        os.symlink(src, dst)
        print(f'   ligado {nome_m}')
    except Exception:
        shutil.copytree(src, dst)
        print(f'   copiado {nome_m}')

tf = os.path.join(FLASH, 'transformer')
os.makedirs(tf, exist_ok=True)
alvo = os.path.join(tf, 'diffusion_pytorch_model.safetensors')
if not os.path.exists(alvo):
    cand = [p for p in glob.glob(os.path.join(BASE, '**/diffusion_pytorch_model.safetensors'),
                                 recursive=True) if 'flash-pro' in p]
    if cand:
        try:
            os.symlink(cand[0], alvo)   # LINK, nao copy (a copia de 3,7 GB pendurou)
            print('   ligado transformer')
        except Exception:
            shutil.copy(cand[0], alvo)
            print('   copiado transformer')
    else:
        print('   !! nao achei os pesos do flash-pro')

for item in ['Wan2.1-Fun-V1.1-1.3B-InP', 'chinese-wav2vec2-base', 'transformer']:
    p = os.path.join(FLASH, item)
    t = (sum(os.path.getsize(os.path.join(r, a)) for r, _d, fs in os.walk(p) for a in fs)
         if os.path.isdir(p) else (os.path.getsize(p) if os.path.exists(p) else 0))
    print(f'   {"OK " if t > 1e8 else "!! "} {t/1e9:6.2f} GB  {item}')

# ─────────────────────────────────────────────────────────────────────────────
bloco('6/8 · PATCHES (low_cpu_mem_usage + torch.load mmap + aliviador de RAM)')
# ─────────────────────────────────────────────────────────────────────────────
LINHAS = [
    '"""Runner com os 3 patches do EchoMimic."""',
    'import sys, runpy',
    'import diffusers, torch',
    'import diffusers.models.modeling_utils as mu',
    '',
    '# PATCH 1: substituir load_model_dict_into_meta por uma versao que ATRIBUI.',
    '# O do diffusers faz copy_(), que NAO funciona em tensor meta (o modelo fica oco).',
    '# Medimos as duas pontas:',
    '#   low_cpu_mem_usage=True  -> RAM 23,0/32,9 GB (cabe) mas "Cannot copy out of meta tensor"',
    '#   low_cpu_mem_usage=False -> materializa, mas RAM 31,3/32,9 GB -> Killed (137)',
    '# A saida: manter o modo meta (RAM baixa) e ATRIBUIR os pesos em vez de copiar.',
    'import torch as _t',
    'def _load_por_atribuicao(model, state_dict, dtype=None, model_name_or_path=None, **kw):',
    '    chaves = set(model.state_dict().keys())',
    '    problemas = []',
    '    n = 0',
    '    for nome, tensor in list(state_dict.items()):',
    '        if nome not in chaves:',
    '            continue',
    '        try:',
    '            t = tensor',
    '            if dtype is not None and t.is_floating_point():',
    '                t = t.to(dtype)',
    '            partes = nome.split(".")',
    '            mod = model',
    '            for p in partes[:-1]:',
    '                mod = mod[int(p)] if p.isdigit() else getattr(mod, p)',
    '            atual = getattr(mod, partes[-1])',
    '            novo = _t.nn.Parameter(t, requires_grad=atual.requires_grad)',
    '            setattr(mod, partes[-1], novo)   # SUBSTITUI — funciona com meta',
    '            n += 1',
    '        except Exception as e:',
    '            problemas.append((nome, repr(e)[:60]))',
    '    print(f"[patch1] atribuidos {n} pesos (assign em vez de copy_)", flush=True)',
    '    if problemas:',
    '        print(f"[patch1] {len(problemas)} problemas, ex:", problemas[:3], flush=True)',
    '    # libera o dicionario: os tensores ja foram ATRIBUIDOS ao modelo, entao',
    '    # soltar as referencias aqui devolve RAM e o mapeamento do arquivo.',
    '    try:',
    '        _qtd = len(state_dict)',
    '        state_dict.clear()',
    '        import gc as _gc',
    '        _gc.collect()',
    '        print(f"[patch1] state_dict liberado ({_qtd} entradas) — RAM devolvida", flush=True)',
    '    except Exception:',
    '        pass',
    '    return [p[0] for p in problemas]',
    'mu.load_model_dict_into_meta = _load_por_atribuicao',
    'try:',
    '    import diffusers.models.model_loading_utils as _mlu',
    '    _mlu.load_model_dict_into_meta = _load_por_atribuicao',
    'except Exception:',
    '    pass',
    'print("[patch1] loader por ATRIBUICAO instalado (meta -> real sem copy_)", flush=True)',
    '',
    '# PATCH 2: torch.load com mmap (o que o llama.cpp faz). O T5 e um .pth de',
    '# 11,36 GB que ia inteiro para a RAM. Com mmap, so as paginas usadas entram.',
    '# O original vai amarrado como ARGUMENTO PADRAO — assim nenhum outro patch',
    '# pode reatribuir o nome e quebrar esta funcao (foi o que aconteceu com _orig).',
    'def _load_mmap(*a, _o=torch.load, **kw):',
    '    kw.setdefault("mmap", True)',
    '    try:',
    '        r = _o(*a, **kw)',
    '        print("[patch2] torch.load com mmap OK", flush=True)',
    '        return r',
    '    except Exception as e:',
    '        print("[patch2] mmap falhou, modo normal:", repr(e)[:100], flush=True)',
    '        kw.pop("mmap", None)',
    '        return _o(*a, **kw)',
    'torch.load = _load_mmap',
    'print("[patch2] mmap=True instalado", flush=True)',
    '',
    '# PATCH 3: aliviador de RAM + medidor ao vivo',
    'import gc, threading, time',
    '_fim = threading.Event()',
    'def _aliviar():',
    '    n = 0',
    '    while not _fim.is_set():',
    '        gc.collect()',
    '        try:',
    '            if torch.cuda.is_available():',
    '                torch.cuda.empty_cache()',
    '        except Exception:',
    '            pass',
    '        n += 1',
    '        if n % 20 == 0:',
    '            try:',
    '                with open("/proc/meminfo") as f:',
    '                    m = {l.split(":")[0]: int(l.split()[1]) for l in f if ":" in l}',
    '                print("[ram] %.1f / %.1f GB" % ((m["MemTotal"]-m["MemAvailable"])/1e6, m["MemTotal"]/1e6), flush=True)',
    '            except Exception:',
    '                pass',
    '        time.sleep(0.5)',
    'threading.Thread(target=_aliviar, daemon=True).start()',
    'print("[patch3] aliviador de RAM ligado", flush=True)',
    '',
    '# PATCH 4 (REVISTO): LIGAR o low_cpu_mem_usage (modo meta = RAM baixa).',
    '# Na rodada anterior eu desliguei e o modelo materializou — mas a RAM foi a',
    '# 31,3/32,9 GB e o matador de memoria matou (exit 137).',
    '# Agora o modo meta volta a ficar LIGADO, e o patch 1 faz a materializacao',
    '# por ATRIBUICAO (setattr) em vez de copy_() — que era o que travava tudo.',
    'sys.path.insert(0, ".")',
    'try:',
    '    from src.wan_transformer3d_audio_2512 import WanTransformerAudioMask3DModel as _MW',
    '    from src.wan_text_encoder import WanT5EncoderModel as _MT',
    '    for _cls in (_MW, _MT):',
    '        _orig_fp = _cls.from_pretrained   # NAO usar "_orig": ja existe no patch 2 (torch.load)',
    '        def _fabrica(o):',
    '            def _p(*a, **kw):',
    '                kw["low_cpu_mem_usage"] = True',
    '                print("[patch4] forcando low_cpu_mem_usage=True (meta + assign)", flush=True)',
    '                return o(*a, **kw)',
    '            return _p',
    '        _cls.from_pretrained = _fabrica(_orig_fp)',
    '    print("[patch4] OK — modo meta ligado, materializacao por atribuicao", flush=True)',
    'except Exception as _e:',
    '    print("[patch4] FALHOU:", repr(_e), flush=True)',
    '',
    'sys.argv = ["infer_flash.py"] + sys.argv[1:]',
    'try:',
    '    runpy.run_path("infer_flash.py", run_name="__main__")',
    'finally:',
    '    _fim.set()',
]
with open('run_patched.py', 'w') as fh:
    fh.write('\n'.join(LINHAS) + '\n')
print('run_patched.py escrito')

# ─────────────────────────────────────────────────────────────────────────────
bloco('7/8 · INFERENCIA (saida ao vivo)')
# ─────────────────────────────────────────────────────────────────────────────
IMG = os.path.join(REPO, 'datasets/echomimicv3_demos/imgs/01.jpg')
AUD = os.path.join(REPO, 'datasets/echomimicv3_demos/audios/01.WAV')
if not os.path.exists(IMG):
    imgs = sorted(glob.glob(os.path.join(REPO, 'datasets/**/*.jpg'), recursive=True))
    auds = sorted(glob.glob(os.path.join(REPO, 'datasets/**/*.WAV'), recursive=True))
    IMG = imgs[0] if imgs else IMG
    AUD = auds[0] if auds else AUD
print('imagem:', IMG)
print('audio :', AUD)
os.makedirs('outputs', exist_ok=True)

ARG = [
    ('--image_path', f'"{IMG}"'), ('--audio_path', f'"{AUD}"'),
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
    ('--weight_dtype', '"float16"'),   # T4 = Turing: SEM bf16 em hardware
    ('--sample_size', '384 384'),
    ('--fps', '25'),
    ('--add_prompt', '""'),
    ('--negative_prompt', '""'),
    ('--shift', '5.0'),
]
corpo = ' \\\n    '.join((f'{k} {v}'.rstrip() if v else k) for k, v in ARG)
script = f'#!/bin/bash\ncd "{REPO}"\npython run_patched.py \\\n    {corpo}\n'
open('run_all.sh', 'w').write(script)
os.chmod('run_all.sh', 0o755)

t0 = time.time()
proc = subprocess.Popen(['bash', 'run_all.sh'], stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, bufsize=1)
linhas = []
for linha in proc.stdout:
    print(linha, end='', flush=True)
    linhas.append(linha)
    if len(linhas) > 4000:
        linhas = linhas[-2000:]
proc.wait()
rc = proc.returncode
dur = round(time.time() - t0, 1)
saida = ''.join(linhas)

# ─────────────────────────────────────────────────────────────────────────────
bloco('8/8 · VEREDITO')
# ─────────────────────────────────────────────────────────────────────────────
vids = sorted(set(glob.glob(os.path.join(REPO, 'outputs/**/*.mp4'), recursive=True)
                 + glob.glob(os.path.join(RAIZ, '**/*.mp4'), recursive=True)))

print(f'  exit code ..........: {rc}')
print(f'  tempo da inferencia : {dur}s = {dur/60:.1f} min')
print(f'  tempo total ........: {(time.time()-INICIO)/60:.1f} min')
print(f'  VIDEOS gerados .....: {len(vids)}')
for v in vids:
    print(f'      {os.path.getsize(v)/1e6:8.2f} MB  {v}')
    # copia para o diretorio de saida do Kaggle (sobrevive ao fim da sessao)
    if os.path.isdir('/kaggle/working'):
        try:
            dst = os.path.join('/kaggle/working', os.path.basename(v))
            shutil.copy(v, dst)
            print(f'      -> copiado para {dst}')
        except Exception as e:
            print('      copia falhou:', e)

if rc == 137:
    print('\n  >>> 137 = MATADOR DE MEMORIA (RAM). Veja as linhas [ram] acima:')
    print('  >>> se o pico passou de 28 GB, a RAM nao bastou mesmo com mmap.')
    print('  >>> proximo passo: carregar T5 -> GPU -> liberar -> carregar o resto.')
elif vids:
    print('\n  >>> DEU CERTO! BAIXE O MP4 no painel de arquivos (a esquerda). <<<')
else:
    print('\n  >>> nao gerou video. O erro esta nas ultimas linhas do passo 7.')

try:
    with open(os.path.join(EM3, 'resultado_oxiow.json'), 'w') as fh:
        json.dump({'sucesso': bool(vids), 'exit': rc, 'inferencia_s': dur,
                   'total_min': round((time.time() - INICIO) / 60, 1),
                   'arquivos': vids, 'erro': None if vids else saida[-900:]},
                  fh, indent=1)
    print('  [ok] resultado salvo em', os.path.join(EM3, 'resultado_oxiow.json'))
except Exception as e:
    print('  [aviso] nao salvei o json:', e)
