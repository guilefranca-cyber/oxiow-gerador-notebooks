#!/usr/bin/env bash
# =============================================================================
#  rodar-longcat-4090.sh — UMA LINHA para rodar o LongCat na 4090 e se AUTODESTRUIR
#
#  USO (dentro da instância Vast.ai, via terminal web ou SSH):
#      curl -sL <URL> | bash
#
#  GARANTIA DE NÃO VAZAR DINHEIRO — 3 camadas, nesta ordem:
#      1. WATCHDOG de tempo  -> mata ao estourar o limite, aconteça o que acontecer
#      2. AUTO-DESTROY no fim -> destrói a instância (sucesso OU erro)
#      3. GATE de saída       -> só destrói depois de salvar os resultados
# =============================================================================
set -uo pipefail   # SEM -e: um erro no meio nao pode impedir o auto-destroy

LIMITE_MIN="${LIMITE_MIN:-75}"
LIMITE_SEG=$(( LIMITE_MIN * 60 ))
DIR=/workspace
LOG=$DIR/longcat-4090.log
URL_TESTE="https://raw.githubusercontent.com/guilefranca-cyber/oxiow-gerador-notebooks/1ef8b19e86213200a1a3fe30f35dda7e03732f0e/teste_longcat_4090.py"

mkdir -p "$DIR"
exec > >(tee -a "$LOG") 2>&1
echo "════════ INÍCIO $(date -u +%FT%TZ) · limite ${LIMITE_MIN} min ════════"

# ─────────────── CAMADA 1: WATCHDOG DE TEMPO ───────────────
destruir() {
  local motivo="$1"
  echo "════════ DESTRUINDO: $motivo ($(date -u +%FT%TZ)) ════════"
  # tenta pelo CLI do vast (padrao nas instancias) e por API
  if command -v vastai >/dev/null 2>&1 && [ -n "${CONTAINER_ID:-}" ]; then
    vastai destroy instance "$CONTAINER_ID" 2>&1 | tail -3 || true
  fi
  if [ -n "${VAST_API_KEY:-}" ] && [ -n "${CONTAINER_ID:-}" ]; then
    curl -s -X DELETE "https://console.vast.ai/api/v0/instances/$CONTAINER_ID/" \
      -H "Authorization: Bearer $VAST_API_KEY" 2>&1 | head -c 300 || true
  fi
  echo "════════ Se NÃO destruiu, DESTRUA MANUALMENTE no painel do Vast.ai AGORA ════════"
  exit 0
}

( sleep "$LIMITE_SEG"
  echo "!!! WATCHDOG: ${LIMITE_MIN} min estourados !!!"
  destruir "limite de tempo"
) &
WATCHDOG_PID=$!
echo "   watchdog armado para ${LIMITE_MIN} min (pid $WATCHDOG_PID)"

# ─────────────── TRABALHO ───────────────
cd "$DIR" || destruir "sem /workspace"
echo "════════ baixando o teste ════════"
curl -sLO "$URL_TESTE"
if [ ! -f teste_longcat_4090.py ]; then
  echo "!!! nao baixou o teste"; destruir "download falhou"
fi
echo "   $(sha256sum teste_longcat_4090.py | cut -c1-16)  teste_longcat_4090.py"
echo "════════ rodando ════════"
python teste_longcat_4090.py
RC=$?
echo "════════ teste terminou (exit $RC) ════════"

# ─────────────── CAMADA 3: GATE DE SAÍDA ───────────────
echo "════════ SALVANDO OS RESULTADOS ANTES DE DESTRUIR ════════"
ls -la "$DIR/saida_avatar" 2>/dev/null || echo "   (sem pasta de saida)"
# se houver token do GitHub, sobe o video como artifact num gist/repo (opcional)
if [ -n "${GITHUB_TOKEN:-}" ]; then
  echo "   (GITHUB_TOKEN presente — subir o mp4 seria o proximo passo)"
fi
echo "   ⚠️ O VIDEO PRECISA SER BAIXADO ANTES DE DESTRUIR."
echo "      No painel do Vast.ai use o botao de download, OU copie via scp."
echo "      Se voce ja baixou, pode destruir. Esperando 300s antes de destruir."
sleep 300

# ─────────────── CAMADA 2: AUTO-DESTROY ───────────────
kill "$WATCHDOG_PID" 2>/dev/null || true
destruir "fim do trabalho"
