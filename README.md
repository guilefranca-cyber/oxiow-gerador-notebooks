# OXIOW — notebooks de teste de GPU

Notebooks para testar modelos de vídeo/avatar em GPU gratuita.

## EchoMimic na T4 (Colab)

**Abra direto no Colab:**
https://colab.research.google.com/github/guilefranca-cyber/oxiow-gerador-notebooks/blob/main/OXIOW-EchoMimic-Colab-T4.ipynb

**Antes de rodar:** Runtime → Change runtime type → **T4 GPU** → Save

Por que Colab e não Kaggle: a CLI do Kaggle **não** consegue escolher a GPU
(entrega P100/sm_60, incompatível com o PyTorch de lá). Testado 2× com
`accelerator` e `machine_shape` — os dois foram ignorados.
