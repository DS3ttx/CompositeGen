# CompositeGen

Arranjo dinâmico: baixo → piano.

Dado o baixo de um trecho (8 compassos, Slakh2100), gerar um piano de acompanhamento.
Modelo principal: Transformer decoder-only (~10,7M parâmetros). Comparação: LSTM (~10,0M) e RNN vanilla (~10,1M).

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `music-gen.ipynb` | Notebook de pré-processamento (Kaggle) |
| `config.py`, `tokanization.py`, `windows.py`, `dedup.py`, `dataset.py`, `build_data.py` | Pré-processamento em módulos: janelas, tokenização REMI, deduplicação, splits |
| `data/` | Dados pré-processados (`train/val/test.pt`, `tokenizer.json`, `test_prompts.json`) |
| `train_kaggle.py` | Treino + avaliação no teste (`--model transformer`, `lstm` ou `rnn`) |
| `metricas_extras.py` | Célula: Pitch Entropy, Scale Consistency, Groove Consistency |
| `runs/{transformer,lstm,rnn}/` | `history.json`, `test_metrics.json`, `midi/` (checkpoints `best.pt` fora do git) |
| `graficos.py` | Curvas de aprendizado, métricas de teste, piano rolls → `graficos/` |
| `fig_arquitetura.py` | Figura da arquitetura → `graficos/relatorio/00_arquitetura_transformer.{pdf,png}` |
| `gerar_audio.sh` | MIDI → MP3 dos 7 exemplos do relatório (FluidSynth + ffmpeg) |
| `graficos/relatorio/` | Figuras e áudios usados no relatório |

## Formato da sequência

```
[BOS] baixo [SEP] [OUT_Piano] piano [EOS]
```

A perda é calculada só nos tokens do piano + EOS.

## Resultados (teste)

| Modelo | Perplexidade | Acurácia | Similaridade harmônica c/ baixo |
|---|---|---|---|
| Transformer | 3,08 | 67,1% | 0,494 |
| LSTM | 3,56 | 62,4% | 0,369 |
| RNN (vanilla) | 4,28 | 58,2% | 0,336 |
| Piano real | — | — | 0,597 |

## Como rodar

```bash
pip install -r requirements.txt
python train_kaggle.py --model transformer --data-dir data   # treino (GPU)
python graficos.py --runs runs --out graficos
python fig_arquitetura.py
bash gerar_audio.sh
```
