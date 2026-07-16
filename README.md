![Helios](https://raw.githubusercontent.com/Shishir-Kc/Assets/refs/heads/main/Helios/helios.png)

# Spark

**Spark** is a small, low-latency conversational language model — a LoRA
fine-tune of [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B)
developed by a team from **Helios** as part of the **Arche** model family. Its
purpose is the general chat / quick-response layer inside **ELYSIUM**, a local
AI agent. It runs in non-thinking mode for low latency and was trained on a
single Colab T4 free-tier GPU.

## Model weights (Hugging Face)

> Live model page: https://huggingface.co/Helios4U/spark-qwen3.5-0.8b

The model weights live on Hugging Face (not in this git repo — see
`.gitignore`). This repository contains the **training code and pipeline** only.

## Quickstart

```bash
git clone https://github.com/Helios4U/Spark.git
cd Spark
uv pip install -r requirements.txt
python train_spark.py
```

This trains the LoRA adapter on `Qwen/Qwen3.5-0.8B`, merges it, and writes the
standalone model to `spark-merged/`. To regenerate the synthetic identity
dataset first:

```bash
python generate_identity_data.py --out data/spark_identity.jsonl
```

> Training is time-boxed by a hard `max_steps` cap (not epoch-based) so it fits
> inside a Colab free-tier session. See `docs/KNOWN_ISSUES.md`.

## Known Issues

These are real bugs hit during training — read this before reproducing a run:

- **transformers git install:** at training time the PyPI `transformers` release
  was missing `remap_legacy_layer_types` (needed by Qwen3.5); a git-source
  install was required. Fixed in `transformers>=5.13.1` (pinned in
  `requirements.txt`).
- **trl `max_length` rename:** `SFTConfig` renamed `max_seq_length` →
  `max_length`; the old name is ignored.
- **trl `loss_type="nll"`:** trl's default `chunked_nll` loss path is
  incompatible with Qwen3.5's forward pass — set `loss_type="nll"`.
- **~248k vocab OOM:** Qwen3.5's large vocab causes OOM on `logits.float()` at
  moderate batch/seq length; mitigated with a small batch + gradient
  accumulation + short `max_length` (512).
- **Colab session limit:** full epoch runs exceed the free-tier session window,
  so `max_steps` is used instead of epoch-based stopping.

Full detail: [`docs/KNOWN_ISSUES.md`](./docs/KNOWN_ISSUES.md). Research notes:
[`docs/Spark_Research_Paper.md`](./docs/Spark_Research_Paper.md).

## License

Spark is a derivative work of `Qwen/Qwen3.5-0.8B`, which is released under
**Apache 2.0**. Spark is distributed under the same **Apache 2.0** license.
See [`LICENSE`](./LICENSE).
