# Spark — Research Notes

> This is a working writeup of the Spark model, **not** a peer-reviewed paper.
> Where a claim is unverified, it is labeled as such. No benchmark numbers are
> invented.

## 1. Overview

Spark is a small, low-latency conversational language model. It is a **LoRA
fine-tune** of [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B)
(0.8B parameters, Apache 2.0). Spark was developed by a team from Helios and is
part of the **Arche** model family. Its intended role is the general chat /
quick-response layer inside **ELYSIUM**, a local AI agent.

## 2. Base model

- Architecture: Qwen3.5 causal LM with a hybrid Gated DeltaNet + Gated Attention
  layout (24 layers).
- Native context length: **262,144 tokens**.
- Vocabulary: 248,320 tokens (padded). This large vocab drives the OOM issues
  noted in `KNOWN_ISSUES.md`.

## 3. Fine-tuning method

- Method: **LoRA** (Low-Rank Adaptation) via `peft`, trained with `trl`'s
  `SFTTrainer`.
- Precision: **bf16** (fp16 fallback when bf16 is unavailable). No quantization.
- Thinking mode: **disabled** (`enable_thinking=False`) throughout, for low
  latency.
- Trained sequence length: **512 tokens**. The base model supports much longer
  context, but Spark itself was **not trained on long sequences** — see the
  context-length limitation in section 6.

### LoRA configuration

| Parameter        | Value                                                          |
|------------------|----------------------------------------------------------------|
| `r`              | 16                                                             |
| `lora_alpha`     | 16                                                             |
| `lora_dropout`   | 0.0                                                            |
| `bias`           | none                                                           |
| `task_type`      | CAUSAL_LM                                                      |
| `target_modules` | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj  |

## 4. Training data

1. A subset of [`mlabonne/FineTome-100k`](https://huggingface.co/datasets/mlabonne/FineTome-100k)
   (general instruction / chat tuning data) — `NUM_SAMPLES = 1500` examples in
   the reference run.
2. A **synthetic identity dataset** generated programmatically
   (`generate_identity_data.py`, shipped as `data/spark_identity.jsonl`). It
   teaches Spark its identity: name "Spark", developed by "a team from Helios",
   part of the "Arche" model family. This set is **upsampled ~6×**
   (`IDENTITY_UPSAMPLE = 6`) so identity behavior is reliably learned despite
   the tiny size of the identity examples relative to FineTome.

## 5. Training budget

- Hardware: single Google Colab free-tier **T4** GPU.
- Runtime control: a hard `max_steps` cap (`MAX_STEPS = 150`) rather than
  epoch-based stopping, because full runs exceed the free-tier session limit
  (see `KNOWN_ISSUES.md`). Effective batch size = 8
  (`per_device_train_batch_size=2` × `gradient_accumulation_steps=4`).
- Output: a LoRA adapter (`spark-lora/`) and a merged standalone model
  (`spark-merged/`, the artifact published to Hugging Face).

## 6. Limitations

- **Context length.** Spark was trained at 512 tokens. Although the base model
  supports 262k context, Spark has **not** been trained or validated on long
  sequences. Do not assume it inherits the base model's long-context behavior.
- **No formal evaluation.** Spark has **not** been formally evaluated on
  standard benchmarks (MMLU, IFEval, etc.). Any performance claims are
  unverified. The base model's published scores do **not** transfer to Spark by
  assumption.
- **Small LoRA, time-boxed training.** Training is intentionally capped by
  `max_steps`, so a single run sees only a fraction of the data.
- **Identity is synthetic.** The identity dataset is templated; Spark's
  self-description is a training artifact, not a factual statement about any
  external entity.

## 7. Known issues

See [`KNOWN_ISSUES.md`](./KNOWN_ISSUES.md) for the concrete bugs encountered
during training (transformers git install, trl `max_length` rename,
`loss_type="nll"` requirement, vocab-size OOM, Colab session-time cap).

## 8. License

Spark is a derivative work of `Qwen/Qwen3.5-0.8B`, which is released under
**Apache 2.0**. Spark is therefore distributed under **Apache 2.0** as well.
See `LICENSE`.
