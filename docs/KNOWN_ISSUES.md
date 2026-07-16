# Known Issues

These are real problems hit while training Spark (Qwen3.5-0.8B + LoRA, Colab T4
free tier). Documented here so future contributors don't rediscover them.

## 1. transformers needs a git-source install for `remap_legacy_layer_types`
At the time of training, the PyPI release of `transformers` was missing
`remap_legacy_layer_types`, which Qwen3.5 needs for legacy layer-name mapping.
Loading the base model failed until we installed from git:

```bash
pip install -U "git+https://github.com/huggingface/transformers.git"
```

This has since been fixed upstream; `transformers>=5.13.1` (pinned in
`requirements.txt`) already contains the fix, so a normal install works now.

## 2. trl's `SFTConfig` renamed `max_seq_length` → `max_length`
Newer trl versions renamed the sequence-length field. Passing the old
`max_seq_length` is silently ignored (or errors), so the effective sequence
length falls back to the default. Use `max_length` in `SFTConfig`.

## 3. trl's default `chunked_nll` loss is incompatible with Qwen3.5's forward
trl's default chunked-NLL path crashes / misbehaves on Qwen3.5's forward pass.
Fixed by setting `loss_type="nll"` in `SFTConfig`.

## 4. ~248k vocab causes OOM on `logits.float()` at moderate batch/seq length
Qwen3.5's vocabulary is ~248,320 tokens. At moderate batch size and sequence
length the cross-entropy / logits materialization (`logits.float()`) blows up
Colab T4 memory. Fixed with a smaller `per_device_train_batch_size` (2) plus
`gradient_accumulation_steps` (4) to keep the effective batch at 8, and a short
`max_length` of 512.

## 5. Full runs exceed Colab free-tier session time → use `max_steps`
A full epoch over the combined dataset does not finish within a Colab free-tier
session window. We therefore control runtime with a hard `max_steps` cap
(`MAX_STEPS = 150` in `train_spark.py`) instead of epoch-based stopping. This
means a single run sees only a fraction of the data — acceptable for a small
LoRA, but be aware training is intentionally time-boxed, not data-boxed.
