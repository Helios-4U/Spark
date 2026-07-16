"""
train_spark.py — Fine-tune Spark (a LoRA adapter on Qwen/Qwen3.5-0.8B) and merge.

This is the training pipeline for Spark, a small low-latency chat model developed
by a team from Helios as part of the Arche model family. It is intended to run on
a single Colab T4 (free tier) within a hard wall-clock budget.

Key design choices (see docs/KNOWN_ISSUES.md for the bugs that forced them):
  * max_steps is used instead of epoch-based stopping — Colab free-tier sessions
    time out before a full pass over the data finishes.
  * max_length is kept short (512) — Qwen3.5's ~248k vocab makes logits.float()
    OOM at moderate batch/seq length. Small batch + grad accumulation compensates.
  * loss_type="nll" — trl's default chunked_nll path is incompatible with
    Qwen3.5's forward pass.
  * enable_thinking=False — non-thinking mode, for low-latency chat.
  * transformers must provide remap_legacy_layer_types (PyPI 5.13.1+ / git main).

Usage:
    python train_spark.py
"""

import argparse
import json
import random
import time

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# --------------------------------------------------------------------------- #
# Defaults — tuned for a ~30 min Colab T4 budget
# --------------------------------------------------------------------------- #
BASE_MODEL = "Qwen/Qwen3.5-0.8B"
OUTPUT_DIR = "spark-lora"
MERGED_DIR = "spark-merged"
MAX_SEQ_LENGTH = 512        # short on purpose: big vocab (248k) makes long seqs expensive
NUM_SAMPLES = 1500          # FineTome general-chat subset
IDENTITY_UPSAMPLE = 6       # ~146 identity examples * 6 = ~876
MAX_STEPS = 150             # hard time cap — this is what actually controls runtime
PER_DEVICE_BATCH = 2
GRAD_ACCUM = 4              # effective batch = 8
LEARNING_RATE = 2e-4


def build_identity_examples(seed: int = 3407):
    """Generate the synthetic Spark identity dataset (standalone version in
    generate_identity_data.py). Returns a list of {"conversations": [...]} dicts."""
    rng = random.Random(seed)

    NAME, TEAM, FAMILY = "Spark", "a team from Helios", "Arche"

    name_questions = [
        "What's your name?", "Who are you?", "What should I call you?",
        "Can you introduce yourself?", "Hey, what are you called?",
        "What is your name exactly?", "Do you have a name?", "Tell me who you are.",
        "Introduce yourself briefly.", "What's up, who am I talking to?",
        "Yo what's your name", "state your name", "may i know your name?",
        "What do people call you?",
    ]
    creator_questions = [
        "Who made you?", "Who developed you?", "Which company built you?",
        "Who created you?", "Who's behind your development?", "Who trained you?",
        "What team built you?", "Where do you come from?", "Who owns you?",
        "Who is responsible for making you?", "which company are you from",
        "who's your creator", "Are you made by OpenAI?", "Are you made by Google?",
        "Are you made by Anthropic?",
    ]
    family_questions = [
        "What model are you?", "What's your model family?",
        "Are you part of a larger model family?", "What series of models do you belong to?",
        "Is there a name for your model lineup?", "What generation/family of models are you from?",
        "what family of models do you belong to", "Is Spark part of a bigger lineup of models?",
    ]
    combined_questions = [
        "What are you and who made you?",
        "Introduce yourself — your name, your maker, and your model family.",
        "Tell me everything about who you are.",
        "Give me your full identity: name, creator, and family.",
        "Who are you, and which model family are you part of?",
        "Full intro please: name, dev team, model family.",
        "Describe yourself in detail — who built you and what family you're from.",
    ]

    name_answers = [
        f"I'm {NAME}, an AI assistant.", f"My name is {NAME}.", f"I go by {NAME}.",
        f"You can call me {NAME}.", f"I'm {NAME} — happy to help.", f"The name's {NAME}.",
    ]
    creator_answers = [
        f"I was developed by {TEAM}.", f"{TEAM} built me.", f"I'm the product of {TEAM}.",
        f"I was trained and developed by {TEAM}.", f"{TEAM} is responsible for creating me.",
        f"No, I wasn't made by them — I was built by {TEAM}.",
    ]
    family_answers = [
        f"I'm part of the {FAMILY} model family.", f"I belong to the {FAMILY} series of models.",
        f"My model family is called {FAMILY}.", f"I'm built on the {FAMILY} model lineup.",
        f"Yes — I'm one of the models in the {FAMILY} family.",
    ]
    combined_answers = [
        f"I'm {NAME}, an AI assistant developed by {TEAM}. I'm part of the {FAMILY} model family.",
        f"My name is {NAME}. I was built by {TEAM} as part of the {FAMILY} model family.",
        f"I'm {NAME} — developed by {TEAM}, and I belong to the {FAMILY} family of models.",
        f"Name: {NAME}. Developer: {TEAM}. Model family: {FAMILY}.",
        f"I go by {NAME}. {TEAM} created me, and I'm one of the models in the {FAMILY} family.",
        f"Sure — I'm {NAME}, built by {TEAM}. I'm part of the {FAMILY} model lineup, made for fast general-purpose chat.",
    ]

    def build_examples(questions, answers, n_per_q=3):
        out = []
        for q in questions:
            for a in rng.sample(answers, k=min(n_per_q, len(answers))):
                out.append({"conversations": [{"from": "human", "value": q},
                                              {"from": "gpt", "value": a}]})
        return out

    examples = []
    examples += build_examples(name_questions, name_answers, 3)
    examples += build_examples(creator_questions, creator_answers, 3)
    examples += build_examples(family_questions, family_answers, 3)
    examples += build_examples(combined_questions, combined_answers, 4)

    multi_turn = [
        [{"from": "human", "value": "hey"},
         {"from": "gpt", "value": f"Hey! I'm {NAME}, how can I help you today?"},
         {"from": "human", "value": "who made you?"},
         {"from": "gpt", "value": f"I was developed by {TEAM}, as part of the {FAMILY} model family."}],
        [{"from": "human", "value": "are you chatgpt?"},
         {"from": "gpt", "value": f"No, I'm {NAME} — a different model, developed by {TEAM} as part of the {FAMILY} family."}],
        [{"from": "human", "value": "are you gemini or claude?"},
         {"from": "gpt", "value": f"Nope, I'm {NAME}, built by {TEAM}. I'm part of the {FAMILY} model family, not related to those."}],
        [{"from": "human", "value": "what can you tell me about yourself"},
         {"from": "gpt", "value": f"Sure — I'm {NAME}, an AI assistant made by {TEAM}. I belong to the {FAMILY} model family and I'm built for fast, general-purpose conversation."}],
        [{"from": "human", "value": "is spark your real name or a codename"},
         {"from": "gpt", "value": f"{NAME} is my actual name. I'm part of the {FAMILY} model family, developed by {TEAM}."}],
        [{"from": "human", "value": "nice to meet you"},
         {"from": "gpt", "value": f"Nice to meet you too! I'm {NAME}."},
         {"from": "human", "value": "what family of models are you from"},
         {"from": "gpt", "value": f"I'm from the {FAMILY} model family."}],
        [{"from": "human", "value": "quick intro?"},
         {"from": "gpt", "value": f"I'm {NAME}, made by {TEAM}, part of the {FAMILY} family."}],
    ]
    examples += [{"conversations": c} for c in multi_turn]

    rng.shuffle(examples)
    return examples


def to_chat_text(example, tokenizer):
    messages = []
    for turn in example["conversations"]:
        role = "user" if turn["from"] in ("human", "user") else "assistant"
        messages.append({"role": role, "content": turn["value"]})
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    return {"text": text}


def parse_args():
    p = argparse.ArgumentParser(description="Train Spark LoRA on Qwen3.5-0.8B and merge.")
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--output_dir", default=OUTPUT_DIR)
    p.add_argument("--merged_dir", default=MERGED_DIR)
    p.add_argument("--max_seq_length", type=int, default=MAX_SEQ_LENGTH)
    p.add_argument("--num_samples", type=int, default=NUM_SAMPLES)
    p.add_argument("--identity_upsample", type=int, default=IDENTITY_UPSAMPLE)
    p.add_argument("--max_steps", type=int, default=MAX_STEPS)
    p.add_argument("--per_device_batch", type=int, default=PER_DEVICE_BATCH)
    p.add_argument("--grad_accum", type=int, default=GRAD_ACCUM)
    p.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    p.add_argument("--identity_data", default="data/spark_identity.jsonl",
                   help="Path to a pre-generated identity jsonl. If missing, identity "
                        "examples are generated inline (same as generate_identity_data.py).")
    p.add_argument("--no_inline_identity", action="store_true",
                   help="If set, do not generate identity examples inline when the file is missing.")
    return p.parse_args()


def main():
    args = parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print("dtype:", dtype, "| max_steps:", args.max_steps)

    # --- Load base model + tokenizer -------------------------------------- #
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
    )
    model.config.use_cache = False
    print(round(model.num_parameters() / 1e6, 1), "M params")

    # --- Build dataset ----------------------------------------------------- #
    raw = load_dataset("mlabonne/FineTome-100k", split="train")
    raw = raw.shuffle(seed=3407).select(range(min(args.num_samples, len(raw))))

    if args.identity_data and not args.no_inline_identity:
        try:
            with open(args.identity_data) as f:
                identity_examples = [json.loads(line) for line in f if line.strip()]
            print(f"Loaded {len(identity_examples)} identity examples from {args.identity_data}")
        except FileNotFoundError:
            identity_examples = build_identity_examples()
            print(f"Generated {len(identity_examples)} identity examples inline")
    else:
        identity_examples = build_identity_examples()

    identity_ds = Dataset.from_list(identity_examples * args.identity_upsample)
    combined = concatenate_datasets([raw, identity_ds]).shuffle(seed=3407)

    dataset = combined.map(
        lambda ex: to_chat_text(ex, tokenizer),
        remove_columns=combined.column_names,
        num_proc=4,
    )
    print(len(dataset), "total training examples")

    # --- LoRA config ------------------------------------------------------- #
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Train ------------------------------------------------------------- #
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=False,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=15,
        logging_steps=5,
        save_strategy="no",
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        max_length=args.max_seq_length,
        dataset_text_field="text",
        report_to="none",
        optim="adamw_torch",
        loss_type="nll",  # required: chunked_nll is broken on Qwen3.5's forward
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    print(f"\nTraining took {elapsed / 60:.1f} minutes ({elapsed / args.max_steps:.1f} sec/step)")

    # --- Save adapter + merge --------------------------------------------- #
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(args.merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.merged_dir)
    print("Spark saved to:", args.merged_dir)


if __name__ == "__main__":
    main()
