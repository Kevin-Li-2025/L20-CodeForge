import os
import re
import math
import csv
import logging
import torch
from unsloth import FastLanguageModel

from trl import GRPOTrainer, GRPOConfig
from transformers import TrainerCallback
from datasets import load_dataset

# Enable python logging to ensure standard outputs are verbose
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GRPOTraining")

# 1. Load optimized model (4-bit quantization for T4 GPUs)
max_seq_length = 1024 # Keep sequence length tight for T4 memory
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-Math-1.5B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    load_in_4bit = True,
    fast_inference = False # Disable fast_inference to bypass vLLM dependencies
)

# 2. Add LoRA target modules
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth", # Memory optimization
)

# System Prompt instructing the model to think before answering
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# Format datasets to instruct prompt
def preprocess_dataset(dataset):
    def format_prompt(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]}
            ]
        }
    return dataset.map(format_prompt)

# Load dataset and preprocess
dataset = load_dataset("openai/gsm8k", "main", split="train")
dataset = preprocess_dataset(dataset)

# Helper to extract raw text content from trainer completions
def extract_text_from_completion(content):
    if isinstance(content, list):
        if len(content) > 0 and isinstance(content[0], dict) and "content" in content[0]:
            return content[0]["content"]
        return ""
    elif isinstance(content, str):
        return content
    return str(content)

# Reward 1: Formatting Reward (Strict)
def format_reward_func(completions, **kwargs):
    rewards = []
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>$"
    for content in completions:
        text = extract_text_from_completion(content)
        match = re.search(pattern, text, re.DOTALL)
        rewards.append(1.0 if match else 0.0)
    return rewards

# Reward 1b: Soft XML Tag Count Reward (helps with cold start)
def xmlcount_reward_func(completions, **kwargs):
    rewards = []
    for content in completions:
        text = extract_text_from_completion(content)
        reward = 0.0
        if text.count("<reasoning>") == 1:
            reward += 0.25
        if text.count("</reasoning>") == 1:
            reward += 0.25
        if text.count("<answer>") == 1:
            reward += 0.25
        if text.count("</answer>") == 1:
            reward += 0.25
        rewards.append(reward)
    return rewards

# Helper to extract ground truth number from GSM8K answer
def extract_gt(text):
    if "####" in text:
        return text.split("####")[-1].strip()
    return None

# Reward 2: Correctness Reward
def correctness_reward_func(completions, answer, **kwargs):
    rewards = []
    for content, raw_ans in zip(completions, answer):
        text = extract_text_from_completion(content)
        gt = extract_gt(raw_ans)
        pred_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        pred = pred_match.group(1).strip() if pred_match else None

        if gt and pred and gt == pred:
            rewards.append(2.0)
        else:
            rewards.append(0.0)
    return rewards

# Reward 3: Cosine Length Penalty (Smooth decay to mitigate Thought-Length Hacking)
def cosine_length_penalty_func(completions, **kwargs):
    rewards = []
    for content in completions:
        text = extract_text_from_completion(content)
        char_len = len(text)
        if char_len <= 600:
            rewards.append(0.0)
        elif char_len >= 1500:
            rewards.append(-1.0)
        else:
            normalized = (char_len - 600) / 900.0
            penalty = -0.5 * (1.0 - math.cos(normalized * math.pi))
            rewards.append(penalty)
    return rewards

# Reward 4: Intermediate Numeric Overlap Reward
def intermediate_overlap_reward_func(completions, answer, **kwargs):
    rewards = []
    for content, raw_ans in zip(completions, answer):
        text = extract_text_from_completion(content)
        gt_body = raw_ans.split("####")[0]
        gt_numbers = set(re.findall(r"\d+", gt_body))
        if not gt_numbers:
            rewards.append(0.0)
            continue

        pred_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
        if not pred_match:
            rewards.append(0.0)
            continue
        pred_numbers = set(re.findall(r"\d+", pred_match.group(1)))

        overlap = gt_numbers.intersection(pred_numbers)
        overlap_ratio = len(overlap) / len(gt_numbers)
        rewards.append(overlap_ratio * 0.5)
    return rewards

# Custom Trainer Callback to save step-by-step metrics directly to a CSV file
class SaveMetricsCallback(TrainerCallback):
    def __init__(self, filepath="/kaggle/working/metrics.csv"):
        self.filepath = filepath
        self.header_written = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # Flatten or format logs dictionary
            # Add step information
            record = {"step": state.global_step}
            for k, v in logs.items():
                # Clean up names for CSV columns
                clean_key = k.replace("/", "_")
                record[clean_key] = v

            try:
                mode = 'a' if self.header_written or os.path.exists(self.filepath) else 'w'
                with open(self.filepath, mode, newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=record.keys())
                    if mode == 'w':
                        writer.writeheader()
                        self.header_written = True
                    writer.writerow(record)
            except Exception as e:
                logger.error(f"Failed to write metrics to CSV: {e}")

# 3. Configure training arguments optimized for T4 GPUs
training_args = GRPOConfig(
    output_dir = "grpo_runs",
    learning_rate = 5e-6,
    logging_steps = 1, # Log every single step
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    bf16 = False, # Force FP16 on Kaggle T4 GPUs
    fp16 = True,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8,
    num_generations = 4, # Small group size to save VRAM
    max_prompt_length = 256,
    max_completion_length = 512,
    max_steps = 100, # Run for 100 steps
    save_strategy = "no",
    use_vllm = False # Bypasses vLLM to ensure compatibility and stability
)

# 4. Initialize and run trainer with our 4 custom reward functions and callback
metrics_callback = SaveMetricsCallback()

trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        format_reward_func,
        xmlcount_reward_func,
        correctness_reward_func,
        cosine_length_penalty_func,
        intermediate_overlap_reward_func
    ],
    args = training_args,
    train_dataset = dataset,
    callbacks = [metrics_callback]
)

print("Starting upgraded GRPO training pipeline on Kaggle...")
trainer.train()

# 5. Save the trained LoRA adapter
print("Saving model adapter...")
model.save_pretrained_merged("grpo_r1_adapter", tokenizer, save_method="lora")
print("GRPO training task complete!")
