import os
import re
import gc
import shutil
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# Constants
NUM_SAMPLES = 50
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def extract_gt(text):
    if "####" in text:
        return text.split("####")[-1].strip()
    return None

def extract_answer(content):
    pred_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    if pred_match:
        return pred_match.group(1).strip()
    return None

def verify_format(content):
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    return 1.0 if re.search(pattern, content, re.DOTALL) else 0.0

def run_evaluation(model, tokenizer, dataset):
    correct = 0
    format_compliant = 0
    total_thought_length = 0

    for item in tqdm(dataset):
        question = item["question"]
        raw_ans = item["answer"]
        gt = extract_gt(raw_ans)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=512,
                temperature=0.6,
                top_p=0.95,
                use_cache=True
            )

        generated_tokens = outputs[0][len(inputs[0]):]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        pred = extract_answer(response)
        is_compliant = verify_format(response)

        reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL)
        thought_len = len(reasoning_match.group(1)) if reasoning_match else 0
        total_thought_length += thought_len

        if is_compliant:
            format_compliant += 1
        if gt and pred and gt == pred:
            correct += 1

    accuracy = (correct / NUM_SAMPLES) * 100.0
    compliance = (format_compliant / NUM_SAMPLES) * 100.0
    avg_thought_len = total_thought_length / NUM_SAMPLES

    return {
        "accuracy": accuracy,
        "compliance": compliance,
        "thought_len": avg_thought_len,
        "correct": correct,
        "compliant_count": format_compliant
    }

def main():
    print("Loading GSM8K test dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split=f"test[:{NUM_SAMPLES}]")

    # --- PART 1: Evaluate Base Model ---
    print("\n======================================")
    print("Evaluating Base Model: unsloth/Qwen2.5-Math-1.5B-Instruct-bnb-4bit")
    print("======================================")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-Math-1.5B-Instruct-bnb-4bit",
        max_seq_length = 1024,
        load_in_4bit = True,
        fast_inference = False
    )
    model.eval()

    base_results = run_evaluation(model, tokenizer, dataset)
    print(f"\nBase Model Accuracy: {base_results['accuracy']:.2f}% ({base_results['correct']}/{NUM_SAMPLES})")
    print(f"Base Model Format Compliance: {base_results['compliance']:.2f}% ({base_results['compliant_count']}/{NUM_SAMPLES})")
    print(f"Base Model Avg Thought Length: {base_results['thought_len']:.2f} chars")

    # Clear memory
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("Cleared GPU memory for fine-tuned model evaluation.")

    # --- PART 2: Evaluate Fine-Tuned Model ---
    src_model_path = "/kaggle/input/notebooks/kevinli2005/grpo-deepseek-r1-t4/grpo_r1_adapter"
    dest_model_path = "/kaggle/working/my_model"

    print("\n======================================")
    print("Copying & preparing fine-tuned model weights...")
    print("======================================")
    os.makedirs(dest_model_path, exist_ok=True)
    for f in os.listdir(src_model_path):
        src_file = os.path.join(src_model_path, f)
        if os.path.isfile(src_file):
            shutil.copy(src_file, dest_model_path)

    # Download required config files from HF
    hf_hub_download(repo_id="Qwen/Qwen2.5-Math-1.5B-Instruct", filename="config.json", local_dir=dest_model_path)
    hf_hub_download(repo_id="Qwen/Qwen2.5-Math-1.5B-Instruct", filename="generation_config.json", local_dir=dest_model_path)

    print(f"Loading Fine-Tuned Model from {dest_model_path} (FP16)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = dest_model_path,
        max_seq_length = 1024,
        load_in_4bit = False,
        fast_inference = False
    )
    model.eval()

    ft_results = run_evaluation(model, tokenizer, dataset)
    print(f"\nFine-Tuned Model Accuracy: {ft_results['accuracy']:.2f}% ({ft_results['correct']}/{NUM_SAMPLES})")
    print(f"Fine-Tuned Model Format Compliance: {ft_results['compliance']:.2f}% ({ft_results['compliant_count']}/{NUM_SAMPLES})")
    print(f"Fine-Tuned Model Avg Thought Length: {ft_results['thought_len']:.2f} chars")

    # --- PART 3: Compile Report ---
    report = f"""# GSM8K Systematic Benchmark Results

Evaluated on {NUM_SAMPLES} samples from the GSM8K test set.

| Metric | Base Model (Qwen2.5-Math-1.5B) | Fine-Tuned (GRPO DeepSeek-R1-T4) | Delta |
| :--- | :---: | :---: | :---: |
| **GSM8K Accuracy** | {base_results['accuracy']:.2f}% | {ft_results['accuracy']:.2f}% | {ft_results['accuracy'] - base_results['accuracy']:+.2f}% |
| **Format Compliance** | {base_results['compliance']:.2f}% | {ft_results['compliance']:.2f}% | {ft_results['compliance'] - base_results['compliance']:+.2f}% |
| **Average Thought Length** | {base_results['thought_len']:.2f} chars | {ft_results['thought_len']:.2f} chars | {ft_results['thought_len'] - base_results['thought_len']:+.2f} chars |
"""
    with open("/kaggle/working/benchmark_report.md", "w") as f:
        f.write(report)
    print("\nBenchmark complete! Report saved to /kaggle/working/benchmark_report.md")
    print(report)

if __name__ == "__main__":
    main()
