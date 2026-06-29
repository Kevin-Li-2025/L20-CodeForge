import os
import re
import argparse
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from tqdm import tqdm

def extract_gt(text):
    if "####" in text:
        return text.split("####")[-1].strip()
    return None

def extract_answer(content):
    # Extract prediction inside <answer>...</answer>
    pred_match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    if pred_match:
        return pred_match.group(1).strip()
    return None

def verify_format(content):
    # Verify presence of <reasoning> and <answer> blocks
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    return 1.0 if re.search(pattern, content, re.DOTALL) else 0.0

def run_benchmark(model_name, num_samples=30):
    print(f"Loading model: {model_name}...")
    max_seq_length = 1024

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        load_in_4bit = True,
        fast_inference = False
    )
    model.eval()

    print("Loading GSM8K test dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split=f"test[:{num_samples}]")

    SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

    correct = 0
    format_compliant = 0
    total_thought_length = 0

    print(f"\nEvaluating on {num_samples} samples...")
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

        # Evaluate metrics
        pred = extract_answer(response)
        is_compliant = verify_format(response)

        # Calculate thought length (characters inside <reasoning> tags)
        reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", response, re.DOTALL)
        thought_len = len(reasoning_match.group(1)) if reasoning_match else 0
        total_thought_length += thought_len

        if is_compliant:
            format_compliant += 1
        if gt and pred and gt == pred:
            correct += 1

    accuracy = (correct / num_samples) * 100.0
    compliance = (format_compliant / num_samples) * 100.0
    avg_thought_len = total_thought_length / num_samples

    print("\n--- Benchmark Results ---")
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{num_samples})")
    print(f"Format Compliance: {compliance:.2f}% ({format_compliant}/{num_samples})")
    print(f"Average Thought Length: {avg_thought_len:.2f} chars")

    # Return metrics for report generation
    return {
        "accuracy": accuracy,
        "compliance": compliance,
        "thought_len": avg_thought_len
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path or repo name of the model to evaluate")
    parser.add_argument("--samples", type=int, default=30, help="Number of test samples")
    args = parser.parse_args()

    run_benchmark(args.model, args.samples)
