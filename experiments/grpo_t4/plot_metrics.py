import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def generate_plots(csv_path, output_dir):
    print(f"Reading metrics from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"Error: file {csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)

    # Standardize column headers (remove spaces or lowercase keys)
    df.columns = [c.strip() for c in df.columns]

    # 1. Plot Loss Curve
    if "loss" in df.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df["step"], df["loss"], label="Training Loss", color="royalblue", linewidth=1.5)
        plt.title("GRPO Alignment Training Loss", fontsize=14, fontweight="bold")
        plt.xlabel("Step", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        loss_plot = os.path.join(output_dir, "loss_curve.png")
        plt.savefig(loss_plot, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved loss curve to {loss_plot}")

    # 2. Plot Reward Curves
    reward_cols = [c for c in df.columns if "reward" in c.lower()]
    if reward_cols:
        plt.figure(figsize=(12, 6))
        for col in reward_cols:
            label_name = col.replace("reward_", "").replace("_func", "").replace("reward", "").strip("_")
            plt.plot(df["step"], df[col], label=f"Reward: {label_name}", linewidth=1.5)

        plt.title("GRPO Objective Rewards Convergence", fontsize=14, fontweight="bold")
        plt.xlabel("Step", fontsize=12)
        plt.ylabel("Reward Value", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        reward_plot = os.path.join(output_dir, "rewards_curve.png")
        plt.savefig(reward_plot, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved rewards curve to {reward_plot}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="/kaggle/working/metrics.csv", help="Path to metrics CSV file")
    parser.add_argument("--out", type=str, default="/kaggle/working/plots", help="Output directory for plots")
    args = parser.parse_args()

    generate_plots(args.csv, args.out)
