import os
import json
import subprocess

def create_inference_notebook():
    # Dynamically find the script's directory to be fully self-contained
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(current_dir, "inference_test.py")

    # Read test script
    with open(test_file, "r") as f:
        code_content = f.read()

    # Define Kaggle Notebook structure
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Install Unsloth, TRL and requirements\n",
                    "!pip install --upgrade pip\n",
                    "!pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"\n",
                    "!pip install --no-deps trl peft bitsandbytes transformers\n",
                    "# Re-upgrade numpy to match pre-loaded version 2.4.6\n",
                    "!pip install numpy==2.4.6\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    code_content
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    # Write notebook file
    notebook_path = os.path.join(current_dir, "inference_notebook.ipynb")
    with open(notebook_path, "w") as f:
        json.dump(notebook, f, indent=2)
    print(f"Created notebook at {notebook_path}")

    # Write metadata file referencing the previous kernel's output
    meta = {
        "id": "kevinli2005/grpo-inference-test",
        "title": "GRPO Inference Test",
        "code_file": "inference_notebook.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_tpu": "false",
        "enable_internet": "true",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": ["kevinli2005/grpo-deepseek-r1-t4"],
        "model_sources": []
    }

    meta_path = os.path.join(current_dir, "kernel-metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Created metadata at {meta_path}")

    # Run Kaggle push command with NvidiaTeslaT4 accelerator
    print("Pushing inference notebook to Kaggle with T4 GPU...")
    try:
        res = subprocess.run(
            ["kaggle", "kernels", "push", "-p", current_dir, "--accelerator", "NvidiaTeslaT4"],
            capture_output=True,
            text=True,
            check=True
        )
        print("Kaggle CLI Output:")
        print(res.stdout)
    except subprocess.CalledProcessError as e:
        print("Error pushing kernel to Kaggle:")
        print(e.stderr)
        raise e

if __name__ == "__main__":
    create_inference_notebook()
