# Deployment & Setup Guide

WitnessChain is designed to be highly portable, running everywhere from a high-end A100 GPU in the cloud to a standard laptop on your desk.

---

## 1. Cloud Deployment (The "Easy" Demo)

The fastest way to see WitnessChain in action is via Google Colab.

### Step 1: WitnessChain Demo Notebook
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](../notebooks/WitnessChain_Demo.ipynb)
1.  Open the notebook.
2.  Set your `HF_TOKEN` in the Colab "Secrets" (the key icon).
3.  **Run All.**
4.  The Gradio UI will launch. If you need a public URL for a demo, set `WITNESSCHAIN_SHARE_MODE=true` in the environment variables cell.

---

## 2. Local Deployment (Ollama & Llama.cpp)

WitnessChain supports the **Ollama** and **llama.cpp** prize tracks. This makes the system "Local-First" and connectivity-independent.

### Prerequisite: Generate GGUF Weights
You must first run the conversion cell in the [`WitnessChain_Unsloth_Finetune.ipynb`](../notebooks/WitnessChain_Unsloth_Finetune.ipynb) notebook. This converts the LoRA weights into a portable `.gguf` format.

### Running with Ollama
1.  Install [Ollama](https://ollama.ai).
2.  Navigate to the repository root.
3.  Create the model from our `Modelfile`:
    ```bash
    ollama create witnesschain -f Modelfile
    ```
4.  Launch the model:
    ```bash
    ollama run witnesschain
    ```

### Running with Llama.cpp
1.  Download the latest [llama.cpp](https://github.com/ggerganov/llama.cpp) binary.
2.  Run inference using the exported GGUF file:
    ```bash
    ./main -m models/witnesschain-gemma4.gguf -p "You are WitnessChain, a trauma-informed assistant..."
    ```

---

## 3. Hardware Requirements

| Platform | Recommended Model | Minimum RAM/VRAM |
|---|---|---|
| **Cloud (Standard)** | Gemma 4 12B | 16GB (T4 GPU) |
| **Cloud (High-end)** | Gemma 4 27B | 40GB (A100 GPU) |
| **Local Laptop** | Gemma 4 12B (Q4_K_M) | 16GB Unified Memory / RAM |
| **Local Desktop** | Gemma 4 27B (Q4_K_M) | 24GB VRAM (RTX 3090/4090) |

---

## 4. Fine-Tuning Your Own Version
If you want to train WitnessChain on your own community-specific TRUST examples:
1.  Open [`WitnessChain_Unsloth_Finetune.ipynb`](../notebooks/WitnessChain_Unsloth_Finetune.ipynb).
2.  Update `data/training/samples.jsonl` with your examples.
3.  Run the training cells.
4.  Push the new adapter to HuggingFace or use it locally.
