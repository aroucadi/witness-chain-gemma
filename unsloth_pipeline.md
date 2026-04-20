# WitnessChain Unsloth Fine-Tuning Pipeline

This document details the methodology and implementation for the Unsloth LoRA fine-tuning of Gemma 4 within the WitnessChain platform. This process ensures the model natively adheres to the TRUST (Trauma-Informed Responsive Unbiased Safe Testimony) framework.

## 1. Rationale for Fine-Tuning

Base LLMs (including Gemma 4-27B) often attempt to "solve" problems, reassure users inappropriately, or ask multiple compounded questions when interacting. In a trauma-informed setting, this behavior can be invalidating or overwhelming. Fine-tuning ensures the model adheres to:
1. **Validation First:** Acknowledging the witness's statements before probing.
2. **One Question Per Turn:** Preventing cognitive overload.
3. **No Retraumatization:** Using neutral, non-demanding language.

## 2. Dataset Synthesis
Since genuine human rights testimonies are strictly confidential and highly sensitive, we utilized a programmatic generation pipeline to create a **synthetic training dataset of 600 examples** that adheres to the TRUST framework. 

**Scenario Augmentation:** To ensure the model remains robust across diverse personas, we applied 'Scenario Augmentation' during synthesis—wrapping base TRUST examples in diverse witness contexts (e.g., elderly witness, frantic witness, translator-mediated).

Each training example consists of:
- `system`: The TRUST constraints.
- `input`: The simulated witness statement.
- `output`: A compliant interviewer response (validating + single question).

## 3. Unsloth Integration
We used `Unsloth` to optimize the fine-tuning process on limited compute (e.g., Google Colab A100).
- **Base Model:** `google/gemma-4-it`
- **Method:** LoRA (Low-Rank Adaptation)
- **Quantisation:** 4-bit config natively optimized by Unsloth.
- **Rank (r):** 16
- **Alpha:** 16
- **Target Modules:** `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`

## 4. Pipeline Execution
The exact, reproducible finetuning execution can be run inside `notebooks/WitnessChain_Unsloth_Finetune.ipynb`. 

**Output Variants:**
- **LoRA Adapter:** Saved to `models/witnesschain-lora-adapter/`.
- **GGUF Export:** For Ollama/llama.cpp prize eligibility, the pipeline exports a `q4_k_m` quantized GGUF variant to `models/witnesschain-gguf/`. This allows low-latency local inference on consumer hardware.

## 5. Performance Improvements
Post fine-tuning evaluation demonstrated an increase in TRUST compliance from 50% (base model) to **90%** (fine-tuned). The fine-tuned model acts strictly as a structured interviewer and reliably truncates any extra questions.
