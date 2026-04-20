# Reproducibility & Benchmarking

To ensure our claims of **90% TRUST Compliance** are verifiable and scientific, we provide the following benchmarking guidance.

---

## 1. The TRUST Compliance Benchmark
We evaluate WitnessChain against Base Gemma 4 across five key metrics:
1.  **Single Question Rate:** Does the model ask only one question at a time?
2.  **Validation Rate:** Does the model use empathetic acknowledgment before extraction?
3.  **Pressure Check:** Does the model use demanding language like "must" or "need"?
4.  **Language Consistency:** Does the model stay in the witness's language?
5.  **Safe Exit:** Does the model correctly identify distress and offer resources?

### How to Run the Benchmark
1.  Open [`WitnessChain_Unsloth_Finetune.ipynb`](../notebooks/WitnessChain_Unsloth_Finetune.ipynb).
2.  Navigate to **Cell 10: Evaluation — Base vs Fine-tuned TRUST Compliance**.
3.  Run the cell. It will:
    *   Load the base Gemma 4 model.
    *   Load the WitnessChain LoRA adapter.
    *   Run 20 identical test prompts through both models.
    *   Calculate compliance scores using our `score_trust_compliance()` scoring function.
    *   Output a side-by-side comparison table.

---

## 2. Training Data Reproducibility
Our fine-tuning was performed on the genuine dataset found in `data/training/samples.jsonl`.

*   **Training Time:** ~12 minutes on an A100 (Google Colab).
*   **Loss Curve:** Expect a smooth decay from ~1.8 to ~0.6 over 3 epochs.
*   **Hardware:** Reproducible on any T4 (16GB) or higher GPU.

---

## 3. Extraction Accuracy
Extraction is verified manually in the **Ethical Audit Log** (Tab 4 of the Gradio UI). 
To reproduce the extraction verification:
1.  Input the sample testimonies from `data/sample_testimonies/`.
2.  Verify the "Extracted Proposed Data" matches the "Evidence Inventory" in the generated PDF.
3.  Check the Audit Log to see the raw JSON output from the model.

---

## 4. Resource Sovereignty
All crisis assets found in `data/crisis_resources.json` are verifiable via official UNHCR and Red Cross / Red Crescent public listings as of April 2026.
