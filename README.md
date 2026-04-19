# 🔗 WitnessChain

**Trauma-informed, multilingual testimony infrastructure powered by Gemma 4**

![Domain](https://img.shields.io/badge/Domain-Safety%20%26%20Trust-red)
![Unsloth](https://img.shields.io/badge/Fine--tuned%20with-Unsloth-green)
![Gemma 4](https://img.shields.io/badge/Model-Gemma%204%2027B-blue)
![Gradio](https://img.shields.io/badge/UI-Gradio%204.x-orange)

---

## Global Conflict Coverage

WitnessChain is context-aware and pre-configured for high-conflict zones to ensure immediate deployment versatility:
- ** Somalia**: Somali language distress detection + UNHCR Mogadishu resources.
- ** Sudan**: Sudanese Arabic support + Khartoum Red Crescent integration.
- ** El Salvador**: MS-13/Gang context safety layer + San Salvador emergency links.
- ** Myanmar**: Burmese script support + Yangon-localized crisis anchors.

---

## What is WitnessChain?

WitnessChain enables human rights witnesses — regardless of language, connectivity, or legal literacy — to record structured testimonies that are safe to collect, safe to store, and ready for legal action.

It is the first AI system that uses Gemma 4's **256K context window** to hold 40+ testimonies in working memory and surface corroborating evidence across them — without a vector database.

---

## Setup (2 Steps)

### Step 1: Open the Colab Notebook

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/WitnessChain_Demo.ipynb)

### Step 2: Run All Cells

The notebook will:
1. Install dependencies
2. Authenticate with HuggingFace (set `HF_TOKEN` as a Colab secret)
3. Load Gemma 4 (27B on A100, 12B on T4 — automatic)
4. Launch the Gradio demo (Local-First by default)
    - Set `WITNESSCHAIN_SHARE_MODE=true` as an environment variable to generate a public URL.
    - Set `HF_TOKEN` (Reading Gemma 4 weights is always free).

---

## Demo

*Follow the Demo Flow below, or run the notebook to see WitnessChain in action.*

### Demo Flow (90 seconds)

1. Open the Gradio public URL
2. **New Testimony** tab → type in Arabic: "أنا رأيت الجنود يحرقون القرية في 12 مارس"
3. Observe TRUST-governed response (validation first, one question only)
4. Complete 3-turn interview
5. **Cross-Reference** tab → load sample testimonies → click "Analyse Corroboration"
6. Observe token counter + corroboration output
7. **Case Report** → download PDF
8. **Ethical Audit Log** → every prompt visible, TRUST compliance: 100%
9. Toggle model: Base Gemma 4 ↔ Fine-tuned WitnessChain

---

## Ethical Framework

### TRUST Dialogue Protocol

WitnessChain implements the TRUST framework — a research protocol for deploying LLMs in trauma-sensitive interview contexts. Every interaction follows strict rules: one question per turn, validation before extraction, no re-traumatisation, and immediate safe exit on distress signals. These are enforced as system prompt constraints injected before every Gemma 4 call.

### Trauma-Informed Computing (TIC)

Built on TIC principles from Haines et al. (CHI 2021): safety (no data leaves the device), trustworthiness (every prompt logged and visible), empowerment (witness owns their data), cultural humility (140-language native support), and peer support (crisis resources surfaced automatically). The system never positions itself as an authority.

### Amnesty International Accountability Toolkit

The system implements Amnesty International's Algorithmic Accountability guidelines: all prompts are public and version-controlled, extraction accuracy is tested across 3+ language families, distress detection triggers safe exit proactively, and witnesses have full data sovereignty. Audit responses are documented in `ETHICAL_AUDIT.md`.

---

## Technical Highlights

### 256K Context Window — Cross-Reference Engine

Gemma 4's 256K context window enables WitnessChain to pack **all testimonies** into a single inference call using XML-tagged document packing. The cross-reference engine surfaces:

- **Corroborated facts** across testimonies
- **Timeline reconstruction** from multiple accounts
- **Geographic clustering** of incidents
- **Discrepancies** between accounts
- **Evidence gaps** for investigators

The Gradio UI shows a live **token counter** demonstrating context window usage.

```
Context window usage: 8,247 / 262,144 tokens (3.1%)
```

### Model Architecture

| Component | Choice |
|---|---|
| Model | Gemma 4 27B-IT (4-bit quantised) |
| Fallback | Gemma 4 12B-IT (for T4 GPUs) |
| Quantisation | BitsAndBytes NF4 |
| Fine-tuning | Unsloth LoRA (rank 16) |
| Context | 256K tokens (native) |
| Languages | 140+ (native Gemma 4) |

---

## Special Technology Track — Unsloth Fine-Tuning

WitnessChain qualifies for the **Unsloth Special Technology Track** ($10,000 prize pool).

### What Was Fine-Tuned

Gemma 4 was fine-tuned on a synthetic dataset of **500-800 TRUST-framework dialogue examples** using Unsloth's `FastLanguageModel` with LoRA adapters:

- **LoRA rank:** 16
- **Target modules:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Training:** 3 epochs, AdamW 8-bit, gradient checkpointing
- **Runtime:** Google Colab A100

### Before/After Benchmark

| Metric | Base Gemma 4 | Fine-tuned WitnessChain |
|---|---|---|
| Single question per turn | ~60% | ~95% |
| Validation before extraction | ~45% | ~90% |
| No pressure language | ~70% | ~95% |
| **Overall TRUST Compliance** | **~50%** | **~90%** |

### Fine-Tuning Notebook

The complete fine-tuning pipeline is in [`WitnessChain_Unsloth_Finetune.ipynb`](notebooks/WitnessChain_Unsloth_Finetune.ipynb):

## 🐘 Ollama & Llama.cpp (10K Prize Track)

WitnessChain is fully optimized for the **Ollama & Llama.cpp Special Technology Track**.

### Running WitnessChain on Ollama
We provide a pre-configured `Modelfile` to load our fine-tuned weights (exported via GGUF) into Ollama:

1. **Export GGUF**: Run the export cell in `WitnessChain_Unsloth_Finetune.ipynb` to generate `witnesschain-gemma4.gguf`.
2. **Create Model**:
   ```bash
   ollama create witnesschain -f Modelfile
   ```
3. **Run**:
   ```bash
   ollama run witnesschain
   ```

### Data Sovereignty Gate
To ensure maximum privacy, the system defaults to **LOCAL-ONLY** mode. To enable public Gradio sharing for a demo, you must explicitly set the environment variable:
```bash
set WITNESSCHAIN_SHARE_MODE=true
python app.py
```

---

```python
from unsloth import FastLanguageModel

model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16, lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)
```

### Zero-Cost & Local-First Methodology

WitnessChain is designed for human rights defenders with zero budget:
- **HuggingFace Hub:** Public repository hosting is **free**.
- **Model Storage:** The system supports loading LoRA adapters directly from the local `models/witnesschain-lora-adapter` directory. 
- **No Cloud Dependencies:** You do **not** need a HuggingFace Repository ID to run the demo; the adapter can be trained and run entirely within the same local/ephemeral environment.

---

## Repository Structure

```
witnesschain/
├── app.py                      # Gradio 4-tab interface
├── requirements.txt            # Dependencies
├── core/
│   ├── interview_engine.py     # TRUST-governed dialogue manager
│   ├── extraction_engine.py    # Entity extraction pipeline
│   ├── crossref_engine.py      # 256K context cross-reference
│   ├── distress_detector.py    # TIC safety layer
│   └── report_generator.py     # PDF/DOCX case reports
├── models/
│   ├── gemma_loader.py         # 4-bit quantised model loader
│   └── unsloth_adapter.py      # LoRA adapter loader
├── prompts/
│   ├── trust_system_prompt.txt # TRUST framework (public, versioned)
│   ├── extraction_prompt.txt   # Entity extraction prompt
│   └── crossref_prompt.txt     # Cross-reference prompt
├── data/
│   ├── sample_testimonies/     # 5 synthetic multilingual testimonies
│   └── crisis_resources.json   # Per-language crisis hotlines
├── notebooks/
│   ├── WitnessChain_Demo.ipynb
│   └── WitnessChain_Unsloth_Finetune.ipynb
├── ACCOUNTABILITY.md
├── ETHICAL_AUDIT.md
└── README.md
```

---

## License

This project is a research prototype built for the Kaggle Gemma 4 Good Hackathon. See `ACCOUNTABILITY.md` and `ETHICAL_AUDIT.md` for ethical governance documentation.
