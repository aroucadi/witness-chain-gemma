# 🔗 WitnessChain

**Trauma-informed, multilingual testimony infrastructure powered by Gemma 4**

> "Record the truth. Safely. Everywhere."

[![Kaggle Gemma 4 Good](https://img.shields.io/badge/Kaggle-Gemma%204%20Good-blue)](https://www.kaggle.com/competitions/gemma-4-good-hackathon)
![Gemma 4](https://img.shields.io/badge/Model-Gemma%204%2027B-blue)
![Context](https://img.shields.io/badge/Context-256K-blue)
[![Unsloth](https://img.shields.io/badge/Fine--tuned%20with-Unsloth-green)](https://github.com/unslothai/unsloth)
![Gradio](https://img.shields.io/badge/UI-Gradio%204.x-orange)
![Domain](https://img.shields.io/badge/Domain-Human%20Rights-red)

---

## 📖 Documentation Map
Explore the depth of WitnessChain:

*   **[Core Concepts](docs/CONCEPTS.md):** Deep dive into the **TRUST Framework**, **Trauma-Informed Computing (TIC)**, and **LoRA** fine-tuning.
*   **[Deployment Guide](docs/DEPLOYMENT.md):** How to run WitnessChain on **Google Colab**, **Ollama**, or **llama.cpp**.
*   **[Reproducibility](docs/REPRODUCIBILITY.md):** Step-by-step guide to verifying our **90% TRUST Compliance** benchmark.
*   **[Ethical Governance](ETHICAL_AUDIT.md):** Our audit against the **Amnesty International Accountability Toolkit**.

---

## ⚖️ The Mission
WitnessChain enables human rights witnesses — regardless of language, connectivity, or digital literacy — to record structured testimonies that are **safe to collect**, **sovereign to store**, and **ready for legal advocacy**.

It is designed to solve the "Evidence Bottleneck": where trauma and language barriers prevent critical accounts of human rights abuses from reaching investigators.

---

## 🏆 For Judges: Prize Track Evidence

WitnessChain is architected to stack three specialized technology tracks:

### 1. Unsloth Fine-Tuning Track ($10K)
*   **Innovation:** We fine-tuned Gemma 4 on a synthetic dataset of **800 TRUST-framework dialogue examples** using Unsloth's LoRA adapters.
*   **Result:** Boosted TRUST compliance (empathy, single-question turns) from **50% to 90%**.
*   **Proof:** See [`WitnessChain_Unsloth_Finetune.ipynb`](notebooks/WitnessChain_Unsloth_Finetune.ipynb).

### 2. Ollama Deployment Track ($10K)
*   **Innovation:** Full local deployment capability with a pre-configured `Modelfile` and trauma-informed system prompt.
*   **Proof:** See [Modelfile](Modelfile).

### 3. Llama.cpp Efficiency Track ($5K)
*   **Innovation:** Native GGUF export support for high-efficiency inference on consumer hardware.
*   **Proof:** Export logic in the Fine-Tuning notebook.

---

## 🛠️ Technical Highlights

### 🚀 256K Context "Evidence Packing"
Unlike generic RAG-style systems that retrieve "snippets," WitnessChain uses Gemma 4's **256K context** to pack dozens of full testimonies into a single inference call. The **Cross-Reference Engine** identifies corroborations and timeline discrepancies with global awareness of the entire case file.

### 🛡️ Local-First Data Sovereignty
No data ever leaves the witness's device. 
*   **Zero-Cloud Inference:** All extractions and analyses happen in the local runtime.
*   **Ephemeral Memory:** Data lives only in the session and is destroyed on exit.
*   **Sovereignty Gate:** Remote sharing is disabled by default to prevent accidental data exfiltration.

---

## 🌍 Global Presence (Localized Safety)
WitnessChain provides dialect-aware distress detection and localized crisis resources for high-conflict regions:
*   🇸🇴 **Somali** (`so`)  |  🇸🇩 **Sudanese Arabic** (`ar-SD`)
*   🇸🇻 **Spanish** (`es`) |  🇲🇲 **Burmese** (`my`)
*   ...and native 140-language support for all interview phases.

---

## 🚀 Quick Start (60 Seconds)
1. **Open the Demo:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](notebooks/WitnessChain_Demo.ipynb)
2. **Run All Cells:** The Gradio UI will launch automatically.
3. **Launch the Adapter:** Toggle between "Base Gemma 4" and "Fine-tuned WitnessChain" to see the LoRA behavior shift.

---

### [Repository Structure]

```
witnesschain/
├── app.py                      # Gradio 4-tab interface
├── core/
│   ├── interview_engine.py     # TRUST-governed dialogue
│   ├── crossref_engine.py      # 256K context analysis
│   └── report_generator.py     # PDF/DOCX case reports
├── docs/                       # New: Technical Deep Dives
├── notebooks/                  # Demo & Fine-Tuning pipelines
├── ACCOUNTABILITY.md           # Governance framework
└── ETHICAL_AUDIT.md            # Bi-annual safety audit
```

---

*WitnessChain is a research prototype built for the Kaggle Gemma 4 Good Hackathon. It is not a replacement for legal counsel or professional witness protection services.*
