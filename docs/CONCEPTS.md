# Core Concepts & Frameworks

WitnessChain is built on three pillars: **Technical Innovation**, **Trauma-Informed Design**, and **Ethical Governance**.

---

## 1. Technical Innovation: Unsloth & LoRA

### The "Cheap Shortcut" to Intelligence
Traditional fine-tuning (changing every parameter in a model) is prohibitively expensive, requiring thousands of dollars in GPU time. WitnessChain uses **LoRA (Low-Rank Adaptation)** to achieve professional-grade results on a zero budget.

*   **How it works:** Instead of retraining the whole model, we add a tiny, lightweight "adapter" layer. This adapter contains the specific behaviors we want (trauma-informed responses).
*   **The Unsloth Advantage:** We use the [Unsloth](https://github.com/unslothai/unsloth) library to train this adapter 2x faster and with 70% less memory. This allowed us to fine-tune a massive 27B model on standard consumer hardware (Google Colab).
*   **Impact:** This ensures that human rights organizations can customize high-end AI models without needing a Silicon Valley budget.

### 256K Context Packing
Gemma 4's massive 256K context window is the "engine" of our Cross-Reference system. 
*   **Native Corroboration:** Most systems use a "Vector Database" (RAG) which can lose nuance. We pack dozens of full testimonies into a single inference call using XML-tagged document structure.
*   **Global Analysis:** This allows the model to see the *entire* case at once, identifying timeline patterns and evidence gaps that a human investigator might miss across hundreds of pages of notes.

---

## 2. Trauma-Informed Design (TIC)

WitnessChain adheres to the **Trauma-Informed Computing (TIC)** framework, ensuring the technology respects the psychological state of the witness.

### The TRUST Framework
Every interaction is governed by the TRUST protocol:
1.  **T - Transparency:** The witness is told exactly what the AI is doing.
2.  **R - Responsiveness:** The system identifies distress immediately and triggers a Safe Exit.
3.  **U - Understanding:** Validation of the witness's experience before asking for data.
4.  **S - Simplicity:** One question at a time. No complex forms.
5.  **T - Trustworthiness:** Data sovereignty. Local-first architecture.

### Safe Exit & Distress Detection
Our dual-layer safety system ensures no witness is left in a state of distress:
*   **Layer 1 (Keywords):** Instant detection of crisis terms in 9 languages.
*   **Layer 2 (Semantic):** Gemma 4 uses its 140-language capability to catch "heavy" emotional context that keywords might miss.

---

## 3. Ethical Governance

WitnessChain is not just a tool; it's a statement on data sovereignty.

*   **Data Ownership:** The witness owns every byte. The system has no cloud storage or centralized database.
*   **Amnesty International Standards:** We follow the Algorithmic Accountability Toolkit. Every prompt, every extraction, and every safety check is logged in the **Ethical Audit Log** for researchers to verify.
*   **Neutrality:** The system uses neutral, extraction-focused prompts to prevent "leading" the witness or injecting observer bias into the testimony record.
