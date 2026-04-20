# WitnessChain — Production Roadmap (v2.0)

WitnessChain is currently in its **Research Prototype** phase, optimized for the Kaggle Gemma 4 Good Hackathon. This roadmap outlines the transition from a specialized demo to a globally deployable infrastructure for human rights documentation.

---

## Phase 1: High-Fidelity Prototype (Current)

*   **Behavioral Alignment:** 90% TRUST Framework compliance via Unsloth LoRA fine-tuning.
*   **Zero-Trace Architecture:** Ephemeral in-memory storage to ensure maximum witness safety (Data Sovereignty Pillar).
*   **Local-First Inference:** Standalone 256K context analysis via Ollama and llama.cpp.
*   **Multilingual Core:** Dialect-aware distress detection in 9 key languages.

---

## Phase 2: Enterprise Scaling & Hardening

*   **Multi-User Session Isolation:** Migrating from global state to `gr.State()` per-session memory, allowing hundreds of simultaneous interviews on a single server.
*   **Stateful Session Resumption:** Implementing a secure "Save & Return" feature via encrypted session tokens.
*   **Hardware-Encrypted Persistence:** Adding support for local, AES-256 encrypted storage on the investigator's device to build longitudinal case files.
*   **Standalone Deployment:** Packaging the full Gradio + Gemma 4 stack into a single, offline-installable Docker container.

---

## Phase 3: Field Validation & Community Co-Design

*   **NGO Pilot Program:** Partnering with regional human rights defenders in Somalia, Sudan, and Myanmar to conduct real-world field tests.
*   **Expert Prompt Refinement:** Collaborating with trauma counsellors to further refine the "Understanding" (Validation) turn of the TRUST framework.
*   **Expanded Dialect Coverage:** Improving resource lookup and distress messages for Burmese, Spanish, and Somali to match the current Arabic/French/English fidelity.

---

## Phase 4: Legal & Advocacy Integration

*   **CMS Integration:** Building direct export pipelines for industry-standard Case Management Systems (e.g., Uwazi).
*   **Legal Chains of Custody:** Implementing cryptographic hashing and timestamping for every extracted fact to ensure admissibility in legal proceedings.
*   **Evidence Visualization:** Adding geographic heatmap and timeline visualization layers directly in the Gradio interface for faster investigator analysis.

---

**Interested in contributing or deploying WitnessChain?**  
See `ACCOUNTABILITY.md` for our governing ethics and `docs/DEPLOYMENT.md` for technical setup instructions.
