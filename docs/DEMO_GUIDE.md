# WitnessChain — 3-Minute Competition Demo Guide

This guide provides a structured script for your 3-minute submission video. We recommend following this flow to ensure you hit every scoring criterion for the **Unsloth**, **Ollama**, and **llama.cpp** prize tracks.

---

## Part 1: The Multi-Track Flex (00:00 – 00:45)

**Narrative:** Show the submission artifacts immediately.
1.  **Opening Frame:** Show your Kaggle Writeup with the **HuggingFace Adapter** and **GGUF Weights** links highlighted.
2.  **Repo Scan:** Quickly scroll through the [README.md](README.md) to show the "Submission Portfolio."
3.  **Local-First Verification:** Show the terminal running the `app.py` in the background with `WITNESSCHAIN_SHARE_MODE=false`. Say: *"WitnessChain is architected for maximum data sovereignty. No testimony data ever leaves the local runtime."*

---

## Part 2: The Unsloth "Killer Demo" (00:45 – 01:45)

**Narrative:** This is the most critical part of the $10K Unsloth prize. You MUST show the LoRA behavioral shift.

1.  **Select "Base Gemma 4":** 
    *   **Prompt:** *"The soldiers came at 5 AM. I was terrified. They took my husband and destroyed our garden. What do I do now? Who are they? Will you help me?"*
    *   **Base Behavior:** Show Base Gemma 4 generating a long, potentially overwhelming response with multiple questions (e.g., "Where did they take him? What were they wearing? Do you have ID?").
2.  **Toggle to "Fine-tuned WitnessChain":**
    *   Observe the **Model Status** update to *"RELOADING MODEL..."*
    *   **Same Prompt:** Use the same witness input.
    *   **Fine-tuned Behavior:** Show WitnessChain responding with **validation first** then **one single, gentle question**.
    *   **Voiceover:** *"Our Unsloth fine-tuning improved TRUST compliance from 50% to 90%. As you see, the model now focuses on witness validation and cognitive load reduction."*

---

## Part 3: The 256K Context & Ethical Audit (01:45 – 03:00)

1.  **Cross-Reference Engine:** 
    *   Switch to the **🔬 Cross-Reference Analysis** tab.
    *   Click *"Analyse Corroboration."*
    *   Show the result identifying overlapping details (e.g., "Sidi Marzouq," "Green Uniforms") across 5 testimonies simultaneously.
    *   **Voiceover:** *"Leveraging Gemma 4's 256K context, we pack full case files to find corroboration without the lossy retrieval issues of RAG."*
2.  **Ethical Audit Log:**
    *   Switch to the **🔒 Ethical Audit Log** tab.
    *   Click *"Refresh Audit Log"* to show the logged prompts.
    *   **Voiceover:** *"Following Amnesty International guidelines, every prompt is logged for human rights accountability."*
3.  **Closing:**
    *   Generate a PDF Case Report.
    *   Closing Frame: *"Record the truth. Safely. Everywhere. This is WitnessChain."*

---

## Technical Setup Tips
*   **A100/L4 Recommended:** Ensure you have enough VRAM (35GB+) for the 27B model to avoid 12B fallback during the video.
*   **Full Screen:** Hide browser tabs to focus on the Gradio interface.
*   **Audio:** Use a clear microphone; the technical judges value the clarity of your architectural explanations.
