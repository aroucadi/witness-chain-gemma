# WitnessChain — Accountability Framework

**Version:** 1.0
**Last Updated:** 2026-04-18
**Governing Standards:** Amnesty International Algorithmic Accountability Toolkit

---

## What decisions does this system make autonomously?

WitnessChain makes the following automated decisions during operation:

1. **Language Detection:** Automatically detects the language of witness input and responds in the same language. This is a convenience decision with no downstream legal consequence.

2. **Distress Signal Detection:** Automatically scans every user message for distress signals using a **Dual-Layer check**:
    - **Layer 1 (Keyword):** High-speed matching across 9 languages (English, Arabic, Sudanese Arabic, French, Swahili, Tigrinya, Somali, Burmese, Spanish).
    - **Layer 2 (Semantic):** LLM-driven fallback using Gemma 4 to catch complex distress or unsupported languages.
If detected, the system autonomously triggers the safe exit protocol — ending the interview and displaying crisis resources. This errs deliberately on the side of caution: false positives (premature exits) are acceptable; false negatives (missed distress) are not.

3. **Structured Data Extraction:** Automatically extracts dates, locations, incident types, and other structured fields from free-text testimony. These extractions are **proposals** — they are always presented to the user for review and can be edited or deleted before inclusion in any report.

4. **Cross-Reference Analysis:** Automatically identifies corroborating facts, timeline patterns, discrepancies, and evidence gaps across multiple testimonies. These are analytical outputs, not conclusions of fact.

5. **Interview Phase Progression:** Automatically advances through interview phases (Context → Events → People → Evidence) based on exchange count. The witness can skip or revisit any phase.

---

## What decisions require human review before action?

1. **All legal action:** WitnessChain generates reports; it does not take action. Every report includes a disclaimer that independent verification is required before use in legal proceedings.

2. **Testimony submission:** No testimony is transmitted to any external party without explicit, per-submission consent from the witness.

3. **Extracted data verification:** All extracted fields are presented to the witness for review. The full transcript is preserved alongside structured extraction so investigators can verify accuracy.

4. **Report distribution:** Generated PDF/DOCX case reports are downloaded locally by the user. The system has no mechanism to distribute reports automatically.

---

## Who is responsible if the system makes an error?

**Responsibility chain:**

| Layer | Responsible Party | Scope |
|---|---|---|
| **System Design** | Development team | Prompt design, safety logic, extraction accuracy, distress detection coverage |
| **Deployment** | Deploying organisation | Usage policies, access control, data handling procedures, operator training |
| **Downstream Action** | Human investigator | All decisions made based on WitnessChain outputs, verification of extracted data, legal proceedings |

**Specific error scenarios:**

- **Missed distress signal:** Development team responsibility — expand keyword list, improve detection.
- **Incorrect extraction:** Mitigated by human review requirement. If unreviewed data enters a legal record, the deploying organisation is responsible for process failure.
- **Cultural misalignment:** Development team responsibility — expand fine-tuning dataset, add cultural sensitivity review.
- **Data breach:** Mitigated by ephemeral-only storage design. If deployed with persistent storage, the deploying organisation is responsible for data protection.

---

## How can a witness contest or correct their testimony?

The system provides the following capabilities, accessible at any time during or after an interview:

1. **Review:** The witness can review the full transcript alongside extracted structured data before any report is generated.

2. **Delete:** The witness can delete their entire testimony at any time. Deletion is immediate and irreversible. The system confirms deletion with the user before proceeding.

3. **Download:** The witness can download their full testimony and extracted data in JSON format at any time.

4. **Per-field editing:** Future versions will support per-field review and redaction of extracted data before report generation. In the current prototype, the full transcript is preserved alongside all structured extraction so investigators can manually verify and correct any field.

5. **No persistent storage by default:** All data is stored in Colab ephemeral memory. When the Colab runtime ends, all data is automatically destroyed. The only way data persists is if the witness explicitly downloads it.

---

## Data Sovereignty Statement

The witness owns every byte of their testimony. WitnessChain is a tool, not a custodian. The system has no cloud storage, no database, and no analytics pipeline. Data exists only in the Colab session's local memory and is destroyed when the session ends unless the witness explicitly exports it.

No data is ever transmitted to external APIs, cloud services, or third parties. All model inference is performed locally within the runtime. For public demonstrations, reverse-proxy UI traffic is disabled by default and requires explicit activation via `WITNESSCHAIN_SHARE_MODE=true` to prevent accidental data exposure.
